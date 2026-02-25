import json
import re
import logging
import hashlib
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings
from config import GEMINI_API_KEYS
import db
from db import sessions, templates, available_fields, collection_meta, db_mongo, ALLOWED_FIELDS, TEMPLATE_TO_COLLECTION
from datetime import datetime
import asyncio
from google.api_core.exceptions import ResourceExhausted


GEMINI_MODELS = [
    "models/gemini-2.5-flash",
    "models/gemini-2.5-flash-lite",
    "models/gemma-3-1b",
    "models/gemma-3-4b",
    "models/gemma-3-12b",
    "models/gemma-3-27b",
    "models/gemini-robotics-er-1.5-preview",
    "models/gemini-2.5-flash-tts",
]

from google import genai
from google.genai import Client, types


logger = logging.getLogger(__name__)

REPLY_SAFE_LIMIT = 4000
MAX_HISTORY = 20


# --- utility helpers
def _hash_key(uid: int, text: str) -> str:
    h = hashlib.sha1()
    h.update(f"{uid}:{text}".encode("utf-8"))
    return h.hexdigest()


def _trim_text(txt: str, n=600):
    txt = txt.strip()
    return txt if len(txt) <= n else txt[:n] + "..."


def _safe_json_extract(txt: str):
    try:
        m = re.search(r'(\[.*\]|\{.*\})', txt, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception:
        return None
    return None


# --- Service classes
class CacheService:
    def __init__(self, db_client=None):
        self._col = (db_client or db_mongo)["ai_cache"]

    async def get(self, key: str):
        doc = await self._col.find_one({"_id": key})
        return doc["value"] if doc else None

    async def set(self, key: str, val: Any):
        await self._col.update_one({"_id": key}, {"$set": {"value": val, "created_at": datetime.utcnow()}}, upsert=True)


class HistoryService:
    def __init__(self, db_client=None, max_history: int = MAX_HISTORY):
        self._col = (db_client or db_mongo)["sessions"]
        self._max = max_history

    async def get(self, uid: int) -> List[Dict]:
        s = await self._col.find_one({"user_id": uid})
        return s.get("history", []) if s else []

    async def append(self, uid: int, role: str, text: str):
        s = await self._col.find_one({"user_id": uid}) or {"user_id": uid, "history": []}
        s["history"].append({"role": role, "parts": [text]})
        s["history"] = s["history"][-self._max:]
        await self._col.update_one({"user_id": uid}, {"$set": {"history": s["history"]}}, upsert=True)


class VectorStoreClient:
    def __init__(self, chroma_collection=None):
        self._col = chroma_collection or db.async_chroma_collection

    async def query(self, text: str, k: int = 4):
        if self._col is None:
            return None
        try:
            return await self._col.query(query_texts=[text], n_results=k)
        except Exception as e:
            logger.warning("Async Chroma search error: %s", e)
            return None


class LLMClient:
    def __init__(self, api_keys: Optional[List[str]] = None, models: Optional[List[str]] = None):
        keys = api_keys if api_keys is not None else GEMINI_API_KEYS or []
        self._clients = [Client(api_key=k, http_options=types.HttpOptions(api_version="v1")) for k in keys]
        self._models = models if models is not None else GEMINI_MODELS

    async def generate_text(self, contents: types.Content) -> str:
        errors: List[str] = []
        final_response: Optional[str] = None

        for client in self._clients:
            try:
                async with client.aio as aclient:
                    for model_name in self._models:
                        try:
                            response = await aclient.models.generate_content(model=model_name, contents=contents)
                            text = (response.text or "").strip()
                            if text:
                                final_response = text
                                logger.info("[%s] success", model_name)
                                break
                            else:
                                err = f"{model_name}: empty response"
                                logger.warning(err)
                                errors.append(err)
                        except Exception as e:
                            err = f"{model_name} failed: {type(e).__name__}: {e}"
                            logger.warning(err)
                            errors.append(err)
            except Exception as e:

                logger.warning("LLM client error: %s", e)

            if final_response:
                break

        if not final_response:
            logger.error("All fallback models failed: %s", "; ".join(errors))
            return "⚠️ Усі моделі недоступні. Спробуйте пізніше."

        return final_response


class PipelineService:
    def __init__(self, allowed_fields_map: Dict[str, List[str]] = ALLOWED_FIELDS):
        self.allowed_map = allowed_fields_map

    def clean_pipeline(self, pipeline: List[Dict], template_name: str) -> Optional[List[Dict]]:

        collection_name = TEMPLATE_TO_COLLECTION.get(template_name)
        allowed = self.allowed_map.get(collection_name, None)
        if allowed is None:
            return pipeline

        cleaned: List[Dict] = []

        for stage in pipeline:
            if not isinstance(stage, dict):
                continue

            if "$match" in stage:
                ok_fields = {}
                for key, val in stage["$match"].items():
                    if key not in allowed:
                        continue
                    if isinstance(val, dict) and "$regex" in val and not isinstance(val["$regex"], str):
                        continue
                    ok_fields[key] = val
                if ok_fields:
                    cleaned.append({"$match": ok_fields})
                else:
                    continue

            elif "$project" in stage:
                ok_fields = {k: v for k, v in stage["$project"].items() if k in allowed}
                if ok_fields:
                    cleaned.append({"$project": ok_fields})
                else:
                    continue
            else:
                cleaned.append(stage)

        if not cleaned:
            logger.warning("[CLEAN PIPELINE WARNING] All stages removed. Returning original pipeline.")
            return pipeline

        return cleaned


class AIAssistant:
    def __init__(self, llm: Optional[LLMClient] = None, vector: Optional[VectorStoreClient] = None,
                 cache: Optional[CacheService] = None, history: Optional[HistoryService] = None,
                 pipeline_svc: Optional[PipelineService] = None):
        self.llm = llm or LLMClient()
        self.vector = vector or VectorStoreClient()
        self.cache = cache or CacheService()
        self.history = history or HistoryService()
        self.pipeline_svc = pipeline_svc or PipelineService()

    # --- LLM wrapper 
    async def gemini_call(self, uid: int, prompt: str, use_history: bool = False) -> str:
        key = _hash_key(uid, prompt)
        cached = await self.cache.get(key)
        if cached:
            return cached

        hist = await self.history.get(uid)
        hist = hist[-1:] if use_history else []

        if hist:
            hist_text = "\n\n".join([f"{h['role']}: {h['parts'][0]}" for h in hist])
            full_prompt = f"{hist_text}\n\nUser: {prompt}"
        else:
            full_prompt = prompt

        contents = types.Content(role="user", parts=[types.Part(text=full_prompt)])
        text = await self.llm.generate_text(contents)

        await self.history.append(uid, "user", prompt)
        await self.history.append(uid, "model", text)
        await self.cache.set(key, text)

        return text


    async def vector_context_from_chroma(self, query_text: str, k: int = 4) -> str:
        results = await self.vector.query(query_text, k=k)
        if not results or not results.get("documents"):
            return ""

        snippets = []
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        for doc, meta in zip(docs, metas):
            part = f"- {meta.get('name','Невідомий товар')} ({meta.get('category','без категорії')}, {meta.get('price','ціна не вказана')}₴)\n{doc[:300]}..."
            snippets.append(part)
        return "\n\nСемантично схожі товари:\n" + "\n".join(snippets)

    async def vector_docs_from_chroma(self, query_text: str, k: int = 4) -> List[str]:
        results = await self.vector.query(query_text, k=k)
        if not results or not results.get("documents"):
            return []
        snippets = []
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        for doc, meta in zip(docs, metas):
            part = f"- {meta.get('name','Невідомий товар')} ({meta.get('category','без категорії')}, {meta.get('price','ціна не вказана')}₴)\n{doc[:300]}..."
            snippets.append(part)
        return snippets


    async def analyze_action(self, uid: int, msg: str):
        p = f"""
Користувач написав: "{_trim_text(msg,400)}"
Визнач дію користувача з цього списку:
product_order, find_order, find_product, find_store, create_service_request, other_action.
Відповідь тільки JSON:
{{ "action": "find_product" }}
"""
        r = await self.gemini_call(uid, p)
        return _safe_json_extract(r)

    async def analyze_message(self, uid: int, msg: str, action: str, template_name: str):
        fields = available_fields.get(template_name, [])
        meta = collection_meta.get(template_name, {})
        meta_short = {k: v[:5] if isinstance(v, list) else v for k, v in meta.items()}

        p = f"""
Користувач написав: "{_trim_text(msg,600)}"
Дія: {action}
Поля колекції: {", ".join(fields)}
Обмеження на деякі поля (застосовуй лише за потреби): {json.dumps(meta_short, ensure_ascii=False)}

Сформуй MongoDB pipeline без $search і $vectorSearch.
Формат відповіді — тільки JSON масив:
[
  {{ ... }},
  {{ ... }}
]
"""

        r = await self.gemini_call(uid, p)
        pipeline = _safe_json_extract(r)

        if isinstance(pipeline, dict):
            pipeline = [pipeline]
        if not isinstance(pipeline, list):
            return None

        cleaned = self.pipeline_svc.clean_pipeline(pipeline, template_name)
        return cleaned


    async def respond_to_data(self, uid: int, data: Any, msg: str):
        if isinstance(data, list):
            data = data[:5]

        json_data = json.dumps(data, ensure_ascii=False, indent=2, default=str)
        MAX_JSON_CHARS = 1200
        if len(json_data) > MAX_JSON_CHARS:
            json_data = json_data[:MAX_JSON_CHARS] + "... (скорочено)"

        vec_ctx = await self.vector_context_from_chroma(msg)
        prompt = f"""
Користувач написав: "{_trim_text(msg,400)}"
Ось знайдені дані з MongoDB:
{json_data}
{vec_ctx}

Сформуй природну і коротку відповідь українською.
"""
        reply = await self.gemini_call(uid, prompt, use_history=True)
        return reply[:REPLY_SAFE_LIMIT]

    async def respond_to_other(self, uid: int, msg: str):
        p = f"""
Користувач написав: "{_trim_text(msg,400)}"
Контекст: компанія продає будівельні інструменти.
Дай коротку (2–3 речення) відповідь українською.
"""
        return await self.gemini_call(uid, p, use_history=False)


_default_assistant = AIAssistant()

# Legacy function names for easier refactoring later if needed
gemini_call = _default_assistant.gemini_call
vector_context_from_chroma = _default_assistant.vector_context_from_chroma
vector_docs_from_chroma = _default_assistant.vector_docs_from_chroma
analyze_action = _default_assistant.analyze_action
analyze_message = _default_assistant.analyze_message
respond_to_data = _default_assistant.respond_to_data
respond_to_other = _default_assistant.respond_to_other

