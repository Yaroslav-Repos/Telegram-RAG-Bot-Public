import pytest
import asyncio

import ai_assistant as aa
import db as dbmod


def test_trim_and_hash_and_safejson():
    s = "a" * 50
    trimmed = aa._trim_text(s, n=10)
    assert trimmed.endswith("...")
    assert len(trimmed) <= 13  # 10 + '...'

    h = aa._hash_key(1, "hello")
    assert isinstance(h, str) and len(h) > 0

    parsed = aa._safe_json_extract('prefix {"a": 1} suffix')
    assert parsed == {"a": 1}

    assert aa._safe_json_extract('no json here') is None


@pytest.mark.asyncio
async def test_analyze_action(monkeypatch):
    async def fake_gemini_call(uid, prompt, use_history=False):
        return '{"action":"find_product"}'

    monkeypatch.setattr(aa, 'gemini_call', fake_gemini_call)

    out = await aa.analyze_action(123, "find drill")
    assert out == {"action": "find_product"}


@pytest.mark.asyncio
async def test_analyze_message(monkeypatch):
    async def fake_gemini_call(uid, prompt, use_history=False):
        # return a pipeline that includes a $match on 'name' and a $project of 'name'
        return '[{"$match": {"name": {"$regex": "drill"}}}, {"$project": {"name": 1}}]'

    monkeypatch.setattr(aa, 'gemini_call', fake_gemini_call)

    pipeline = await aa.analyze_message(1, "find drill", "find_product", "product_template")
    assert isinstance(pipeline, list)
    # Expect at least one stage to be a dict
    assert all(isinstance(s, dict) for s in pipeline)
    # The cleaned pipeline should include allowed fields (name is allowed)
    assert any('$match' in stage or '$project' in stage for stage in pipeline)


@pytest.mark.asyncio
async def test_vector_context_from_chroma(monkeypatch):
    class FakeCol:
        async def query(self, query_texts, n_results):
            return {"documents": [["doc1 content"]], "metadatas": [[{"name": "X", "category": "Y", "price": 100}]]}

    monkeypatch.setattr(dbmod, 'async_chroma_collection', FakeCol())

    res = await aa.vector_context_from_chroma("test query")
    assert isinstance(res, str)
    assert "Семантично схожі товари" in res


@pytest.mark.asyncio
async def test_respond_to_data(monkeypatch):
    async def fake_vector(q, k=4):
        return "- sample"

    async def fake_gem(uid, prompt, use_history=False):
        return "Reply text"

    monkeypatch.setattr(aa, 'vector_context_from_chroma', fake_vector)
    monkeypatch.setattr(aa, 'gemini_call', fake_gem)

    reply = await aa.respond_to_data(1, [{"name": "drill"}], "query")
    assert isinstance(reply, str)
    assert "Reply" in reply
