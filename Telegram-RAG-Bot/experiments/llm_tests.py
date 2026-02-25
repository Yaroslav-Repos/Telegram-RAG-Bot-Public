import asyncio
import numpy as np

async def llm_accuracy_test(llm_func, embed_fn, test_cases, uid=0):
    """
    Оцінка якості LLM через семантичну подібність.

    test_cases: [{"query": "...", "expected": "..."}]
    llm_func(uid, prompt) -> str (async)
    embed_fn([text]) -> [vector]
    """
    sims = []
    raw = []

    for t in test_cases:
        resp = await llm_func(uid, t["query"])

        e_resp = await asyncio.to_thread(embed_fn, [resp])
        e_gold = await asyncio.to_thread(embed_fn, [t["expected"]])

        a = np.array(e_resp[0], dtype=float)
        b = np.array(e_gold[0], dtype=float)
        sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

        sims.append(sim)
        raw.append({
            "query": t["query"],
            "expected": t["expected"],
            "response": resp,
            "similarity": sim
        })

    return {
        "mean": float(np.mean(sims)) if sims else 0.0,
        "std": float(np.std(sims)) if sims else 0.0,
        "raw": raw
    }


async def rag_quality_test(
        rag_func,
        embed_fn,
        rag_test_cases,
        sim_threshold=0.75
    ):
    """
    Повноцінний RAG-тест із embedding-метриками:
    
    1) retrieval_quality      – чи витягнули потрібні документи (semantic)
    2) groundedness           – наскільки відповідь базується на документах
    3) hallucination_penalty  – наскільки відповідь НЕ відповідає документам
    4) answer_quality         – схожість відповіді з очікуваною
    
    rag_func(query) має повертати:
        {
            "answer": str,
            "docs": list[str]   # сирі документи з Chroma
        }
    """

    results = []

    retrieval_scores = []
    grounded_scores = []
    hallucination_penalties = []
    answer_scores = []

    for test in rag_test_cases:
        query = test["query"]
        expected_docs = test["expected_docs"]
        expected_answer = test["expected_answer"]

        # -----------------------------
        # 0. Отримуємо RAG-відповідь
        # -----------------------------
        rag_output = await rag_func(query)
        rag_answer = rag_output["answer"]
        retrieved_docs = rag_output["docs"]

        # Embed RAG answer один раз
        e_answer = (await asyncio.to_thread(embed_fn, [rag_answer]))[0]
        e_answer = np.array(e_answer, dtype=float)

        # -------------------------------------------------
        # 1. Retrieval Quality (expected_docs → retrieved_docs)
        # -------------------------------------------------
        retr_hits = 0
        for target in expected_docs:
            e_target = (await asyncio.to_thread(embed_fn, [target]))[0]
            e_target = np.array(e_target, dtype=float)

            # max cosine similarity із документами
            sims = []
            for doc in retrieved_docs:
                e_doc = (await asyncio.to_thread(embed_fn, [doc]))[0]
                e_doc = np.array(e_doc, dtype=float)

                sim = float(
                    np.dot(e_target, e_doc) /
                    (np.linalg.norm(e_target) * np.linalg.norm(e_doc) + 1e-9)
                )
                sims.append(sim)

            if sims and max(sims) > sim_threshold:
                retr_hits += 1

        retrieval_quality = retr_hits / len(expected_docs)
        retrieval_scores.append(retrieval_quality)

        # -------------------------------------------------
        # 2. Groundedness (answer → retrieved_docs)
        # -------------------------------------------------
        grounded_list = []
        for doc in retrieved_docs:
            e_doc = (await asyncio.to_thread(embed_fn, [doc]))[0]
            e_doc = np.array(e_doc, dtype=float)

            sim = float(
                np.dot(e_answer, e_doc) /
                (np.linalg.norm(e_answer) * np.linalg.norm(e_doc) + 1e-9)
            )
            grounded_list.append(sim)

        groundedness = float(np.mean(grounded_list)) if grounded_list else 0.0
        grounded_scores.append(groundedness)

        # -------------------------------------------------
        # 3. Hallucination penalty
        #    1 - max similarity(answer, docs)
        # -------------------------------------------------
        max_sim_with_docs = max(grounded_list) if grounded_list else 0.0
        hallucination_penalty = 1 - max_sim_with_docs

        hallucination_penalties.append(hallucination_penalty)

        # -------------------------------------------------
        # 4. Answer quality (answer → expected_answer)
        # -------------------------------------------------
        e_expected = (await asyncio.to_thread(embed_fn, [expected_answer]))[0]
        e_expected = np.array(e_expected, dtype=float)

        answer_quality = float(
            np.dot(e_answer, e_expected) /
            (np.linalg.norm(e_answer) * np.linalg.norm(e_expected) + 1e-9)
        )

        answer_scores.append(answer_quality)

        # -------------------------------------------------
        # Зберігаємо результат
        # -------------------------------------------------
        results.append({
            "query": query,
            "retrieval_quality": retrieval_quality,
            "groundedness": groundedness,
            "hallucination_penalty": hallucination_penalty,
            "answer_quality": answer_quality,
            "rag_answer": rag_answer,
            "retrieved_docs": retrieved_docs,
            "expected_docs": expected_docs,
            "expected_answer": expected_answer
        })

    # -------------------------------------------------
    # Підсумковий результат
    # -------------------------------------------------
    return {
        "mean_retrieval_quality": float(np.mean(retrieval_scores)),
        "mean_groundedness": float(np.mean(grounded_scores)),
        "mean_hallucination_penalty": float(np.mean(hallucination_penalties)),
        "mean_answer_quality": float(np.mean(answer_scores)),
        "details": results
    }


from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas import evaluate
from ragas.metrics import (
    ContextRecall,
    Faithfulness,
    ResponseRelevancy,
    FactualCorrectness,
)
import pandas as pd
import numpy as np

from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas import evaluate
from ragas.metrics import ResponseRelevancy, FactualCorrectness
import pandas as pd
import numpy as np

async def ragas_llm_test(
        llm_func,        # async uid, prompt -> str
        llm,                  # LangChain LLM (ChatGoogleGenerativeAI)
        ragas_embeddings,      # LangChain Embeddings (HuggingFaceEmbeddings)
        test_cases,      # [{"query": "...", "expected": "..."}]
        uid=0
    ):
    """
    Оцінка якості LLM через RAGAS-БЕЗ-документів.

    llm_func(uid, prompt) -> str (async)
    test_cases: [{"query": str, "expected": str}]
    """

    samples = []
    extra_info = []

    for case in test_cases:
        query = case["query"]
        expected = case["expected"]

        # Отримуємо відповідь від твоєї LLM-функції
        resp = await llm_func(uid, query)

        samples.append(
            SingleTurnSample(
                user_input=query,
                retrieved_contexts=[],   # НІЯКИХ ДОКУМЕНТІВ
                response=resp,
                reference=expected,
            )
        )

        extra_info.append({
            "query": query,
            "expected": expected,
            "response": resp,
        })

    if not samples:
        return {"summary": {}, "details": []}

    dataset = EvaluationDataset(samples=samples)

    metrics = [
        ResponseRelevancy(),      # наскільки відповідь релевантна референсу
        FactualCorrectness(),     # наскільки відповідь збігається з reference
    ]

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,              # LangChain LLM як суддя
        embeddings=ragas_embeddings,
        raise_exceptions=False,
    )

    df = result.to_pandas()
    metric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    summary = {col: float(df[col].mean()) for col in metric_cols}

    details = []
    for i, row in df.iterrows():
        row_dict = {
            "user_input": row.get("user_input"),
            "response": row.get("response"),
            "reference": row.get("reference"),
        }
        for col in metric_cols:
            row_dict[col] = float(row[col])
        row_dict.update(extra_info[i])
        details.append(row_dict)

    return {
        "summary": summary,   # середні значення метрик
        "details": details,   # покейсно
    }

async def ragas_quality_test(
        rag_func,
        rag_test_cases,
        llm,                  # LangChain LLM (ChatGoogleGenerativeAI)
        ragas_embeddings      # LangChain Embeddings (HuggingFaceEmbeddings)
    ):
    """
    RAGAS-оцінка без кастомних wrapper'ів.

    rag_func(query) -> { "answer": str, "docs": list[str] }

    rag_test_cases:
      {
        "query": str,
        "expected_answer": str,
        "expected_docs": list[str]
      }
    """

    samples = []
    extra_info = []

    for case in rag_test_cases:
        query = case["query"]
        expected_answer = case["expected_answer"]
        expected_docs = case.get("expected_docs", [])

        rag_output = await rag_func(query)
        answer = rag_output["answer"]
        docs = rag_output["docs"]

        samples.append(
            SingleTurnSample(
                user_input=query,
                retrieved_contexts=docs,
                response=answer,
                reference=expected_answer,
            )
        )

        extra_info.append({
            "query": query,
            "expected_answer": expected_answer,
            "expected_docs": expected_docs,
            "rag_answer": answer,
            "retrieved_docs": docs,
        })

    if not samples:
        return {"summary": {}, "details": []}

    dataset = EvaluationDataset(samples=samples)

    metrics = [
        ContextRecall(),
        Faithfulness(),
        ResponseRelevancy(),
        FactualCorrectness(),
    ]

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=ragas_embeddings, 
        raise_exceptions=False,
    )

    df = result.to_pandas()
    metric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    summary = {col: float(df[col].mean()) for col in metric_cols}

    details = []
    for i, row in df.iterrows():
        row_dict = {
            "user_input": row.get("user_input"),
            "response": row.get("response"),
            "reference": row.get("reference"),
        }
        for col in metric_cols:
            row_dict[col] = float(row[col])
        row_dict.update(extra_info[i])
        details.append(row_dict)

    return {
        "summary": summary,
        "details": details,
    }