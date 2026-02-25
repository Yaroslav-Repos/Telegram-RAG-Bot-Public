import asyncio
from datetime import datetime

from .exporters import (
    export_json, export_csv, save_plot_png,
    quick_hist, quick_line
)

from .queue_tests import (
    arrival_process_estimator,
    service_time_estimator,
    mmn_metrics,
    mm1_metrics
)

from .db_tests import (
    measure_read_latency,
    measure_write_latency,
    measure_throughput,
    measure_index_vs_nonindex,
    measure_aggregation
)
from .functional_tests import pipeline_validity_test
from .llm_tests import llm_accuracy_test, rag_quality_test, ragas_llm_test, ragas_quality_test
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")


async def run_single_experiment_batch(context):
    """
    context передається з sim_DniproM:

        context = {
            "latency_log": [...],
            "arrival_log": [...],
            "service_log": [...],
            "db_collection": MongoCollection,
            "llm_func": fn,
            "rag_func": fn,
            "embed_fn": fn,
            "test_cases": [...],
            "pipeline_analyze": fn
        }

    """

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {"timestamp": ts}

    latency_log = context.get("latency_log", [])
    arrival_log = context.get("arrival_log", [])
    service_log = context.get("service_log", [])

    # ----------------------------------------------------------------------
    # 1. LATENCY: Histogram + mean
    # ----------------------------------------------------------------------
    if latency_log:
        await export_csv(
            f"{RESULTS_DIR}/latency_{ts}.csv",
            ["latency_sec"],
            [[v] for v in latency_log]
        )
        
        fig = quick_hist(latency_log, "Telegram Send Latency", "seconds")
        await save_plot_png(f"{RESULTS_DIR}/latency_{ts}.png", fig)

        summary["latency_mean"] = float(sum(latency_log) / len(latency_log))


    # ----------------------------------------------------------------------
    # 2. QUEUE THEORY (λ, μ, M/M/?)
    # ----------------------------------------------------------------------
    if len(arrival_log) >= 1 and len(service_log) >= 1:
        arr = arrival_process_estimator(arrival_log)
        srv = service_time_estimator(service_log)

        summary["arrival_est"] = arr
        summary["service_est"] = srv

    if service_log:
        now_ts = datetime.now().isoformat()

        await export_csv(
            f"{RESULTS_DIR}/service_time_{ts}.csv",
            ["timestamp", "service_sec"],
            [[now_ts, v] for v in service_log]
        )

        # ----------------- ВСТАНОВЛЮЄМО КІЛЬКІСТЬ СЕРВЕРІВ n -------------------
        n_servers = context.get("mmn_servers")

        # ----------------------------------------------------------------------
        # Обчислення M/M/?
        # ----------------------------------------------------------------------
        if arr and srv and arr.get("lambda") and srv.get("mu"):
            mmn = mmn_metrics(arr["lambda"], srv["mu"], n_servers)

            mm1 = mm1_metrics(arr["lambda"], srv["mu"])

            summary["mm1"] = mm1
            summary["mmn"] = mmn
            summary["mmn"]["n"] = n_servers

            # графік зміни λ
            fig_l = quick_line(
                range(len(arrival_log) - 1),
                list(arrival_log)[1:],
                xlabel="request index",
                ylabel="arrival timestamp",
                title="Arrival Timestamps Evolution"
            )
            await save_plot_png(f"{RESULTS_DIR}/arrival_evo_{ts}.png", fig_l)

            # графік service time
            fig_s = quick_hist(service_log, "Service Time Distribution", "seconds")
            await save_plot_png(f"{RESULTS_DIR}/service_hist_{ts}.png", fig_s)


    # ----------------------------------------------------------------------
    # 3. DB LOAD TESTS
    # ----------------------------------------------------------------------
    db_find = context.get("db_find_collection")
    db_insert = context.get("db_insert_collection")

    if (db_find is not None) and (db_insert is not None):
        print("[EXPERIMENT] Running extended DB metrics...")

        reads = await measure_read_latency(db_find, {})
        writes = await measure_write_latency(db_insert)
        t_read = await measure_throughput(db_find, mode="read")
        t_write = await measure_throughput(db_insert, mode="write")
        idx = await measure_index_vs_nonindex(db_find)
        aggs = await measure_aggregation(db_find)

        summary["db_metrics"] = {
            "read_latency": reads,
            "write_latency": writes,
            "read_throughput": t_read,
            "write_throughput": t_write,
            "index_vs_nonindex": idx,
            "aggregation": aggs,
        }


    # ----------------------------------------------------------------------
    # 4. FUNCTIONAL PIPELINE TEST
    # ----------------------------------------------------------------------
    pipeline_analyze = context.get("pipeline_analyze")
    test_queries = context.get("test_queries", [])

    if pipeline_analyze and test_queries:
        print("[EXPERIMENT] Testing pipeline validity...")
        pipe_res = await pipeline_validity_test(pipeline_analyze, test_queries)
        summary["pipeline_test"] = pipe_res


    # ----------------------------------------------------------------------
    # 5. LLM ACCURACY TESTS (опціонально)
    # ----------------------------------------------------------------------
    if context.get("llm_func") and context.get("embed_fn") and context.get("test_cases") and context.get("llm") and context.get("ragas_embeddings") :
        print("[EXPERIMENT] Running LLM accuracy test...")
        acc = await llm_accuracy_test(
            context["llm_func"],
            context["embed_fn"],
            context["test_cases"]
        )

        ragas_llm_metrics = await ragas_llm_test(
            llm_func=context["llm_func"],
            llm=context["llm"],              # LangChain LLM
            ragas_embeddings=context["ragas_embeddings"],
            test_cases=context["test_cases"],
            uid=0,
        )

        summary["llm_quality"] = {
            "embedding_based": acc,
            "ragas_based": ragas_llm_metrics,
        }


    # ----------------------------------------------------------------------
    # 6. RAG TEST
    # ----------------------------------------------------------------------
    if context.get("rag_func") and context.get("embed_fn") and context.get("ragas_embeddings") and context.get("llm") and context.get("rag_questions"):
        print("[EXPERIMENT] Running RAG test...")
        rag_metrics = await rag_quality_test(
            rag_func=context["rag_func"],    # ← БЕРЕМО З context
            embed_fn=context["embed_fn"],    # ← ТЕЖ БЕРЕМО З context
            rag_test_cases=context["rag_questions"]
        )


        ragas_result = await ragas_quality_test(
            context["rag_func"],
            context["rag_questions"],
            context["llm"],                   
            context["ragas_embeddings"]
            )

        summary["rag_quality"] = {
            "embedding_based": rag_metrics,
            "ragas_based": ragas_result
        }


    # ----------------------------------------------------------------------
    # 7. SAVE SUMMARY JSON
    # ----------------------------------------------------------------------
    await export_json(f"{RESULTS_DIR}/summary_{ts}.json", summary)
    return summary


async def periodic_experiments(interval_seconds=60, context_builder=None):
    """
    Автоматичний запуск експериментів раз на N секунд.
    """
    while True:
        try:
            context = context_builder() if context_builder else {}
            await run_single_experiment_batch(context)
        except Exception as e:
            print("[Experiments] ERROR:", e)

        await asyncio.sleep(interval_seconds)