import time
import statistics
from datetime import datetime

async def measure_read_latency(collection, query, n=500):
    latencies = []
    errors = 0

    for _ in range(n):
        t0 = time.perf_counter()
        try:
            await collection.find_one(query)
        except:
            errors += 1
            continue
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # ms

    return {
        "p50": statistics.median(latencies) if latencies else None,
        "p95": statistics.quantiles(latencies, n=20)[-1] if len(latencies) >= 20 else None,
        "p99": sorted(latencies)[int(len(latencies)*0.99)] if len(latencies) >= 100 else None,
        "avg": statistics.mean(latencies) if latencies else None,
        "errors": errors,
        "samples": len(latencies)
    }


async def measure_write_latency(collection, n=500):
    latencies = []
    errors = 0

    for _ in range(n):
        t0 = time.perf_counter()
        try:
            await collection.insert_one({
                "ts": datetime.now().isoformat(),
                "v": _
            })
        except:
            errors += 1
            continue
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    return {
        "p50": statistics.median(latencies) if latencies else None,
        "p95": statistics.quantiles(latencies, n=20)[-1] if len(latencies) >= 20 else None,
        "p99": sorted(latencies)[int(len(latencies)*0.99)] if len(latencies) >= 100 else None,
        "avg": statistics.mean(latencies) if latencies else None,
        "errors": errors,
        "samples": len(latencies)
    }


async def measure_throughput(collection, mode="read", duration=12.0):
    """
    Вимірює кількість read/write за секунду.
    """
    start = time.perf_counter()
    count = 0

    if mode == "read":
        query = {}
        while time.perf_counter() - start < duration:
            await collection.find_one(query)
            count += 1

    elif mode == "write":
        while time.perf_counter() - start < duration:
            await collection.insert_one({"ts": datetime.utcnow().isoformat()})
            count += 1

    ops = count / duration
    return {"ops_per_sec": ops, "samples": count}


async def measure_index_vs_nonindex(collection):
    """
    Вимірює різницю між доступом до індексованого поля і неіндексованого.
    """
    # припустимо, що поле "name" індексоване
    indexed = await measure_read_latency(collection, {"name": "Bosch"})
    nonindexed = await measure_read_latency(collection, {"description": "дриль"})

    return {
        "indexed": indexed,
        "nonindexed": nonindexed
    }


async def measure_aggregation(collection):
    """
    Легкий aggregation benchmark.
    """
    import time
    latencies = []

    for _ in range(500):
        t0 = time.perf_counter()
        cursor = collection.aggregate([
            {"$match": {}},
            {"$group": {"_id": None, "count": {"$sum": 1}}}
        ])
        try:
            await cursor.to_list(length=1)
        except:
            continue
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    return {
        "avg": statistics.mean(latencies),
        "p95": statistics.quantiles(latencies, n=20)[-1] if len(latencies) >= 20 else None,
        "samples": len(latencies)
    }
