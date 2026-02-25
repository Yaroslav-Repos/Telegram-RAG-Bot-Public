def pipeline_safety_check(pipeline):
    forbidden = {"$where", "$function", "$accumulator", "$out", "$merge"}
    try:
        for stage in pipeline:
            if not isinstance(stage, dict):
                return False
            for key in stage.keys():
                if key in forbidden:
                    return False
        return True
    except Exception:
        return False


async def pipeline_validity_test(analyze_func, queries):
    """
    analyze_func(uid, text, action, template) -> pipeline
    """
    results = {"ok": [], "bad": []}

    for q in queries:
        pipeline = await analyze_func(0, q, "find_product", "product_template")

        if not pipeline:
            results["bad"].append({"query": q, "reason": "no pipeline"})
            continue

        if not pipeline_safety_check(pipeline):
            results["bad"].append({"query": q, "reason": "unsafe", "pipeline": pipeline})
        else:
            results["ok"].append({"query": q, "pipeline": pipeline})

    return results
