import pytest
from db import is_pipeline_safe


def test_safe_pipeline_products():
    pipeline = [{"$match": {"name": "Drill"}}, {"$project": {"name": 1}}]
    assert is_pipeline_safe(pipeline, "products") is True


def test_forbidden_pattern():
    pipeline = [{"$match": {"$where": "this.price > 0"}}]
    assert is_pipeline_safe(pipeline, "products") is False


def test_disallowed_field():
    pipeline = [{"$match": {"secret_field": "x"}}]
    assert is_pipeline_safe(pipeline, "products") is False
