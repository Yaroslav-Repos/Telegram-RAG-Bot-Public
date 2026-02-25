import pytest
import asyncio
from types import SimpleNamespace

import repositories


class FakeCollection:
    def __init__(self):
        self._data = [{"_id": "oid1", "product_id": "p1", "name": "Prod1", "price": 100, "stock": 5}]

    async def distinct(self, key):
        return ["CatA", "CatB"]

    async def count_documents(self, q):
        return 7

    def find(self, q):
        class Cursor:
            def __init__(self, data):
                self._data = data

            def skip(self, n):
                return self

            def limit(self, n):
                return self

            async def to_list(self, length):
                return self._data[:length]

        return Cursor(self._data)

    async def find_one(self, q):
        return self._data[0]

    async def update_one(self, q, u):
        return SimpleNamespace(matched_count=1)

    async def insert_one(self, doc):
        return SimpleNamespace(inserted_id="newid")


@pytest.mark.asyncio
async def test_product_repository(monkeypatch):
    fake = FakeCollection()
    monkeypatch.setattr(repositories, 'products', fake)

    repo = repositories.ProductRepository()

    cats = await repo.get_categories()
    assert isinstance(cats, list) and "CatA" in cats

    cnt = await repo.count_by_category("CatA")
    assert cnt == 7

    prods = await repo.get_products_by_category("CatA", 0, 10)
    assert isinstance(prods, list) and prods[0]["name"] == "Prod1"

    p = await repo.get_by_objectid("oid1")
    assert p["product_id"] == "p1"

    res = await repo.decrease_stock("oid1", 2)
    assert getattr(res, 'matched_count', 0) == 1


@pytest.mark.asyncio
async def test_order_repository(monkeypatch):
    fake = FakeCollection()
    monkeypatch.setattr(repositories, 'orders', fake)

    repo = repositories.OrderRepository()

    last = await repo.get_last_order()
    # FakeCollection.find_one returns a product; still should be truthy
    assert last is not None

    ins = await repo.insert({"order_id": "1"})
    assert getattr(ins, 'inserted_id', None) == "newid"
