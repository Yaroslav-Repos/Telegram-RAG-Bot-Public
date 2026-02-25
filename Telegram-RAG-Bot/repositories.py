from typing import List, Optional
from bson.objectid import ObjectId
import db

products = db.products
orders = db.orders


class ProductRepository:
    async def get_categories(self) -> List[str]:
        return await products.distinct("category")

    async def count_by_category(self, category: str) -> int:
        return await products.count_documents({"category": category})

    async def get_products_by_category(self, category: str, page: int, page_size: int) -> List[dict]:
        start = page * page_size
        cursor = products.find({"category": category}).skip(start).limit(page_size)
        return await cursor.to_list(length=page_size)

    async def get_by_objectid(self, oid: ObjectId) -> Optional[dict]:
        return await products.find_one({"_id": oid})

    async def decrease_stock(self, oid: ObjectId, quantity: int):
        return await products.update_one({"_id": oid}, {"$inc": {"stock": -quantity}})


class OrderRepository:
    async def get_last_order(self) -> Optional[dict]:
        return await orders.find_one(sort=[("order_id", -1)])

    async def insert(self, order: dict):
        return await orders.insert_one(order)
