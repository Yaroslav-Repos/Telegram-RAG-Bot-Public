from motor.motor_asyncio import AsyncIOMotorClient
from config import MONGO_URI

import chromadb
from chromadb.config import Settings

client = AsyncIOMotorClient(MONGO_URI)
db_mongo = client["dniprom"]

orders = db_mongo["orders"]
products = db_mongo["products"]
stores = db_mongo["stores"]
services = db_mongo["service_requests"]
sessions = db_mongo["sessions"]

async_chroma = None
async_chroma_collection = None

from chromadb.utils import embedding_functions
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2", device=device)

async def init_chroma():
    global async_chroma, async_chroma_collection

    async_chroma = await chromadb.AsyncHttpClient(
        host="localhost",
        port=8000,
        settings=Settings()
    )

    async_chroma_collection = await async_chroma.get_or_create_collection(
        name="products_rag",
        embedding_function=ef
    )

    print("[Chroma] Async client initialized")

order_template = '''orders:{
  "_id": ObjectId("..."),
  "order_id": "12345",
  "customer_id": "456",
  "product_id": "789",
  "quantity": 2,
  "total_price": 1500,
  "order_date": "2025-05-13",
  "status": "В дорозі",
  "shipping_address": "Дніпро",
  "contact_info": {
    "phone": "+380123456789",
    "email": "customer@example.com"
  }
}'''

product_template = '''products:{
  "_id": ObjectId("..."),
  "product_id": "789",
  "name": "Bosch Шуруповерт Pro 3000",
  "category": "Будівельні інструменти",
  "description": "Bosch Шуруповерт Pro 3000 — надійний будівельний інструмент для домашнього та професійного використання. Особливості: Ergonomic handle, Overheat protection.",
  "specs": {
      "power": "1200 Вт",
      "voltage": "220 В",
      "rotation_speed": "3000 об/хв",
      "weight": "2.3 кг"
  },
  "features": ["Ergonomic handle", "Overheat protection", "Durable casing"],
  "package_contents": ["Інструмент", "Акумулятор", "Інструкція користувача"],
  "price": 3450.50,
  "stock": 25,
  "discount": 0.1,
  "warranty": "24 місяці офіційної гарантії",
  "country_of_origin": "Німеччина",
  "rating": 4.7,
  "reviews_count": 126,
  "image_url": "https://example.com/images/screwdriver_3000.jpg",
  "created_at": "2025-11-12T00:00:00Z"
}'''

store_template = '''stores:{
  "_id": ObjectId("..."),
  "store_id": "001",
  "name": "Дніпро-М Дніпро",
  "address": "Дніпро",
  "contact_info": {
    "phone": "",
    "email": ""
}}'''

service_request_template = '''service_requests:{
  "_id": ObjectId("..."),
  "service_request_id": "SR001",
  "customer_id": "456",
  "product_id": "789",
  "issue_description": "",
  "request_date": "2025-05-13",
  "status": "В обробці",
  "service_center": "Дніпро-М Дніпро",
  "resolution_date": "2025-05-15"
}'''

# --- Визначаємо доступні поля для кожної колекції ---
templates = {
    "order_template": order_template,
    "product_template": product_template,
    "store_template": store_template,
    "service_request_template": service_request_template
}

available_fields = {
    "order_template": ["order_id", "customer_id", "product_id", "quantity", "total_price", "order_date", "status", "shipping_address", "contact_info: {phone , email}"],
    "product_template": ["product_id", "name", "category", "description", "specs: {power, voltage, rotation_speed, weight}", "features", "package_contents", "price", "stock", "discount", "warranty", "country_of_origin", "rating", "reviews_count", "image_url", "created_at"],
    "store_template": ["store_id", "name", "address", "contact_info: {phone , email}"],
    "service_request_template": ["service_request_id", "customer_id", "product_id", "issue_description", "request_date", "status", "service_center", "resolution_date"]
}

# --- Метадані для кожної колекції ---
collection_meta = {
    "order_template": {"status": ["Очікує", "В обробці", "В дорозі", "Доставлено", "Скасовано"]},
    "product_template": {
        "category": [
            "Будівельні інструменти",
            "Ручний інструмент",
            "Вимірювальні прилади",
            "Садові інструменти",
            "Зварювальне обладнання",
            "Аксесуари та витратні матеріали",
            "Електроінструменти для дому",
            "Освітлювальне обладнання",
            "Пневматичні інструменти",
            "Засоби безпеки та захисту"
        ],
        "warranty": ["12 місяців офіційної гарантії", "24 місяці офіційної гарантії"],
        "country_of_origin": ["Німеччина", "Польща", "Китай", "Японія", "Україна", "Туреччина", "Італія", "Франція", "Чехія"]
    },
    "store_template": {"address": ["Дніпро", "Київ", "Львів", "Харків", "Одеса"]},
    "service_request_template": {"status": ["Очікує", "В обробці", "Завершено", "Відхилено"], "service_center": ["Дніпро-М Дніпро", "Дніпро-М Київ", "Дніпро-М Львів", "Дніпро-М Одеса"]}
}

ALLOWED_FIELDS = {
    "orders": [
        "order_id", "customer_id", "product_id", "quantity",
        "total_price", "order_date", "status", "shipping_address",
        "contact_info.phone", "contact_info.email"
    ],
    "products": [
        "product_id", "name", "category", "description",
        "specs.power", "specs.voltage", "specs.rotation_speed", "specs.weight",
        "features", "package_contents",
        "price", "stock", "discount", "warranty",
        "country_of_origin", "rating", "reviews_count",
        "image_url", "created_at"
    ],
    "stores": [
        "store_id", "name", "address",
        "contact_info.phone", "contact_info.email"
    ],
    "service_requests": [
        "service_request_id", "customer_id", "product_id",
        "issue_description", "request_date", "status",
        "service_center", "resolution_date"
    ]
}

TEMPLATE_TO_COLLECTION = {
    "product_template": "products",
    "order_template": "orders",
    "store_template": "stores",
    "service_request_template": "service_requests"
    }


import json
from typing import Any, Dict, List, Set

FORBIDDEN_PATTERNS = [
    "$where",
    "function(",
    "eval(",
    "$accumulator",
    "$function",
    "mapReduce",
    "__proto__",
    "prototype",
]


def _collect_field_names(obj: Any, out: Set[str]) -> None:
    """Recursively collect potential field names used in pipeline stage.

    This is conservative — it may collect some operator names but the
    validation later ignores keys starting with '$'.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and not k.startswith("$"):
                out.add(k)
            _collect_field_names(v, out)
    elif isinstance(obj, list):
        for item in obj:
            _collect_field_names(item, out)
    elif isinstance(obj, str):
        # expressions like "$field.sub" -> record field without leading $
        if obj.startswith("$"):
            out.add(obj[1:])


def is_pipeline_safe(pipeline: Any, collection_name: str) -> bool:
    """Basic safety checks for a MongoDB aggregation pipeline produced by AI.

    - Pipeline must be a list of dicts
    - Disallow known dangerous substrings and operators
    - Ensure referenced fields exist in ALLOWED_FIELDS for the target collection
    This function is intentionally conservative.
    """
    if not isinstance(pipeline, list):
        return False

    try:
        dumped = json.dumps(pipeline).lower()
    except Exception:
        return False

    for pat in FORBIDDEN_PATTERNS:
        if pat in dumped:
            return False

    allowed = ALLOWED_FIELDS.get(collection_name)
    if not allowed:
        # If we don't have a whitelist for this collection, refuse by default
        return False

    referenced: Set[str] = set()
    for stage in pipeline:
        if not isinstance(stage, dict):
            return False
        _collect_field_names(stage, referenced)

    # Verify each referenced field is allowed (either exact dotted name or root field)
    allowed_set = set(allowed)
    for ref in referenced:
        # skip empty or operator-like keys
        if not ref or ref.startswith("$") or ref.startswith("_"):
            continue

        # Check exact match or root match (e.g., 'specs.power' vs 'specs')
        if ref in allowed_set:
            continue

        root = ref.split(".")[0]
        if root in allowed_set:
            continue

        # Also accept dotted allowed fields that match the prefix
        match = False
        for a in allowed_set:
            if a.startswith(root + "."):
                match = True
                break
        if match:
            continue

        # Nothing matched -> unsafe
        return False

    return True


async def safe_aggregate(collection, pipeline: List[Dict], limit: int = 5):
    """Validate pipeline and run aggregation safely.

    Raises ValueError when pipeline is rejected.
    """
    # Motor collection has .name attribute
    coll_name = getattr(collection, "name", None)
    if not coll_name:
        raise ValueError("Unknown collection provided to safe_aggregate")

    if not is_pipeline_safe(pipeline, coll_name):
        raise ValueError("Pipeline rejected by safety validator")

    cursor = collection.aggregate(pipeline)
    return await cursor.to_list(length=limit)
