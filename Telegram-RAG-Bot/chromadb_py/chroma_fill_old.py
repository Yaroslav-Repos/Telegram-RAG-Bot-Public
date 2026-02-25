import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import chromadb
from chromadb.config import Settings
from bson import ObjectId
from chromadb.utils import embedding_functions


# --------------------------
# 1) ПЕРЕТВОРЕННЯ MONGO-ДОКУМЕНТА У ТЕКСТ ДЛЯ CHROMA
# --------------------------
def build_product_document(product: dict) -> str:
    """
    Створює RAG-документ з товару.
    Структура: Назва → Опис → Характеристики → Особливості → Комплектація → Гарантія → Інше
    """
    specs_str = "\n".join(
        f"- {k}: {v}" for k, v in product.get("specs", {}).items()
    )

    features_str = ", ".join(product.get("features", []))
    package_str = ", ".join(product.get("package_contents", []))

    text = f"""
Товар: {product.get("name")}
Категорія: {product.get("category")}

Опис:
{product.get("description")}

Характеристики:
{specs_str}

Особливості:
{features_str}

Комплектація:
{package_str}

Ціна: {product.get("price")} грн
Залишок на складі: {product.get("stock")}
Знижка: {product.get("discount") * 100 if product.get("discount") else 0}%
Гарантія: {product.get("warranty")}
Країна-виробник: {product.get("country_of_origin")}

Рейтинг: {product.get("rating")} із {product.get("reviews_count")} відгуків

Дата додавання: {product.get("created_at")}
""".strip()

    return text

ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# --------------------------
# 2) ФУНКЦІЯ ДОДАВАННЯ ДОКУМЕНТІВ У CHROMA
# --------------------------
async def sync_products_to_chroma(
    mongo_uri: str,
    chroma_host: str = "localhost",
    chroma_port: int = 8000,
    collection_name: str = "products_rag",
):
    # MongoDB async client
    mongo = AsyncIOMotorClient(mongo_uri)
    db = mongo["dniprom"]
    collection = db["products"]

    # Chroma async client
    chroma = await chromadb.AsyncHttpClient(
        host=chroma_host,
        port=chroma_port,
        settings=Settings()
    )

    chroma_collection = await chroma.get_or_create_collection(
        name=collection_name,
        embedding_function=ef
    )

    # Читаємо всі товари
    cursor = collection.find({})
    async for product in cursor:
        chroma_doc = build_product_document(product)
        chroma_id = str(product["_id"])  # id для Chroma

        await chroma_collection.add(
            ids=[chroma_id],
            documents=[chroma_doc],
            metadatas=[{
                "product_id": product.get("product_id"),
                "name": product.get("name"),
                "category": product.get("category"),
                "price": product.get("price"),
                "rating": product.get("rating"),
                "stock": product.get("stock"),
            }]
        )

        print(f"[OK] Added product {product.get('name')} -> Chroma")

    print("\n✔ Синхронізація Mongo → Chroma завершена.")


# --------------------------
# 3) ЗАПУСК
# --------------------------
if __name__ == "__main__":
    asyncio.run(sync_products_to_chroma(
        mongo_uri="mongodb://localhost:27017/"
    ))
