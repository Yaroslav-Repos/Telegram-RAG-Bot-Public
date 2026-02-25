import asyncio
import chromadb
from chromadb.config import Settings

async def query_document(chroma_client, collection_name: str, query: str, n_results: int = 3):
    """
    Semantic search по тексту.
    """
    collection = await chroma_client.get_collection(collection_name)

    result = await collection.query(
        query_texts=[query],
        n_results=n_results
    )

    # Розгортаємо відповідь у читабельну структуру
    hits = []
    for doc, meta, dist, doc_id in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
        result["ids"][0]
    ):
        hits.append({
            "id": doc_id,
            "metadata": meta,
            "document": doc,
            "distance": dist
        })
    return hits

async def main():
    chroma = await chromadb.AsyncHttpClient(
        host="localhost",
        port=8000,
        settings=Settings()
    )
    hits = await query_document(
        chroma,
        "products_rag",
        "шукаю садові інструменти для пиляння гілок, порадь найкращі",
        n_results=2
    )
    print("\n=== SEARCH RESULTS ===")
    for h in hits:
        print(h)

asyncio.run(main())
