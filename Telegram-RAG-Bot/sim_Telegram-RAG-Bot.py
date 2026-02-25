API_ID =  12345678
API_HASH = "*********************************"
BOT_USERNAME = "**************************" 

import os
os.environ["GOOGLE_API_KEY"] = "**********************************"

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
)

# from langchain_community.embeddings import HuggingFaceEmbeddings

#ragas_hf_embeddings = HuggingFaceEmbeddings(
#    model_name="sentence-transformers/all-MiniLM-L6-v2"
#)

# Параметри навантаження
NUM_CLIENTS = 2
MESSAGES_PER_CLIENT = 10
MAX_RPS = 200

import asyncio
import random
import time
from motor.motor_asyncio import AsyncIOMotorClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ==== експериментальний стенд ====
from experiments.runner import run_single_experiment_batch
from experiments.queue_tests import build_qsystem_samples

# ==== асистент ====
from ai_assistant import gemini_call, vector_context_from_chroma, vector_docs_from_chroma, analyze_message

# ==== Pyrogram (опційно) ====
USE_PYROGRAM = True

if USE_PYROGRAM:
    from pyrogram_load.client_manager import PyroClientWrapper
    from pyrogram_load.load_generator import PyroLoadGenerator
    from pyrogram_load.rate_limiter import RPSLimiter

# ==== Параметри MongoDB ====
# client = AsyncIOMotorClient("mongodb://localhost:27017/")
# db = client["dniprom"]
# products = db["products"]
# orders = db["orders"]


# ============================================================
# 1. LLM / RAG / Pipeline wrappers
# ============================================================

# async def llm_answer(uid, query):
#     return await gemini_call(uid, query, use_history=False)

# async def rag_answer(query: str):
#     # 1) Сирі документи для метрик
#     docs = await vector_docs_from_chroma(query, k=5)

#     # 2) форматований контекст для LLM
#     context = await vector_context_from_chroma(query, k=5)

#     prompt = (
#         f"Питання: {query}\n"
#         f"Ти — консультант магазину будівельного обладнання.\n"
#         f"Використай ТІЛЬКИ наведений контекст:\n{context}\n\n"
#         "Сформуй відповідь СТРОГО у форматі:\n"
#         "<Назва товару> (<Категорія>, <Ціна>)\n"
#         "<Назва товару> (<Категорія>, <Ціна>)\n"
#         "...\n\n"
#         "Вимоги:\n"
#         "- без вступних фраз (не пиши «Ось», «Пропоную», «В наявності»);\n"
#         "- без пояснень, описів чи характеристик;\n"
#         "- без markdown (**жирного тексту**, списків, заголовків);\n"
#         "- лише один товар на рядок;\n"
#         "Виведи ЛИШЕ список товарів у вказаному форматі. Нічого більше."
#     )

#     answer = await gemini_call(0, prompt, use_history=False)

#     return {
#         "answer": answer,
#         "docs": docs
#     }

# #chroma_embedder = SentenceTransformerEmbeddingFunction(
# #    model_name="sentence-transformers/all-MiniLM-L6-v2"
# #)

# #def embedding_function(texts):
# #    return chroma_embedder(texts)
# #

# async def pipeline_analyze(uid, text, action="find_product",
#                            template="product_template"):
#     return await analyze_message(uid, text, action, template)


# ============================================================
# 2. Тестові набори
# ============================================================

LLM_TEST_CASES = [
    {"query": "Покажи всі доступні в каталозі товари бренду Bosch.", "expected": "Ось доступні товари Bosch з наявного каталогу: Bosch Викрутка S 929 (Ручний інструмент, 1273.42₴), Bosch Розвідний ключ X 913 (Ручний інструмент, 558.73₴), Bosch Маска респіратор X 202 (Засоби безпеки та захисту, 1010.46₴), Bosch Рулетка Pro 784 (Ручний інструмент, 810.34₴)"},
    {"query": "Перелічіть перфоратори з каталогу разом з цінами.", "expected": "Ось популярні моделі перфораторів: Hyundai Перфоратор Lite 526 (Будівельні інструменти, 11892.43₴), Black+Decker Перфоратор 2.0 566 (Будівельні інструменти, 16123.15₴)."},
]

PIPELINE_QUERIES = [
    "Покажи дрилі Bosch",
    "Покажи перфоратори",
    "Знайди акумуляторні інструменти"
]

RAG_TEST_CASES = [
    {
        "query": "Перерахуйте всі товари бренду Bosch, доступні в каталозі магазину.",
        "expected_docs": [
            "- Bosch Маска респіратор X 202 (Засоби безпеки та захисту, 1010.46₴)\nBosch Маска респіратор X 202 | Засоби безпеки та захисту | Bosch Маска респіратор X 202 — надійний засоби безпеки та захисту для домашнього та професійного використання. Особливості: Water resistant, ...",
            "- Bosch Викрутка S 929 (Ручний інструмент, 1273.42₴)\nBosch Викрутка S 929 | Ручний інструмент | Bosch Викрутка S 929 — надійний ручний інструмент для домашнього та професійного використання. Особливості: Water resistant, Ergonomic handle. | Water resist...",
            "- Bosch Болгарка компакт 149 (Електроінструменти для дому, 10435.6₴)\nBosch Болгарка компакт 149 | Електроінструменти для дому | Bosch Болгарка компакт 149 — надійний електроінструменти для дому для домашнього та професійного використання. Особливості: Ergonomic handle,...",
            "- Bosch Секатор X 115 (Садові інструменти, 5439.1₴)\nBosch Секатор X 115 | Садові інструменти | Bosch Секатор X 115 — надійний садові інструменти для домашнього та професійного використання. Особливості: Overheat protection, Low noise, Ergonomic handle....",
            "- Bosch Ліхтар акумуляторний X 223 (Освітлювальне обладнання, 5589.37₴)\nBosch Ліхтар акумуляторний X 223 | Освітлювальне обладнання | Bosch Ліхтар акумуляторний X 223 — надійний освітлювальне обладнання для домашнього та професійного використання. Особливості: Overheat pr..."
        ],
        "expected_answer":
"""
Bosch Маска респіратор X 202 (Засоби безпеки та захисту, 1010.46₴)
Bosch Викрутка S 929 (Ручний інструмент, 1273.42₴)
Bosch Болгарка компакт 149 (Електроінструменти для дому, 10435.6₴)
Bosch Секатор X 115 (Садові інструменти, 5439.1₴)
Bosch Ліхтар акумуляторний X 223 (Освітлювальне обладнання, 5589.37₴)
"""
    },

    {
        "query": "Покажіть шілфувальні круги у вашому каталозі",
        "expected_docs": [
            "- Einhell Шліфувальний круг Lite 382 (Аксесуари та витратні матеріали, 2403.55₴)\nEinhell Шліфувальний круг Lite 382 | Аксесуари та витратні матеріали | Einhell Шліфувальний круг Lite 382 — надійний аксесуари та витратні матеріали для домашнього та професійного використання. Особли...",
            "- Krausmann Дриль-шуруповерт S 157 (Електроінструменти для дому, 8535.86₴)\nKrausmann Дриль-шуруповерт S 157 | Електроінструменти для дому | Krausmann Дриль-шуруповерт S 157 — надійний електроінструменти для дому для домашнього та професійного використання. Особливості: Ergon...",
            "- Intertool Шліфувальний круг 2.0 798 (Аксесуари та витратні матеріали, 1392.33₴)\nIntertool Шліфувальний круг 2.0 798 | Аксесуари та витратні матеріали | Intertool Шліфувальний круг 2.0 798 — надійний аксесуари та витратні матеріали для домашнього та професійного використання. Особ...",
            "- Einhell Шуруповерт Pro 461 (Будівельні інструменти, 13880.85₴)\nEinhell Шуруповерт Pro 461 | Будівельні інструменти | Einhell Шуруповерт Pro 461 — надійний будівельні інструменти для домашнього та професійного використання. Особливості: Water resistant, Ergonomic ...",
            "- Einhell Шуруповерт Lite 414 (Будівельні інструменти, 9990.04₴)\nEinhell Шуруповерт Lite 414 | Будівельні інструменти | Einhell Шуруповерт Lite 414 — надійний будівельні інструменти для домашнього та професійного використання. Особливості: Durable casing, Ergonomic..."
            ],
        "expected_answer": 
"""
Einhell Шліфувальний круг Lite 382 (Аксесуари та витратні матеріали, 2403.55₴)
Intertool Шліфувальний круг 2.0 798 (Аксесуари та витратні матеріали, 1392.33₴)
"""
    },
]


# ============================================================
# 3. SYNTHETIC METRICS (коли Pyrogram вимкнено)
# ============================================================

# def generate_synthetic_arrivals(n=80, rate=0.8):
#     """
#     Імітує часові мітки надходження запитів (arrival_log).
#     """
#     t = time.perf_counter()
#     out = []
#     for _ in range(n):
#         out.append(t)
#         t += random.random() / rate
#     return out


# def generate_synthetic_service(n=80, mean=0.4):
#     """
#     Імітує час обслуговування сервісом (service_log).
#     """
#     return [random.random() * mean for _ in range(n)]


# def generate_synthetic_latency(n=80):
#     return [random.random() * 0.2 for _ in range(n)]


# ============================================================
# 4. MAIN
# ============================================================

async def main():

    if USE_PYROGRAM:
        print("[SIM] Running REAL Telegram load test...")

        latency_log = {}
        arrival_log = {}
        service_log = {}

        rps_limiter = RPSLimiter(MAX_RPS)

        clients = []
        for i in range(NUM_CLIENTS):
            session = f"test_user_{i}"
            c = PyroClientWrapper(
                session,
                API_ID,
                API_HASH,
                latency_log,
                arrival_log,
                service_log,
                rps_limiter
            )
            await c.start(bot_username=BOT_USERNAME)
            clients.append(c)

        generator = PyroLoadGenerator(
            clients,
            BOT_USERNAME,
            messages_per_client=MESSAGES_PER_CLIENT,
            delay=(0.01, 0.1)
        )

        await generator.run_all()

        for c in clients:
            await c.wait_until_all_processed(timeout=100)

        for c in clients:
            await c.stop()
        
        samples = build_qsystem_samples(arrival_log, latency_log, service_log)

        if samples:
            latency_log = samples["latencies"]
            arrival_log = samples["arrival_ts"]
            service_log = samples["S_est"]
    else:
        print("[SIM] Pyrogram load test is DISABLED. Using synthetic data...")

        # Фейкові дані замість Pyrogram
        # latency_log = generate_synthetic_latency()
        # arrival_log = generate_synthetic_arrivals()
        # service_log = generate_synthetic_service()

    # =======================================================
    # RUN EXPERIMENT
    # =======================================================

    print(latency_log)
    print(arrival_log)
    print(service_log)

    print("[SIM] Running EXPERIMENT batch...")

    summary = await run_single_experiment_batch({
        "latency_log": latency_log,
        "arrival_log": arrival_log,
        "service_log": service_log,

        "mmn_servers": MAX_RPS

        #"db_find_collection": products,
        #"db_insert_collection": orders,

        #"llm_func": llm_answer,
        #"embed_fn": embedding_function,
        #"test_cases": LLM_TEST_CASES,
        #"rag_func": rag_answer,
        #"llm": llm,
        #"ragas_embeddings": ragas_hf_embeddings,
        #"rag_questions": RAG_TEST_CASES,

        #"pipeline_analyze": pipeline_analyze,
        #"test_queries": PIPELINE_QUERIES,
    })


    print("[SIM] Experiment done.")
    print(summary)


if __name__ == "__main__":
    asyncio.run(main())

