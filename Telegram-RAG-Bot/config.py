import os
from typing import List

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def _split_keys(env_var: str) -> List[str]:
    raw = os.getenv(env_var, "")
    if not raw:
        return []
    return [k.strip() for k in raw.split(",") if k.strip()]


TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
MONGO_URI = os.getenv("MONGO_URI", "")


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


GEMINI_API_KEYS = _split_keys("GEMINI_API_KEYS") or ([GEMINI_API_KEY] if GEMINI_API_KEY else [])


