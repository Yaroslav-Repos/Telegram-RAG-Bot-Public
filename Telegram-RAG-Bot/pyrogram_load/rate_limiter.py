# pyrogram_load/rate_limiter.py
import asyncio
import time


class RPSLimiter:
    def __init__(self, rps: float):
        self.capacity = rps
        self.tokens = rps
        self.last = time.perf_counter()
        self.lock = asyncio.Lock()

    async def acquire(self):
        while True:
            async with self.lock:
                now = time.perf_counter()
                elapsed = now - self.last
                self.last = now

                # Поповнюємо токени
                self.tokens = min(self.capacity, self.tokens + elapsed * self.capacity)

                if self.tokens >= 1:
                    self.tokens -= 1
                    return

                # Потрібно трохи почекати для появи 1 токена
                wait_time = (1 - self.tokens) / self.capacity

            await asyncio.sleep(wait_time)