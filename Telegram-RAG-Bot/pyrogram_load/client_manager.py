import asyncio
import time
from time import perf_counter
from pyrogram import Client, filters
from .rate_limiter import RPSLimiter


class PyroClientWrapper:

    def __init__(
        self,
        session_name,
        api_id,
        api_hash,
        latency_log,
        arrival_log,
        service_log,
        rps: RPSLimiter,
    ):
        self.client = Client(session_name, api_id, api_hash)

        self.rps = rps

        # Logs: msg_id -> value
        self.latency_log = latency_log   # час send_message
        self.arrival_log = arrival_log   # коли відправили
        self.service_log = service_log   # повний service time (send->reply)

        # Pending: msg_id -> send_ts
        self.pending = {}

        # Queue для відповідей бота
        self.reply_queue = asyncio.Queue()

        # Лок для MTProto (послідовність викликів)
        self.pyro_lock = asyncio.Lock()

        # Конфіг бота
        self.bot_id = None
        self.bot_username = None


    async def start(self, bot_username=None):

        if bot_username is None:
            raise ValueError("bot_username is required")

        self.bot_username = bot_username

        await self.client.start()

        @self.client.on_message(filters.incoming)
        async def handler(_, msg):
            # Ловимо ТІЛЬКИ повідомлення від бота
            if self.bot_id and msg.from_user and msg.from_user.id == self.bot_id:
                await self.reply_queue.put((msg, time.perf_counter()))

        # get_users під локом
        if self.bot_id is None:
            async with self.pyro_lock:
                bot = await self.client.get_users(self.bot_username)
            self.bot_id = bot.id

        asyncio.create_task(self._reply_consumer())


    async def stop(self):
        await self.client.stop()


    async def send_to_bot(self, bot_username, text):
        """
        - завжди шлемо ПРЯМО в бота за username (приватний чат)
        """

        # RPS limit
        await self.rps.acquire()

        send_ts = time.perf_counter()

        # Вимірюємо latency send_message
        t0 = time.perf_counter()
        async with self.pyro_lock:
            # бот у приваті за username
            msg = await self.client.send_message(self.bot_username, text)
        t1 = time.perf_counter()

        send_latency = t1 - t0
        msg_id = msg.id

        # Логи
        self.arrival_log[msg_id] = send_ts
        self.latency_log[msg_id] = send_latency

        # Для подальшого матчу відповіді
        self.pending[msg_id] = send_ts


    async def _reply_consumer(self):
        """
        - матч через reply_to_message_id
        - рахуємо service_time = resp_ts - send_ts
        """
        while True:
            reply_msg, resp_ts = await self.reply_queue.get()

            send_id = reply_msg.reply_to_message_id
            if send_id is None:
                continue  # відповідь не як reply

            send_ts = self.pending.pop(send_id, None)
            if send_ts is None:
                continue  # немає такого запиту в pending

            service = resp_ts - send_ts

            # лог по msg_id
            self.service_log[send_id] = service

            print(
                f"[REPLY] id={reply_msg.id} reply_to={send_id} "
                f"service={service:.4f}s "
                f"text={repr(reply_msg.text)[:40]}..."
            )


    async def wait_until_all_processed(self, timeout=100):
        start = time.perf_counter()
        while True:
            if len(self.pending) == 0 and self.reply_queue.empty():
                return True

            if time.perf_counter() - start > timeout:
                return False

            await asyncio.sleep(0.05)