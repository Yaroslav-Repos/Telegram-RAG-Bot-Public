import asyncio
import random

# розширювати за необхідністю
TEST_MESSAGES = [
    "/rule",
    "/start",
    "/order"
]


class PyroLoadGenerator:

    def __init__(self, clients, bot_username, messages_per_client=20, delay=(3, 5)):
        self.clients = clients
        self.bot_username = bot_username
        self.messages_per_client = messages_per_client
        self.delay_min, self.delay_max = delay

    async def _run_one(self, client):
        for _ in range(self.messages_per_client):
            text = random.choice(TEST_MESSAGES)
            # Стара сигнатура: send_to_bot(bot_username, text)
            asyncio.create_task(client.send_to_bot(self.bot_username, text))
            await asyncio.sleep(random.uniform(self.delay_min, self.delay_max))

    async def run_all(self):
        tasks = [asyncio.create_task(self._run_one(c)) for c in self.clients]
        await asyncio.gather(*tasks)