import time
import asyncio
import logging
from aiogram.types import BotCommand, BotCommandScopeDefault
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from config import TELEGRAM_TOKEN
from handlers import handle_message, handle_pagination_callback, PageCallbackFilter
import db
from db import sessions, init_chroma
from order_fsm import router, start_order
from aiogram.fsm.context import FSMContext


logger = logging.getLogger(__name__)


class BotApp:

    def __init__(self, token: str):
        self._validate_token(token)
        self.token = token
        self.bot: Bot | None = None
        self.dp: Dispatcher | None = None

    def _validate_token(self, token: str) -> None:
        if not token or not isinstance(token, str) or token.strip() == "":
            logger.error("TELEGRAM_TOKEN is missing or empty. Aborting startup.")
            raise RuntimeError("Invalid TELEGRAM_TOKEN in config. Set TELEGRAM_TOKEN environment variable.")

    def create_clients(self) -> None:
        """Створює Bot та Dispatcher під час старту програми."""


        self.bot = Bot(token=self.token)
        self.dp = Dispatcher()

    async def set_commands(self) -> None:
        commands = [
            BotCommand(command="/start", description="Старт бота"),
            BotCommand(command="/clear_history", description="Очистити історію чату"),
            BotCommand(command="/rule", description="Правила спілкування з ботом"),
            BotCommand(command="/order", description="Оформити замовлення"),
        ]

        assert self.bot is not None
        await self.bot.set_my_commands(commands, BotCommandScopeDefault())

    def register_handlers(self) -> None:

        assert self.dp is not None

        # callback pagination
        self.dp.callback_query.register(handle_pagination_callback, PageCallbackFilter())

        # FSM router
        self.dp.include_router(router)


        @self.dp.message(Command("order"))
        async def _cmd_order(message: types.Message, state: FSMContext):
            await start_order(message, state)

        @self.dp.message(Command("start"))
        async def _cmd_start(message: types.Message):
            t1 = time.perf_counter()
            await message.reply(
                "👋 Вітаємо!\n🤖 У чаті використано генеративну AI-модель!\n💬 Бот спілкується природною мовою\n✍️ Просто напишіть запит, що Вас цікавить —\n🧠 Ми опрацюємо його як типовий AI-чат-бот!"
            )
            t2 = time.perf_counter()
            logger.debug("Handled /start in %.3fs", (t2 - t1))

        @self.dp.message(Command("clear_history"))
        async def _clear_history(message: types.Message):

            user = message.from_user
            if user is None or not hasattr(user, "id"):
                await message.reply("📭 Користувача не знайдено.")
                return

            user_id = int(user.id)

            try:
                result = await sessions.update_one({"user_id": user_id}, {"$set": {"history": []}})
            except Exception as e:
                logger.exception("Failed to clear history for user %s", user_id)
                await message.reply("❌ Помилка при очищенні історії.")
                return

            if getattr(result, "matched_count", 0):
                await message.reply("🧹 Історію чат-боту очищено.")
            else:
                await message.reply("📭 Історія була порожня або не знайдена.")

        @self.dp.message(Command("rule"))
        async def _send_rule(message: types.Message):
            instruction_text = (
                "📋 **Як правильно ставити запитання боту:**\n\n"
                "1. Формулюй чітко та конкретно, описуючи свою проблему або запит.\n"
                "2. Для пошуку замовлення або продукту — вкажи назву, характеристики або артикул.\n"
                "3. Використовуй просту і зрозумілу мову.\n"
                "4. Якщо сумніваєшся, пиши коротко, бот допоможе уточнити.\n\n"
                "Це допоможе боту краще зрозуміти твоє питання і дати точну відповідь."
            )
            await message.reply(instruction_text, parse_mode="Markdown")

        @self.dp.message()
        async def _universal_handler(message: types.Message):
            await handle_message(message)

    async def start(self) -> None:

        self.create_clients()
        assert self.dp is not None and self.bot is not None


        self.register_handlers()


        await init_chroma()
        logger.info("Chroma initialized (client ready)")


        await self.set_commands()

        logger.info("Starting polling")
        await self.dp.start_polling(self.bot, skip_updates=True)


async def main():

    logging.basicConfig(level=logging.INFO)

    try:
        app = BotApp(TELEGRAM_TOKEN)
        await app.start()
    except Exception:
        logger.exception("Failed to start BotApp")


if __name__ == "__main__":
    asyncio.run(main())
