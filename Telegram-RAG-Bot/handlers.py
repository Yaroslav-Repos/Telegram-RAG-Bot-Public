import asyncio
from aiogram import types
import db
from db import orders, products, stores, services
from ai_assistant import (
    analyze_action,
    analyze_message,
    respond_to_data,
    respond_to_other,
    vector_context_from_chroma,
    gemini_call
)
import traceback
import json

ACTION_MAP = {
    "find_order": ("order_template", orders),
    "find_product": ("product_template", products),
    "find_store": ("store_template", stores),
}

from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.filters import BaseFilter

PAGE_SIZE = 3


class ChatHandler:

    def __init__(self):
        self.user_results: dict = {}

    def make_pagination_keyboard(self, results: list, page: int, total: int):
        buttons = []
        if page > 0:
            buttons.append(InlineKeyboardButton(text="⬅️ Назад", callback_data=f"pgn:{page-1}"))
        if (page + 1) * PAGE_SIZE < total:
            buttons.append(InlineKeyboardButton(text="➡️ Далі", callback_data=f"pgn:{page+1}"))

        if not buttons:
            return None

        inline_keyboard = [buttons[i:i+2] for i in range(0, len(buttons), 2)]
        return InlineKeyboardMarkup(inline_keyboard=inline_keyboard)

    async def handle_message(self, message: types.Message):
        try:
            user_id = message.from_user.id
            text = message.text.strip()

            # Очищення попередньої пагінації
            if user_id in self.user_results:
                ctx = self.user_results[user_id]
                msg_id = ctx.get("message_id")
                if msg_id:
                    try:
                        await message.bot.edit_message_reply_markup(
                            chat_id=message.chat.id,
                            message_id=msg_id,
                            reply_markup=None,
                        )
                    except Exception as e:
                        print(f"Помилка видалення старої клавіатури: {e}")

                del self.user_results[user_id]

            # Аналіз дії
            action_data = await analyze_action(user_id, text)

            if not action_data or not action_data.get("action"):
                await message.reply("⚠️ Не вдалося визначити дію. Спробуйте інакше.")
                return

            action = action_data["action"]

            # Простий тип дій
            if action == "create_service_request":
                await message.reply("✏️ Ми зв'яжемо вас з нашим оператором, очікуйте!")
                return

            if action == "product_order":
                await message.reply("Зацікавив продукт?\n✏️ Для створення замовлення введіть команду:\n/order")
                return

            if action == "other_action":
                reply = await respond_to_other(user_id, text)
                await message.reply(reply)
                return

            if action not in ACTION_MAP:
                await message.reply("⚠️ Невідома дія. Спробуйте сформулювати запит інакше.")
                return

            # Якщо шукаємо замовлення — додаємо фільтр по користувачу
            if action == "find_order":
                text += f'. Запит тільки для користувача: customer_id: "{user_id}"'

            template_name, collection = ACTION_MAP[action]

            # AI формує Mongo pipeline 
            pipeline = await analyze_message(user_id, text, action, template_name)

            try:
                if not pipeline or not isinstance(pipeline, list):
                    raise ValueError("Invalid or empty pipeline")

                # Motor: async aggregate (validated)
                result = await db.safe_aggregate(collection, pipeline, limit=5)

                if len(result) == 0:
                    raise ValueError("Empty pipeline result")

                total_results = len(result)
                first_page = result[:PAGE_SIZE]


                reply = await respond_to_data(user_id, first_page, text)

                if total_results > PAGE_SIZE:
                    keyboard = self.make_pagination_keyboard(result, page=0, total=total_results)

                    sent_message = await message.reply(reply, reply_markup=keyboard)
                    self.user_results[user_id] = {
                        "data": result,
                        "text": text,
                        "message_id": sent_message.message_id,
                    }
                else:
                    await message.reply(reply)

            except Exception as e:
                print(f"Mongo pipeline error or no results: {e}")

                # Семантичний пошук Chroma
                vec_ctx = await vector_context_from_chroma(text)
                print(vec_ctx)

                if vec_ctx:
                    prompt = f"""
Користувач написав: "{text}"
{vec_ctx}

Сформуй коротку, логічну відповідь українською на основі наведених семантично схожих товарів.
"""
                    reply = await gemini_call(user_id, prompt, True)
                    await message.reply(reply)

                else:
                    await message.reply("⚠️ Не вдалося знайти результати навіть через семантичний пошук.")

        except Exception:
            traceback.print_exc()
            await message.reply("❌ Виникла внутрішня помилка. Спробуйте сформувати запит по іншому.")
    async def handle_pagination_callback(self, callback_query: types.CallbackQuery):
        """Handle pagination callbacks for this chat handler instance."""
        user_id = callback_query.from_user.id
        user_data = self.user_results.get(user_id)

        if not user_data:
            await callback_query.answer("⚠️ Результати більше не доступні.")
            return

        try:
            page = int(callback_query.data.split(":")[1])
        except (IndexError, ValueError):
            await callback_query.answer("Помилка пагінації.")
            return

        start = page * PAGE_SIZE
        end = start + PAGE_SIZE
        data = user_data["data"]
        text = user_data["text"]

        if start >= len(data):
            await callback_query.answer("⚠️ Немає такої сторінки.")
            return

        page_data = data[start:end]

        reply = await respond_to_data(user_id, page_data, text)

        keyboard = self.make_pagination_keyboard(data, page, len(data))

        if keyboard:
            edited_message = await callback_query.message.edit_text(reply, reply_markup=keyboard)
        else:
            edited_message = await callback_query.message.edit_text(reply)

        self.user_results[user_id]["message_id"] = edited_message.message_id

        await callback_query.answer()


class PageCallbackFilter(BaseFilter):
    async def __call__(self, callback: types.CallbackQuery) -> bool:
        return callback.data is not None and callback.data.startswith("pgn:")


chat_handler = ChatHandler()


handle_message = chat_handler.handle_message
handle_pagination_callback = chat_handler.handle_pagination_callback
