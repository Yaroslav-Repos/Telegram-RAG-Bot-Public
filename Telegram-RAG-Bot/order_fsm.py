from aiogram.fsm.state import State, StatesGroup
from aiogram import types, Router, F
from aiogram.fsm.context import FSMContext
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery
from bson.objectid import ObjectId
from datetime import datetime
from aiogram.filters import Command

from repositories import ProductRepository, OrderRepository


product_repo = ProductRepository()
order_repo = OrderRepository()


class OrderStates(StatesGroup):
    waiting_for_category = State()
    waiting_for_product = State()
    waiting_for_price_confirmation = State()
    waiting_for_quantity = State()
    waiting_for_confirmation = State()


router = Router()


class OrderService:
    def __init__(self, router: Router, product_repo: ProductRepository, order_repo: OrderRepository):
        self.router = router
        self.product_repo = product_repo
        self.order_repo = order_repo
        self.category_map: dict = {}
        self.CATEGORY_PREFIX = "cat"
        self.PAGE_SIZE = 5


        router.callback_query.register(self.select_category, F.data.startswith("category:"))
        router.callback_query.register(self.paginate_products, F.data.startswith("prodpage:"))
        router.callback_query.register(self.select_product, F.data.startswith("product:"))
        router.callback_query.register(self.confirm_price, F.data.startswith("confirm_price"))
        router.callback_query.register(self.set_quantity, F.data.startswith("quantity:"))
        router.callback_query.register(self.place_order, F.data == "place_order")
        router.callback_query.register(self.restart_order, F.data == "restart")
        router.message.register(self.catch_all_messages)

    async def start_order(self, message: types.Message, state: FSMContext):
        data = await state.get_data()
        old_msg = data.get("active_message")

        if old_msg:
            try:
                await message.bot.edit_message_text(
                    chat_id=message.chat.id,
                    message_id=old_msg,
                    text="❌ Попереднє замовлення перервано. Створено нове.",
                )
            except Exception:
                pass

        await state.clear()
        await state.set_state(OrderStates.waiting_for_category)

        self.category_map.clear()
        categories = await self.product_repo.get_categories()
        keyboard_buttons = []

        for i, cat in enumerate(categories):
            short_id = f"{self.CATEGORY_PREFIX}{i}"
            self.category_map[short_id] = cat
            keyboard_buttons.append([
                InlineKeyboardButton(text=cat, callback_data=f"category:{short_id}")
            ])

        keyboard = InlineKeyboardMarkup(inline_keyboard=keyboard_buttons)
        sent = await message.reply("📂 Оберіть категорію товару:", reply_markup=keyboard)
        await state.update_data(active_message=sent.message_id)

    async def select_category(self, callback: CallbackQuery, state: FSMContext):
        short_id = callback.data.split(":")[1]
        category = self.category_map.get(short_id)

        if not category:
            await callback.message.edit_text("⚠️ Категорію не знайдено.")
            await state.update_data(active_message=callback.message.message_id)
            return

        total_count = await self.product_repo.count_by_category(category)

        if total_count == 0:
            await callback.message.edit_text("⚠️ У цій категорії немає товарів.")
            await state.update_data(active_message=callback.message.message_id)
            return

        await state.update_data(category=category, page=0, total_count=total_count)
        await state.set_state(OrderStates.waiting_for_product)
        await self.render_product_page(callback, state)

    async def render_product_page(self, callback: CallbackQuery, state: FSMContext):
        data = await state.get_data()
        category = data["category"]
        page = data["page"]
        total_count = data["total_count"]

        products_list = await self.product_repo.get_products_by_category(category, page, self.PAGE_SIZE)

        keyboard_rows = [
            [
                InlineKeyboardButton(text=f"{p['name']} – {p['price']}₴", callback_data=f"product:{str(p['_id'])}")
            ]
            for p in products_list
        ]

        pagination = []
        if page > 0:
            pagination.append(InlineKeyboardButton(text="⬅️", callback_data=f"prodpage:{page-1}"))
        if (page + 1) * self.PAGE_SIZE < total_count:
            pagination.append(InlineKeyboardButton(text="➡️", callback_data=f"prodpage:{page+1}"))
        if pagination:
            keyboard_rows.append(pagination)

        keyboard = InlineKeyboardMarkup(inline_keyboard=keyboard_rows)

        await callback.message.edit_text(
            f"🛒 Оберіть товар з категорії: {category}\n"
            f"📄 Сторінка {page+1} із {((total_count - 1) // self.PAGE_SIZE) + 1}",
            reply_markup=keyboard,
        )
        await state.update_data(active_message=callback.message.message_id)

    async def paginate_products(self, callback: CallbackQuery, state: FSMContext):
        new_page = int(callback.data.split(":")[1])
        data = await state.get_data()
        total_pages = (data["total_count"] - 1) // self.PAGE_SIZE

        if new_page < 0 or new_page > total_pages:
            await callback.answer("⚠️ Сторінка недоступна.")
            return

        await state.update_data(page=new_page)
        await self.render_product_page(callback, state)

    async def select_product(self, callback: CallbackQuery, state: FSMContext):
        product_id = callback.data.split(":")[1]
        product = await self.product_repo.get_by_objectid(ObjectId(product_id))

        if not product:
            await callback.message.edit_text("⚠️ Товар не знайдено.")
            await state.update_data(active_message=callback.message.message_id)
            return

        if product.get("stock", 0) == 0:
            await callback.message.edit_text("⚠️ На жаль, цей товар наразі відсутній в наявності.")
            await state.update_data(active_message=callback.message.message_id)
            return

        await state.update_data(product=product)

        price = product["price"]
        discount = product.get("discount", 0)
        final_price = round(price * (1 - discount))

        msg = (
            f"✅ Ви обрали: {product['name']}\n"
            f"Ціна: {price}₴\n"
            f"{'💸 Знижка: ' + str(int(discount * 100)) + '%' if discount else ''}\n"
            f"💰 До оплати за одиницю: {final_price}₴\n\n"
            f"Підтверджуєте вибір?"
        )

        keyboard = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="Так", callback_data="confirm_price"), InlineKeyboardButton(text="Назад", callback_data="restart")]])

        await state.set_state(OrderStates.waiting_for_price_confirmation)
        await callback.message.edit_text(msg, reply_markup=keyboard)
        await state.update_data(active_message=callback.message.message_id)

    async def confirm_price(self, callback: CallbackQuery, state: FSMContext):
        data = await state.get_data()
        product = data["product"]
        stock = product["stock"]

        max_quantity = min(5, stock)
        keyboard = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text=str(i), callback_data=f"quantity:{i}")] for i in range(1, max_quantity + 1)])

        await state.set_state(OrderStates.waiting_for_quantity)
        await callback.message.edit_text(f"🔢 Скільки одиниць '{product['name']}' ви хочете замовити?", reply_markup=keyboard)
        await state.update_data(active_message=callback.message.message_id)

    async def set_quantity(self, callback: CallbackQuery, state: FSMContext):
        quantity = int(callback.data.split(":")[1])
        data = await state.get_data()
        product = data["product"]

        price = product["price"]
        discount = product.get("discount", 0)
        final_price = round(price * (1 - discount)) * quantity

        await state.update_data(quantity=quantity, total_price=final_price)

        keyboard = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="✅ Підтвердити замовлення", callback_data="place_order")], [InlineKeyboardButton(text="❌ Скасувати", callback_data="restart")]])

        await state.set_state(OrderStates.waiting_for_confirmation)
        await callback.message.edit_text(f"🧾 Ви замовляєте: {product['name']} x {quantity}\n💰 Загальна сума: {final_price} грн\n\nНатисніть нижче, щоб підтвердити замовлення:", reply_markup=keyboard)
        await state.update_data(active_message=callback.message.message_id)

    async def place_order(self, callback: CallbackQuery, state: FSMContext):
        data = await state.get_data()
        user_id = str(callback.from_user.id)

        product = data["product"]
        quantity = data["quantity"]
        total_price = data["total_price"]

        last_order = await self.order_repo.get_last_order()

        if last_order and last_order.get("order_id") and last_order["order_id"].isdigit():
            next_order_id = str(int(last_order["order_id"]) + 1)
        else:
            next_order_id = "1"

        order = {
            "order_id": next_order_id,
            "customer_id": user_id,
            "product_id": product["product_id"],
            "quantity": quantity,
            "total_price": total_price,
            "order_date": datetime.now().strftime("%Y-%m-%d"),
            "status": "Очікує підтвердження",
            "shipping_address": "Не вказано",
            "contact_info": {"phone": "", "email": ""},
        }

        await self.order_repo.insert(order)
        await self.product_repo.decrease_stock(product["_id"], quantity)

        await state.clear()
        await state.update_data(active_message=callback.message.message_id)

        await callback.message.edit_text(
            f"✅ Ваше замовлення прийнято!\n🆔 Номер: {order['order_id']}\n📦 Товар: {product['name']}\n🔢 Кількість: {quantity}\n💰 Сума: {total_price} грн\n💸 Перейдіть до оплати"
        )

    async def restart_order(self, callback: CallbackQuery, state: FSMContext):
        current_state = await state.get_state()
        if current_state is None:
            await callback.answer("Немає активної операції, яку можна скасувати.", show_alert=True)
            return

        await state.clear()
        await callback.message.edit_reply_markup(reply_markup=None)
        await callback.message.answer("❌ Замовлення скасовано. Ви можете почати спочатку.")

    async def catch_all_messages(self, message: types.Message, state: FSMContext):
        current_state = await state.get_state()
        if current_state is not None:
            await message.reply("Використовуйте кнопки нижче для вибору.")


order_service = OrderService(router, product_repo, order_repo)
start_order = order_service.start_order
