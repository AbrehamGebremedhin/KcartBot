"""High-level chat service orchestrating sessions for KCartBot."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.agents.agent import AgentTurn, KcartAgent
from app.db.models import (
	FlashSaleStatus,
	SupplierProductStatus,
	TransactionStatus,
	UserRole,
)
from app.db.repository.flash_sale_repository import FlashSaleRepository
from app.db.repository.order_item_repository import OrderItemRepository
from app.db.repository.supplier_product_repository import SupplierProductRepository
from app.db.repository.user_repository import UserRepository


@dataclass
class SessionState:
	stage: str = "await_role"
	user_role: Optional[str] = None
	has_account: Optional[bool] = None
	name: Optional[str] = None
	phone: Optional[str] = None
	user_id: Optional[int] = None
	context: Dict[str, Any] = field(default_factory=dict)
	summary_sent: bool = False


@dataclass
class LoginResult:
	handled: bool
	response: Optional[str] = None
	login_completed: bool = False


class ChatService:
	"""Stateful service managing chat sessions with the LangChain agent."""

	def __init__(self) -> None:
		self._agent = KcartAgent()
		self._history: Dict[str, List[BaseMessage]] = {}
		self._states: Dict[str, SessionState] = {}

	async def send_message(
		self,
		session_id: str,
		message: str,
		*,
		context: Optional[Dict[str, Any]] = None,
	) -> Dict[str, Any]:
		"""Process a user message for a session and return agent output."""
		history = self._history.get(session_id, [])
		state = self._states.setdefault(session_id, SessionState())

		login_result = await self._handle_login_flow(state, message)
		if login_result.handled:
			history.append(HumanMessage(content=message))
			if login_result.response:
				history.append(AIMessage(content=login_result.response))
			self._history[session_id] = history
			return {
				"response": login_result.response or "",
				"intent": None,
				"flow": None,
				"classifier_output": None,
				"tool_calls": [],
				"trace": {
					"login_flow": True,
					"stage": state.stage,
					"login_completed": login_result.login_completed,
				},
			}

		merged_context: Dict[str, Any] = {}
		if state.context:
			merged_context.update(state.context)
		if context:
			merged_context.update(context)

		turn: AgentTurn = await self._agent.ainvoke(
			message,
			chat_history=history,
			context=merged_context or None,
		)
		self._history[session_id] = turn.history

		return {
			"response": turn.response,
			"intent": turn.intent,
			"flow": turn.flow,
			"classifier_output": turn.classifier_output,
			"tool_calls": turn.tool_calls,
			"trace": turn.trace,
		}

	def reset_session(self, session_id: str) -> None:
		"""Clear stored history for a session."""
		self._history.pop(session_id, None)
		self._states.pop(session_id, None)

	def get_session_history(self, session_id: str) -> List[Dict[str, str]]:
		"""Expose stored history as serialised entries for external use."""
		history = self._history.get(session_id, [])
		serialised: List[Dict[str, str]] = []
		for message in history:
			if isinstance(message, HumanMessage):
				serialised.append({"role": "user", "content": message.content})
			elif isinstance(message, AIMessage):
				serialised.append({"role": "assistant", "content": message.content})
			else:
				serialised.append({"role": "system", "content": getattr(message, "content", "")})
		return serialised

	async def aclose(self) -> None:
		await self._agent.aclose()

	def close(self) -> None:
		import asyncio

		try:
			loop = asyncio.get_running_loop()
			if loop.is_running():
				raise RuntimeError("Synchronous close not supported while an event loop is running. Await 'aclose()'.")
		except RuntimeError:
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
		try:
			loop.run_until_complete(self.aclose())
		finally:
			if loop.is_running():
				return
			loop.close()
			asyncio.set_event_loop(None)

	async def _handle_login_flow(self, state: SessionState, message: str) -> LoginResult:
		text = (message or "").strip()
		lower = text.lower()

		if state.stage == "await_role":
			if "supplier" in lower:
				state.user_role = "supplier"
			elif "customer" in lower:
				state.user_role = "customer"
			else:
				return LoginResult(
					handled=True,
					response="Hi there! Are you shopping as a customer or managing stock as a supplier?",
				)
			state.stage = "await_account_status"
			prompt = (
				"Great! Do you already have an account or are you brand new?"
				if state.user_role == "customer"
				else "Awesome. Do you already have a supplier account or are you getting started?"
			)
			return LoginResult(handled=True, response=prompt)

		if state.stage == "await_account_status":
			new_keywords = [
				"new",
				"don't have",
				"do not have",
				"no account",
				"not yet",
				"getting started",
				"get started",
			]
			existing_keywords = ["have", "account", "existing", "yes", "sure"]
			if any(keyword in lower for keyword in new_keywords):
				state.has_account = False
				state.stage = "authenticated"
				state.context = {
					"user": {
						"role": state.user_role,
						"status": "new",
					},
				}
				message_out = (
					"Perfect! I'll set you up as we chat and capture any details when you're ready."
					if state.user_role == "customer"
					else "Great! We'll help you onboard and get your catalog ready."
				)
				return LoginResult(handled=True, response=message_out, login_completed=True)
			if any(keyword in lower for keyword in existing_keywords):
				state.has_account = True
				state.stage = "await_name"
				return LoginResult(handled=True, response="Perfect! What's the name on the account?")
			return LoginResult(
				handled=True,
				response="Just let me know if you're brand new or already have an account, and we'll get moving!",
			)

		if state.stage == "await_name":
			if not text:
				return LoginResult(handled=True, response="Could you share the name on the account?")
			state.name = text
			state.stage = "await_phone"
			return LoginResult(
				handled=True,
				response=f"Thanks {text}! What's the phone number linked to your account?",
			)

		if state.stage == "await_phone":
			digits = re.sub(r"\D", "", text)
			if len(digits) < 9:
				return LoginResult(
					handled=True,
					response="I didn't catch a valid phone number. Could you resend it (including country code if possible)?",
				)
			state.phone = digits
			user = await self._match_user(state)
			if not user:
				state.stage = "await_account_status"
				state.name = None
				state.phone = None
				return LoginResult(
					handled=True,
					response=(
						"Hmm, I couldn't find an account with that name and number. Are you new, or do you want to try again with different details?"
					),
				)

			state.stage = "authenticated"
			state.user_id = user.user_id
			state.name = user.name
			state.phone = user.phone
			state.user_role = user.role.value
			state.context = {
				"user": {
					"id": user.user_id,
					"role": user.role.value,
					"status": "existing",
					"name": user.name,
					"phone": user.phone,
					"default_location": user.default_location,
				},
			}
			welcome = f"Welcome back, {user.name}!"
			if state.user_role == "supplier":
				summary = await self._build_supplier_summary(user.user_id)
				response_text = f"{welcome} {summary}"
			else:
				response_text = f"{welcome} Let's get you what you need today."
			return LoginResult(handled=True, response=response_text, login_completed=True)

		return LoginResult(handled=False)

	async def _match_user(self, state: SessionState):
		filters: Dict[str, Any] = {}
		if state.name:
			filters["name"] = state.name
		if state.phone:
			filters["phone"] = state.phone
		if state.user_role:
			filters["role"] = UserRole(state.user_role)
		users = await UserRepository.list_users(filters)
		return users[0] if users else None

	async def _build_supplier_summary(self, supplier_id: int) -> str:
		expiring_products = await SupplierProductRepository.generate_flash_sale_proposals(
			supplier_id,
			within_days=3,
		)
		pending_items = await OrderItemRepository.list_order_items(
			{
				"supplier_id": supplier_id,
				"order__status": TransactionStatus.PENDING,
			}
		)
		pending_orders = len({item.order_id for item in pending_items})
		active_sales = await FlashSaleRepository.list_flash_sales(
			{
				"supplier_id": supplier_id,
				"status": FlashSaleStatus.ACTIVE,
			}
		)
		scheduled_sales = await FlashSaleRepository.list_flash_sales(
			{
				"supplier_id": supplier_id,
				"status": FlashSaleStatus.SCHEDULED,
			}
		)
		proposed_sales = await FlashSaleRepository.list_flash_sales(
			{
				"supplier_id": supplier_id,
				"status": FlashSaleStatus.PROPOSED,
			}
		)
		valid_proposals: List[Any] = []
		for sale in proposed_sales:
			await sale.fetch_related("product", "supplier_product")
			supplier_product = getattr(sale, "supplier_product", None)
			sp_status = getattr(supplier_product, "status", None)
			sp_expiry = getattr(supplier_product, "expiry_date", None)
			if (
				sp_status in {SupplierProductStatus.ACTIVE, SupplierProductStatus.ON_SALE}
				and sp_expiry is not None
				and sp_expiry >= date.today()
			):
				valid_proposals.append(sale)
			else:
				await FlashSaleRepository.cancel_flash_sale(sale.id)

		lines: List[str] = []
		if pending_orders:
			lines.append(
				f"You have {pending_orders} pending order{'s' if pending_orders != 1 else ''} awaiting action "
				f"covering {len(pending_items)} line item{'s' if len(pending_items) != 1 else ''}."
			)
		else:
			lines.append("No new orders waiting right now — nice and clear!")

		if active_sales:
			top_discount = max(sale.discount_percent for sale in active_sales)
			lines.append(
				f"Active flash sales: {len(active_sales)} (top discount {top_discount:.0f}%)."
			)
		else:
			lines.append("No active flash sales at the moment.")

		if scheduled_sales:
			next_sale = min(scheduled_sales, key=lambda sale: sale.start_date)
			lines.append(
				f"Next scheduled flash sale kicks off on {next_sale.start_date.strftime('%d %b %Y %H:%M')}."
			)

		if valid_proposals:
			product_names = [sale.product.product_name_en for sale in valid_proposals if getattr(sale, "product", None)]
			recommendation = ", ".join(product_names[:3]) if product_names else f"{len(valid_proposals)} items"
			lines.append(
				f"⚠️ {len(valid_proposals)} item{'s' if len(valid_proposals) != 1 else ''} close to expiry ready for flash sale ({recommendation}). "
				"Tell me to accept or decline a proposal when you're ready."
			)
		elif expiring_products:
			lines.append(
				"Heads-up: you have stock approaching expiry. I can prep flash sale offers whenever you say the word."
			)

		return " ".join(lines)
