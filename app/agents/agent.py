"""KCartBot Agent - Orchestrates LLM and tools for customer/supplier conversations."""

from __future__ import annotations

import datetime
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from app.services.llm_service import LLMService
from app.tools.database_tool import DatabaseAccessTool
from app.tools.date_tool import DateResolverTool
from app.tools.generate_image import ImageGeneratorTool
from app.tools.intent_classifier import IntentClassifierTool
from app.tools.search_context import VectorSearchTool

logger = logging.getLogger(__name__)


class Agent:
    """Main agent that orchestrates LLM and tools for KCartBot conversations."""

    def __init__(self) -> None:
        """Initialize the agent with all required tools and services."""
        self.llm_service = LLMService()
        self.intent_classifier = IntentClassifierTool()  # Create its own LLM service instance
        self.database_tool = DatabaseAccessTool()
        self.vector_search = VectorSearchTool()
        self.date_resolver = DateResolverTool(llm_service=self.llm_service)
        self.image_generator = ImageGeneratorTool()

        # Tool registry for dynamic access
        self.tools = {
            "intent_classifier": self.intent_classifier,
            "database_access": self.database_tool,
            "vector_search": self.vector_search,
            "date_resolver": self.date_resolver,
            "image_generator": self.image_generator,
        }

    async def process_message(
        self,
        user_message: str,
        session_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a user message and return a response.

        Args:
            user_message: The user's input message
            session_context: Session context including chat history, user info, etc.

        Returns:
            Dict containing response and updated context
        """
        try:
            # Initialize context if not provided
            if session_context is None:
                session_context = {}

            # Extract chat history
            chat_history = session_context.get("chat_history", [])

            # Step 1: Classify intent
            intent_result = await self.intent_classifier.run(
                {"text": user_message},
                context={"chat_history": chat_history}
            )

            intent = intent_result.get("intent", "intent.unknown")
            flow = intent_result.get("flow", "unknown")
            filled_slots = intent_result.get("filled_slots", {})
            missing_slots = intent_result.get("missing_slots", [])
            suggested_tools = intent_result.get("suggested_tools", [])

            logger.info(f"Classified intent: {intent}, flow: {flow}, missing_slots: {missing_slots}")

            # Update session context with current intent and flow
            session_context.update({
                "current_intent": intent,
                "current_flow": flow,
                "filled_slots": filled_slots,
                "missing_slots": missing_slots,
                "last_user_message": user_message,  # Store the original user message
            })

            # Step 2: Handle the conversation based on flow and intent
            if flow == "unknown" or intent == "intent.unknown":
                response = await self._handle_unknown_intent(user_message, chat_history)
            elif flow == "customer":
                response = await self._handle_customer_flow(
                    intent, filled_slots, missing_slots, suggested_tools, session_context
                )
            elif flow == "supplier":
                response = await self._handle_supplier_flow(
                    intent, filled_slots, missing_slots, suggested_tools, session_context
                )
            elif flow == "onboarding":
                response = await self._handle_onboarding_flow(
                    intent, filled_slots, missing_slots, suggested_tools, session_context
                )
            else:
                response = "I'm sorry, I couldn't understand your request. Could you please clarify?"

            # Step 3: Update chat history
            chat_history.append({"role": "user", "content": user_message})
            chat_history.append({"role": "assistant", "content": response})

            # Keep only last 10 exchanges to avoid token overflow
            if len(chat_history) > 20:
                chat_history = chat_history[-20:]

            session_context["chat_history"] = chat_history

            return {
                "response": response,
                "session_context": session_context,
                "intent_info": {
                    "intent": intent,
                    "flow": flow,
                    "filled_slots": filled_slots,
                    "missing_slots": missing_slots,
                }
            }

        except Exception as exc:
            logger.error(f"Error processing message: {exc}")
            return {
                "response": "I'm sorry, I encountered an error. Please try again.",
                "session_context": session_context or {},
                "error": str(exc)
            }

    async def _handle_unknown_intent(
        self, user_message: str, chat_history: List[Dict[str, str]]
    ) -> str:
        """Handle unknown intents by asking for clarification."""
        # Check for simple greetings
        greeting_words = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "greetings"]
        if any(word in user_message.lower() for word in greeting_words):
            return "Hello! Welcome to KCartBot, your fresh produce marketplace assistant. Are you a customer looking to place an order, or a supplier managing inventory?"

        # Check if this might be a confirmation of previous context
        if chat_history:
            last_assistant_msg = None
            for msg in reversed(chat_history):
                if msg.get("role") == "assistant":
                    last_assistant_msg = msg.get("content", "")
                    break

            if any(word in user_message.lower() for word in ["yes", "okay", "sure", "go ahead", "confirm"]):
                return "Great! Could you please provide more details about what you'd like to do?"

    async def _handle_unknown_intent(
        self, user_message: str, chat_history: List[Dict[str, str]]
    ) -> str:
        """Handle unknown intents by asking for clarification."""
        # Check for simple greetings
        greeting_words = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "greetings"]
        if any(word in user_message.lower() for word in greeting_words):
            return "Hello! Welcome to KCartBot, your fresh produce marketplace assistant. Are you a customer looking to place an order, or a supplier managing inventory?"

        # Check if this might be a confirmation of previous context
        if chat_history:
            last_assistant_msg = None
            for msg in reversed(chat_history):
                if msg.get("role") == "assistant":
                    last_assistant_msg = msg.get("content", "")
                    break

            if any(word in user_message.lower() for word in ["yes", "okay", "sure", "go ahead", "confirm"]):
                return "Great! Could you please provide more details about what you'd like to do?"

        return "I'm not sure what you mean. Are you a customer looking to place an order, or a supplier managing inventory?"

    async def _handle_customer_flow(
        self,
        intent: str,
        filled_slots: Dict[str, Any],
        missing_slots: List[str],
        suggested_tools: List[str],
        session_context: Dict[str, Any]
    ) -> str:
        """Handle customer flow intents."""
        try:
            if intent == "intent.customer.register":
                return await self._handle_customer_registration(filled_slots, missing_slots, session_context)

            elif intent == "intent.customer.check_availability":
                return await self._handle_product_availability(filled_slots, session_context)

            elif intent == "intent.customer.storage_advice":
                return await self._handle_storage_advice(filled_slots, session_context)

            elif intent == "intent.customer.nutrition_query":
                return await self._handle_nutrition_query(filled_slots, session_context)

            elif intent == "intent.customer.seasonal_query":
                return await self._handle_seasonal_query(filled_slots, session_context)

            elif intent == "intent.customer.what_is_in_season":
                return await self._handle_in_season_query(filled_slots, session_context)

            elif intent == "intent.customer.general_advisory":
                return await self._handle_general_advisory(filled_slots, session_context)

            elif intent == "intent.customer.place_order":
                return await self._handle_place_order(filled_slots, missing_slots, session_context)

            elif intent == "intent.customer.set_delivery_date":
                return await self._handle_set_delivery_date(filled_slots, session_context)

            elif intent == "intent.customer.set_delivery_location":
                return await self._handle_set_delivery_location(filled_slots, session_context)

            elif intent == "intent.customer.confirm_payment":
                return await self._handle_confirm_payment(filled_slots, session_context)

            elif intent == "intent.customer.check_deliveries":
                return await self._handle_customer_check_deliveries(filled_slots, session_context)

            else:
                return "I'm here to help with your fresh produce needs. What would you like to do?"

        except Exception as exc:
            logger.error(f"Error in customer flow: {exc}")
            return "I encountered an issue processing your request. Please try again."

    async def _handle_supplier_flow(
        self,
        intent: str,
        filled_slots: Dict[str, Any],
        missing_slots: List[str],
        suggested_tools: List[str],
        session_context: Dict[str, Any]
    ) -> str:
        """Handle supplier flow intents."""
        try:
            if intent == "intent.supplier.register":
                return await self._handle_supplier_registration(filled_slots, missing_slots, session_context)

            elif intent == "intent.supplier.add_product":
                return await self._handle_add_product(filled_slots, missing_slots, session_context)

            elif intent == "intent.supplier.set_quantity":
                return await self._handle_set_quantity(filled_slots, session_context)

            elif intent == "intent.supplier.update_inventory":
                return await self._handle_update_inventory(filled_slots, session_context)

            elif intent == "intent.supplier.set_delivery_dates":
                return await self._handle_set_delivery_dates(filled_slots, session_context)

            elif intent == "intent.supplier.set_expiry_date":
                return await self._handle_set_expiry_date(filled_slots, session_context)

            elif intent == "intent.supplier.set_price":
                return await self._handle_set_price(filled_slots, session_context)

            elif intent == "intent.supplier.request_pricing_insight":
                return await self._handle_pricing_insight(filled_slots, session_context)

            elif intent == "intent.supplier.generate_product_image":
                return await self._handle_generate_image(filled_slots, session_context)

            elif intent == "intent.supplier.check_deliveries":
                return await self._handle_supplier_check_deliveries(filled_slots, session_context)

            elif intent == "intent.supplier.check_stock":
                return await self._handle_check_stock(session_context)

            elif intent == "intent.supplier.view_expiring_products":
                return await self._handle_view_expiring_products(filled_slots, session_context)

            elif intent == "intent.supplier.accept_flash_sale":
                return await self._handle_accept_flash_sale(filled_slots, session_context)

            elif intent == "intent.supplier.decline_flash_sale":
                return await self._handle_decline_flash_sale(filled_slots, session_context)

            elif intent == "intent.supplier.view_delivery_schedule":
                return await self._handle_view_delivery_schedule(filled_slots, session_context)

            elif intent == "intent.supplier.check_deliveries_by_date":
                return await self._handle_check_deliveries_by_date(filled_slots, session_context)

            elif intent == "intent.supplier.add_to_existing":
                return await self._handle_add_to_existing(session_context)

            elif intent == "intent.supplier.create_new_listing":
                return await self._handle_create_new_listing(session_context)

            elif intent == "intent.customer.nutrition_query":
                return await self._handle_nutrition_query(filled_slots, session_context)

            else:
                return "I'm here to help you manage your inventory and sales. What would you like to do?"

        except Exception as exc:
            logger.error(f"Error in supplier flow: {exc}")
            return "I encountered an issue processing your request. Please try again."

    async def _handle_onboarding_flow(
        self,
        intent: str,
        filled_slots: Dict[str, Any],
        missing_slots: List[str],
        suggested_tools: List[str],
        session_context: Dict[str, Any]
    ) -> str:
        """Handle user onboarding flow."""
        try:
            if intent == "intent.user.is_customer":
                session_context["user_role"] = "customer"
                return "Great! Are you a new customer or do you already have an account with us?"

            elif intent == "intent.user.is_supplier":
                session_context["user_role"] = "supplier"
                return "Excellent! Are you a new supplier or do you already have an account with us?"

            elif intent == "intent.user.has_account":
                user_role = session_context.get("user_role")
                if not user_role:
                    return "Please tell me if you're a customer or supplier first."

                return f"I see you already have an account. What's your name and phone number so I can verify your {user_role} account?"

            elif intent == "intent.user.new_user":
                return await self._handle_new_user_registration(filled_slots, session_context)

            elif intent == "intent.user.verify_account":
                return await self._handle_account_verification(filled_slots, session_context)

            else:
                return "I'm not sure about that. Are you a customer or supplier?"

        except Exception as exc:
            logger.error(f"Error in onboarding flow: {exc}")
            return "I encountered an issue during onboarding. Please try again."

    # Onboarding flow handlers
    async def _handle_account_verification(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Handle verification of existing user account."""
        user_name = filled_slots.get("user_name")
        phone_number = filled_slots.get("phone_number")
        user_role = session_context.get("user_role")

        if not user_name or not phone_number:
            return "I need both your name and phone number to verify your account."

        try:
            # Check if user exists in database
            result = await self.database_tool.run({
                "table": "users",
                "method": "list_users",
                "args": [],
                "kwargs": {"filters": {"name__icontains": user_name, "phone": phone_number, "role": user_role}}
            })

            if result and len(result) > 0:
                user = result[0]
                session_context["user_id"] = user["user_id"]
                session_context["authenticated"] = True

                if user_role == "customer":
                    return f"Welcome back, {user_name}! Your customer account has been verified. How can I help you with your fresh produce needs today?"
                else:
                    # Supplier dashboard - show pending orders and expiring products
                    dashboard_info = await self._get_supplier_dashboard_info(user["user_id"])
                    return f"Welcome back, {user_name}! Your supplier account has been verified. {dashboard_info} How can I help you manage your inventory today?"
            else:
                return f"I couldn't find an account with that name and phone number. Would you like to create a new {user_role} account instead?"

        except Exception as exc:
            logger.error(f"Failed to verify account: {exc}")
            return "I couldn't verify your account right now. Please try again."

    async def _get_supplier_dashboard_info(self, supplier_id: int) -> str:
        """Get dashboard information for supplier login."""
        try:
            dashboard_parts = []

            # Get pending orders
            pending_orders = await self._get_supplier_pending_orders(supplier_id)
            if pending_orders:
                dashboard_parts.append(pending_orders)

            # Get expiring products and flash sale suggestions
            expiring_info = await self._get_supplier_expiring_products_and_suggestions(supplier_id)
            if expiring_info:
                dashboard_parts.append(expiring_info)

            if not dashboard_parts:
                return "You have no pending orders or expiring products at this time."

            return " ".join(dashboard_parts)

        except Exception as exc:
            logger.error(f"Failed to get supplier dashboard info: {exc}")
            return "I couldn't load your dashboard information right now."

    async def _get_supplier_pending_orders(self, supplier_id: int) -> str:
        """Get pending orders for a supplier."""
        try:
            # Query order items for this supplier
            order_items = await self.database_tool.run({
                "table": "order_items",
                "method": "list_order_items",
                "args": [],
                "kwargs": {"filters": {"supplier": supplier_id}}
            })

            if not order_items:
                return ""

            # Filter for pending/confirmed orders
            pending_orders = []
            for item in order_items:
                # Get the transaction for this order item
                transaction = await self.database_tool.run({
                    "table": "transactions",
                    "method": "get_transaction_by_id",
                    "args": [item["order"]["order_id"]],
                    "kwargs": {}
                })

                if transaction and transaction.get("status") in ["Pending", "Confirmed"]:
                    pending_orders.append({
                        "order_id": transaction["order_id"],
                        "customer": transaction.get("user", "Unknown customer"),
                        "product": item.get("product", "Unknown product"),
                        "quantity": item.get("quantity", 0),
                        "unit": item.get("unit", "kg"),
                        "delivery_date": transaction.get("delivery_date"),
                        "status": transaction.get("status", "Unknown")
                    })

            if not pending_orders:
                return ""

            # Format the response as a single line paragraph
            order_descriptions = []
            for order in pending_orders[:5]:  # Show up to 5 orders
                product_name = "Unknown product"
                if isinstance(order["product"], dict):
                    product_name = order["product"].get("product_name_en", "Unknown product")

                customer_name = "Unknown customer"
                if isinstance(order["customer"], dict):
                    customer_name = order["customer"].get("name", "Unknown customer")

                customer_location = "Unknown location"
                if isinstance(order["customer"], dict):
                    customer_location = order["customer"].get("default_location", "Unknown location")

                delivery_info = ""
                if order["delivery_date"]:
                    try:
                        from datetime import datetime
                        if isinstance(order["delivery_date"], str):
                            if 'T' in order["delivery_date"]:
                                delivery_date = order["delivery_date"].split('T')[0]
                            else:
                                delivery_date = order["delivery_date"]
                            date_obj = datetime.fromisoformat(delivery_date.replace('Z', '+00:00'))
                            delivery_info = f" - Delivery: {date_obj.strftime('%b %d')}"
                    except:
                        delivery_info = f" - Delivery: {order['delivery_date']}"

                order_descriptions.append(
                    f"Order {str(order['order_id'])[:8]}...: {product_name} "
                    f"({order['quantity']} {order['unit']}) for {customer_name} in {customer_location}{delivery_info} - {order['status']}"
                )

            if len(pending_orders) > 5:
                order_descriptions.append(f"and {len(pending_orders) - 5} more pending orders")

            return "📦 **Pending Orders:** " + ", ".join(order_descriptions) + "."

        except Exception as exc:
            logger.error(f"Failed to get supplier pending orders: {exc}")
            return ""

    async def _get_supplier_expiring_products_and_suggestions(self, supplier_id: int) -> str:
        """Get expiring products and flash sale suggestions for a supplier."""
        try:
            # Get expiring products (within 7 days)
            expiring_products = await self.database_tool.run({
                "table": "supplier_products",
                "method": "get_expiring_products",
                "args": [supplier_id, 7],
                "kwargs": {}
            })

            if not expiring_products:
                return ""

            # Generate flash sale suggestions
            await self.database_tool.run({
                "table": "supplier_products",
                "method": "generate_flash_sale_proposals",
                "args": [supplier_id],
                "kwargs": {"within_days": 7, "default_discount": 25.0}
            })

            # Get proposed flash sales
            proposed_sales = await self.database_tool.run({
                "table": "flash_sales",
                "method": "list_flash_sales",
                "args": [],
                "kwargs": {"filters": {"supplier": supplier_id, "status": "proposed"}}
            })

            # Format the response
            response_parts = ["⚠️ **Expiring Products (next 7 days):**"]

            for product in expiring_products[:5]:  # Show up to 5 products
                product_name = "Unknown product"
                if isinstance(product.get("product"), dict):
                    product_name = product["product"].get("product_name_en", "Unknown product")

                expiry_date = product.get("expiry_date")
                expiry_info = "Unknown expiry"
                if expiry_date:
                    try:
                        from datetime import datetime
                        if isinstance(expiry_date, str):
                            date_obj = datetime.fromisoformat(expiry_date.replace('Z', '+00:00'))
                            expiry_info = date_obj.strftime('%b %d')
                        else:
                            expiry_info = str(expiry_date)
                    except:
                        expiry_info = str(expiry_date)

                quantity = product.get("quantity_available", 0)
                unit = product.get("unit", "kg")

                response_parts.append(
                    f"• {product_name}: {quantity} {unit} expires {expiry_info}"
                )

            if len(expiring_products) > 5:
                response_parts.append(f"... and {len(expiring_products) - 5} more expiring products.")

            # Add flash sale suggestions
            if proposed_sales:
                response_parts.append("💡 **Flash Sale Suggestions:**")
                response_parts.append("I've created flash sale proposals for your expiring products with 25% discount.")
                response_parts.append("You can accept these to attract customers and reduce waste.")

                for sale in proposed_sales[:3]:  # Show up to 3 suggestions
                    product_name = "Unknown product"
                    if isinstance(sale.get("product"), dict):
                        product_name = sale["product"].get("product_name_en", "Unknown product")

                    discount = sale.get("discount_percent", 0)
                    response_parts.append(f"• {product_name}: {discount}% off until expiry")

            return " ".join(response_parts)

        except Exception as exc:
            logger.error(f"Failed to get supplier expiring products: {exc}")
            return ""

    async def _handle_new_user_registration(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Handle registration of new user."""
        user_role = session_context.get("user_role")
        if not user_role:
            return "Please tell me if you're a customer or supplier first."

        if user_role == "customer":
            user_name = filled_slots.get("customer_name")
            phone_number = filled_slots.get("phone_number")
            default_location = filled_slots.get("default_location")

            if not user_name:
                return "What's your name?"
            if not phone_number:
                return "What's your phone number?"
            if not default_location:
                return "What's your default delivery location?"

            try:
                result = await self.database_tool.run({
                    "table": "users",
                    "method": "create_user",
                    "args": [],
                    "kwargs": {
                        "name": user_name,
                        "phone": phone_number,
                        "default_location": default_location,
                        "preferred_language": "English",  # Default language
                        "role": "customer",
                        "joined_date": datetime.date.today()  # Explicitly set joined date
                    }
                })
                session_context["user_id"] = result.get("user_id")
                session_context["authenticated"] = True
                return f"Welcome {user_name}! Your customer account has been created. How can I help you with your fresh produce needs today?"

            except Exception as exc:
                logger.error(f"Failed to register customer: {exc}")
                return "I couldn't create your account. Please try again."

        else:  # supplier
            supplier_name = filled_slots.get("supplier_name")
            phone_number = filled_slots.get("phone_number")

            if not supplier_name:
                return "What's your business name?"
            if not phone_number:
                return "What's your phone number?"

            try:
                result = await self.database_tool.run({
                    "table": "users",
                    "method": "create_user",
                    "args": [],
                    "kwargs": {
                        "name": supplier_name,
                        "phone": phone_number,
                        "default_location": "",  # Suppliers don't need a default location
                        "preferred_language": "English",  # Default language
                        "role": "supplier",
                        "joined_date": datetime.date.today()  # Explicitly set joined date
                    }
                })
                session_context["user_id"] = result.get("user_id")
                session_context["authenticated"] = True
                return f"Welcome {supplier_name}! Your supplier account has been created. How can I help you manage your inventory today?"

            except Exception as exc:
                logger.error(f"Failed to register supplier: {exc}")
                return "I couldn't create your account. Please try again."

    # Customer flow handlers
    async def _handle_customer_registration(
        self, filled_slots: Dict[str, Any], missing_slots: List[str], session_context: Dict[str, Any]
    ) -> str:
        """Handle customer registration."""
        if missing_slots:
            slot = missing_slots[0]
            if slot == "customer_name":
                return "Welcome! What's your name?"
            elif slot == "phone_number":
                return "Great! What's your phone number?"
            elif slot == "default_location":
                return "Perfect! What's your default delivery location?"

        # All slots filled, register the customer
        try:
            result = await self.database_tool.run({
                "table": "users",
                "method": "create_user",
                "args": [],
                "kwargs": {
                    "name": filled_slots["customer_name"],
                    "phone": filled_slots["phone_number"],
                    "default_location": filled_slots["default_location"],
                    "preferred_language": "English",  # Default language
                    "role": "customer",
                    "joined_date": datetime.date.today()  # Explicitly set joined date
                }
            })
            session_context["user_id"] = result.get("user_id")
            return f"Welcome {filled_slots['customer_name']}! Your account has been created. How can I help you today?"
        except Exception as exc:
            logger.error(f"Failed to register customer: {exc}")
            return "I couldn't create your account. Please try again."

    async def _handle_product_availability(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Check product availability."""
        product_name = filled_slots.get("product_name")
        if not product_name:
            return "What product are you looking for?"

        try:
            # First, try to find the product by name
            product = await self.database_tool.run({
                "table": "products",
                "method": "find_product_by_any_name",
                "args": [product_name],
                "kwargs": {}
            })

            if product:
                # Product found - show availability from suppliers
                result = await self.database_tool.run({
                    "table": "supplier_products",
                    "method": "list_supplier_products",
                    "args": [],
                    "kwargs": {"filters": {"product": product["product_id"]}}
                })

                available_products = []
                for item in result:
                    if item.get("quantity_available", 0) > 0:
                        available_products.append(item)

                if not available_products:
                    return f"Sorry, {product_name} is currently out of stock."

                # Show available options
                response_parts = [f"Yes, {product_name} is available from:"]
                for item in available_products[:3]:  # Show up to 3 options
                    supplier_name = item.get("supplier", {}).get("name", "Unknown supplier")
                    price = item.get("unit_price_etb", 0)
                    unit = item.get("unit", "kg")
                    quantity = item.get("quantity_available", 0)
                    response_parts.append(f"- {supplier_name}: {price} ETB per {unit} ({quantity} {unit} available)")

                return " ".join(response_parts)

            # Product not found - check if it's a supplier name
            supplier = await self.database_tool.run({
                "table": "users",
                "method": "get_user_by_name",
                "args": [product_name],
                "kwargs": {}
            })

            if supplier and supplier.get("role") == "supplier":
                # Supplier found - show products available from this supplier
                result = await self.database_tool.run({
                    "table": "supplier_products",
                    "method": "list_supplier_products",
                    "args": [],
                    "kwargs": {"filters": {"supplier": supplier["user_id"]}}
                })

                available_products = []
                for item in result:
                    if item.get("quantity_available", 0) > 0:
                        available_products.append(item)

                if not available_products:
                    return f"Sorry, {product_name} doesn't have any products available right now."

                # Set supplier_id in session context for future orders
                session_context["supplier_id"] = supplier["user_id"]

                # Show products from this supplier
                response_parts = [f"Products available from {product_name}:"]
                for item in available_products[:5]:  # Show up to 5 products
                    product_name_display = item.get("product", {}).get("product_name_en", "Unknown product")
                    price = item.get("unit_price_etb", 0)
                    unit = item.get("unit", "kg")
                    quantity = item.get("quantity_available", 0)
                    response_parts.append(f"- {product_name_display}: {price} ETB per {unit} ({quantity} {unit} available)")

                return " ".join(response_parts)

            # Neither product nor supplier found
            return f"Sorry, {product_name} is not currently available."

        except Exception as exc:
            logger.error(f"Failed to check availability: {exc}")
            return "I couldn't check availability right now. Please try again."

    async def _handle_storage_advice(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Provide storage advice using RAG (Retrieval-Augmented Generation)."""
        product_name = filled_slots.get("product_name")
        if not product_name:
            return "What product do you need storage advice for?"

        try:
            # First, find the product in the database to get its English name
            product = await self.database_tool.run({
                "table": "products",
                "method": "find_product_by_any_name",
                "args": [product_name],
                "kwargs": {}
            })

            # Use English name for vector search since the vector DB only has English content
            search_name = product_name
            if product and product.get("product_name_en") and product["product_name_en"] != "Unknown":
                search_name = product["product_name_en"]

            # Retrieve relevant context
            context = await self.vector_search.run({
                "query": f"storage advice for {search_name}",
                "top_k": 5  # Get more results for better context
            })

            if context.get("error"):
                return f"I don't have specific storage advice for {product_name}, but generally keep fresh produce in a cool, dry place."

            results = context.get("results", [])
            if not results:
                return f"I don't have specific storage advice for {product_name}, but generally keep fresh produce in a cool, dry place."

            # Extract relevant text from results
            context_texts = []
            for result in results[:3]:  # Use top 3 results for context
                text = result.get("text", "").strip()
                if text and len(text) > 10:
                    context_texts.append(text)

            if not context_texts:
                return f"I don't have specific storage advice for {product_name} in my knowledge base, but here are some general tips for storing fresh produce: Keep most vegetables and fruits in a cool, dry place away from direct sunlight. Refrigerate leafy greens and cut produce in airtight containers. Wash fruits and vegetables just before eating, not before storing. Store different types of produce separately to prevent ethylene gas from speeding up ripening. Check regularly and remove any spoiled items to prevent them from affecting others."

            # Use RAG: Combine retrieved context with LLM generation
            context_combined = "\n".join(context_texts)
            user_question = f"How should I store {product_name}?"

            # Create RAG prompt
            rag_prompt = f"""
You are a food storage expert providing objective storage advice. Do not mention ordering, marketplace, suppliers, customers, or KCartBot. Focus only on storage recommendations.

Based on the following storage information, provide helpful and practical advice for storing {product_name}.

Storage Information:
{context_combined}

User Question: {user_question}

Please provide clear, concise storage advice that incorporates the relevant information above. Focus on practical tips that will help preserve freshness and quality.
"""

            # Use LLM to generate response
            llm = self.llm_service.clone(system_prompt="")
            llm_response = await llm.acomplete(rag_prompt)

            if llm_response and llm_response.strip():
                return llm_response.strip()
            else:
                # Fallback to direct context if LLM fails
                return f"For {product_name}: {context_texts[0]}"

        except Exception as exc:
            logger.error(f"Failed to get storage advice: {exc}")
            return f"Generally, keep {product_name} in a cool, dry place away from direct sunlight. For best results, store fresh produce in the refrigerator for leafy greens and cut items, wash just before eating, and check regularly for spoilage."

    async def _handle_nutrition_query(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Handle nutrition comparison queries using RAG."""
        product_a = filled_slots.get("product_a")
        product_b = filled_slots.get("product_b")

        if not product_a or not product_b:
            return "Which two products would you like to compare nutritionally?"

        try:
            # Get English names for both products
            english_names = []
            for product_name in [product_a, product_b]:
                product = await self.database_tool.run({
                    "table": "products",
                    "method": "find_product_by_any_name",
                    "args": [product_name],
                    "kwargs": {}
                })
                if product and product.get("product_name_en") and product["product_name_en"] != "Unknown":
                    english_names.append(product["product_name_en"])
                else:
                    english_names.append(product_name)

            # Retrieve relevant context
            context = await self.vector_search.run({
                "query": f"nutritional comparison between {english_names[0]} and {english_names[1]}",
                "top_k": 5
            })

            if context.get("error") or not context.get("results"):
                return f"I don't have specific nutritional data comparing {product_a} and {product_b}."

            # Extract relevant text from results
            context_texts = []
            for result in context["results"][:3]:  # Use top 3 results
                text = result.get("text", "").strip()
                if text and len(text) > 10:
                    context_texts.append(text)

            if not context_texts:
                return f"I don't have specific nutritional data comparing {product_a} and {product_b}."

            # Use RAG: Combine retrieved context with LLM generation
            context_combined = "\n".join(context_texts)
            user_question = f"How do {product_a} and {product_b} compare nutritionally?"

            # Create RAG prompt
            rag_prompt = f"""
You are a nutrition expert providing objective nutritional information. Do not mention ordering, marketplace, suppliers, customers, or KCartBot. Focus only on the nutritional comparison.

Based on the following nutritional information, provide a helpful comparison between {product_a} and {product_b}.

Nutritional Information:
{context_combined}

User Question: {user_question}

Please provide a clear, balanced nutritional comparison that highlights the key differences and similarities between these two products. Include specific nutritional benefits or considerations for each.
"""

            # Use LLM to generate response
            llm = self.llm_service.clone(system_prompt="")
            llm_response = await llm.acomplete(rag_prompt)

            if llm_response and llm_response.strip():
                return llm_response.strip()
            else:
                # Fallback to direct context
                return f"Nutritional comparison: {context_texts[0]}"

        except Exception as exc:
            logger.error(f"Failed to get nutrition info: {exc}")
            return f"I couldn't find nutritional information comparing {product_a} and {product_b}."

    async def _handle_seasonal_query(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Handle seasonal availability queries using RAG."""
        season = filled_slots.get("season")
        location = filled_slots.get("location")

        query = "seasonal produce availability"
        if season:
            query += f" in {season}"
        if location:
            query += f" in {location}"

        user_question = f"What produce is available in {season or 'different seasons'}{' in ' + location if location else ''}?"

        try:
            # Retrieve relevant context
            context = await self.vector_search.run({
                "query": query,
                "top_k": 5
            })

            if context.get("error") or not context.get("results"):
                return "I don't have specific seasonal information right now."

            # Extract relevant text from results
            context_texts = []
            for result in context["results"][:3]:  # Use top 3 results
                text = result.get("text", "").strip()
                if text and len(text) > 10:
                    context_texts.append(text)

            if not context_texts:
                return "I don't have specific seasonal information in my knowledge base right now."

            # Use RAG: Combine retrieved context with LLM generation
            context_combined = "\n".join(context_texts)

            # Create RAG prompt
            rag_prompt = f"""
You are a seasonal produce expert providing objective information about produce availability. Do not mention ordering, marketplace, suppliers, customers, or KCartBot. Focus only on seasonal availability.

Based on the following seasonal produce information, provide helpful advice about produce availability.

Seasonal Information:
{context_combined}

User Question: {user_question}

Please provide clear, organized information about seasonal produce availability. Include specific fruits and vegetables that are typically available during this time, and any relevant tips about quality or selection.
"""

            # Use LLM to generate response
            llm = self.llm_service.clone(system_prompt="")
            llm_response = await llm.acomplete(rag_prompt)

            if llm_response and llm_response.strip():
                return llm_response.strip()
            else:
                # Fallback to direct context
                return f"Seasonal information: {context_texts[0]}"

        except Exception as exc:
            logger.error(f"Failed to get seasonal info: {exc}")
            return "I couldn't find seasonal availability information."

    async def _handle_in_season_query(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Handle 'what's in season' queries using RAG."""
        location = filled_slots.get("location")
        query = "what produce is currently in season"
        if location:
            query += f" in {location}"

        user_question = f"What produce is currently in season{' in ' + location if location else ''}?"

        try:
            # Retrieve relevant context
            context = await self.vector_search.run({
                "query": query,
                "top_k": 5
            })

            if context.get("error") or not context.get("results"):
                return "I don't have current seasonal information."

            # Extract relevant text from results
            context_texts = []
            for result in context["results"][:3]:  # Use top 3 results
                text = result.get("text", "").strip()
                if text and len(text) > 10:
                    context_texts.append(text)

            if not context_texts:
                return "I don't have current seasonal information in my knowledge base right now."

            # Use RAG: Combine retrieved context with LLM generation
            context_combined = "\n".join(context_texts)

            # Create RAG prompt
            rag_prompt = f"""
You are a seasonal produce expert providing objective information about current seasonal produce. Do not mention ordering, marketplace, suppliers, customers, or KCartBot. Focus only on what's currently in season.

Based on the following information about seasonal produce, provide helpful information about what's currently in season.

Seasonal Information:
{context_combined}

User Question: {user_question}

Please provide clear, organized information about produce that is currently in season. Include specific fruits and vegetables, and any relevant tips about quality, selection, or availability.
"""

            # Use LLM to generate response
            llm = self.llm_service.clone(system_prompt="")
            llm_response = await llm.acomplete(rag_prompt)

            if llm_response and llm_response.strip():
                return llm_response.strip()
            else:
                # Fallback to direct context
                return f"Currently in season: {context_texts[0]}"

        except Exception as exc:
            logger.error(f"Failed to get in-season info: {exc}")
            return "I couldn't find information about what's currently in season."

    async def _handle_general_advisory(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Handle general product advisory questions using RAG."""
        question = filled_slots.get("question")
        if not question:
            return "What question do you have about our products?"

        try:
            # Retrieve relevant context
            context = await self.vector_search.run({
                "query": question,
                "top_k": 5
            })

            if context.get("error") or not context.get("results"):
                return "I don't have information about that topic."

            # Extract relevant text from results
            context_texts = []
            for result in context["results"][:3]:  # Use top 3 results
                text = result.get("text", "").strip()
                if text and len(text) > 10:
                    context_texts.append(text)

            if not context_texts:
                return "I don't have information about that topic in my knowledge base right now."

            # Use RAG: Combine retrieved context with LLM generation
            context_combined = "\n".join(context_texts)

            # Create RAG prompt
            rag_prompt = f"""
You are a fresh produce expert providing objective advice about fruits and vegetables. Do not mention ordering, marketplace, suppliers, customers, or KCartBot. Focus only on the produce-related question.

Based on the following information about fresh produce, provide a helpful and accurate answer to the user's question.

Reference Information:
{context_combined}

User Question: {question}

Please provide a clear, helpful answer that addresses the user's question using the information provided above. Focus on practical, actionable advice related to fresh produce.
"""

            # Use LLM to generate response
            llm = self.llm_service.clone(system_prompt="")
            llm_response = await llm.acomplete(rag_prompt)

            if llm_response and llm_response.strip():
                return llm_response.strip()
            else:
                # Fallback to direct context
                return f"Here's what I know: {context_texts[0]}"

        except Exception as exc:
            logger.error(f"Failed to get advisory info: {exc}")
            return "I couldn't find information about that."

    async def _handle_place_order(
        self, filled_slots: Dict[str, Any], missing_slots: List[str], session_context: Dict[str, Any]
    ) -> str:
        """Handle order placement."""
        user_id = session_context.get("user_id")
        if not user_id:
            return "Please register first before placing an order."

        order_items = filled_slots.get("order_items", [])
        if not order_items:
            return "What would you like to order?"

        # Handle case where order_items is a string (product name) instead of list
        if isinstance(order_items, str):
            # Convert string to expected list format
            order_items = [{"product_name": order_items}]

        # Check if we have missing slots that need to be filled
        if missing_slots:
            if "quantity" in missing_slots:
                return "How much would you like to order?"
            elif "preferred_delivery_date" in missing_slots:
                return "When would you like this delivered?"

        try:
            # Create the order
            total_price = 0
            order_details = []

            for item in order_items:
                # Handle both dict format and string format
                if isinstance(item, dict):
                    product_name = item.get("product_name", "")
                    quantity = item.get("quantity", 1)
                else:
                    # If item is a string, treat it as product name
                    product_name = str(item)
                    quantity = 1  # Default quantity

                # Find the product by any name
                product = await self.database_tool.run({
                    "table": "products",
                    "method": "find_product_by_any_name",
                    "args": [product_name],
                    "kwargs": {}
                })

                if not product:
                    return f"Sorry, {product_name} is not available."

                # Get supplier products for this product
                supplier_id = session_context.get("supplier_id")
                if supplier_id:
                    supplier_products = await self.database_tool.run({
                        "table": "supplier_products",
                        "method": "list_supplier_products",
                        "args": [],
                        "kwargs": {"filters": {"product": product["product_id"], "supplier": supplier_id}}
                    })
                else:
                    supplier_products = await self.database_tool.run({
                        "table": "supplier_products",
                        "method": "list_supplier_products",
                        "args": [],
                        "kwargs": {"filters": {"product": product["product_id"]}}
                    })

                if not supplier_products:
                    return f"Sorry, {product_name} is not available."

                # Use first available supplier for now
                supplier_product = supplier_products[0]
                unit_price = supplier_product.get("unit_price_etb", 0)
                subtotal = quantity * unit_price
                total_price += subtotal

                order_details.append({
                    "product": supplier_product["product"],
                    "supplier": supplier_product["supplier"],
                    "quantity": quantity,
                    "unit_price": unit_price,
                    "subtotal": subtotal
                })

            # Create transaction
            # Get the user model instance
            user_instance = await self.database_tool.run({
                "table": "users",
                "method": "get_user_by_id",
                "args": [user_id],
                "kwargs": {},
                "raw_instances": True
            })

            if not user_instance:
                return "User not found. Please register first."

            # Handle delivery date if provided
            delivery_date = filled_slots.get("delivery_date")
            resolved_delivery_date = None
            if delivery_date:
                try:
                    resolved_delivery_date = await self.date_resolver.run(delivery_date)
                except Exception as exc:
                    logger.warning(f"Failed to resolve delivery date '{delivery_date}': {exc}")
                    # Continue without delivery date if resolution fails

            transaction = await self.database_tool.run({
                "table": "transactions",
                "method": "create_transaction",
                "args": [],
                "kwargs": {
                    "user": user_instance,  # Pass the User model instance
                    "date": datetime.date.today(),  # Required order date
                    "total_price": total_price,
                    "payment_method": "COD",
                    "status": "Pending",
                    **({"delivery_date": resolved_delivery_date.isoformat()} if resolved_delivery_date else {})
                },
                "raw_instances": True  # Keep the Transaction model instance
            })

            # Create order items
            for detail in order_details:
                logger.info(f"Creating order item for product {detail['product']['product_id']} from supplier {detail['supplier']['user_id']}")
                # Get the product and supplier model instances
                product_instance = await self.database_tool.run({
                    "table": "products",
                    "method": "get_product_by_id",
                    "args": [detail["product"]["product_id"]],
                    "kwargs": {},
                    "raw_instances": True
                })

                supplier_instance = await self.database_tool.run({
                    "table": "users",
                    "method": "get_user_by_id",
                    "args": [detail["supplier"]["user_id"]],
                    "kwargs": {},
                    "raw_instances": True
                })

                if not product_instance or not supplier_instance:
                    logger.error(f"Failed to get product or supplier instances: product_instance={product_instance}, supplier_instance={supplier_instance}")
                    return "Product or supplier information not found."

                logger.info(f"Creating order item with order={transaction}, product={product_instance}, supplier={supplier_instance}")
                await self.database_tool.run({
                    "table": "order_items",
                    "method": "create_order_item",
                    "args": [],
                    "kwargs": {
                        "order": transaction,  # Pass the Transaction model instance
                        "product": product_instance,  # Pass the Product model instance
                        "supplier": supplier_instance,  # Pass the User model instance
                        "quantity": detail["quantity"],
                        "unit": "kg",  # Assume kg for now
                        "price_per_unit": detail["unit_price"],
                        "subtotal": detail["subtotal"]
                    }
                })
                logger.info(f"Order item created successfully")

            return f"Order placed successfully! Total: {total_price} ETB. Payment will be Cash on Delivery."

        except Exception as exc:
            logger.error(f"Failed to place order: {exc}")
            return "I couldn't place your order. Please try again."

    async def _handle_set_delivery_date(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Handle delivery date setting."""
        delivery_date = filled_slots.get("delivery_date")
        if not delivery_date:
            return "When would you like your delivery?"

        try:
            # Resolve the date
            resolved_date = await self.date_resolver.run(delivery_date)

            # Update the most recent pending order
            user_id = session_context.get("user_id")
            if not user_id:
                return "Please register first."

            orders = await self.database_tool.run({
                "table": "transactions",
                "method": "list_transactions",
                "args": [],
                "kwargs": {"filters": {"user": user_id, "status": "Pending"}}
            })

            if not orders:
                return "You don't have any pending orders."

            latest_order = orders[0]  # Assuming sorted by date desc

            await self.database_tool.run({
                "table": "transactions",
                "method": "update_transaction",
                "args": [latest_order["order_id"]],
                "kwargs": {"delivery_date": resolved_date.isoformat()}
            })

            return f"Delivery date set to {resolved_date.strftime('%B %d, %Y')}."

        except Exception as exc:
            logger.error(f"Failed to set delivery date: {exc}")
            return "I couldn't set the delivery date. Please try again."

    async def _handle_set_delivery_location(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Handle delivery location setting."""
        location = filled_slots.get("delivery_location")
        if not location:
            return "Where would you like your delivery?"

        # For now, just acknowledge - in practice might need validation
        return f"Delivery location set to {location}."

    async def _handle_confirm_payment(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Handle payment confirmation."""
        order_ref = filled_slots.get("order_reference")
        if not order_ref:
            return "Which order would you like to confirm payment for?"

        try:
            # Update order status to Confirmed
            await self.database_tool.run({
                "table": "transactions",
                "method": "update_transaction",
                "args": [order_ref],
                "kwargs": {"status": "Confirmed"}
            })

            return "Payment confirmed! Your order is now being prepared for delivery."

        except Exception as exc:
            logger.error(f"Failed to confirm payment: {exc}")
            return "I couldn't confirm the payment. Please try again."

    async def _handle_customer_check_deliveries(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Handle customer checking their deliveries."""
        user_id = session_context.get("user_id")
        if not user_id:
            return "Please log in as a customer first."

        date_filter = filled_slots.get("date")
        order_ref = filled_slots.get("order_reference")

        try:
            # Query transactions (orders) for this customer
            transactions = await self.database_tool.run({
                "table": "transactions",
                "method": "list_transactions",
                "args": [],
                "kwargs": {"filters": {"user": user_id}}
            })

            if not transactions:
                return "You have no deliveries scheduled."

            # If a specific date is requested, filter by delivery date
            if date_filter:
                resolved_date = await self.date_resolver.run(date_filter)
                date_str = resolved_date.strftime('%Y-%m-%d')
                
                filtered_transactions = []
                for transaction in transactions:
                    delivery_date = transaction.get("delivery_date")
                    if delivery_date:
                        # Convert delivery_date to date string for comparison
                        if isinstance(delivery_date, str):
                            if 'T' in delivery_date:
                                delivery_date = delivery_date.split('T')[0]
                        if delivery_date == date_str:
                            filtered_transactions.append(transaction)
                transactions = filtered_transactions

                if not transactions:
                    return f"You have no deliveries scheduled for {resolved_date.strftime('%B %d, %Y')}."

            # If a specific order reference is requested, filter to that order
            if order_ref:
                transactions = [t for t in transactions if t.get("order_id", "").startswith(order_ref)]
                if not transactions:
                    return f"I couldn't find an order with reference '{order_ref}'."

            # Format the response
            response_parts = ["Your deliveries:"]

            for transaction in transactions[:10]:  # Limit to 10 orders
                order_id = transaction.get("order_id", "Unknown")[:8]
                delivery_date = transaction.get("delivery_date")
                status = transaction.get("status", "Unknown")
                total_price = transaction.get("total_price", 0)

                date_str = "Date not set"
                if delivery_date:
                    try:
                        # If delivery_date is already a date string, format it
                        if isinstance(delivery_date, str):
                            if 'T' in delivery_date:
                                delivery_date = delivery_date.split('T')[0]
                            # Try to parse and format the date
                            from datetime import datetime
                            date_obj = datetime.fromisoformat(delivery_date.replace('Z', '+00:00'))
                            date_str = date_obj.strftime('%B %d, %Y')
                        else:
                            date_str = str(delivery_date)
                    except Exception:
                        date_str = str(delivery_date)

                response_parts.append(
                    f"- Order {order_id}...: {date_str} - {status} - {total_price} ETB"
                )

            if len(transactions) > 10:
                response_parts.append(f"... and {len(transactions) - 10} more orders.")

            return " ".join(response_parts)

        except Exception as exc:
            logger.error(f"Failed to check customer deliveries: {exc}")
            return "I couldn't check your deliveries right now. Please try again."

    # Supplier flow handlers
    async def _handle_supplier_registration(
        self, filled_slots: Dict[str, Any], missing_slots: List[str], session_context: Dict[str, Any]
    ) -> str:
        """Handle supplier registration."""
        if missing_slots:
            slot = missing_slots[0]
            if slot == "supplier_name":
                return "Welcome! What's your business name?"
            elif slot == "phone_number":
                return "Great! What's your phone number?"

        # All slots filled, register the supplier
        try:
            result = await self.database_tool.run({
                "table": "users",
                "method": "create_user",
                "args": [],
                "kwargs": {
                    "name": filled_slots["supplier_name"],
                    "phone": filled_slots["phone_number"],
                    "default_location": "",  # Suppliers don't need a default location
                    "preferred_language": "English",  # Default language
                    "role": "supplier",
                    "joined_date": datetime.date.today()  # Explicitly set joined date
                }
            })
            session_context["user_id"] = result.get("user_id")
            return f"Welcome {filled_slots['supplier_name']}! Your supplier account has been created. How can I help you manage your inventory?"
        except Exception as exc:
            logger.error(f"Failed to register supplier: {exc}")
            return "I couldn't create your account. Please try again."

    async def _handle_add_product(
        self, filled_slots: Dict[str, Any], missing_slots: List[str], session_context: Dict[str, Any]
    ) -> str:
        """Handle adding a new product."""
        user_id = session_context.get("user_id")
        if not user_id:
            return "Please register as a supplier first."

        product_name = filled_slots.get("product_name")
        if not product_name:
            return "What product would you like to add?"

        # Check if product exists using any name field
        try:
            product = await self.database_tool.run({
                "table": "products",
                "method": "find_product_by_any_name",
                "args": [product_name],
                "kwargs": {}
            })

            if not product:
                # Create new product - determine which name field to set based on input
                create_kwargs = {
                    "category": "Vegetable",  # Default category
                    "unit": "kg",  # Default unit
                    "base_price_etb": 0.0,  # Default base price
                    "in_season_start": "January",  # Default season start
                    "in_season_end": "December",  # Default season end
                }

                # Simple language detection: if contains non-ASCII, assume Amharic
                if product_name.isascii():
                    create_kwargs["product_name_en"] = product_name
                    create_kwargs["product_name_am"] = "Unknown"
                    create_kwargs["product_name_am_latin"] = "Unknown"
                else:
                    create_kwargs["product_name_en"] = "Unknown"
                    create_kwargs["product_name_am"] = product_name
                    create_kwargs["product_name_am_latin"] = "Unknown"

                product = await self.database_tool.run({
                    "table": "products",
                    "method": "create_product",
                    "args": [],
                    "kwargs": create_kwargs
                })

            product_id = product["product_id"]

            # If this supplier already has this product, prompt whether to add to existing or create new listing
            supplier_products = await self.database_tool.run({
                "table": "supplier_products",
                "method": "list_supplier_products",
                "args": [],
                "kwargs": {"filters": {"supplier": user_id, "product": product_id}}
            })

            if supplier_products:
                # Save existing supplier product info to session so next steps can use it
                existing = supplier_products[0]
                if "pending_product" not in session_context:
                    session_context["pending_product"] = {}

                session_context["pending_product"].update({
                    "product_id": product_id,
                    "product_name": product_name,
                    "existing_inventory_id": existing.get("inventory_id"),
                    "existing_quantity": existing.get("quantity_available", 0),
                    "existing_price": existing.get("unit_price_etb"),
                    "update_existing": None  # wait for user's choice
                })

                # Ask the user whether to add to existing or create a separate listing
                existing_qty = existing.get("quantity_available", 0)
                existing_price = existing.get("unit_price_etb", "Unknown")
                return (
                    f"You already have {product_name}: {existing_qty} kg at {existing_price} ETB/kg. "
                    "Would you like to add to the existing listing or create a separate listing? "
                    "Reply 'add' to increase existing inventory or 'new' to create a separate listing."
                )

            return f"Product '{product_name}' is ready. What's the quantity you have available?"

        except Exception as exc:
            logger.error(f"Failed to add product: {exc}")
            return "I couldn't add the product. Please try again."

    async def _handle_set_quantity(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Handle setting product quantity."""
        user_id = session_context.get("user_id")
        if not user_id:
            return "Please register first."

        product_name = filled_slots.get("product_name")
        quantity = filled_slots.get("quantity")

        if not product_name or quantity is None:
            return "What product and how much quantity?"

        # Check if this supplier already has this product and user wants to add to existing
        user_message = session_context.get("last_user_message", "").lower()
        is_existing_inventory_request = any(keyword in user_message for keyword in [
            "existing", "inventory", "add more", "add to my", "more to my", "increase my"
        ])

        if is_existing_inventory_request:
            try:
                # Find the product
                product = await self.database_tool.run({
                    "table": "products",
                    "method": "find_product_by_any_name",
                    "args": [product_name],
                    "kwargs": {}
                })

                if product:
                    # Check if supplier already has this product
                    supplier_products = await self.database_tool.run({
                        "table": "supplier_products",
                        "method": "list_supplier_products",
                        "args": [],
                        "kwargs": {"filters": {"supplier": user_id, "product": product["product_id"]}}
                    })

                    if supplier_products:
                        # User wants to add to existing inventory - update quantity directly
                        existing_product = supplier_products[0]
                        current_quantity = existing_product.get("quantity_available", 0)
                        new_quantity = current_quantity + quantity

                        update_kwargs = {"quantity_available": new_quantity}

                        await self.database_tool.run({
                            "table": "supplier_products",
                            "method": "update_supplier_product",
                            "args": [existing_product["inventory_id"]],
                            "kwargs": update_kwargs
                        })

                        return f"Added {quantity} kg to your existing {product_name} inventory. Total quantity now: {new_quantity} kg at {existing_product.get('unit_price_etb', 'current price')} ETB per kg, deliverable {existing_product.get('available_delivery_days', 'existing schedule')}."

            except Exception as exc:
                logger.error(f"Failed to check existing inventory: {exc}")
                # Fall through to normal flow if check fails

        # Store the quantity information in session context for later use
        if "pending_product" not in session_context:
            session_context["pending_product"] = {}

        session_context["pending_product"]["product_name"] = product_name
        session_context["pending_product"]["quantity"] = quantity

        # Get pricing suggestions before asking for price
        try:
            # Find the product to get its ID for competitor price lookup
            product = await self.database_tool.run({
                "table": "products",
                "method": "find_product_by_any_name",
                "args": [product_name],
                "kwargs": {}
            })

            suggestion_text = ""
            if product:
                competitor_prices = await self.database_tool.run({
                    "table": "competitor_prices",
                    "method": "list_competitor_prices",
                    "args": [],
                    "kwargs": {"filters": {"product": product["product_id"]}}
                })

                if competitor_prices:
                    prices = [cp.get("price_etb_per_kg", 0) for cp in competitor_prices if cp.get("price_etb_per_kg", 0) > 0]
                    if prices:
                        avg_price = sum(prices) / len(prices)
                        min_price = min(prices)
                        max_price = max(prices)

                        suggestion_text = f" 📊 **Market Insights for {product_name}:** • Average market price: {avg_price:.1f} ETB/kg • Price range: {min_price:.1f} - {max_price:.1f} ETB/kg • Suggested competitive range: {max(min_price * 0.9, avg_price * 0.85):.1f} - {min(max_price * 1.1, avg_price * 1.15):.1f} ETB/kg"

            return f"I'll add {quantity} kg of {product_name}.{suggestion_text} What's the price per kg in ETB you'd like to set?"

        except Exception as exc:
            logger.error(f"Failed to get pricing suggestions: {exc}")
            return f"I'll add {quantity} kg of {product_name}. What's the price per kg in ETB?"

    async def _handle_set_delivery_dates(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Handle setting delivery dates."""
        delivery_dates = filled_slots.get("delivery_dates")
        pending_product = session_context.get("pending_product")

        if not pending_product:
            return "Please start by adding a product first."

        product_name = pending_product.get("product_name", "this product")

        if not delivery_dates:
            return f"What days can you deliver {product_name}? (e.g., 'Monday to Friday' or 'weekdays')"

        # Store delivery dates
        pending_product["delivery_dates"] = delivery_dates

        # Now we have all the information, create the supplier product
        user_id = session_context.get("user_id")
        if not user_id:
            return "Please register first."

        try:
            # Find the product
            product = await self.database_tool.run({
                "table": "products",
                "method": "find_product_by_any_name",
                "args": [product_name],
                "kwargs": {}
            })

            if not product:
                return f"Product '{product_name}' not found. Please add it first."

            product_id = product["product_id"]

            # Get the actual model instances for foreign key relationships
            user = await self.database_tool.run({
                "table": "users",
                "method": "get_user_by_id",
                "args": [user_id],
                "kwargs": {},
                "raw_instances": True
            })

            product = await self.database_tool.run({
                "table": "products",
                "method": "get_product_by_id",
                "args": [product_id],
                "kwargs": {},
                "raw_instances": True
            })

            if not user or not product:
                return "I couldn't find the required user or product information."

            quantity = pending_product.get("quantity")
            unit_price = pending_product.get("unit_price")
            expiry_date = pending_product.get("expiry_date")
            update_existing = pending_product.get("update_existing")

            # Handle based on user's choice for existing inventory
            if update_existing is True:
                # User chose to add to existing inventory - only update quantity
                supplier_products = await self.database_tool.run({
                    "table": "supplier_products",
                    "method": "list_supplier_products",
                    "args": [],
                    "kwargs": {"filters": {"supplier": user_id, "product": product_id}}
                })

                if supplier_products:
                    # Add to existing quantity
                    existing_product = supplier_products[0]
                    current_quantity = existing_product.get("quantity_available", 0)
                    new_quantity = current_quantity + quantity

                    update_kwargs = {
                        "quantity_available": new_quantity
                    }

                    await self.database_tool.run({
                        "table": "supplier_products",
                        "method": "update_supplier_product",
                        "args": [existing_product["inventory_id"]],
                        "kwargs": update_kwargs
                    })
                    # Clear pending product info
                    session_context.pop("pending_product", None)
                    return f"Added {quantity} kg to your existing {product_name} inventory. Total quantity now: {new_quantity} kg at {existing_product.get('unit_price_etb', 'current price')} ETB per kg, deliverable {existing_product.get('available_delivery_days', 'existing schedule')}."
                else:
                    # No existing inventory found, create new anyway
                    create_kwargs = {
                        "supplier": user,  # Pass the user model instance
                        "product": product,  # Pass the product model instance
                        "quantity_available": quantity,
                        "unit": "kg",
                        "unit_price_etb": unit_price,
                        "available_delivery_days": delivery_dates,
                        "status": "active"  # Required status field
                    }
                    if expiry_date is not None:
                        create_kwargs["expiry_date"] = expiry_date

                    await self.database_tool.run({
                        "table": "supplier_products",
                        "method": "create_supplier_product",
                        "args": [],
                        "kwargs": create_kwargs
                    })
                    # Clear pending product info
                    session_context.pop("pending_product", None)
                    expiry_info = ""
                    if expiry_date:
                        try:
                            from datetime import datetime
                            if isinstance(expiry_date, str) and 'T' not in expiry_date:
                                date_obj = datetime.fromisoformat(expiry_date)
                                expiry_info = f", expires {date_obj.strftime('%B %d, %Y')}"
                            else:
                                expiry_info = f", expires {expiry_date}"
                        except:
                            expiry_info = f", expires {expiry_date}"
                    return f"Added {product_name} to your inventory: {quantity} kg at {unit_price} ETB per kg{expiry_info}, deliverable {delivery_dates}."

            elif update_existing is False:
                # User chose to create a new listing
                create_kwargs = {
                    "supplier": user,  # Pass the user model instance
                    "product": product,  # Pass the product model instance
                    "quantity_available": quantity,
                    "unit": "kg",
                    "unit_price_etb": unit_price,
                    "available_delivery_days": delivery_dates,
                    "status": "active"  # Required status field
                }
                if expiry_date is not None:
                    create_kwargs["expiry_date"] = expiry_date

                await self.database_tool.run({
                    "table": "supplier_products",
                    "method": "create_supplier_product",
                    "args": [],
                    "kwargs": create_kwargs
                })
                # Clear pending product info
                session_context.pop("pending_product", None)
                expiry_info = ""
                if expiry_date:
                    try:
                        from datetime import datetime
                        if isinstance(expiry_date, str) and 'T' not in expiry_date:
                            date_obj = datetime.fromisoformat(expiry_date)
                            expiry_info = f", expires {date_obj.strftime('%B %d, %Y')}"
                        else:
                            expiry_info = f", expires {expiry_date}"
                    except:
                        expiry_info = f", expires {expiry_date}"
                return f"Created new listing for {product_name}: {quantity} kg at {unit_price} ETB per kg{expiry_info}, deliverable {delivery_dates}."

            else:
                # Fallback: update_existing not set (shouldn't happen in normal flow)
                supplier_products = await self.database_tool.run({
                    "table": "supplier_products",
                    "method": "list_supplier_products",
                    "args": [],
                    "kwargs": {"filters": {"supplier": user_id, "product": product_id}}
                })

                if supplier_products:
                    # Update existing
                    update_kwargs = {
                        "quantity_available": quantity,
                        "unit_price_etb": unit_price,
                        "available_delivery_days": delivery_dates
                    }
                    if expiry_date is not None:
                        update_kwargs["expiry_date"] = expiry_date

                    await self.database_tool.run({
                        "table": "supplier_products",
                        "method": "update_supplier_product",
                        "args": [supplier_products[0]["inventory_id"]],
                        "kwargs": update_kwargs
                    })
                    # Clear pending product info
                    session_context.pop("pending_product", None)
                    return f"Updated {product_name}: {quantity} kg at {unit_price} ETB per kg, deliverable {delivery_dates}."
                else:
                    # Create new supplier product with all information
                    create_kwargs = {
                        "supplier": user,  # Pass the user model instance
                        "product": product,  # Pass the product model instance
                        "quantity_available": quantity,
                        "unit": "kg",
                        "unit_price_etb": unit_price,
                        "available_delivery_days": delivery_dates,
                        "status": "active"  # Required status field
                    }
                    if expiry_date is not None:
                        create_kwargs["expiry_date"] = expiry_date

                    await self.database_tool.run({
                        "table": "supplier_products",
                        "method": "create_supplier_product",
                        "args": [],
                        "kwargs": create_kwargs
                    })
                    # Clear pending product info
                    session_context.pop("pending_product", None)
                    expiry_info = ""
                    if expiry_date:
                        try:
                            from datetime import datetime
                            if isinstance(expiry_date, str) and 'T' not in expiry_date:
                                date_obj = datetime.fromisoformat(expiry_date)
                                expiry_info = f", expires {date_obj.strftime('%B %d, %Y')}"
                            else:
                                expiry_info = f", expires {expiry_date}"
                        except Exception:
                            expiry_info = f", expires {expiry_date}"
                    return f"Added {product_name} to your inventory: {quantity} kg at {unit_price} ETB per kg{expiry_info}, deliverable {delivery_dates}."

        except Exception as exc:
            logger.error(f"Failed to create supplier product: {exc}")
            return "I couldn't add the product to your inventory. Please try again."

    async def _handle_set_expiry_date(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Handle setting expiry date."""
        expiry_date_input = filled_slots.get("expiry_date")
        pending_product = session_context.get("pending_product")

        if not pending_product:
            return "Please start by adding a product first."

        product_name = pending_product.get("product_name", "this product")

        # Handle cases where user says no expiry or similar
        if expiry_date_input and any(word in expiry_date_input.lower() for word in ["no", "none", "doesn't", "never", "no expiry"]):
            pending_product["expiry_date"] = None
            return f"Noted - {product_name} has no expiry date. What days can you deliver {product_name}? (e.g., 'Monday to Friday' or 'weekdays')"

        if not expiry_date_input:
            return f"When does {product_name} expire? (You can say 'no expiry' if it doesn't expire)"

        try:
            resolved_date = await self.date_resolver.run(expiry_date_input)
            pending_product["expiry_date"] = resolved_date.isoformat()
            return f"Expiry date set to {resolved_date.strftime('%B %d, %Y')}. What days can you deliver {product_name}? (e.g., 'Monday to Friday' or 'weekdays')"
        except Exception as exc:
            logger.error(f"Failed to resolve expiry date: {exc}")
            # Store the raw input if date resolution fails
            pending_product["expiry_date"] = expiry_date_input
            return f"Expiry date noted as: {expiry_date_input}. What days can you deliver {product_name}? (e.g., 'Monday to Friday' or 'weekdays')"

    async def _handle_set_price(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Handle setting product price."""
        user_id = session_context.get("user_id")
        if not user_id:
            return "Please register first."

        unit_price = filled_slots.get("unit_price")

        if unit_price is None:
            return "What's the price per kg in ETB?"

        # Check if we have pending product information
        pending_product = session_context.get("pending_product")
        if not pending_product:
            return "Please tell me which product you want to set the price for."

        product_name = pending_product.get("product_name")
        quantity = pending_product.get("quantity")

        if not product_name or quantity is None:
            return "I need the product name and quantity first."

        # Store the price in pending product info
        pending_product["unit_price"] = unit_price

        return f"Great! I'll set the price at {unit_price} ETB per kg. When does this {product_name} expire? (You can say 'no expiry' if it doesn't expire)"

    async def _handle_pricing_insight(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Provide pricing insights."""
        product_name = filled_slots.get("product_name")
        if not product_name:
            return "Which product do you need pricing insights for?"

        try:
            # Get competitor prices
            competitor_prices = await self.database_tool.run({
                "table": "competitor_prices",
                "method": "list_competitor_prices",
                "args": [],
                "kwargs": {"filters": {"product__product_name_en__icontains": product_name}}
            })

            if not competitor_prices:
                return f"No competitor pricing data available for {product_name}."

            avg_price = sum(cp.get("price_etb_per_kg", 0) for cp in competitor_prices) / len(competitor_prices)
            return f"Average competitor price for {product_name}: {avg_price:.2f} ETB per kg."

        except Exception as exc:
            logger.error(f"Failed to get pricing insight: {exc}")
            return f"I couldn't get pricing insights for {product_name}."

    async def _handle_generate_image(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Generate product image."""
        product_name = filled_slots.get("product_name")
        if not product_name:
            return "Which product would you like an image for?"

        try:
            result = await self.image_generator.run(product_name)
            return f"Image generated for {product_name}: {result}"
        except Exception as exc:
            logger.error(f"Failed to generate image: {exc}")
            return f"I couldn't generate an image for {product_name}."

    async def _handle_check_stock(
        self, session_context: Dict[str, Any]
    ) -> str:
        """Check supplier's stock."""
        user_id = session_context.get("user_id")
        if not user_id:
            return "Please register first."

        try:
            products = await self.database_tool.run({
                "table": "supplier_products",
                "method": "list_supplier_products",
                "args": [],
                "kwargs": {"filters": {"supplier": user_id}}
            })

            if not products:
                return "You don't have any products in inventory yet."

            response_parts = ["📦 **Your Current Inventory:**"]
            for product in products:
                name = product.get("product", {}).get("product_name_en", "Unknown")
                quantity = product.get("quantity_available", 0)
                unit = product.get("unit", "kg")
                price = product.get("unit_price_etb", 0)
                expiry_date = product.get("expiry_date")
                delivery_days = product.get("available_delivery_days", "Not specified")
                status = product.get("status", "unknown")

                # Format expiry date
                expiry_info = ""
                if expiry_date:
                    try:
                        from datetime import datetime
                        if isinstance(expiry_date, str):
                            if 'T' in expiry_date:
                                expiry_date = expiry_date.split('T')[0]
                            date_obj = datetime.fromisoformat(expiry_date.replace('Z', '+00:00'))
                            expiry_info = f" • Expires: {date_obj.strftime('%b %d, %Y')}"
                        else:
                            expiry_info = f" • Expires: {expiry_date}"
                    except:
                        expiry_info = f" • Expires: {expiry_date}"

                # Format status with emoji
                status_emoji = "✅" if status == "active" else "⏸️" if status == "on_sale" else "❌"

                response_parts.append(
                    f"{status_emoji} **{name}** • Quantity: {quantity} {unit} • Price: {price} ETB/{unit} • Delivery: {delivery_days}{expiry_info}"
                )

            return " ".join(response_parts)

        except Exception as exc:
            logger.error(f"Failed to check stock: {exc}")
            return "I couldn't check your inventory."

    async def _handle_view_expiring_products(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """View expiring products."""
        time_horizon = filled_slots.get("time_horizon", "1 week")
        return f"Checking products expiring within {time_horizon}."

    async def _handle_accept_flash_sale(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Accept flash sale."""
        product_name = filled_slots.get("product_name")
        if not product_name:
            return "Which product flash sale would you like to accept?"

        return f"Flash sale accepted for {product_name}."

    async def _handle_decline_flash_sale(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Decline flash sale."""
        product_name = filled_slots.get("product_name")
        if not product_name:
            return "Which product flash sale would you like to decline?"

        return f"Flash sale declined for {product_name}."

    async def _handle_view_delivery_schedule(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """View delivery schedule."""
        date_range = filled_slots.get("date_range")
        query = "your delivery schedule"
        if date_range:
            query += f" for {date_range}"

        return f"Here's {query}."

    async def _handle_check_deliveries_by_date(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Check deliveries by date."""
        user_id = session_context.get("user_id")
        if not user_id:
            return "Please log in as a supplier first."

        date = filled_slots.get("date")
        if not date:
            return "Which date would you like to check deliveries for?"

        try:
            # Resolve the date to ensure proper format
            resolved_date = await self.date_resolver.run(date)
            date_str = resolved_date.strftime('%Y-%m-%d')

            # Query order items for this supplier with deliveries on the specified date
            order_items = await self.database_tool.run({
                "table": "order_items",
                "method": "list_order_items",
                "args": [],
                "kwargs": {"filters": {"supplier": user_id}}
            })

            if not order_items:
                return f"You have no deliveries scheduled for {resolved_date.strftime('%B %d, %Y')}."

            # Filter by delivery date from transactions
            deliveries_today = []
            for item in order_items:
                # Get the transaction for this order item
                transaction = await self.database_tool.run({
                    "table": "transactions",
                    "method": "get_transaction_by_id",
                    "args": [item["order"]["order_id"]],
                    "kwargs": {}
                })

                if transaction and transaction.get("delivery_date"):
                    # Check if delivery date matches (compare date parts only)
                    delivery_date = transaction["delivery_date"]
                    if isinstance(delivery_date, str):
                        delivery_date = delivery_date.split('T')[0]  # Remove time part if present

                    if delivery_date == date_str and transaction.get("status") in ["Confirmed", "Pending"]:
                        deliveries_today.append({
                            "order_id": transaction["order_id"],
                            "customer": transaction.get("user", "Unknown customer"),
                            "product": item.get("product", "Unknown product"),
                            "quantity": item.get("quantity", 0),
                            "unit": item.get("unit", "kg"),
                            "status": transaction.get("status", "Unknown")
                        })

            if not deliveries_today:
                return f"You have no deliveries scheduled for {resolved_date.strftime('%B %d, %Y')}."

            # Format the response
            response_parts = [f"Your deliveries for {resolved_date.strftime('%B %d, %Y')}:"]

            for delivery in deliveries_today[:10]:  # Limit to 10 deliveries
                product_name = "Unknown product"
                if isinstance(delivery["product"], dict):
                    product_name = delivery["product"].get("product_name_en", "Unknown product")
                elif hasattr(delivery["product"], 'get'):
                    product_name = delivery["product"].get("product_name_en", "Unknown product")

                customer_name = "Unknown customer"
                if isinstance(delivery["customer"], dict):
                    customer_name = delivery["customer"].get("name", "Unknown customer")

                response_parts.append(
                    f"- Order {delivery['order_id'][:8]}...: {product_name} "
                    f"({delivery['quantity']} {delivery['unit']}) for {customer_name} - {delivery['status']}"
                )

            if len(deliveries_today) > 10:
                response_parts.append(f"... and {len(deliveries_today) - 10} more deliveries.")

            return " ".join(response_parts)

        except Exception as exc:
            logger.error(f"Failed to check deliveries by date: {exc}")
            return f"I couldn't check your deliveries for {date}. Please try again."

    async def _handle_supplier_check_deliveries(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Handle supplier checking their deliveries/orders."""
        user_id = session_context.get("user_id")
        if not user_id:
            return "Please log in as a supplier first."

        date_filter = filled_slots.get("date")
        order_ref = filled_slots.get("order_reference")

        try:
            # Get pending orders for this supplier
            pending_orders = await self._get_supplier_pending_orders(user_id)

            if not pending_orders:
                return "You have no pending orders at this time."

            # If a specific date is requested, we could filter further, but for now just return all pending
            # The _get_supplier_pending_orders already filters for pending/confirmed status

            # If a specific order reference is requested, filter to that order
            if order_ref:
                # This is a simple implementation - in practice might need better matching
                if order_ref.lower() not in pending_orders.lower():
                    return f"I couldn't find an order with reference '{order_ref}' in your pending orders."

            return pending_orders

        except Exception as exc:
            logger.error(f"Failed to check supplier deliveries: {exc}")
            return "I couldn't check your deliveries right now. Please try again."

    async def _handle_accept_order(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Accept order."""
        order_ref = filled_slots.get("order_reference")
        if not order_ref:
            return "Which order would you like to accept?"

        return f"Order {order_ref} accepted."

    async def _handle_update_inventory(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Handle updating existing inventory quantity."""
        user_id = session_context.get("user_id")
        if not user_id:
            return "Please register first."

        product_name = filled_slots.get("product_name")
        quantity = filled_slots.get("quantity")

        if not product_name or quantity is None:
            return "What product and how much quantity do you want to add?"

        try:
            # Find the product
            product = await self.database_tool.run({
                "table": "products",
                "method": "find_product_by_any_name",
                "args": [product_name],
                "kwargs": {}
            })

            if not product:
                return f"I couldn't find {product_name} in your inventory. Please add it as a new product first."

            # Check if supplier already has this product
            supplier_products = await self.database_tool.run({
                "table": "supplier_products",
                "method": "list_supplier_products",
                "args": [],
                "kwargs": {"filters": {"supplier": user_id, "product": product["product_id"]}}
            })

            if not supplier_products:
                return f"You don't have {product_name} in your inventory yet. Please add it as a new product first."

            # Update existing inventory - add to current quantity
            existing_product = supplier_products[0]
            current_quantity = existing_product.get("quantity_available", 0)
            new_quantity = current_quantity + quantity

            update_kwargs = {"quantity_available": new_quantity}

            await self.database_tool.run({
                "table": "supplier_products",
                "method": "update_supplier_product",
                "args": [existing_product["inventory_id"]],
                "kwargs": update_kwargs
            })

            return f"Added {quantity} kg to your existing {product_name} inventory. Total quantity now: {new_quantity} kg at {existing_product.get('unit_price_etb', 'current price')} ETB per kg, deliverable {existing_product.get('available_delivery_days', 'existing schedule')}."

        except Exception as exc:
            logger.error(f"Failed to update inventory: {exc}")
            return f"I couldn't update your {product_name} inventory. Please try again."
