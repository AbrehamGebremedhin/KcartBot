"""KCartBot Agent - Orchestrates LLM and tools for customer/supplier conversations."""

from __future__ import annotations

import datetime
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from app.services.llm_service import LLMService
from app.tools.database_tool import DatabaseAccessTool
from app.tools.date_tool import DateResolverTool
from app.tools.generate_image import ImageGeneratorTool
from app.tools.intent_classifier import IntentClassifierTool
from app.tools.search_context import VectorSearchTool
from app.agents.multilingual_responses import get_multilingual_response_dictionary

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

    def _detect_language(self, text: str) -> str:
        """
        Detect the language of the input text.
        
        Returns:
            'amharic' for Amharic script (አማርኛ)
            'phonetic_amharic' for phonetic/latinized Amharic (e.g., "selam", "neger")
            'english' for English
        """
        if not text or not text.strip():
            return "english"  # Default fallback
        
        text = text.strip()
        
        # Check for Amharic script characters (Ethiopic script)
        # Amharic uses characters in the range U+1200 to U+137F
        amharic_chars = sum(1 for char in text if '\u1200' <= char <= '\u137f')
        total_chars = len(text.replace(' ', ''))
        
        if amharic_chars > total_chars * 0.3:  # More than 30% Amharic characters
            return "amharic"
        
        # Check for phonetic Amharic patterns
        # Common phonetic Amharic words and patterns
        phonetic_amharic_indicators = [
            # Greetings
            r'\b(selam|salam|tena|tenayistilign|dehna|dehna hun|selam neger|neger)\b',
            # Common words
            r'\b(ine|min|ande|neger|ay|aydelem|meskerem|tikimt|betam|nech|min chu|ey|eyu|konjo|min alebet|min lij|min lijoch)\b',
            # Question words
            r'\b(min|ande|yet|lema|ke|kem|kena|ken|kegna|kegne|kegnal|kegnaleh)\b',
            # Numbers in phonetic
            r'\b(and|hulet|hulet and|sost|arba|arba and|arba sost|arba arba|ammist|ammist and)\b',
            # Common phrases
            r'\b(neger ale|neger lij|neger lijoch|betam neger|betam lij|betam lijoch)\b',
            # Product related
            r'\b(tomato|mango|orange|banana|potato|onion|garlic|pepper|carrot|cabbage|lettuce|spinach)\b',
            # Units
            r'\b(kilo|kg|liter|liters|quintal|ton)\b'
        ]
        
        # Check if text matches phonetic patterns
        combined_pattern = '|'.join(f'(?:{pattern})' for pattern in phonetic_amharic_indicators)
        if re.search(combined_pattern, text.lower(), re.IGNORECASE):
            return "phonetic_amharic"
        
        # Additional check: if text contains many common Amharic syllable patterns
        # Amharic syllables often end with specific patterns
        amharic_syllable_patterns = [
            r'\b\w*[aeiou][aeiou]\w*\b',  # Double vowels (common in phonetic Amharic)
            r'\b\w*esh\w*\b',  # Common ending
            r'\b\w*och\w*\b',  # Common ending
            r'\b\w*gn\w*\b',   # Common consonant combination
            r'\b\w*ch\w*\b',   # Common consonant combination
            r'\b\w*sh\w*\b',   # Common consonant combination
        ]
        
        phonetic_matches = 0
        for pattern in amharic_syllable_patterns:
            if re.search(pattern, text.lower()):
                phonetic_matches += 1
        
        # If multiple phonetic patterns match, likely phonetic Amharic
        if phonetic_matches >= 2:
            return "phonetic_amharic"
        
        # Default to English
        return "english"

    def _get_multilingual_response(self, key: str, language: str, **kwargs) -> str:
        """
        Get a multilingual response based on the language.
        
        Args:
            key: The response key
            language: The detected language ('amharic', 'phonetic_amharic', 'english')
            **kwargs: Format arguments for the response
        """
        responses = self._get_response_dictionary()
        
        if key not in responses:
            logger.warning(f"Response key '{key}' not found in multilingual dictionary")
            return f"Response key '{key}' not found."
        
        response_dict = responses[key]
        response_text = response_dict.get(language, response_dict.get('english', f"Missing response for key: {key}"))
        
        if kwargs:
            try:
                response_text = response_text.format(**kwargs)
            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to format response '{key}' with kwargs {kwargs}: {e}")
        
        return response_text

    def _get_response_dictionary(self) -> Dict[str, Dict[str, str]]:
        """Get the complete multilingual response dictionary."""
        return get_multilingual_response_dictionary()

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

            # Detect language from user message
            detected_language = self._detect_language(user_message)
            session_context["detected_language"] = detected_language
            logger.info(f"Detected language: {detected_language}")

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
                response = await self._handle_unknown_intent(user_message, chat_history, detected_language)
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
                response = self._get_multilingual_response("error_unknown", detected_language)

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
            detected_language = session_context.get("detected_language", "english") if session_context else "english"
            return {
                "response": self._get_multilingual_response("error_generic", detected_language),
                "session_context": session_context or {},
                "error": str(exc)
            }

    async def _handle_unknown_intent(
        self, user_message: str, chat_history: List[Dict[str, str]], language: str = "english"
    ) -> str:
        """Handle unknown intents by asking for clarification."""
        # Check for simple greetings
        greeting_words = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "greetings"]
        phonetic_greetings = ["selam", "salam", "tenayistilign", "dehna", "dehna hun"]
        amharic_greetings = ["ሰላም", "ጤና ይስጥልኝ", "ደህና"]
        
        all_greetings = greeting_words + phonetic_greetings + amharic_greetings
        
        if any(word in user_message.lower() for word in all_greetings):
            return self._get_multilingual_response("greeting", language)

        # Check if this might be a confirmation of previous context
        if chat_history:
            last_assistant_msg = None
            for msg in reversed(chat_history):
                if msg.get("role") == "assistant":
                    last_assistant_msg = msg.get("content", "")
                    break

            if any(word in user_message.lower() for word in ["yes", "okay", "sure", "go ahead", "confirm", "aw", "ey", "eyu"]):
                return self._get_multilingual_response("confirmation_response", language)

        return self._get_multilingual_response("unknown_intent", language)

    async def _handle_customer_flow(
        self,
        intent: str,
        filled_slots: Dict[str, Any],
        missing_slots: List[str],
        suggested_tools: List[str],
        session_context: Dict[str, Any]
    ) -> str:
        """Handle customer flow intents."""
        language = session_context.get("detected_language", "english")
        
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
                return self._get_multilingual_response("error_unknown", language)

        except Exception as exc:
            logger.error(f"Error in customer flow: {exc}")
            return self._get_multilingual_response("error_generic", language)

    async def _handle_supplier_flow(
        self,
        intent: str,
        filled_slots: Dict[str, Any],
        missing_slots: List[str],
        suggested_tools: List[str],
        session_context: Dict[str, Any]
    ) -> str:
        language = session_context.get("detected_language", "english")
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
                return self._get_multilingual_response("error_supplier", language)

        except Exception as exc:
            logger.error(f"Error in supplier flow: {exc}")
            return self._get_multilingual_response("error_generic", language)

    async def _handle_onboarding_flow(
        self,
        intent: str,
        filled_slots: Dict[str, Any],
        missing_slots: List[str],
        suggested_tools: List[str],
        session_context: Dict[str, Any]
    ) -> str:
        """Handle user onboarding flow."""
        language = session_context.get("detected_language", "english")
        
        try:
            if intent == "intent.user.is_customer":
                session_context["user_role"] = "customer"
                return self._get_multilingual_response("is_customer", language)

            elif intent == "intent.user.is_supplier":
                session_context["user_role"] = "supplier"
                return self._get_multilingual_response("is_supplier", language)

            elif intent == "intent.user.has_account":
                user_role = session_context.get("user_role")
                if not user_role:
                    return self._get_multilingual_response("unknown_intent", language)

                return self._get_multilingual_response("has_account", language, user_role=user_role)

            elif intent == "intent.user.new_user":
                return await self._handle_new_user_registration(filled_slots, session_context)

            elif intent == "intent.user.verify_account":
                return await self._handle_account_verification(filled_slots, session_context)

            else:
                return self._get_multilingual_response("unknown_intent", language)

        except Exception as exc:
            logger.error(f"Error in onboarding flow: {exc}")
            return self._get_multilingual_response("error_generic", language)

    # Onboarding flow handlers
    async def _handle_account_verification(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Handle verification of existing user account."""
        language = session_context.get("detected_language", "english")
        
        user_name = filled_slots.get("user_name")
        phone_number = filled_slots.get("phone_number")
        user_role = session_context.get("user_role")

        if not user_name or not phone_number:
            return self._get_multilingual_response("need_name_phone", language)

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
                    return self._get_multilingual_response("account_verified_customer", language, user_name=user_name)
                else:
                    # Supplier dashboard - show pending orders and expiring products
                    dashboard_info = await self._get_supplier_dashboard_info(user["user_id"])
                    return self._get_multilingual_response("account_verified_supplier", language, user_name=user_name, dashboard_info=dashboard_info)
            else:
                return self._get_multilingual_response("account_not_found", language, user_role=user_role)

        except Exception as exc:
            logger.error(f"Failed to verify account: {exc}")
            return self._get_multilingual_response("error_generic", language)

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
        language = session_context.get("detected_language", "english")
        
        user_role = session_context.get("user_role")
        if not user_role:
            return self._get_multilingual_response("unknown_intent", language)

        if user_role == "customer":
            user_name = filled_slots.get("customer_name")
            phone_number = filled_slots.get("phone_number")
            default_location = filled_slots.get("default_location")

            if not user_name:
                return self._get_multilingual_response("ask_customer_name", language)
            if not phone_number:
                return self._get_multilingual_response("ask_phone_number", language)
            if not default_location:
                return self._get_multilingual_response("ask_default_location", language)

            try:
                result = await self.database_tool.run({
                    "table": "users",
                    "method": "create_user",
                    "args": [],
                    "kwargs": {
                        "name": user_name,
                        "phone": phone_number,
                        "default_location": default_location,
                        "preferred_language": language,  # Store detected language
                        "role": "customer",
                        "joined_date": datetime.date.today()  # Explicitly set joined date
                    }
                })
                session_context["user_id"] = result.get("user_id")
                session_context["authenticated"] = True
                return self._get_multilingual_response("customer_registered", language, customer_name=user_name)

            except Exception as exc:
                logger.error(f"Failed to register customer: {exc}")
                return self._get_multilingual_response("registration_failed", language)

        else:  # supplier
            supplier_name = filled_slots.get("supplier_name")
            phone_number = filled_slots.get("phone_number")

            if not supplier_name:
                return self._get_multilingual_response("ask_business_name", language)
            if not phone_number:
                return self._get_multilingual_response("ask_phone_number", language)

            try:
                result = await self.database_tool.run({
                    "table": "users",
                    "method": "create_user",
                    "args": [],
                    "kwargs": {
                        "name": supplier_name,
                        "phone": phone_number,
                        "default_location": "",  # Suppliers don't need a default location
                        "preferred_language": language,  # Store detected language
                        "role": "supplier",
                        "joined_date": datetime.date.today()  # Explicitly set joined date
                    }
                })
                session_context["user_id"] = result.get("user_id")
                session_context["authenticated"] = True
                return self._get_multilingual_response("supplier_registered", language, supplier_name=supplier_name)

            except Exception as exc:
                logger.error(f"Failed to register supplier: {exc}")
                return self._get_multilingual_response("registration_failed", language)

    # Customer flow handlers
    async def _handle_customer_registration(
        self, filled_slots: Dict[str, Any], missing_slots: List[str], session_context: Dict[str, Any]
    ) -> str:
        """Handle customer registration."""
        language = session_context.get("detected_language", "english")
        
        if missing_slots:
            slot = missing_slots[0]
            if slot == "customer_name":
                return self._get_multilingual_response("ask_customer_name", language)
            elif slot == "phone_number":
                return self._get_multilingual_response("ask_phone_number", language)
            elif slot == "default_location":
                return self._get_multilingual_response("ask_default_location", language)

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
                    "preferred_language": language,  # Store detected language
                    "role": "customer",
                    "joined_date": datetime.date.today()  # Explicitly set joined date
                }
            })
            session_context["user_id"] = result.get("user_id")
            return self._get_multilingual_response("customer_registered", language, customer_name=filled_slots['customer_name'])
        except Exception as exc:
            logger.error(f"Failed to register customer: {exc}")
            return self._get_multilingual_response("registration_failed", language)

    async def _handle_product_availability(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Check product availability."""
        language = session_context.get("detected_language", "english")
        
        product_name = filled_slots.get("product_name")
        if not product_name:
            return self._get_multilingual_response("what_product", language)

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
                    return self._get_multilingual_response("product_not_available", language, product_name=product_name)

                # Show available options
                response_parts = [self._get_multilingual_response("product_available", language, product_name=product_name)]
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
                    return self._get_multilingual_response("supplier_no_products", language, supplier_name=product_name)

                # Set supplier_id in session context for future orders
                session_context["supplier_id"] = supplier["user_id"]

                # Show products from this supplier
                response_parts = [self._get_multilingual_response("supplier_products", language, supplier_name=product_name)]
                for item in available_products[:5]:  # Show up to 5 products
                    product_name_display = item.get("product", {}).get("product_name_en", "Unknown product")
                    price = item.get("unit_price_etb", 0)
                    unit = item.get("unit", "kg")
                    quantity = item.get("quantity_available", 0)
                    response_parts.append(f"- {product_name_display}: {price} ETB per {unit} ({quantity} {unit} available)")

                return " ".join(response_parts)

            # Neither product nor supplier found
            return self._get_multilingual_response("product_not_available", language, product_name=product_name)

        except Exception as exc:
            logger.error(f"Failed to check availability: {exc}")
            return self._get_multilingual_response("error_generic", language)

    async def _handle_storage_advice(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Provide storage advice using RAG (Retrieval-Augmented Generation)."""
        language = session_context.get("detected_language", "english")
        
        product_name = filled_slots.get("product_name")
        if not product_name:
            return self._get_multilingual_response("what_product_storage", language)

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
        language = session_context.get("detected_language", "english")
        product_a = filled_slots.get("product_a")
        product_b = filled_slots.get("product_b")

        if not product_a or not product_b:
            return self._get_multilingual_response("nutrition_query_missing_products", language)

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
                return self._get_multilingual_response("nutrition_no_data", language, product_a=product_a, product_b=product_b)

            # Extract relevant text from results
            context_texts = []
            for result in context["results"][:3]:  # Use top 3 results
                text = result.get("text", "").strip()
                if text and len(text) > 10:
                    context_texts.append(text)

            if not context_texts:
                return self._get_multilingual_response("nutrition_no_data", language, product_a=product_a, product_b=product_b)

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
            return self._get_multilingual_response("nutrition_error", language, product_a=product_a, product_b=product_b)

    async def _handle_seasonal_query(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Handle seasonal availability queries using RAG."""
        language = session_context.get("detected_language", "english")
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
                return self._get_multilingual_response("seasonal_not_found", language)

            # Extract relevant text from results
            context_texts = []
            for result in context["results"][:3]:  # Use top 3 results
                text = result.get("text", "").strip()
                if text and len(text) > 10:
                    context_texts.append(text)

            if not context_texts:
                return self._get_multilingual_response("seasonal_no_data", language)

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
                return self._get_multilingual_response("seasonal_fallback", language, context=context_texts[0])

        except Exception as exc:
            logger.error(f"Failed to get seasonal info: {exc}")
            return self._get_multilingual_response("seasonal_error", language)

    async def _handle_in_season_query(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Handle 'what's in season' queries using RAG."""
        language = session_context.get("detected_language", "english")
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
                return self._get_multilingual_response("seasonal_not_found", language)

            # Extract relevant text from results
            context_texts = []
            for result in context["results"][:3]:  # Use top 3 results
                text = result.get("text", "").strip()
                if text and len(text) > 10:
                    context_texts.append(text)

            if not context_texts:
                return self._get_multilingual_response("seasonal_no_data", language)

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
                return self._get_multilingual_response("seasonal_fallback", language, context=context_texts[0])

        except Exception as exc:
            logger.error(f"Failed to get in-season info: {exc}")
            return self._get_multilingual_response("seasonal_error", language)

    async def _handle_general_advisory(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Handle general product advisory questions using RAG."""
        language = session_context.get("detected_language", "english")
        question = filled_slots.get("question")
        if not question:
            return self._get_multilingual_response("general_advisory_no_query", language)

        try:
            # Retrieve relevant context
            context = await self.vector_search.run({
                "query": question,
                "top_k": 5
            })

            if context.get("error") or not context.get("results"):
                return self._get_multilingual_response("general_advisory_not_found", language)

            # Extract relevant text from results
            context_texts = []
            for result in context["results"][:3]:  # Use top 3 results
                text = result.get("text", "").strip()
                if text and len(text) > 10:
                    context_texts.append(text)

            if not context_texts:
                return self._get_multilingual_response("general_advisory_no_data", language)

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
                return self._get_multilingual_response("general_advisory_fallback", language, context=context_texts[0])

        except Exception as exc:
            logger.error(f"Failed to get advisory info: {exc}")
            return self._get_multilingual_response("general_advisory_error", language)

    async def _handle_place_order(
        self, filled_slots: Dict[str, Any], missing_slots: List[str], session_context: Dict[str, Any]
    ) -> str:
        """Handle order placement."""
        language = session_context.get("detected_language", "english")
        user_id = session_context.get("user_id")
        if not user_id:
            return self._get_multilingual_response("register_first", language)

        order_items = filled_slots.get("order_items", [])
        if not order_items:
            return self._get_multilingual_response("what_order", language)

        # Handle case where order_items is a string (product name) instead of list
        if isinstance(order_items, str):
            # Try to parse quantity, unit, and product from string like "2 kilo mango"
            import re
            match = re.match(r'(\d+(?:\.\d+)?)\s*(kilo|kg|liter|liters?)\s+(.+)', order_items.lower())
            if match:
                quantity = float(match.group(1))
                unit = match.group(2)
                if unit in ['kilo', 'kg']:
                    unit = 'kg'
                elif unit in ['liter', 'liters']:
                    unit = 'liter'
                product_name = match.group(3).strip()
                order_items = [{"product_name": product_name, "quantity": quantity, "unit": unit}]
            else:
                # Fallback: treat the whole string as product name
                order_items = [{"product_name": order_items}]

        # Check if we have missing slots that need to be filled
        if missing_slots:
            if "quantity" in missing_slots:
                return self._get_multilingual_response("how_much", language)
            elif "preferred_delivery_date" in missing_slots:
                return self._get_multilingual_response("when_delivery", language)

        try:
            # Create the order
            total_price = 0
            order_details = []

            # Group order items by product to handle multiple quantities of same product
            product_groups = {}
            for item in order_items:
                if isinstance(item, dict):
                    product_name = item.get("product_name", "")
                    quantity = item.get("quantity", 1)
                else:
                    product_name = str(item)
                    quantity = 1

                if product_name not in product_groups:
                    product_groups[product_name] = 0
                product_groups[product_name] += quantity

            # Process each unique product
            for product_name, total_quantity in product_groups.items():

                # Find the product by any name
                product = await self.database_tool.run({
                    "table": "products",
                    "method": "find_product_by_any_name",
                    "args": [product_name],
                    "kwargs": {}
                })

                if not product:
                    return self._get_multilingual_response("product_not_available", language, product_name=product_name)

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
                    return self._get_multilingual_response("product_not_available", language, product_name=product_name)

                # Use first available supplier for now
                supplier_product = supplier_products[0]
                available_quantity = supplier_product.get("quantity_available", 0)
                
                if total_quantity > available_quantity:
                    return self._get_multilingual_response("insufficient_quantity", language, product_name=product_name, available_quantity=available_quantity, unit=supplier_product.get('unit', 'kg'))
                
                unit_price = supplier_product.get("unit_price_etb", 0)
                subtotal = total_quantity * unit_price
                total_price += subtotal

                order_details.append({
                    "product": supplier_product["product"],
                    "supplier": supplier_product["supplier"],
                    "quantity": total_quantity,
                    "unit_price": unit_price,
                    "unit": supplier_product.get("unit", "kg"),
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
                return self._get_multilingual_response("user_not_found", language)

            # Handle delivery date if provided
            delivery_date = filled_slots.get("delivery_date") or filled_slots.get("preferred_delivery_date")
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
                    return self._get_multilingual_response("product_supplier_not_found", language)

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
                        "unit": detail["unit"],
                        "price_per_unit": detail["unit_price"],
                        "subtotal": detail["subtotal"]
                    }
                })
                logger.info(f"Order item created successfully")

                # Update supplier inventory
                new_quantity = available_quantity - detail["quantity"]
                await self.database_tool.run({
                    "table": "supplier_products",
                    "method": "update_supplier_product",
                    "args": [supplier_product["inventory_id"]],
                    "kwargs": {"quantity_available": new_quantity}
                })
                logger.info(f"Updated supplier inventory: {supplier_product['inventory_id']} now has {new_quantity} {detail['unit']} available")

            return self._get_multilingual_response("order_placed", language, total_price=total_price)

        except Exception as exc:
            logger.error(f"Failed to place order: {exc}")
            return self._get_multilingual_response("order_failed", language)

    async def _handle_set_delivery_date(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        language = session_context.get("detected_language", "english")
        """Handle delivery date setting."""
        delivery_date = filled_slots.get("delivery_date")
        if not delivery_date:
            return self._get_multilingual_response("when_delivery_date", language)

        try:
            language = session_context.get("detected_language", "english")
            # Resolve the date
            resolved_date = await self.date_resolver.run(delivery_date)

            # Update the most recent pending order
            user_id = session_context.get("user_id")
            if not user_id:
                return self._get_multilingual_response("register_first", language)

            orders = await self.database_tool.run({
                "table": "transactions",
                "method": "list_transactions",
                "args": [],
                "kwargs": {"filters": {"user": user_id, "status": "Pending"}}
            })

            if not orders:
                return self._get_multilingual_response("no_deliveries", language)

            latest_order = orders[0]  # Assuming sorted by date desc

            await self.database_tool.run({
                "table": "transactions",
                "method": "update_transaction",
                "args": [latest_order["order_id"]],
                "kwargs": {"delivery_date": resolved_date.isoformat()}
            })

            return self._get_multilingual_response("delivery_date_set", language, delivery_date=resolved_date.strftime('%B %d, %Y'))

        except Exception as exc:
            logger.error(f"Failed to set delivery date: {exc}")
            return self._get_multilingual_response("delivery_date_failed", language)

    async def _handle_set_delivery_location(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Handle delivery location setting."""
        language = session_context.get("detected_language", "english")
        location = filled_slots.get("delivery_location")
        if not location:
            return self._get_multilingual_response("where_delivery", language)

        # For now, just acknowledge - in practice might need validation
        return self._get_multilingual_response("delivery_location_set", language, location=location)

    async def _handle_confirm_payment(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Handle payment confirmation."""
        language = session_context.get("detected_language", "english")
        order_ref = filled_slots.get("order_reference")
        if not order_ref:
            return self._get_multilingual_response("which_order_payment", language)

        try:
            # Update order status to Confirmed
            await self.database_tool.run({
                "table": "transactions",
                "method": "update_transaction",
                "args": [order_ref], # This assumes order_ref is the ID, which might be incorrect
                "kwargs": {"status": "Confirmed"}
            })

            return self._get_multilingual_response("payment_confirmed", language)

        except Exception as exc:
            logger.error(f"Failed to confirm payment: {exc}")
            return self._get_multilingual_response("payment_failed", language)

    async def _handle_customer_check_deliveries(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Handle customer checking their deliveries."""
        user_id = session_context.get("user_id")
        if not user_id:
            return self._get_multilingual_response("register_first", "english") # Should be customer specific

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
                return self._get_multilingual_response("no_deliveries", "english")

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
                    return self._get_multilingual_response("no_deliveries_date", "english", date=resolved_date.strftime('%B %d, %Y'))

            # If a specific order reference is requested, filter to that order
            if order_ref:
                transactions = [t for t in transactions if t.get("order_id", "").startswith(order_ref)]
                if not transactions:
                    return f"I couldn't find an order with reference '{order_ref}'." # Needs translation

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
            return self._get_multilingual_response("deliveries_failed", "english")

    # Supplier flow handlers
    async def _handle_supplier_registration(
        self, filled_slots: Dict[str, Any], missing_slots: List[str], session_context: Dict[str, Any]
    ) -> str:
        """Handle supplier registration."""
        language = session_context.get("detected_language", "english")
        
        if missing_slots:
            slot = missing_slots[0]
            if slot == "supplier_name":
                return self._get_multilingual_response("ask_business_name", language)
            elif slot == "phone_number":
                return self._get_multilingual_response("ask_phone_number", language)

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
                    "preferred_language": language,  # Store detected language
                    "role": "supplier",
                    "joined_date": datetime.date.today()  # Explicitly set joined date
                }
            })
            session_context["user_id"] = result.get("user_id")
            return self._get_multilingual_response("supplier_registered", language, supplier_name=filled_slots['supplier_name'])
        except Exception as exc:
            logger.error(f"Failed to register supplier: {exc}")
            return self._get_multilingual_response("registration_failed", language)

    async def _handle_add_product(
        self, filled_slots: Dict[str, Any], missing_slots: List[str], session_context: Dict[str, Any]
    ) -> str:
        """Handle adding a new product."""
        language = session_context.get("detected_language", "english")
        user_id = session_context.get("user_id")
        if not user_id:
            return self._get_multilingual_response("register_first", language)

        product_name = filled_slots.get("product_name")
        if not product_name:
            return self._get_multilingual_response("what_product_add", language)

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

            return self._get_multilingual_response("what_quantity", language)

        except Exception as exc:
            logger.error(f"Failed to add product: {exc}")
            return self._get_multilingual_response("error_generic", language)

    async def _handle_set_quantity(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Handle setting product quantity."""
        language = session_context.get("detected_language", "english")
        user_id = session_context.get("user_id")
        if not user_id:
            return self._get_multilingual_response("register_first", language)

        product_name = filled_slots.get("product_name")
        quantity = filled_slots.get("quantity")

        if not product_name or quantity is None:
            return self._get_multilingual_response("what_quantity", language)

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

            return f"I'll add {quantity} kg of {product_name}.{suggestion_text} {self._get_multilingual_response('what_price', language)}"

        except Exception as exc:
            logger.error(f"Failed to get pricing suggestions: {exc}")
            return f"I'll add {quantity} kg of {product_name}. {self._get_multilingual_response('what_price', language)}"

    async def _handle_set_delivery_dates(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Handle setting delivery dates."""
        language = session_context.get("detected_language", "english")
        delivery_dates = filled_slots.get("delivery_dates")
        pending_product = session_context.get("pending_product")

        if not pending_product:
            return self._get_multilingual_response("error_generic", language)

        product_name = pending_product.get("product_name", "this product")

        if not delivery_dates:
            return self._get_multilingual_response("what_delivery_days", language, product_name=product_name)

        # Store delivery dates
        pending_product["delivery_dates"] = delivery_dates

        # Now we have all the information, create the supplier product
        user_id = session_context.get("user_id")
        if not user_id:
            return self._get_multilingual_response("register_first", language)

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
                return self._get_multilingual_response("error_generic", language)

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
            return self._get_multilingual_response("error_generic", language)

    async def _handle_set_expiry_date(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Handle setting expiry date."""
        language = session_context.get("detected_language", "english")
        expiry_date_input = filled_slots.get("expiry_date")
        pending_product = session_context.get("pending_product")

        if not pending_product:
            return self._get_multilingual_response("register_first", language)

        product_name = pending_product.get("product_name", "this product")

        # Handle cases where user says no expiry or similar
        if expiry_date_input and any(word in expiry_date_input.lower() for word in ["no", "none", "doesn't", "never", "no expiry"]):
            pending_product["expiry_date"] = None
            return self._get_multilingual_response("no_expiry_noted", language, product_name=product_name)

        if not expiry_date_input:
            return self._get_multilingual_response("what_expiry_date", language, product_name=product_name)

        try:
            resolved_date = await self.date_resolver.run(expiry_date_input)
            pending_product["expiry_date"] = resolved_date.isoformat()
            return self._get_multilingual_response("expiry_date_set", language, date=resolved_date.strftime('%B %d, %Y'), product_name=product_name)
        except Exception as exc:
            logger.error(f"Failed to resolve expiry date: {exc}")
            # Store the raw input if date resolution fails
            pending_product["expiry_date"] = expiry_date_input
            return self._get_multilingual_response("expiry_date_noted", language, input=expiry_date_input, product_name=product_name)

    async def _handle_set_price(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Handle setting product price."""
        language = session_context.get("detected_language", "english")
        user_id = session_context.get("user_id")
        if not user_id:
            return self._get_multilingual_response("register_first", language)

        unit_price = filled_slots.get("unit_price")

        if unit_price is None:
            return self._get_multilingual_response("what_price", language)

        # Check if we have pending product information
        pending_product = session_context.get("pending_product")
        if not pending_product:
            return self._get_multilingual_response("need_product_quantity", language)

        product_name = pending_product.get("product_name")
        quantity = pending_product.get("quantity")

        if not product_name or quantity is None:
            return self._get_multilingual_response("need_product_quantity", language)

        # Store the price in pending product info
        pending_product["unit_price"] = unit_price

        return self._get_multilingual_response("price_set", language, unit_price=unit_price, product_name=product_name)

    async def _handle_pricing_insight(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Provide pricing insights."""
        language = session_context.get("detected_language", "english")
        product_name = filled_slots.get("product_name")
        if not product_name:
            return self._get_multilingual_response("what_product_pricing", language)

        try:
            # Get competitor prices
            competitor_prices = await self.database_tool.run({
                "table": "competitor_prices",
                "method": "list_competitor_prices",
                "args": [],
                "kwargs": {"filters": {"product__product_name_en__icontains": product_name}}
            })

            if not competitor_prices:
                return self._get_multilingual_response("no_competitor_data", language, product_name=product_name)

            avg_price = sum(cp.get("price_etb_per_kg", 0) for cp in competitor_prices) / len(competitor_prices)
            return self._get_multilingual_response("competitor_price", language, product_name=product_name, avg_price=f"{avg_price:.2f}")

        except Exception as exc:
            logger.error(f"Failed to get pricing insight: {exc}")
            return self._get_multilingual_response("pricing_error", language, product_name=product_name)

    async def _handle_generate_image(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Generate product image."""
        language = session_context.get("detected_language", "english")
        product_name = filled_slots.get("product_name")
        if not product_name:
            return self._get_multilingual_response("what_product_image", language)

        try:
            result = await self.image_generator.run(product_name)
            return self._get_multilingual_response("image_generated", language, product_name=product_name, result=result)
        except Exception as exc:
            logger.error(f"Failed to generate image: {exc}")
            return self._get_multilingual_response("image_error", language, product_name=product_name)

    async def _handle_check_stock(
        self, session_context: Dict[str, Any]
    ) -> str:
        """Check supplier's stock."""
        language = session_context.get("detected_language", "english")
        user_id = session_context.get("user_id")
        if not user_id:
            return self._get_multilingual_response("register_first", language)

        try:
            products = await self.database_tool.run({
                "table": "supplier_products",
                "method": "list_supplier_products",
                "args": [],
                "kwargs": {"filters": {"supplier": user_id}}
            })

            if not products:
                return self._get_multilingual_response("no_inventory", language)

            response_parts = [self._get_multilingual_response("inventory_header", language)]
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
                    self._get_multilingual_response("inventory_item", language,
                        status_emoji=status_emoji, name=name, quantity=quantity, unit=unit,
                        price=price, delivery_days=delivery_days, expiry_info=expiry_info)
                )

            return " ".join(response_parts)

        except Exception as exc:
            logger.error(f"Failed to check stock: {exc}")
            return self._get_multilingual_response("inventory_error", language)

    async def _handle_view_expiring_products(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """View expiring products."""
        language = session_context.get("detected_language", "english")
        time_horizon = filled_slots.get("time_horizon", "1 week")
        return self._get_multilingual_response("expiring_products", language, time_horizon=time_horizon)

    async def _handle_accept_flash_sale(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Accept flash sale."""
        language = session_context.get("detected_language", "english")
        product_name = filled_slots.get("product_name")
        if not product_name:
            return self._get_multilingual_response("what_flash_sale_accept", language)

        return self._get_multilingual_response("flash_sale_accepted", language, product_name=product_name)

    async def _handle_decline_flash_sale(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Decline flash sale."""
        language = session_context.get("detected_language", "english")
        product_name = filled_slots.get("product_name")
        if not product_name:
            return self._get_multilingual_response("what_flash_sale_decline", language)

        return self._get_multilingual_response("flash_sale_declined", language, product_name=product_name)

    async def _handle_view_delivery_schedule(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """View delivery schedule."""
        language = session_context.get("detected_language", "english")
        date_range = filled_slots.get("date_range")
        query = "your delivery schedule"
        if date_range:
            query += f" for {date_range}"

        return self._get_multilingual_response("delivery_schedule", language, date_range=date_range or "all dates")

    async def _handle_check_deliveries_by_date(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Check deliveries by date."""
        language = session_context.get("detected_language", "english")
        user_id = session_context.get("user_id")
        if not user_id:
            return self._get_multilingual_response("login_supplier", language)

        date = filled_slots.get("date")
        if not date:
            return self._get_multilingual_response("what_date_deliveries", language)

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
                return self._get_multilingual_response("no_deliveries_date", language, date=resolved_date.strftime('%B %d, %Y'))

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
                return self._get_multilingual_response("no_deliveries_date", language, date=resolved_date.strftime('%B %d, %Y'))

            # Format the response
            response_parts = [self._get_multilingual_response("deliveries_date", language, date=resolved_date.strftime('%B %d, %Y'))]

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
            return self._get_multilingual_response("deliveries_error", language, date=date)

    async def _handle_supplier_check_deliveries(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Handle supplier checking their deliveries/orders."""
        language = session_context.get("detected_language", "english")
        user_id = session_context.get("user_id")
        if not user_id:
            return self._get_multilingual_response("login_supplier", language)

        date_filter = filled_slots.get("date")
        order_ref = filled_slots.get("order_reference")

        try:
            # Get pending orders for this supplier
            pending_orders = await self._get_supplier_pending_orders(user_id)

            if not pending_orders:
                return self._get_multilingual_response("no_pending_orders", language)

            # If a specific date is requested, we could filter further, but for now just return all pending
            # The _get_supplier_pending_orders already filters for pending/confirmed status

            # If a specific order reference is requested, filter to that order
            if order_ref:
                # This is a simple implementation - in practice might need better matching
                if order_ref.lower() not in pending_orders.lower():
                    return self._get_multilingual_response("order_not_found", language, order_ref=order_ref)

            return pending_orders

        except Exception as exc:
            logger.error(f"Failed to check supplier deliveries: {exc}")
            return self._get_multilingual_response("deliveries_error", language, date="this period")

    async def _handle_accept_order(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Accept order."""
        language = session_context.get("detected_language", "english")
        order_ref = filled_slots.get("order_reference")
        if not order_ref:
            return self._get_multilingual_response("what_order_accept", language)

        return self._get_multilingual_response("order_accepted", language, order_ref=order_ref)

    async def _handle_update_inventory(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Handle updating existing inventory quantity."""
        language = session_context.get("detected_language", "english")
        user_id = session_context.get("user_id")
        if not user_id:
            return self._get_multilingual_response("register_first", language)

        product_name = filled_slots.get("product_name")
        quantity = filled_slots.get("quantity")

        if not product_name or quantity is None:
            return self._get_multilingual_response("what_product_quantity_update", language)

        try:
            # Find the product
            product = await self.database_tool.run({
                "table": "products",
                "method": "find_product_by_any_name",
                "args": [product_name],
                "kwargs": {}
            })

            if not product:
                return self._get_multilingual_response("product_not_found", language, product_name=product_name)

            # Check if supplier already has this product
            supplier_products = await self.database_tool.run({
                "table": "supplier_products",
                "method": "list_supplier_products",
                "args": [],
                "kwargs": {"filters": {"supplier": user_id, "product": product["product_id"]}}
            })

            if not supplier_products:
                return self._get_multilingual_response("not_in_inventory", language, product_name=product_name)

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

            return self._get_multilingual_response("inventory_updated", language,
                quantity=quantity, product_name=product_name, new_quantity=new_quantity,
                current_price=existing_product.get('unit_price_etb', 'current price'),
                delivery_days=existing_product.get('available_delivery_days', 'existing schedule'))

        except Exception as exc:
            logger.error(f"Failed to update inventory: {exc}")
            return self._get_multilingual_response("inventory_update_error", language, product_name=product_name)
