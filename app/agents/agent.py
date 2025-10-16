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
from app.agents.customer_handlers import CustomerHandlers
from app.agents.supplier_handlers import SupplierHandlers

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

        # Initialize handler classes
        self.customer_handlers = CustomerHandlers(
            database_tool=self.database_tool,
            vector_search=self.vector_search,
            date_resolver=self.date_resolver,
            llm_service=self.llm_service
        )
        self.supplier_handlers = SupplierHandlers(
            database_tool=self.database_tool,
            vector_search=self.vector_search,
            date_resolver=self.date_resolver,
            llm_service=self.llm_service,
            image_generator=self.image_generator
        )

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

            # Check if we can fill missing slots from a previous intent
            current_intent = session_context.get("current_intent")
            current_filled_slots = session_context.get("filled_slots", {})
            current_missing_slots = session_context.get("missing_slots", [])

            if current_intent and current_missing_slots:
                # If the new message can fill slots for the current intent, merge them
                can_fill_slots = False
                slot_mapping = {
                    "delivery_date": "preferred_delivery_date",  # Map delivery_date to preferred_delivery_date
                    # Add other mappings as needed
                }
                
                for slot in current_missing_slots:
                    if slot in filled_slots:
                        current_filled_slots[slot] = filled_slots[slot]
                        can_fill_slots = True
                    elif slot_mapping.get(slot) in filled_slots:
                        current_filled_slots[slot] = filled_slots[slot_mapping[slot]]
                        can_fill_slots = True

                if can_fill_slots:
                    # Update missing slots
                    updated_missing_slots = [slot for slot in current_missing_slots if slot not in current_filled_slots]
                    # Keep the current intent if slots were filled
                    intent = current_intent
                    flow = session_context.get("current_flow", flow)
                    filled_slots = current_filled_slots
                    missing_slots = updated_missing_slots
                    suggested_tools = session_context.get("suggested_tools", suggested_tools)

            logger.info(f"Final intent: {intent}, flow: {flow}, missing_slots: {missing_slots}")

            # Update session context with current intent and flow
            session_context.update({
                "current_intent": intent,
                "current_flow": flow,
                "filled_slots": filled_slots,
                "missing_slots": missing_slots,
                "suggested_tools": suggested_tools,
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

        return "I'm not sure what you mean. Are you a customer looking to place an order, or a supplier managing inventory?"

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
                    dashboard_info = await self.supplier_handlers._get_supplier_dashboard_info(user["user_id"])
                    return f"Welcome back, {user_name}! Your supplier account has been verified. {dashboard_info} How can I help you manage your inventory today?"
            else:
                return f"I couldn't find an account with that name and phone number. Would you like to create a new {user_role} account instead?"

        except Exception as exc:
            logger.error(f"Failed to verify account: {exc}")
            return "I couldn't verify your account right now. Please try again."



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
    async def _handle_customer_flow(
        self,
        intent: str,
        filled_slots: Dict[str, Any],
        missing_slots: List[str],
        suggested_tools: List[str],
        session_context: Dict[str, Any]
    ) -> str:
        """Handle customer flow by delegating to customer handlers."""
        return await self.customer_handlers.handle_flow(intent, filled_slots, missing_slots, suggested_tools, session_context)

    # Supplier flow handlers
    async def _handle_supplier_flow(
        self,
        intent: str,
        filled_slots: Dict[str, Any],
        missing_slots: List[str],
        suggested_tools: List[str],
        session_context: Dict[str, Any]
    ) -> str:
        """Handle supplier flow by delegating to supplier handlers."""
        return await self.supplier_handlers.handle_flow(intent, filled_slots, missing_slots, suggested_tools, session_context)
