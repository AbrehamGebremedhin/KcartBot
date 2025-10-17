"""Customer flow handlers for KCartBot Agent."""

from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, List

from app.services.llm_service import LLMService
from app.tools.database_tool import DatabaseAccessTool
from app.tools.date_tool import DateResolverTool
from app.utils.language_utils import LanguageDetector, TranslationService, MultilingualResponseFormatter

logger = logging.getLogger(__name__)


class CustomerHandlers:
    """Handles all customer flow intents and operations."""

    def __init__(self, database_tool: DatabaseAccessTool, vector_search, date_resolver: DateResolverTool, llm_service: LLMService):
        """Initialize customer handlers with required tools and services."""
        self.database_tool = database_tool
        self.vector_search = vector_search
        self.date_resolver = date_resolver
        self.llm_service = llm_service

        # Initialize language utilities
        self.language_detector = LanguageDetector()
        self.translation_service = TranslationService(llm_service)
        self.response_formatter = MultilingualResponseFormatter(self.translation_service)

    async def _get_language_preferences(self, session_context: Dict[str, Any]) -> tuple:
        """Get detected language and user preferred language."""
        detected_language = self.language_detector.detect_language(session_context.get("last_user_message", ""))

        # Get user preferred language if available
        user_id = session_context.get("user_id")
        user_preferred_language = None
        if user_id:
            try:
                user_data = await self.database_tool.run({
                    "table": "users",
                    "method": "get_user_by_id",
                    "args": [user_id],
                    "kwargs": {}
                })
                user_preferred_language = self.translation_service.get_language_from_user(user_data)
            except Exception:
                pass  # Fall back to detected language

        return detected_language, user_preferred_language

    async def _handle_customer_registration(
        self, filled_slots: Dict[str, Any], missing_slots: List[str], session_context: Dict[str, Any]
    ) -> str:
        """Handle customer registration."""
        detected_language, user_preferred_language = await self._get_language_preferences(session_context)

        if missing_slots:
            slot = missing_slots[0]
            if slot == "customer_name":
                response = "Welcome! What's your name?"
            elif slot == "phone_number":
                response = "Great! What's your phone number?"
            elif slot == "default_location":
                response = "Perfect! What's your default delivery location?"

            return await self.response_formatter.format_response(
                response, detected_language, user_preferred_language
            )

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

            response = f"Welcome {filled_slots['customer_name']}! Your account has been created. How can I help you today?"
            return await self.response_formatter.format_response(
                response, detected_language, user_preferred_language
            )
        except Exception as exc:
            logger.error(f"Failed to register customer: {exc}")
            response = "I couldn't create your account. Please try again."
            return await self.response_formatter.format_response(
                response, detected_language, user_preferred_language
            )

    async def _handle_product_availability(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Check product availability."""
        detected_language = self.language_detector.detect_language(session_context.get("last_user_message", ""))

        # Get user preferred language if available
        user_id = session_context.get("user_id")
        user_preferred_language = None
        if user_id:
            try:
                user_data = await self.database_tool.run({
                    "table": "users",
                    "method": "get_user_by_id",
                    "args": [user_id],
                    "kwargs": {}
                })
                user_preferred_language = self.translation_service.get_language_from_user(user_data)
            except Exception:
                pass

        product_name = filled_slots.get("product_name")
        if not product_name:
            response = "What product are you looking for?"
            return await self.response_formatter.format_response(
                response, detected_language, user_preferred_language
            )

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
                    response = f"Sorry, {product_name} is currently out of stock."
                    return await self.response_formatter.format_response(
                        response, detected_language, user_preferred_language
                    )

                # Show available options
                response_parts = [f"Yes, {product_name} is available from:"]
                for item in available_products[:3]:  # Show up to 3 options
                    supplier_name = item.get("supplier", {}).get("name", "Unknown supplier")
                    price = item.get("unit_price_etb", 0)
                    unit = item.get("unit", "kg")
                    quantity = item.get("quantity_available", 0)
                    response_parts.append(f"- {supplier_name}: {price} ETB per {unit} ({quantity} {unit} available)")

                response = " ".join(response_parts)
                return await self.response_formatter.format_response(
                    response, detected_language, user_preferred_language
                )

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
                    response = f"Sorry, {product_name} doesn't have any products available right now."
                    return await self.response_formatter.format_response(
                        response, detected_language, user_preferred_language
                    )

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

                response = " ".join(response_parts)
                return await self.response_formatter.format_response(
                    response, detected_language, user_preferred_language
                )

            # Neither product nor supplier found
            response = f"Sorry, {product_name} is not currently available."
            return await self.response_formatter.format_response(
                response, detected_language, user_preferred_language
            )

        except Exception as exc:
            logger.error(f"Failed to check availability: {exc}")
            response = "I couldn't check availability right now. Please try again."
            return await self.response_formatter.format_response(
                response, detected_language, user_preferred_language
            )

    async def _handle_storage_advice(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Provide storage advice using RAG (Retrieval-Augmented Generation)."""
        detected_language = self.language_detector.detect_language(session_context.get("last_user_message", ""))

        # Get user preferred language if available
        user_id = session_context.get("user_id")
        user_preferred_language = None
        if user_id:
            try:
                user_data = await self.database_tool.run({
                    "table": "users",
                    "method": "get_user_by_id",
                    "args": [user_id],
                    "kwargs": {}
                })
                user_preferred_language = self.translation_service.get_language_from_user(user_data)
            except Exception:
                pass

        product_name = filled_slots.get("product_name")
        if not product_name:
            response = "What product do you need storage advice for?"
            return await self.response_formatter.format_response(
                response, detected_language, user_preferred_language
            )

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
                response = f"I don't have specific storage advice for {product_name}, but generally keep fresh produce in a cool, dry place."
                return await self.response_formatter.format_response(
                    response, detected_language, user_preferred_language
                )

            results = context.get("results", [])
            if not results:
                response = f"I don't have specific storage advice for {product_name}, but generally keep fresh produce in a cool, dry place."
                return await self.response_formatter.format_response(
                    response, detected_language, user_preferred_language
                )

            # Extract relevant text from results
            context_texts = []
            for result in results[:3]:  # Use top 3 results for context
                text = result.get("text", "").strip()
                if text and len(text) > 10:
                    context_texts.append(text)

            if not context_texts:
                response = f"I don't have specific storage advice for {product_name} in my knowledge base, but here are some general tips for storing fresh produce: Keep most vegetables and fruits in a cool, dry place away from direct sunlight. Refrigerate leafy greens and cut produce in airtight containers. Wash fruits and vegetables just before eating, not before storing. Store different types of produce separately to prevent ethylene gas from speeding up ripening. Check regularly and remove any spoiled items to prevent them from affecting others."
                return await self.response_formatter.format_response(
                    response, detected_language, user_preferred_language
                )

            # Use RAG: Combine retrieved context with LLM generation
            context_combined = "\n".join(context_texts)
            user_question = f"How should I store {product_name}?"

            # Create RAG prompt
            rag_prompt = f"""
You are a food storage expert providing objective storage advice. Do not mention ordering, marketplace, suppliers, customers, or KCartBot. Focus only on storage recommendations.

Based on the following storage information, provide helpful and practical advice for storing {product_name}. The response should be in English as it will be translated later if needed.

Storage Information:
{context_combined}

User Question: {user_question}

Please provide clear, concise storage advice that incorporates the relevant information above. Focus on practical tips that will help preserve freshness and quality.
"""

            # Use LLM to generate response
            llm = self.llm_service.clone(system_prompt="")
            llm_response = await llm.acomplete(rag_prompt)

            if llm_response and llm_response.strip():
                response = llm_response.strip()
            else:
                # Fallback to direct context if LLM fails
                response = f"For {product_name}: {context_texts[0]}"

            return await self.response_formatter.format_response(
                response, detected_language, user_preferred_language
            )

        except Exception as exc:
            logger.error(f"Failed to get storage advice: {exc}")
            response = f"Generally, keep {product_name} in a cool, dry place away from direct sunlight. For best results, store fresh produce in the refrigerator for leafy greens and cut items, wash just before eating, and check regularly for spoilage."
            return await self.response_formatter.format_response(
                response, detected_language, user_preferred_language
            )

    async def _handle_nutrition_query(
        self, filled_slots: Dict[str, Any], session_context: Dict[str, Any]
    ) -> str:
        """Handle nutrition comparison queries using RAG."""
        detected_language, user_preferred_language = await self._get_language_preferences(session_context)

        product_a = filled_slots.get("product_a")
        product_b = filled_slots.get("product_b")

        if not product_a or not product_b:
            response = "Which two products would you like to compare nutritionally?"
            return await self.response_formatter.format_response(
                response, detected_language, user_preferred_language
            )

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
                response = f"I don't have specific nutritional data comparing {product_a} and {product_b}."
                return await self.response_formatter.format_response(
                    response, detected_language, user_preferred_language
                )

            # Extract relevant text from results
            context_texts = []
            for result in context["results"][:3]:  # Use top 3 results
                text = result.get("text", "").strip()
                if text and len(text) > 10:
                    context_texts.append(text)

            if not context_texts:
                response = f"I don't have specific nutritional data comparing {product_a} and {product_b}."
                return await self.response_formatter.format_response(
                    response, detected_language, user_preferred_language
                )

            # Use RAG: Combine retrieved context with LLM generation
            context_combined = "\n".join(context_texts)
            user_question = f"How do {product_a} and {product_b} compare nutritionally?"

            # Create RAG prompt
            rag_prompt = f"""
You are a nutrition expert providing objective nutritional information. Do not mention ordering, marketplace, suppliers, customers, or KCartBot. Focus only on the nutritional comparison.

Based on the following nutritional information, provide a helpful comparison between {product_a} and {product_b}. The response should be in English as it will be translated later if needed.

Nutritional Information:
{context_combined}

User Question: {user_question}

Please provide a clear, balanced nutritional comparison that highlights the key differences and similarities between these two products. Include specific nutritional benefits or considerations for each.
"""

            # Use LLM to generate response
            llm = self.llm_service.clone(system_prompt="")
            llm_response = await llm.acomplete(rag_prompt)

            if llm_response and llm_response.strip():
                response = llm_response.strip()
            else:
                # Fallback to direct context
                response = f"Nutritional comparison: {context_texts[0]}"

            return await self.response_formatter.format_response(
                response, detected_language, user_preferred_language
            )

        except Exception as exc:
            logger.error(f"Failed to get nutrition info: {exc}")
            response = f"I couldn't find nutritional information comparing {product_a} and {product_b}."
            return await self.response_formatter.format_response(
                response, detected_language, user_preferred_language
            )

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
        detected_language, user_preferred_language = await self._get_language_preferences(session_context)

        user_id = session_context.get("user_id")
        if not user_id:
            response = "Please register first before placing an order."
            return await self.response_formatter.format_response(
                response, detected_language, user_preferred_language
            )

        order_items = filled_slots.get("order_items", [])
        if not order_items:
            response = "What would you like to order?"
            return await self.response_formatter.format_response(
                response, detected_language, user_preferred_language
            )

        # Handle case where order_items is a string (product name) instead of list
        if isinstance(order_items, str):
            # Convert string to expected list format
            order_items = [{"product_name": order_items}]

        # Check if we have missing slots that need to be filled
        if missing_slots:
            if "quantity" in missing_slots:
                response = "How much would you like to order?"
                return await self.response_formatter.format_response(
                    response, detected_language, user_preferred_language
                )
            elif "preferred_delivery_date" in missing_slots:
                response = "When would you like this delivered?"
                return await self.response_formatter.format_response(
                    response, detected_language, user_preferred_language
                )

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
                    response = f"Sorry, {product_name} is not available."
                    return await self.response_formatter.format_response(
                        response, detected_language, user_preferred_language
                    )

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
                    response = f"Sorry, {product_name} is not available."
                    return await self.response_formatter.format_response(
                        response, detected_language, user_preferred_language
                    )

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
                response = "User not found. Please register first."
                return await self.response_formatter.format_response(
                    response, detected_language, user_preferred_language
                )

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
                    response = "Product or supplier information not found."
                    return await self.response_formatter.format_response(
                        response, detected_language, user_preferred_language
                    )

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

            response = f"Order placed successfully! Total: {total_price} ETB. Payment will be Cash on Delivery."
            return await self.response_formatter.format_response(
                response, detected_language, user_preferred_language
            )

        except Exception as exc:
            logger.error(f"Failed to place order: {exc}")
            response = "I couldn't place your order. Please try again."
            return await self.response_formatter.format_response(
                response, detected_language, user_preferred_language
            )

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

    async def handle_flow(
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