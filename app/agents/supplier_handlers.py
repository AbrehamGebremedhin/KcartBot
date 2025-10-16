"""Supplier flow handlers for KCartBot Agent."""

from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, List

from app.services.llm_service import LLMService
from app.tools.database_tool import DatabaseAccessTool
from app.tools.date_tool import DateResolverTool
from app.tools.generate_image import ImageGeneratorTool
from app.utils.language_utils import LanguageDetector, TranslationService, MultilingualResponseFormatter

logger = logging.getLogger(__name__)


class SupplierHandlers:
    """Handles all supplier flow intents and operations."""

    def __init__(self, database_tool: DatabaseAccessTool, vector_search, date_resolver: DateResolverTool, llm_service: LLMService, image_generator):
        """Initialize supplier handlers with required tools and services."""
        self.database_tool = database_tool
        self.vector_search = vector_search
        self.date_resolver = date_resolver
        self.llm_service = llm_service
        self.image_generator = image_generator

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

    async def _handle_supplier_registration(
        self, filled_slots: Dict[str, Any], missing_slots: List[str], session_context: Dict[str, Any]
    ) -> str:
        """Handle supplier registration."""
        detected_language, user_preferred_language = await self._get_language_preferences(session_context)

        if missing_slots:
            slot = missing_slots[0]
            if slot == "supplier_name":
                response = "Welcome! What's your business name?"
            elif slot == "phone_number":
                response = "Great! What's your phone number?"

            return await self.response_formatter.format_response(
                response, detected_language, user_preferred_language
            )

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
            response = f"Welcome {filled_slots['supplier_name']}! Your supplier account has been created. How can I help you manage your inventory?"
            return await self.response_formatter.format_response(
                response, detected_language, user_preferred_language
            )
        except Exception as exc:
            logger.error(f"Failed to register supplier: {exc}")
            response = "I couldn't create your account. Please try again."
            return await self.response_formatter.format_response(
                response, detected_language, user_preferred_language
            )

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

    async def _handle_add_to_existing(self, session_context: Dict[str, Any]) -> str:
        """Handle adding to existing inventory."""
        pending_product = session_context.get("pending_product")
        if not pending_product:
            return "Please start by adding a product first."

        # Set flag to indicate user wants to add to existing
        pending_product["update_existing"] = True
        return "Got it! I'll add to your existing inventory. What's the quantity you want to add?"

    async def _handle_create_new_listing(self, session_context: Dict[str, Any]) -> str:
        """Handle creating a new listing."""
        pending_product = session_context.get("pending_product")
        if not pending_product:
            return "Please start by adding a product first."

        # Set flag to indicate user wants to create new listing
        pending_product["update_existing"] = False
        return "Got it! I'll create a new listing. What's the quantity for this new listing?"

    async def handle_flow(
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
                # This seems to be a shared intent, but we'll handle it in customer handlers
                return "Nutrition queries are typically for customers. What would you like to do with your inventory?"

            else:
                return "I'm here to help you manage your inventory and sales. What would you like to do?"

        except Exception as exc:
            logger.error(f"Error in supplier flow: {exc}")
            return "I encountered an issue processing your request. Please try again."