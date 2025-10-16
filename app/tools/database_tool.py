"""Tool for accessing database tables using repositories."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from app.core.tortoise_config import init_db, close_db
from app.db.repository.competitor_price_repository import CompetitorPriceRepository
from app.db.repository.flash_sale_repository import FlashSaleRepository
from app.db.repository.order_item_repository import OrderItemRepository
from app.db.repository.product_repository import ProductRepository
from app.db.repository.supplier_product_repository import SupplierProductRepository
from app.db.repository.transaction_repository import TransactionRepository
from app.db.repository.user_repository import UserRepository
from app.tools.base import ToolBase

logger = logging.getLogger(__name__)


class DatabaseAccessTool(ToolBase):
    """Tool that provides access to database tables via repositories with full CRUD operations."""

    def __init__(self) -> None:
        self.repositories = {
            "users": UserRepository,
            "products": ProductRepository,
            "supplier_products": SupplierProductRepository,
            "competitor_prices": CompetitorPriceRepository,
            "transactions": TransactionRepository,
            "order_items": OrderItemRepository,
            "flash_sales": FlashSaleRepository,
        }

        description = self._build_description()
        super().__init__(
            name="database_access",
            description=description,
        )

    def _build_description(self) -> str:
        """Build a comprehensive description of all tables and their attributes."""
        return """
Access database tables using their respective repositories. Supports CRUD operations: create, get_by_id, update, delete, list, and table-specific methods.

Available tables and their attributes:

1. users - Stores information about users (customers and suppliers)
   - user_id: Integer (primary key)
   - name: String (user's full name)
   - phone: String (unique phone number)
   - default_location: String (user's default location)
   - preferred_language: Enum (English, Amharic)
   - role: Enum (customer, supplier)
   - joined_date: Date (when user joined the platform)
   - created_at: Datetime (record creation timestamp)

2. products - Stores product information with multilingual names
   - product_id: UUID (primary key)
   - product_name_en: String (English product name)
   - product_name_am: String (Amharic product name)
   - product_name_am_latin: String (Latin transliteration of Amharic name)
   - category: Enum (Vegetable, Fruit, Dairy)
   - unit: Enum (kg, liter)
   - base_price_etb: Float (base price in Ethiopian Birr)
   - in_season_start: Enum (month when product is in season)
   - in_season_end: Enum (month when product season ends)
   - image_url: String (optional product image URL)
   - created_at: Datetime (record creation timestamp)

3. supplier_products - Stores supplier inventory for specific products
   - inventory_id: UUID (primary key)
   - supplier: ForeignKey (reference to User model)
   - product: ForeignKey (reference to Product model)
   - quantity_available: Float (available quantity)
   - unit: Enum (kg, liter)
   - unit_price_etb: Float (price per unit in Ethiopian Birr)
   - expiry_date: Date (optional expiry date)
   - available_delivery_days: String (days available for delivery)
   - last_updated: Datetime (last update timestamp)
   - status: Enum (active, expired, on_sale)

4. competitor_prices - Stores competitor pricing data
   - id: UUID (primary key)
   - product: ForeignKey (reference to Product model)
   - tier: Enum (Local_Shop, Supermarket, Distribution_Center)
   - date: Date (price date)
   - price_etb_per_kg: Float (price per kg in Ethiopian Birr)
   - source_location: String (location of competitor)
   - created_at: Datetime (record creation timestamp)

5. transactions - Stores order/transaction information
   - order_id: UUID (primary key)
   - user: ForeignKey (reference to User model)
   - date: Date (transaction date)
   - delivery_date: Date (optional delivery date)
   - total_price: Float (total price in Ethiopian Birr)
   - payment_method: Enum (COD - Cash on Delivery)
   - status: Enum (Pending, Confirmed, Delivered, Cancelled)
   - created_at: Datetime (record creation timestamp)

6. order_items - Stores individual items within orders
   - id: UUID (primary key)
   - order: ForeignKey (reference to Transaction model)
   - product: ForeignKey (reference to Product model)
   - supplier: ForeignKey (optional reference to User model)
   - quantity: Float (ordered quantity)
   - unit: Enum (kg, liter)
   - price_per_unit: Float (price per unit in Ethiopian Birr)
   - subtotal: Float (subtotal for this item)

7. flash_sales - Stores flash sale information
   - id: Integer (primary key)
   - supplier_product: ForeignKey (optional reference to SupplierProduct model)
   - supplier: ForeignKey (reference to User model)
   - product: ForeignKey (reference to Product model)
   - start_date: Datetime (sale start time)
   - end_date: Datetime (sale end time)
   - discount_percent: Float (discount percentage)
   - status: Enum (proposed, scheduled, active, expired, cancelled)
   - auto_generated: Boolean (whether auto-generated)
   - created_at: Datetime (record creation timestamp)
   - updated_at: Datetime (last update timestamp)

Input format: {"table": "table_name", "method": "method_name", "args": [], "kwargs": {}}
Example: {"table": "users", "method": "list_users", "args": [], "kwargs": {"filters": {"role": "customer"}}}
"""

    async def run(self, input: Any, context: Optional[Dict[str, Any]] = None) -> Any:
        """Execute database operations via repositories.

        Args:
            input: Dictionary with 'table', 'method', 'args', 'kwargs', and optionally 'raw_instances'
            context: Optional context dictionary.

        Returns:
            Result of the repository method call. If 'raw_instances' is True, returns raw model instances.
            Otherwise, returns serialized dictionaries for JSON compatibility.

        Raises:
            ValueError: If input is invalid or table/method not found.
        """
        if not isinstance(input, dict):
            raise ValueError("Input must be a dictionary with 'table', 'method', 'args', 'kwargs'")

        table = input.get("table")
        method = input.get("method")
        args = input.get("args", [])
        kwargs = input.get("kwargs", {})
        raw_instances = input.get("raw_instances", False)  # New parameter

        if not table or not method:
            raise ValueError("Input must include 'table' and 'method'")

        if table not in self.repositories:
            raise ValueError(f"Unknown table: {table}. Available tables: {list(self.repositories.keys())}")

        repo_class = self.repositories[table]

        if not hasattr(repo_class, method):
            raise ValueError(f"Unknown method '{method}' for table '{table}'")

        repo_method = getattr(repo_class, method)

        try:
            if args and kwargs:
                result = await repo_method(*args, **kwargs)
            elif args:
                result = await repo_method(*args)
            elif kwargs:
                result = await repo_method(**kwargs)
            else:
                result = await repo_method()

            # Return raw instances if requested, otherwise serialize
            if raw_instances:
                return result
            else:
                return self._serialize_result(result)
        except Exception as exc:
            logger.error("Database operation failed: %s", exc)
            raise ValueError(f"Database operation failed: {exc}") from exc

    def _serialize_result(self, result):
        """Convert Tortoise ORM results to JSON-serializable dictionaries."""
        if result is None:
            return None

        # Handle single model instance
        if hasattr(result, '_meta'):  # Tortoise model instance
            return self._model_to_dict(result)

        # Handle queryset/list of models
        if hasattr(result, '__iter__') and not isinstance(result, (str, dict)):
            try:
                return [self._model_to_dict(item) for item in result]
            except (TypeError, AttributeError):
                # If it's not a list of models, return as-is
                return result

        # Return primitive types as-is
        return result

    def _model_to_dict(self, model):
        """Convert a Tortoise model instance to a dictionary."""
        if not hasattr(model, '_meta'):
            return model

        # Use the model's built-in dict conversion if available
        if hasattr(model, 'dict'):
            return model.dict()

        # Fallback: manually extract fields based on model type
        data = {}

        # Get the model class name to handle specific fields
        model_name = model.__class__.__name__

        if model_name == 'User':
            data = {
                'user_id': getattr(model, 'user_id', None),
                'name': getattr(model, 'name', None),
                'phone': getattr(model, 'phone', None),
                'default_location': getattr(model, 'default_location', None),
                'preferred_language': getattr(model, 'preferred_language', None).value if getattr(model, 'preferred_language', None) else None,
                'role': getattr(model, 'role', None).value if getattr(model, 'role', None) else None,
                'joined_date': getattr(model, 'joined_date', None),
                'created_at': getattr(model, 'created_at', None),
            }
        elif model_name == 'Product':
            data = {
                'product_id': str(getattr(model, 'product_id', None)),
                'product_name_en': getattr(model, 'product_name_en', None),
                'product_name_am': getattr(model, 'product_name_am', None),
                'product_name_am_latin': getattr(model, 'product_name_am_latin', None),
                'category': getattr(model, 'category', None).value if getattr(model, 'category', None) else None,
                'unit': getattr(model, 'unit', None).value if getattr(model, 'unit', None) else None,
                'base_price_etb': getattr(model, 'base_price_etb', None),
                'in_season_start': getattr(model, 'in_season_start', None).value if getattr(model, 'in_season_start', None) else None,
                'in_season_end': getattr(model, 'in_season_end', None).value if getattr(model, 'in_season_end', None) else None,
                'image_url': getattr(model, 'image_url', None),
                'created_at': getattr(model, 'created_at', None),
            }
        elif model_name == 'SupplierProduct':
            supplier = getattr(model, 'supplier', None)
            product = getattr(model, 'product', None)
            data = {
                'inventory_id': str(getattr(model, 'inventory_id', None)),
                'supplier': self._model_to_dict(supplier) if supplier else None,
                'product': self._model_to_dict(product) if product else None,
                'quantity_available': getattr(model, 'quantity_available', None),
                'unit': getattr(model, 'unit', None).value if getattr(model, 'unit', None) else None,
                'unit_price_etb': getattr(model, 'unit_price_etb', None),
                'expiry_date': getattr(model, 'expiry_date', None),
                'available_delivery_days': getattr(model, 'available_delivery_days', None),
                'last_updated': getattr(model, 'last_updated', None),
                'status': getattr(model, 'status', None).value if getattr(model, 'status', None) else None,
            }
        elif model_name == 'Transaction':
            user = getattr(model, 'user', None)
            data = {
                'order_id': str(getattr(model, 'order_id', None)),
                'user': self._model_to_dict(user) if user else None,
                'date': getattr(model, 'date', None),
                'delivery_date': getattr(model, 'delivery_date', None),
                'total_price': getattr(model, 'total_price', None),
                'payment_method': getattr(model, 'payment_method', None).value if getattr(model, 'payment_method', None) else None,
                'status': getattr(model, 'status', None).value if getattr(model, 'status', None) else None,
                'created_at': getattr(model, 'created_at', None),
            }
        elif model_name == 'OrderItem':
            order = getattr(model, 'order', None)
            product = getattr(model, 'product', None)
            supplier = getattr(model, 'supplier', None)
            data = {
                'id': str(getattr(model, 'id', None)),
                'order': self._model_to_dict(order) if order else None,
                'product': self._model_to_dict(product) if product else None,
                'supplier': self._model_to_dict(supplier) if supplier else None,
                'quantity': getattr(model, 'quantity', None),
                'unit': getattr(model, 'unit', None).value if getattr(model, 'unit', None) else None,
                'price_per_unit': getattr(model, 'price_per_unit', None),
                'subtotal': getattr(model, 'subtotal', None),
            }
        elif model_name == 'CompetitorPrice':
            product = getattr(model, 'product', None)
            data = {
                'id': str(getattr(model, 'id', None)),
                'product': self._model_to_dict(product) if product else None,
                'tier': getattr(model, 'tier', None).value if getattr(model, 'tier', None) else None,
                'date': getattr(model, 'date', None),
                'price_etb_per_kg': getattr(model, 'price_etb_per_kg', None),
                'source_location': getattr(model, 'source_location', None),
                'created_at': getattr(model, 'created_at', None),
            }
        elif model_name == 'FlashSale':
            supplier_product = getattr(model, 'supplier_product', None)
            supplier = getattr(model, 'supplier', None)
            product = getattr(model, 'product', None)
            data = {
                'id': getattr(model, 'id', None),
                'supplier_product': self._model_to_dict(supplier_product) if supplier_product else None,
                'supplier': self._model_to_dict(supplier) if supplier else None,
                'product': self._model_to_dict(product) if product else None,
                'start_date': getattr(model, 'start_date', None),
                'end_date': getattr(model, 'end_date', None),
                'discount_percent': getattr(model, 'discount_percent', None),
                'status': getattr(model, 'status', None).value if getattr(model, 'status', None) else None,
                'auto_generated': getattr(model, 'auto_generated', None),
                'created_at': getattr(model, 'created_at', None),
                'updated_at': getattr(model, 'updated_at', None),
            }
        else:
            # Generic fallback for other models - get all non-private attributes
            for attr in dir(model):
                if not attr.startswith('_') and not callable(getattr(model, attr)):
                    value = getattr(model, attr)
                    if not hasattr(value, '_meta'):  # Skip related models for now
                        data[attr] = value

        return data
