from typing import Any, Dict, List, Optional
import json
import asyncio
from datetime import datetime, timedelta, date
from decimal import Decimal
from uuid import UUID

from tortoise import Tortoise
from tortoise.exceptions import IntegrityError, ConfigurationError
from tortoise.models import Model
from tortoise.queryset import QuerySet
from tortoise.fields.relational import (
    BackwardFKRelation,
    ForeignKeyFieldInstance,
    ManyToManyRelation,
)

from app.tools.base import ToolBase
from app.db.models import (
    PreferredLanguage,
    UserRole,
    FlashSaleStatus,
    SupplierProductStatus,
    UnitType,
    ProductCategory,
    Month,
    SupplierProduct,
    CompetitorTier,
)
from app.db.repository.user_repository import UserRepository
from app.db.repository.product_repository import ProductRepository
from app.db.repository.supplier_product_repository import SupplierProductRepository
from app.db.repository.competitor_price_repository import CompetitorPriceRepository
from app.db.repository.transaction_repository import TransactionRepository
from app.db.repository.order_item_repository import OrderItemRepository
from app.db.repository.flash_sale_repository import FlashSaleRepository
from pydantic import BaseModel, Field, ValidationError, field_validator
from app.core.tortoise_config import init_db
from app.utils.schedule import parse_delivery_schedule


_db_init_lock = asyncio.Lock()


async def ensure_db_initialized() -> None:
    """Initialise Tortoise ORM lazily when tools run outside FastAPI lifecycle."""
    try:
        Tortoise.get_connection("default")
        return
    except ConfigurationError:
        pass

    async with _db_init_lock:
        try:
            Tortoise.get_connection("default")
            return
        except ConfigurationError:
            await init_db()


class DataAccessRequest(BaseModel):
    entity: str = Field(description="Target entity name, e.g. products")
    operation: str = Field(default="list", description="Operation to execute: list|get|search|create|update|delete")
    id: Optional[Any] = Field(default=None, description="Identifier for get operations")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Filter criteria")
    data: Dict[str, Any] = Field(default_factory=dict, description="Payload for create operations")
    limit: Optional[int] = Field(default=None, ge=1, le=500, description="Optional row limit")

    @field_validator("entity", mode="before")
    @classmethod
    def _lower_entity(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("entity must be a string")
        return value.strip().lower()

    @field_validator("operation", mode="before")
    @classmethod
    def _lower_operation(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("operation must be a string")
        return value.strip().lower()


class AnalyticsRequest(BaseModel):
    operation: str = Field(description="Analytics operation to execute")
    product_id: Optional[str] = None
    product_name: Optional[str] = None
    user_id: Optional[int] = None
    supplier_id: Optional[int] = None
    supplier_name: Optional[str] = None
    filters: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("operation", mode="before")
    @classmethod
    def _lower_operation(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("operation must be a string")
        return value.strip().lower()


class DataAccessTool(ToolBase):
    """
    Tool for AI agent to access and manage data from various repositories.
    Supports querying (list/get/search) and full CRUD operations across
    users, products, supplier products, competitor prices, transactions,
    order items, and flash sales.
    """

    def __init__(self):
        super().__init__(
            name="data_access",
            description=(
                "Access and manage data within the marketplace. "
                "Supports operations: 'list', 'get', 'search', 'create', 'update', 'delete' on entities: "
                "'users', 'products', 'supplier_products', 'competitor_prices', 'transactions', "
                "'order_items', 'flash_sales'. "
                "Example input: {\"entity\": \"products\", \"operation\": \"list\", \"filters\": {\"category\": \"Vegetable\"}}"
            ),
        )
        self.repositories = {
            "users": UserRepository,
            "products": ProductRepository,
            "supplier_products": SupplierProductRepository,
            "competitor_prices": CompetitorPriceRepository,
            "transactions": TransactionRepository,
            "order_items": OrderItemRepository,
            "flash_sales": FlashSaleRepository,
        }
        self._supported_operations = ["list", "get", "search", "create", "update", "delete"]
        self._repository_methods: Dict[str, Dict[str, str]] = {
            "users": {
                "list": "list_users",
                "create": "create_user",
                "update": "update_user",
                "delete": "delete_user",
            },
            "products": {
                "list": "list_products",
                "create": "create_product",
                "update": "update_product",
                "delete": "delete_product",
            },
            "supplier_products": {
                "list": "list_supplier_products",
                "create": "create_supplier_product",
                "update": "update_supplier_product",
                "delete": "delete_supplier_product",
            },
            "competitor_prices": {
                "list": "list_competitor_prices",
                "create": "create_competitor_price",
                "update": "update_competitor_price",
                "delete": "delete_competitor_price",
            },
            "transactions": {
                "list": "list_transactions",
                "create": "create_transaction",
                "update": "update_transaction",
                "delete": "delete_transaction",
            },
            "order_items": {
                "list": "list_order_items",
                "create": "create_order_item",
                "update": "update_order_item",
                "delete": "delete_order_item",
            },
            "flash_sales": {
                "list": "list_flash_sales",
                "create": "create_flash_sale",
                "update": "update_flash_sale",
                "delete": "delete_flash_sale",
            },
        }

    async def run(self, input: Dict[str, Any], context: Dict[str, Any] = None) -> Any:
        """Execute validated data access operations and surface friendly errors."""
        await ensure_db_initialized()
        try:
            if isinstance(input, str):
                try:
                    input = json.loads(input)
                except json.JSONDecodeError:
                    return {"error": "Invalid JSON payload for data_access tool."}
            request = DataAccessRequest.model_validate(input)
        except ValidationError as exc:
            return {
                "error": "Invalid data_access request.",
                "details": exc.errors(),
            }
        except Exception as exc:  # pragma: no cover - defensive logging
            return {"error": f"Unexpected validation error: {exc}"}

        entity = request.entity
        operation = request.operation
        repository = self.repositories.get(entity)

        if repository is None:
            return {
                "error": f"Unknown entity: {entity}.",
                "valid_entities": list(self.repositories.keys()),
            }

        try:
            if operation == "get":
                return await self._get_by_id(repository, entity, request.id)
            if operation == "list":
                return await self._list_entities(repository, request.filters, request.limit)
            if operation == "search":
                return await self._search_entities(repository, request.filters, request.limit)
            if operation == "create":
                return await self._create_entity(repository, entity, request.data)
            if operation == "update":
                return await self._update_entity(entity, request.id, request.data)
            if operation == "delete":
                return await self._delete_entity(entity, request.id)
            return {
                "error": f"Unknown operation: {operation}.",
                "valid_operations": self._supported_operations,
            }
        except Exception as exc:  # pragma: no cover - repository level errors
            return {"error": f"Error accessing data: {exc}"}

    async def _get_by_id(self, repository, entity: str, entity_id: Any) -> Dict[str, Any]:
        """Retrieve a single entity by ID."""
        if not entity_id:
            return {"error": "ID is required for 'get' operation"}
        
        # Determine the appropriate method based on entity
        if entity == 'users':
            result = await repository.get_user_by_id(entity_id)
        elif entity == 'products':
            result = await repository.get_product_by_id(entity_id)
        elif entity == 'supplier_products':
            result = await repository.get_supplier_product_by_id(entity_id)
        elif entity == 'competitor_prices':
            result = await repository.get_competitor_price_by_id(entity_id)
        elif entity == 'transactions':
            result = await repository.get_transaction_by_id(entity_id)
        elif entity == 'order_items':
            result = await repository.get_order_item_by_id(entity_id)
        elif entity == 'flash_sales':
            result = await repository.get_flash_sale_by_id(entity_id)
        else:
            return {"error": f"Unsupported entity: {entity}"}
        
        if result:
            return await self._serialize_model(result)
        return {"error": f"{entity} with ID {entity_id} not found"}

    async def _list_entities(
        self, 
        repository, 
        filters: Optional[Dict] = None, 
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """List entities with optional filters."""
        # Find the appropriate list method for this repository
        list_methods = [m for m in dir(repository) if m.startswith('list_')]
        if not list_methods:
            return {"error": "No list method found for repository"}
        
        # Call the list method
        list_method = getattr(repository, list_methods[0])
        entities = await list_method(filters)
        
        # Apply limit if specified
        if limit and isinstance(entities, list):
            entities = entities[:limit]
        
        # Serialize results
        results = []
        for entity_obj in entities:
            results.append(await self._serialize_model(entity_obj))
        
        return {
            "count": len(results),
            "results": results
        }

    async def _search_entities(
        self, 
        repository, 
        filters: Dict, 
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Search entities with filters (similar to list but emphasizes filtering)."""
        if not filters:
            return {
                "error": "Filters are required for 'search' operation. "
                "Use 'list' operation without filters to get all entities."
            }
        return await self._list_entities(repository, filters, limit)

    async def _serialize_model(self, model) -> Dict[str, Any]:
        """Convert Tortoise ORM model instances to plain dictionaries."""
        if model is None:
            return None
        if isinstance(model, dict):
            return model
        if not isinstance(model, Model):
            return self._normalise_value(model)

        data: Dict[str, Any] = {}
        for field_name, field in model._meta.fields_map.items():
            if isinstance(field, BackwardFKRelation):
                continue
            if isinstance(field, ManyToManyRelation):
                continue
            if isinstance(field, ForeignKeyFieldInstance):
                source = field.source_field
                fk_value = getattr(model, source, None)
                data[source] = self._normalise_value(fk_value)
                continue

            value = getattr(model, field_name, None)
            data[field_name] = self._normalise_value(value)

        if isinstance(model, SupplierProduct):
            product = getattr(model, "product", None)
            if product is not None:
                data.setdefault("product_id", getattr(model, "product_id", None))
                data["product_name"] = getattr(product, "product_name_en", None)
                data["product_name_en"] = getattr(product, "product_name_en", None)
                data["product_name_am"] = getattr(product, "product_name_am", None)
                data["product_name_am_latin"] = getattr(product, "product_name_am_latin", None)
                category = getattr(product, "category", None)
                data["product_category"] = getattr(category, "value", category)
            expiry_date = getattr(model, "expiry_date", None)
            if expiry_date is not None:
                is_expired = expiry_date < date.today()
            else:
                is_expired = False
            data["is_expired"] = is_expired
            status_value = data.get("status")
            if is_expired and status_value != SupplierProductStatus.EXPIRED.value:
                data["effective_status"] = SupplierProductStatus.EXPIRED.value
            else:
                data["effective_status"] = status_value

        return data

        return data

    @staticmethod
    def _normalise_value(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (UUID, Decimal)):
            return str(value)
        if hasattr(value, "isoformat"):
            try:
                return value.isoformat()
            except Exception:  # pragma: no cover - defensive
                pass
        if hasattr(value, "value"):
            return value.value
        if isinstance(value, Model):
            return getattr(value, value._meta.pk_attr, None)
        if isinstance(value, QuerySet):
            return []
        if isinstance(value, (list, tuple, set)):
            return [DataAccessTool._normalise_value(item) for item in value]
        return str(value)

    async def _create_flash_sale(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a manual flash sale entry for a supplier product."""
        if not data:
            return {"error": "Data payload is required for flash sale creation."}

        supplier_product_id = data.get("supplier_product_id") or data.get("inventory_id")
        discount_percent = data.get("discount_percent")
        start_date_raw = data.get("start_date")
        end_date_raw = data.get("end_date")
        status_raw = data.get("status")

        if not supplier_product_id:
            return {"error": "supplier_product_id is required to create a flash sale."}
        if discount_percent is None:
            return {"error": "discount_percent is required to create a flash sale."}

        supplier_product = await SupplierProductRepository.get_supplier_product_by_id(supplier_product_id)
        if not supplier_product:
            return {"error": f"Supplier product {supplier_product_id} not found."}
        await supplier_product.fetch_related("supplier", "product")

        start_date = self._parse_datetime(start_date_raw) or datetime.utcnow()
        end_date = self._parse_datetime(end_date_raw) or (start_date + timedelta(hours=6))
        status = self._normalise_flash_status(status_raw)

        flash_sale = await FlashSaleRepository.create_flash_sale(
            supplier_product=supplier_product,
            supplier=supplier_product.supplier,
            product=supplier_product.product,
            start_date=start_date,
            end_date=end_date,
            discount_percent=float(discount_percent),
            status=status,
            auto_generated=False,
        )

        return {
            "message": "Flash sale created successfully.",
            "record": await self._serialize_model(flash_sale),
        }

    async def _create_supplier_product(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create supplier inventory entries for onboarding flows."""
        if not data:
            return {"error": "Data payload is required for supplier product creation."}

        supplier_id = data.get("supplier_id") or data.get("supplier")
        product_id = data.get("product_id") or data.get("product")
        product_name = data.get("product_name") or data.get("product_label")
        quantity_raw = (
            data.get("quantity_available")
            or data.get("quantity")
            or data.get("available_quantity")
        )
        price_raw = data.get("unit_price") or data.get("price_per_unit") or data.get("price")
        unit_raw = data.get("unit")
        status_raw = data.get("status")
        expiry_raw = data.get("expiry_date")
        delivery_days = data.get("available_delivery_days") or data.get("delivery_days")

        if not supplier_id:
            return {"error": "supplier_id is required to add inventory."}

        supplier = await UserRepository.get_user_by_id(supplier_id)
        if not supplier or supplier.role.value != UserRole.SUPPLIER.value:
            return {"error": f"Supplier {supplier_id} not found or not a supplier account."}

        if not expiry_raw:
            return {"error": "expiry_date is required for perishable inventory."}

        if delivery_days is None:
            return {"error": "available_delivery_days is required for supplier inventory."}

        price_val = self._parse_float(price_raw)
        if price_val is None:
            return {"error": "unit price must be a numeric value."}
        if price_val <= 0:
            return {"error": "unit price must be greater than zero."}

        unit = self._normalise_unit(unit_raw) or UnitType.KG

        product = await self._resolve_product_reference(
            product_id,
            product_name,
            price_hint=price_val,
            unit_hint=unit,
            category_hint=data.get("category"),
        )
        if not product:
            return {
                "error": "Product could not be resolved for supplier inventory.",
                "hint": "Ensure the product exists or provide a valid product_id/product_name.",
            }

        quantity_val = self._parse_float(quantity_raw)
        if quantity_val is None:
            return {"error": "quantity must be a numeric value."}
        if quantity_val <= 0:
            return {"error": "quantity must be greater than zero."}

        status = self._normalise_supplier_product_status(status_raw) or SupplierProductStatus.ACTIVE
        expiry_date = self._parse_date(expiry_raw)
        if expiry_date is None:
            return {"error": "expiry_date must be a valid ISO date (YYYY-MM-DD)."}

        delivery_days_norm = self._normalise_delivery_days(delivery_days)
        if delivery_days_norm is None:
            return {
                "error": "available_delivery_days could not be understood.",
                "hint": "Examples: 'Daily', 'Mon,Wed,Fri', 'Weekends', 'all week'.",
            }

        payload = {
            "supplier_id": supplier.user_id,
            "product_id": product.product_id,
            "quantity_available": quantity_val,
            "unit": unit,
            "unit_price_etb": price_val,
            "status": status,
            "available_delivery_days": delivery_days_norm,
            "expiry_date": expiry_date,
        }

        pricing_guidance: Optional[Dict[str, Any]] = None
        try:
            await ensure_db_initialized()
            analytics_tool = AnalyticsDataTool()
            pricing_guidance = await analytics_tool._get_price_guidance(
                str(product.product_id),
                product.product_name_en,
            )
        except Exception as guidance_exc:  # pragma: no cover - defensive
            pricing_guidance = {
                "error": "pricing_guidance_unavailable",
                "details": str(guidance_exc),
            }

        supplier_product = await SupplierProductRepository.create_supplier_product(**payload)
        result: Dict[str, Any] = {
            "message": "Supplier product created successfully.",
            "record": await self._serialize_model(supplier_product),
        }
        if pricing_guidance:
            result["pricing_guidance"] = pricing_guidance
        if not pricing_guidance or pricing_guidance.get("error"):
            result.setdefault("notes", []).append(
                "Pricing guidance could not be generated prior to creation."
            )
        return result

    async def _create_entity(self, repository, entity: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch create requests with entity-specific normalisation."""
        if not data:
            return {"error": "Data payload is required for 'create' operation."}

        if entity == "supplier_products":
            return await self._create_supplier_product(data)
        if entity == "products":
            return await self._create_product(data)
        if entity == "flash_sales":
            return await self._create_flash_sale(data)

        if entity == "users":
            payload = dict(data)
            if "preferred_language" in payload:
                payload["preferred_language"] = self._normalise_preferred_language(payload["preferred_language"])
            if "role" in payload:
                payload["role"] = self._normalise_role(payload["role"])
            if "joined_date" in payload:
                joined_date = self._parse_date(payload["joined_date"])
                if joined_date is None:
                    return {"error": "joined_date must be a valid ISO date."}
                payload["joined_date"] = joined_date
            data = payload

        method = self._get_repo_method(entity, "create")
        if method is None:
            return {"error": f"'create' operation is not available for entity: {entity}."}

        try:
            instance = await method(**data)
        except IntegrityError as exc:
            return {"error": f"Create failed due to integrity error: {exc}"}

        return {
            "message": "Record created successfully.",
            "record": await self._serialize_model(instance),
        }

    async def _create_product(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a product with sensible defaults when minimal data is provided."""
        if not data:
            return {"error": "Data payload is required for product creation."}

        name_en = data.get("product_name_en") or data.get("name")
        if not name_en:
            return {"error": "product_name_en is required to create a product."}
        name_en = name_en.strip()

        base_price_raw = (
            data.get("base_price_etb")
            or data.get("base_price")
            or data.get("price")
            or data.get("unit_price")
        )
        base_price = self._parse_float(base_price_raw) or 0.0
        if base_price <= 0:
            return {"error": "base_price_etb must be greater than zero."}

        category = self._normalise_product_category(data.get("category"))
        if category is None:
            category = self._infer_product_category(name_en)

        unit = self._normalise_unit(data.get("unit")) or UnitType.KG
        in_season_start = self._normalise_month(data.get("in_season_start")) or Month.JANUARY
        in_season_end = self._normalise_month(data.get("in_season_end")) or Month.DECEMBER

        payload = {
            "product_name_en": name_en,
            "product_name_am": data.get("product_name_am") or name_en,
            "product_name_am_latin": data.get("product_name_am_latin") or name_en,
            "category": category,
            "unit": unit,
            "base_price_etb": base_price,
            "in_season_start": in_season_start,
            "in_season_end": in_season_end,
            "image_url": data.get("image_url"),
        }

        product = await ProductRepository.create_product(**payload)
        return {
            "message": "Product created successfully.",
            "record": await self._serialize_model(product),
        }

    async def _update_entity(self, entity: str, entity_id: Any, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update records across repositories with light normalisation."""
        if not entity_id:
            return {"error": "ID is required for 'update' operation."}
        if not data:
            return {"error": "Data payload is required for 'update' operation."}

        update_payload = dict(data)

        if entity == "users":
            if "preferred_language" in update_payload:
                update_payload["preferred_language"] = self._normalise_preferred_language(update_payload["preferred_language"])
            if "role" in update_payload:
                update_payload["role"] = self._normalise_role(update_payload["role"])
            if "joined_date" in update_payload:
                joined_date = self._parse_date(update_payload["joined_date"])
                if joined_date is None:
                    return {"error": "joined_date must be a valid ISO date."}
                update_payload["joined_date"] = joined_date

        elif entity == "supplier_products":
            normalised_payload: Dict[str, Any] = {}
            if any(key in update_payload for key in ("quantity", "quantity_available")):
                quantity_val = self._parse_float(update_payload.get("quantity_available") or update_payload.get("quantity"))
                if quantity_val is None or quantity_val < 0:
                    return {"error": "quantity must be a non-negative number."}
                normalised_payload["quantity_available"] = quantity_val
            if any(key in update_payload for key in ("unit_price", "price_per_unit", "price", "unit_price_etb")):
                price_val = self._parse_float(
                    update_payload.get("unit_price_etb")
                    or update_payload.get("unit_price")
                    or update_payload.get("price_per_unit")
                    or update_payload.get("price")
                )
                if price_val is None or price_val < 0:
                    return {"error": "unit price must be a non-negative number."}
                normalised_payload["unit_price_etb"] = price_val
            if "unit" in update_payload:
                unit = self._normalise_unit(update_payload["unit"])
                if unit is None:
                    return {"error": "unit is not recognised."}
                normalised_payload["unit"] = unit
            if "status" in update_payload:
                status = self._normalise_supplier_product_status(update_payload["status"])
                if status is None:
                    return {"error": "status is not recognised."}
                normalised_payload["status"] = status
            if any(key in update_payload for key in ("product_id", "product", "product_name")):
                product = None
                if update_payload.get("product_id"):
                    product = await ProductRepository.get_product_by_id(update_payload["product_id"])
                else:
                    product = await ProductRepository.get_product_by_name(update_payload.get("product_name") or update_payload.get("product"))
                if not product:
                    return {"error": "Product not found for update."}
                normalised_payload["product_id"] = product.product_id
            if "expiry_date" in update_payload:
                expiry_date = self._parse_date(update_payload["expiry_date"])
                if expiry_date is None:
                    return {"error": "expiry_date must be a valid ISO date."}
                normalised_payload["expiry_date"] = expiry_date
            if "available_delivery_days" in update_payload or "delivery_days" in update_payload:
                delivery_days = update_payload.get("available_delivery_days") or update_payload.get("delivery_days")
                delivery_days = self._normalise_delivery_days(delivery_days)
                if delivery_days is None:
                    return {
                        "error": "available_delivery_days could not be understood for update.",
                        "hint": "Examples: 'Daily', 'Mon,Wed,Fri', 'Weekends', 'all week'.",
                    }
                normalised_payload["available_delivery_days"] = delivery_days
            update_payload = normalised_payload or {}
            if not update_payload:
                return {"error": "No valid supplier product fields supplied for update."}

        elif entity == "flash_sales":
            normalised_payload = {}
            if "status" in update_payload:
                status = self._normalise_flash_status(update_payload["status"])
                if status is None:
                    return {"error": "status is not recognised for flash sale."}
                normalised_payload["status"] = status
            for field in ("discount_percent", "discount", "discount_rate"):
                if field in update_payload:
                    discount_value = self._parse_float(update_payload[field])
                    if discount_value is None or discount_value < 0:
                        return {"error": "discount must be a non-negative number."}
                    normalised_payload["discount_percent"] = discount_value
                    break
            if "start_date" in update_payload:
                start_date = self._parse_datetime(update_payload["start_date"])
                if start_date is None:
                    return {"error": "start_date must be a valid ISO datetime."}
                normalised_payload["start_date"] = start_date
            if "end_date" in update_payload:
                end_date = self._parse_datetime(update_payload["end_date"])
                if end_date is None:
                    return {"error": "end_date must be a valid ISO datetime."}
                normalised_payload["end_date"] = end_date
            if "auto_generated" in update_payload:
                normalised_payload["auto_generated"] = bool(update_payload["auto_generated"])
            update_payload = normalised_payload or {}
            if not update_payload:
                return {"error": "No valid flash sale fields supplied for update."}

        method = self._get_repo_method(entity, 'update')
        if method is None:
            return {"error": f"'update' operation is not available for entity: {entity}."}

        instance = await method(entity_id, **update_payload)
        if not instance:
            return {"error": f"{entity} with ID {entity_id} not found."}

        return {
            "message": "Record updated successfully.",
            "record": await self._serialize_model(instance),
        }

    async def _delete_entity(self, entity: str, entity_id: Any) -> Dict[str, Any]:
        """Delete records for supported entities."""
        if not entity_id:
            return {"error": "ID is required for 'delete' operation."}

        method = self._get_repo_method(entity, 'delete')
        if method is None:
            return {"error": f"'delete' operation is not available for entity: {entity}."}

        deleted = await method(entity_id)
        if not deleted:
            return {"error": f"{entity} with ID {entity_id} not found."}

        return {"message": f"{entity} record deleted successfully."}

    def _get_repo_method(self, entity: str, operation: str):
        methods = self._repository_methods.get(entity, {})
        method_name = methods.get(operation)
        if not method_name:
            return None
        repository = self.repositories.get(entity)
        if repository and hasattr(repository, method_name):
            return getattr(repository, method_name)
        return None

    @staticmethod
    def _normalise_preferred_language(value: Any) -> PreferredLanguage:
        if isinstance(value, PreferredLanguage):
            return value
        if isinstance(value, str):
            normalised = value.strip().lower()
            for option in PreferredLanguage:
                if option.value.lower() == normalised:
                    return option
        return PreferredLanguage.ENGLISH

    @staticmethod
    def _normalise_role(value: Any) -> UserRole:
        if isinstance(value, UserRole):
            return value
        if isinstance(value, str):
            normalised = value.strip().lower()
            for option in UserRole:
                if option.value == normalised:
                    return option
        return UserRole.SUPPLIER

    @staticmethod
    def _parse_date(value: Any) -> Optional[date]:
        if value is None:
            return None
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                return date.fromisoformat(value.split("T")[0])
            except ValueError:
                return None
        return None

    @staticmethod
    def _parse_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalise_unit(value: Any) -> Optional[UnitType]:
        if value is None:
            return None
        if isinstance(value, UnitType):
            return value
        if isinstance(value, str):
            normalised = value.strip().lower()
            for unit in UnitType:
                if unit.value.lower() == normalised:
                    return unit
        return None

    @staticmethod
    def _normalise_supplier_product_status(value: Any) -> Optional[SupplierProductStatus]:
        if value is None:
            return None
        if isinstance(value, SupplierProductStatus):
            return value
        if isinstance(value, str):
            normalised = value.strip().lower()
            for status in SupplierProductStatus:
                if status.value == normalised:
                    return status
        return None

    @staticmethod
    def _normalise_product_category(value: Any) -> Optional[ProductCategory]:
        if value is None:
            return None
        if isinstance(value, ProductCategory):
            return value
        if isinstance(value, str):
            normalised = value.strip().lower()
            for category in ProductCategory:
                if category.value.lower() == normalised:
                    return category
        return None

    @staticmethod
    def _normalise_month(value: Any) -> Optional[Month]:
        if value is None:
            return None
        if isinstance(value, Month):
            return value
        if isinstance(value, str):
            normalised = value.strip().lower()
            for month in Month:
                if month.value.lower() == normalised:
                    return month
        return None

    @staticmethod
    def _normalise_delivery_days(value: Any) -> Optional[str]:
        if value is None:
            return None

        if isinstance(value, (list, tuple, set)):
            joined = ",".join(str(item) for item in value if item)
            if not joined:
                return None
            value = joined

        if isinstance(value, str):
            parsed = parse_delivery_schedule(value)
            if parsed.normalized_days:
                return parsed.normalized_days
            if (
                parsed.schedule_type.startswith("relative")
                and parsed.start_date is not None
                and parsed.end_date is not None
            ):
                return (
                    f"{parsed.schedule_type}:{parsed.start_date.isoformat()}-"
                    f"{parsed.end_date.isoformat()}"
                )
        return None

    def _infer_product_category(self, name: str) -> ProductCategory:
        lookup = name.strip().lower()
        vegetable_keywords = {"tomato", "onion", "carrot", "cabbage", "lettuce", "spinach", "kale", "pepper", "potato"}
        fruit_keywords = {"banana", "apple", "mango", "orange", "avocado", "strawberry", "grape", "pineapple"}
        dairy_keywords = {"milk", "butter", "cheese", "yogurt"}
        if any(word in lookup for word in vegetable_keywords):
            return ProductCategory.VEGETABLE
        if any(word in lookup for word in fruit_keywords):
            return ProductCategory.FRUIT
        if any(word in lookup for word in dairy_keywords):
            return ProductCategory.DAIRY
        return ProductCategory.VEGETABLE

    async def _resolve_product_reference(
        self,
        product_id: Optional[str],
        product_name: Optional[str],
        *,
        price_hint: Optional[float] = None,
        unit_hint: Optional[UnitType] = None,
        category_hint: Optional[str] = None,
    ):
        if product_id:
            product = await ProductRepository.get_product_by_id(product_id)
            if product:
                return product

        name = (product_name or "").strip()
        if not name:
            return None

        product = await ProductRepository.get_product_by_name(name)
        if product:
            return product

        if price_hint is None:
            return None

        category = self._normalise_product_category(category_hint)
        if category is None:
            category = self._infer_product_category(name)

        unit = unit_hint or UnitType.KG

        payload = {
            "product_name_en": name,
            "product_name_am": name,
            "product_name_am_latin": name,
            "category": category,
            "unit": unit,
            "base_price_etb": price_hint,
            "in_season_start": Month.JANUARY,
            "in_season_end": Month.DECEMBER,
        }

        return await ProductRepository.create_product(**payload)

    @staticmethod
    def _parse_datetime(value: Any) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return None
        return None

    @staticmethod
    def _normalise_flash_status(value: Any) -> Optional[FlashSaleStatus]:
        if value is None:
            return None
        if isinstance(value, FlashSaleStatus):
            return value
        if isinstance(value, str):
            normalised = value.strip().lower()
            for status in FlashSaleStatus:
                if status.value == normalised:
                    return status
        return None


class AnalyticsDataTool(ToolBase):
    """
    Tool for AI agent to perform analytical queries on the data.
    Provides aggregated insights, statistics, and cross-entity analysis.
    """

    def __init__(self):
        super().__init__(
            name="analytics_data",
            description=(
                "Perform analytical queries and get insights from the database. "
                "Supports operations: 'product_stats', 'user_stats', 'transaction_stats', "
                "'price_comparison', 'pricing_guidance', 'supplier_inventory'. "
                "Example input: {'operation': 'product_stats', 'product_id': 'uuid-here'}"
            )
        )

    async def run(self, input: Dict[str, Any], context: Dict[str, Any] = None) -> Any:
        """Execute validated analytics operations."""
        await ensure_db_initialized()
        try:
            if isinstance(input, str):
                try:
                    input = json.loads(input)
                except json.JSONDecodeError:
                    return {"error": "Invalid JSON payload for analytics_data tool."}

            request = AnalyticsRequest.model_validate(input)
        except ValidationError as exc:
            return {
                "error": "Invalid analytics_data request.",
                "details": exc.errors(),
            }
        except Exception as exc:  # pragma: no cover - defensive logging
            return {"error": f"Unexpected validation error: {exc}"}

        operation = request.operation
        try:
            if operation == 'product_stats':
                return await self._get_product_stats(request.product_id)
            if operation == 'user_stats':
                return await self._get_user_stats(request.user_id)
            if operation == 'transaction_stats':
                return await self._get_transaction_stats(request.filters)
            if operation == 'price_comparison':
                return await self._get_price_comparison(request.product_id, request.product_name)
            if operation == 'pricing_guidance':
                return await self._get_price_guidance(request.product_id, request.product_name)
            if operation == 'supplier_inventory':
                return await self._get_supplier_inventory(request.supplier_id, request.supplier_name)
            return {
                "error": f"Unknown operation: {operation}.",
                "valid_operations": [
                    'product_stats',
                    'user_stats',
                    'transaction_stats',
                    'price_comparison',
                    'pricing_guidance',
                    'supplier_inventory',
                ],
            }
        except Exception as exc:  # pragma: no cover - repository errors
            return {"error": f"Error performing analytics: {exc}"}

    async def _get_product_stats(self, product_id: str) -> Dict[str, Any]:
        """Get statistics for a specific product."""
        if not product_id:
            return {"error": "product_id is required"}
        
        # Get product details
        product = await ProductRepository.get_product_by_id(product_id)
        if not product:
            return {"error": f"Product {product_id} not found"}
        
        # Get supplier products for this product
        supplier_products = await SupplierProductRepository.list_supplier_products(
            {'product_id': product_id}
        )
        
        # Get competitor prices
        competitor_prices = await CompetitorPriceRepository.list_competitor_prices(
            {'product_id': product_id}
        )
        
        # Calculate stats
        total_available = sum(sp.quantity_available for sp in supplier_products)
        avg_supplier_price = (
            sum(sp.unit_price_etb for sp in supplier_products) / len(supplier_products)
            if supplier_products else 0
        )
        avg_competitor_price = (
            sum(cp.price_etb_per_kg for cp in competitor_prices) / len(competitor_prices)
            if competitor_prices else 0
        )
        
        return {
            "product_id": str(product_id),
            "product_name": product.product_name_en,
            "category": product.category.value,
            "base_price": product.base_price_etb,
            "suppliers_count": len(supplier_products),
            "total_available_quantity": total_available,
            "average_supplier_price": round(avg_supplier_price, 2),
            "average_competitor_price": round(avg_competitor_price, 2),
            "competitor_prices_count": len(competitor_prices)
        }

    async def _get_user_stats(self, user_id: int) -> Dict[str, Any]:
        """Get statistics for a specific user."""
        if not user_id:
            return {"error": "user_id is required"}
        
        user = await UserRepository.get_user_by_id(user_id)
        if not user:
            return {"error": f"User {user_id} not found"}
        
        # Get transactions for this user
        transactions = await TransactionRepository.list_transactions(
            {'user_id': user_id}
        )
        
        # Calculate stats
        total_orders = len(transactions)
        total_spent = sum(t.total_price for t in transactions)
        avg_order_value = total_spent / total_orders if total_orders > 0 else 0
        
        # Count by status
        status_counts = {}
        for t in transactions:
            status = t.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "user_id": user_id,
            "name": user.name,
            "role": user.role.value,
            "total_orders": total_orders,
            "total_spent": round(total_spent, 2),
            "average_order_value": round(avg_order_value, 2),
            "orders_by_status": status_counts
        }

    async def _get_transaction_stats(self, filters: Dict) -> Dict[str, Any]:
        """Get aggregated transaction statistics."""
        transactions = await TransactionRepository.list_transactions(filters)
        
        if not transactions:
            return {"message": "No transactions found matching the filters"}
        
        total_transactions = len(transactions)
        total_revenue = sum(t.total_price for t in transactions)
        avg_transaction_value = total_revenue / total_transactions
        
        # Group by status
        by_status = {}
        for t in transactions:
            status = t.status.value
            if status not in by_status:
                by_status[status] = {"count": 0, "revenue": 0}
            by_status[status]["count"] += 1
            by_status[status]["revenue"] += t.total_price
        
        return {
            "total_transactions": total_transactions,
            "total_revenue": round(total_revenue, 2),
            "average_transaction_value": round(avg_transaction_value, 2),
            "breakdown_by_status": by_status,
            "filters_applied": filters
        }

    async def _get_price_comparison(self, product_id: Optional[str], product_name: Optional[str]) -> Dict[str, Any]:
        """Compare prices across suppliers and competitors."""
        product = await self._resolve_product(product_id, product_name)
        if not product:
            identifier = product_id or product_name or "(unknown product)"
            return {"error": f"Product {identifier} not found"}

        target_product_id = product.product_id

        supplier_products = await SupplierProductRepository.list_supplier_products({'product_id': target_product_id})

        competitor_prices = await CompetitorPriceRepository.list_competitor_prices({'product_id': target_product_id})

        supplier_prices = [
            {
                "supplier_id": sp.supplier_id,
                "price": sp.unit_price_etb,
                "quantity_available": sp.quantity_available,
                "status": sp.status.value,
            }
            for sp in supplier_products
        ]

        competitor_price_data: List[Dict[str, Any]] = []
        tier_prices: Dict[str, List[float]] = {tier.value: [] for tier in CompetitorTier}
        tier_entries: Dict[str, List[Dict[str, Any]]] = {tier.value: [] for tier in CompetitorTier}

        for cp in competitor_prices:
            tier_value = cp.tier.value if hasattr(cp.tier, "value") else str(cp.tier)
            entry = {
                "tier": tier_value,
                "tier_label": self._format_competitor_tier_label(tier_value),
                "price": cp.price_etb_per_kg,
                "location": cp.source_location,
                "date": cp.date.isoformat(),
            }
            competitor_price_data.append(entry)
            tier_prices.setdefault(tier_value, []).append(cp.price_etb_per_kg)
            tier_entries.setdefault(tier_value, []).append(entry)

        competitor_prices_by_tier: List[Dict[str, Any]] = []
        for tier in CompetitorTier:
            tier_value = tier.value
            prices = tier_prices.get(tier_value, [])
            competitor_prices_by_tier.append(
                {
                    "tier": tier_value,
                    "tier_label": self._format_competitor_tier_label(tier_value),
                    "count": len(prices),
                    "average_price": round(sum(prices) / len(prices), 2) if prices else None,
                    "lowest_price": round(min(prices), 2) if prices else None,
                    "highest_price": round(max(prices), 2) if prices else None,
                    "prices": tier_entries.get(tier_value, []),
                }
            )

        return {
            "product_id": str(target_product_id),
            "product_name": product.product_name_en,
            "base_price": product.base_price_etb,
            "supplier_prices": supplier_prices,
            "competitor_prices": competitor_price_data,
            "competitor_prices_by_tier": competitor_prices_by_tier,
            "lowest_supplier_price": min((sp.unit_price_etb for sp in supplier_products), default=None),
            "highest_supplier_price": max((sp.unit_price_etb for sp in supplier_products), default=None),
            "lowest_competitor_price": min((cp.price_etb_per_kg for cp in competitor_prices), default=None),
            "highest_competitor_price": max((cp.price_etb_per_kg for cp in competitor_prices), default=None)
        }

    async def _get_price_guidance(self, product_id: Optional[str], product_name: Optional[str]) -> Dict[str, Any]:
        """Provide pricing guidance based on supplier and competitor data."""
        product = await self._resolve_product(product_id, product_name)
        if not product:
            identifier = product_id or product_name or "(unknown product)"
            return {"error": f"Product {identifier} not found"}

        supplier_products = await SupplierProductRepository.list_supplier_products({'product_id': product.product_id})
        competitor_prices = await CompetitorPriceRepository.list_competitor_prices({'product_id': product.product_id})

        supplier_price_values = [sp.unit_price_etb for sp in supplier_products]
        competitor_price_values = [cp.price_etb_per_kg for cp in competitor_prices]

        def _avg(values: List[float]) -> Optional[float]:
            return sum(values) / len(values) if values else None

        supplier_stats = {
            "count": len(supplier_products),
            "average": round(_avg(supplier_price_values), 2) if supplier_price_values else None,
            "lowest": round(min(supplier_price_values), 2) if supplier_price_values else None,
            "highest": round(max(supplier_price_values), 2) if supplier_price_values else None,
        }
        competitor_stats = {
            "count": len(competitor_prices),
            "average": round(_avg(competitor_price_values), 2) if competitor_price_values else None,
            "lowest": round(min(competitor_price_values), 2) if competitor_price_values else None,
            "highest": round(max(competitor_price_values), 2) if competitor_price_values else None,
        }

        recommended_price: Optional[float] = None
        rationale_parts: List[str] = []

        if competitor_stats["average"] is not None:
            recommended_price = competitor_stats["average"]
            rationale_parts.append(
                f"Aligning with the average competitor price of {competitor_stats['average']} ETB/kg"
            )
        elif supplier_stats["average"] is not None:
            recommended_price = supplier_stats["average"]
            rationale_parts.append(
                f"Matching the current supplier average of {supplier_stats['average']} ETB/kg"
            )
        else:
            base_price = product.base_price_etb or 0
            recommended_price = base_price if base_price > 0 else None
            if recommended_price is not None:
                rationale_parts.append("Falling back to catalog base price data")

        low_anchor = competitor_stats["lowest"] or supplier_stats["lowest"]
        high_anchor = competitor_stats["highest"] or supplier_stats["highest"]

        if recommended_price is not None:
            recommended_price = round(recommended_price, 2)

        price_band = None
        if low_anchor is not None or high_anchor is not None:
            floor = low_anchor if low_anchor is not None else high_anchor
            ceiling = high_anchor if high_anchor is not None else floor
            price_band = {
                "floor": round(floor, 2) if isinstance(floor, (int, float)) else floor,
                "ceiling": round(ceiling, 2) if isinstance(ceiling, (int, float)) else ceiling,
            }
        elif recommended_price is not None:
            price_band = {
                "floor": recommended_price,
                "ceiling": recommended_price,
            }

        rationale = ", ".join(rationale_parts) if rationale_parts else "Limited market data available"

        confidence = "low"
        if competitor_stats["count"] and supplier_stats["count"]:
            confidence = "high"
        elif competitor_stats["count"] or supplier_stats["count"]:
            confidence = "medium"

        data_sources = []
        if supplier_stats["count"]:
            data_sources.append("supplier_inventory")
        if competitor_stats["count"]:
            data_sources.append("competitor_prices")
        if not data_sources:
            data_sources.append("catalog_base_price")

        return {
            "product_id": str(product.product_id),
            "product_name": product.product_name_en,
            "unit": product.unit.value,
            "recommended_price": recommended_price,
            "price_band": price_band,
            "currency": "ETB",
            "supplier_stats": supplier_stats,
            "competitor_stats": competitor_stats,
            "base_price": product.base_price_etb,
            "rationale": rationale,
            "confidence": confidence,
            "data_sources": data_sources,
        }

    async def _resolve_product(self, product_id: Optional[str], product_name: Optional[str]):
        if product_id:
            product = await ProductRepository.get_product_by_id(product_id)
            if product:
                return product

        resolved = await ProductRepository.find_product_by_any_name(product_name)
        if resolved:
            return resolved

        return None

    @staticmethod
    def _format_competitor_tier_label(tier_value: str) -> str:
        if not tier_value:
            return ""
        return tier_value.replace("_", " ").title()

    async def _get_supplier_inventory(self, supplier_id: Optional[int], supplier_name: Optional[str]) -> Dict[str, Any]:
        """Get inventory details for a specific supplier."""
        supplier = None

        if supplier_id:
            supplier = await UserRepository.get_user_by_id(supplier_id)
        elif supplier_name:
            supplier = await UserRepository.get_user_by_name(supplier_name)

        if not supplier:
            identifier = supplier_id if supplier_id is not None else supplier_name or "(unknown)"
            return {"error": f"Supplier {identifier} not found"}

        resolved_supplier_id = supplier.user_id

        if supplier.role.value != 'supplier':
            return {"error": f"User {resolved_supplier_id} is not a supplier"}
        
        # Get all supplier products
        supplier_products = await SupplierProductRepository.list_supplier_products(
            {'supplier_id': resolved_supplier_id}
        )
        
        total_products = len(supplier_products)
        total_quantity = sum(sp.quantity_available for sp in supplier_products)
        total_value = sum(sp.quantity_available * sp.unit_price_etb for sp in supplier_products)
        
        # Group by status
        by_status = {}
        current_date = date.today()
        for sp in supplier_products:
            effective_status = sp.status.value
            if sp.expiry_date and sp.expiry_date < current_date:
                effective_status = SupplierProductStatus.EXPIRED.value
            if effective_status not in by_status:
                by_status[effective_status] = {"count": 0, "quantity": 0}
            by_status[effective_status]["count"] += 1
            by_status[effective_status]["quantity"] += sp.quantity_available
        
        inventory_items = []
        for sp in supplier_products:
            product = getattr(sp, "product", None)
            product_name_en = getattr(product, "product_name_en", None) if product else None
            product_name_am = getattr(product, "product_name_am", None) if product else None
            product_name_am_latin = getattr(product, "product_name_am_latin", None) if product else None
            category = getattr(product, "category", None)
            expiry_date = sp.expiry_date
            is_expired = bool(expiry_date and expiry_date < current_date)
            effective_status = sp.status.value
            if is_expired and effective_status != SupplierProductStatus.EXPIRED.value:
                effective_status = SupplierProductStatus.EXPIRED.value
            inventory_items.append(
                {
                    "inventory_id": str(sp.inventory_id),
                    "product_id": str(sp.product_id),
                    "product_name": product_name_en,
                    "product_name_en": product_name_en,
                    "product_name_am": product_name_am,
                    "product_name_am_latin": product_name_am_latin,
                    "product_category": getattr(category, "value", category),
                    "quantity_available": sp.quantity_available,
                    "unit": sp.unit.value,
                    "unit_price": sp.unit_price_etb,
                    "status": sp.status.value,
                    "effective_status": effective_status,
                    "is_expired": is_expired,
                    "expiry_date": expiry_date.isoformat() if expiry_date else None,
                }
            )
        
        return {
            "supplier_id": resolved_supplier_id,
            "supplier_name": supplier.name,
            "total_products": total_products,
            "total_quantity": total_quantity,
            "total_inventory_value": round(total_value, 2),
            "breakdown_by_status": by_status,
            "inventory": inventory_items
        }
