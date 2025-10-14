from typing import Any, Dict, List, Optional, Tuple
import json
import asyncio
import calendar
import re
from datetime import datetime, timedelta, date
from decimal import Decimal, InvalidOperation
import uuid
from uuid import UUID

from tortoise import Tortoise
from tortoise.exceptions import IntegrityError, ConfigurationError
from tortoise.models import Model
from tortoise.queryset import QuerySet
from tortoise.transactions import in_transaction
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
    PaymentMethod,
    TransactionStatus,
    OrderItem,
)
from app.db.repository.user_repository import UserRepository
from app.db.repository.product_repository import ProductRepository
from app.db.repository.supplier_product_repository import SupplierProductRepository
from app.db.repository.competitor_price_repository import CompetitorPriceRepository
from app.db.repository.transaction_repository import TransactionRepository
from app.db.repository.order_item_repository import OrderItemRepository
from app.db.repository.flash_sale_repository import FlashSaleRepository
from app.tools.schedule_tool import ScheduleTool
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
        self._schedule_tool = ScheduleTool()
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
        self._active_context: Dict[str, Any] = {}

    async def run(self, input: Dict[str, Any], context: Dict[str, Any] = None) -> Any:
        """Execute validated data access operations and surface friendly errors."""
        self._active_context = context or {}
        try:
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
                    if (
                        entity == "order_items"
                        and (
                            "delivery_date" in request.data
                            or "preferred_delivery_date" in request.data
                        )
                    ):
                        return await self._update_order_delivery(request)
                    return await self._update_entity(
                        entity,
                        request.id,
                        request.data,
                        request.filters,
                    )
                if operation == "delete":
                    return await self._delete_entity(entity, request.id)
                return {
                    "error": f"Unknown operation: {operation}.",
                    "valid_operations": self._supported_operations,
                }
            except Exception as exc:  # pragma: no cover - repository level errors
                return {"error": f"Error accessing data: {exc}"}
        finally:
            self._active_context = {}

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

        normalised_filters = await self._maybe_expand_special_filters(repository, filters or {})
        if normalised_filters is None:
            return {"count": 0, "results": []}

        # Call the list method
        list_method = getattr(repository, list_methods[0])
        entities = await list_method(normalised_filters)
        
        # Apply limit if specified
        if limit and isinstance(entities, list):
            entities = entities[:limit]
        
        # Serialize results
        results = []
        for entity_obj in entities:
            serialized = await self._serialize_model(entity_obj)
            if serialized is not None:
                results.append(serialized)

        return {
            "count": len(results),
            "results": results,
        }

    def _get_list_method(self, repository):
        list_methods = [m for m in dir(repository) if m.startswith("list_")]
        if not list_methods:
            return None
        return getattr(repository, list_methods[0])

    async def _fetch_repository_records(
        self,
        repository,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> List[Any]:
        list_method = self._get_list_method(repository)
        if not list_method:
            return []
        normalised_filters = await self._maybe_expand_special_filters(repository, filters or {})
        if normalised_filters is None:
            return []
        records = await list_method(normalised_filters)
        if isinstance(records, list) and limit is not None:
            return records[:limit]
        return records if isinstance(records, list) else []

    @staticmethod
    def _extract_pk(model: Any) -> Any:
        meta = getattr(model, "_meta", None)
        pk_attr = getattr(meta, "pk_attr", None) if meta else None
        if pk_attr and hasattr(model, pk_attr):
            return getattr(model, pk_attr)
        return getattr(model, "id", None)

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

        normalised_filters = await self._maybe_expand_special_filters(repository, filters)
        if normalised_filters is None:
            # Special filter expansion determined that no results are possible.
            return {"count": 0, "results": []}

        return await self._list_entities(repository, filters, limit)

    async def _maybe_expand_special_filters(
        self,
        repository,
        filters: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Handle repository-specific filter conveniences (e.g., product name lookups)."""
        if not filters:
            return {}

        entity_name = getattr(repository, "__name__", "")
        working = dict(filters)

        if entity_name == "ProductRepository":
            name_terms: List[str] = []
            for key in ("name", "product_name"):
                if key in working:
                    value = working.pop(key)
                    if isinstance(value, (list, tuple, set)):
                        name_terms.extend(str(item).strip() for item in value if str(item).strip())
                    elif isinstance(value, str) and value.strip():
                        name_terms.append(value.strip())
            if name_terms:
                product_ids: List[Any] = []
                seen: set[Any] = set()
                for term in name_terms:
                    product = await ProductRepository.find_product_by_any_name(term)
                    if product and product.product_id not in seen:
                        seen.add(product.product_id)
                        product_ids.append(product.product_id)
                if not product_ids:
                    return None
                working.setdefault("product_id", {"lookup": "in", "value": product_ids})
        
        elif entity_name == "SupplierProductRepository":
            name_terms: List[str] = []
            for key in ("product_name", "name"):
                if key in working:
                    value = working.pop(key)
                    if isinstance(value, (list, tuple, set)):
                        name_terms.extend(str(item).strip() for item in value if str(item).strip())
                    elif isinstance(value, str) and value.strip():
                        name_terms.append(value.strip())
            if name_terms:
                product_ids: List[Any] = []
                seen: set[Any] = set()
                for term in name_terms:
                    product = await ProductRepository.find_product_by_any_name(term)
                    if product and product.product_id not in seen:
                        seen.add(product.product_id)
                        product_ids.append(product.product_id)
                if not product_ids:
                    return None
                working.setdefault("product_id", {"lookup": "in", "value": product_ids})

            supplier_terms: List[str] = []
            for key in ("supplier_name", "supplier_label"):
                if key in working:
                    value = working.pop(key)
                    if isinstance(value, (list, tuple, set)):
                        supplier_terms.extend(str(item).strip() for item in value if str(item).strip())
                    elif isinstance(value, str) and value.strip():
                        supplier_terms.append(value.strip())

            raw_supplier = working.pop("supplier", None)
            if isinstance(raw_supplier, dict):
                working.setdefault("supplier_id", raw_supplier)
            elif isinstance(raw_supplier, (int, float)):
                working.setdefault("supplier_id", int(raw_supplier))
            elif isinstance(raw_supplier, str) and raw_supplier.strip():
                supplier_terms.append(raw_supplier.strip())
            elif isinstance(raw_supplier, (list, tuple, set)):
                supplier_terms.extend(str(item).strip() for item in raw_supplier if str(item).strip())

            if supplier_terms:
                supplier_ids: List[int] = []
                seen_suppliers: set[int] = set()
                for name in supplier_terms:
                    supplier = await UserRepository.get_user_by_name(name)
                    if supplier and getattr(supplier, "role", None) == UserRole.SUPPLIER:
                        supplier_id_val = int(supplier.user_id)
                        if supplier_id_val not in seen_suppliers:
                            seen_suppliers.add(supplier_id_val)
                            supplier_ids.append(supplier_id_val)
                if not supplier_ids:
                    return None
                if "supplier_id" in working and isinstance(working["supplier_id"], dict):
                    existing = working["supplier_id"].get("value")
                    if isinstance(existing, list):
                        merged = list({*existing, *supplier_ids})
                        working["supplier_id"]["value"] = merged
                    else:
                        working["supplier_id"] = {"lookup": working["supplier_id"].get("lookup", "in"), "value": supplier_ids}
                else:
                    working.setdefault("supplier_id", {"lookup": "in", "value": supplier_ids})

            delivery_override = working.pop("delivery_location", None)
            if delivery_override:
                if isinstance(delivery_override, dict):
                    working.setdefault("supplier__default_location", delivery_override)
                else:
                    working.setdefault(
                        "supplier__default_location",
                        {"lookup": "iexact", "value": str(delivery_override).strip()},
                    )

            quantity_filters: Dict[str, Any] = {}
            for key in list(working.keys()):
                if key in ("available_quantity", "quantity"):
                    quantity_filters[key] = working.pop(key)

            if quantity_filters:
                normalised_lookup: Dict[str, Any] = {}
                comparison_aliases = {
                    "gte": "gte",
                    "gt": "gt",
                    "lte": "lte",
                    "lt": "lt",
                    "eq": "exact",
                    "exact": "exact",
                }
                for _, value in quantity_filters.items():
                    lookup_name: Optional[str] = None
                    lookup_value: Any = None
                    if isinstance(value, dict):
                        lookup_name = value.get("lookup") or value.get("op")
                        lookup_value = value.get("value")
                        if not lookup_name and len(value) == 1:
                            single_key, single_val = next(iter(value.items()))
                            if single_key in comparison_aliases:
                                lookup_name = comparison_aliases[single_key]
                                lookup_value = single_val
                        if lookup_name and lookup_value is None and lookup_name in value:
                            lookup_value = value.get(lookup_name)
                    else:
                        lookup_value = value
                        lookup_name = "gte"
                    if lookup_name and lookup_value is not None:
                        normalised_lookup[f"quantity_available__{lookup_name}"] = lookup_value
                working.update(normalised_lookup)

        elif entity_name == "OrderItemRepository":
            order_reference = working.pop("order_reference", None)
            if order_reference is not None and "order_id" not in working:
                working["order_id"] = order_reference

            order_id_value = working.get("order_id")
            if isinstance(order_id_value, dict):
                values = order_id_value.get("value")
                if isinstance(values, (list, tuple, set)):
                    normalised_values: List[str] = []
                    for candidate in values:
                        try:
                            normalised_values.append(str(uuid.UUID(str(candidate).strip())))
                        except (ValueError, AttributeError):
                            continue
                    if not normalised_values:
                        return None
                    order_id_value = {"lookup": order_id_value.get("lookup", "in"), "value": normalised_values}
                    working["order_id"] = order_id_value
                else:
                    candidate = order_id_value.get("value")
                    if candidate is None and order_id_value.get("lookup") in order_id_value:
                        candidate = order_id_value.get(order_id_value.get("lookup"))
                    if candidate is None:
                        return None
                    try:
                        working["order_id"] = str(uuid.UUID(str(candidate).strip()))
                    except (ValueError, AttributeError):
                        return None
            elif order_id_value is not None:
                try:
                    working["order_id"] = str(uuid.UUID(str(order_id_value).strip()))
                except ValueError:
                    return None

            user_filter = working.pop("user_id", None)
            if user_filter is not None:
                working["order__user_id"] = user_filter

            status_filter = working.pop("status", None)
            if status_filter is not None:
                status_enum = self._normalise_transaction_status(status_filter)
                if status_enum is not None:
                    working["order__status"] = status_enum.value

            delivery_filter = None
            delivery_lookup = "exact"
            for key in ("delivery_date", "preferred_delivery_date"):
                if key in working:
                    raw_value = working.pop(key)
                    if isinstance(raw_value, dict):
                        delivery_lookup = raw_value.get("lookup") or raw_value.get("op") or "exact"
                        delivery_filter = raw_value.get("value")
                        if delivery_filter is None and delivery_lookup in raw_value:
                            delivery_filter = raw_value.get(delivery_lookup)
                    else:
                        delivery_filter = raw_value
                    break
            if delivery_filter is not None:
                parsed_date = self._parse_date(delivery_filter)
                if parsed_date is not None:
                    if delivery_lookup and delivery_lookup.lower() not in {"exact", "eq"}:
                        working["order__delivery_date"] = {"lookup": delivery_lookup, "value": parsed_date}
                    else:
                        working["order__delivery_date"] = parsed_date

            product_name_filter = working.pop("product_name", None)
            name_terms: List[str] = []
            if product_name_filter is not None:
                if isinstance(product_name_filter, (list, tuple, set)):
                    name_terms.extend(
                        str(item).strip() for item in product_name_filter if str(item).strip()
                    )
                elif isinstance(product_name_filter, str) and product_name_filter.strip():
                    name_terms.append(product_name_filter.strip())
            if name_terms:
                product_ids: List[Any] = []
                seen: set[Any] = set()
                for term in name_terms:
                    product = await ProductRepository.find_product_by_any_name(term)
                    if product and product.product_id not in seen:
                        seen.add(product.product_id)
                        product_ids.append(product.product_id)
                if not product_ids:
                    return None
                working.setdefault("product_id", {"lookup": "in", "value": product_ids})

        elif entity_name == "TransactionRepository":
            order_reference = working.pop("order_reference", None)
            if order_reference is not None and "order_id" not in working:
                working["order_id"] = order_reference

            order_id_value = working.get("order_id")
            if isinstance(order_id_value, dict):
                lookup = order_id_value.get("lookup") or order_id_value.get("op") or "in"
                values = order_id_value.get("value")
                if values is None and lookup in order_id_value:
                    values = order_id_value.get(lookup)
                if isinstance(values, (list, tuple, set)):
                    normalised_values: List[str] = []
                    for candidate in values:
                        try:
                            normalised_values.append(str(uuid.UUID(str(candidate).strip())))
                        except (ValueError, AttributeError):
                            continue
                    if not normalised_values:
                        return None
                    working["order_id"] = {"lookup": lookup, "value": normalised_values}
                elif values is not None:
                    try:
                        working["order_id"] = str(uuid.UUID(str(values).strip()))
                    except (ValueError, AttributeError):
                        return None
                else:
                    return None
            elif order_id_value is not None:
                try:
                    working["order_id"] = str(uuid.UUID(str(order_id_value).strip()))
                except ValueError:
                    return None

            status_filter = working.get("status")
            if status_filter is not None:
                status_enum = self._normalise_transaction_status(status_filter)
                if status_enum is not None:
                    working["status"] = status_enum.value
                else:
                    working.pop("status", None)

            preferred_delivery = working.pop("preferred_delivery_date", None)
            if "delivery_date" in working:
                existing = working["delivery_date"]
                delivery_lookup = "exact"
                delivery_value = existing
                if isinstance(existing, dict):
                    delivery_lookup = existing.get("lookup") or existing.get("op") or "exact"
                    delivery_value = existing.get("value")
                    if delivery_value is None and delivery_lookup in existing:
                        delivery_value = existing.get(delivery_lookup)
                parsed_delivery = self._parse_date(delivery_value)
                if parsed_delivery is not None:
                    if isinstance(existing, dict) and delivery_lookup.lower() not in {"exact", "eq"}:
                        working["delivery_date"] = {"lookup": delivery_lookup, "value": parsed_delivery}
                    else:
                        working["delivery_date"] = parsed_delivery
                else:
                    working.pop("delivery_date", None)
            if preferred_delivery is not None:
                parsed_preferred = self._parse_date(preferred_delivery)
                if parsed_preferred is not None:
                    working["delivery_date"] = parsed_preferred

        return self._normalise_filters(working)

    def _normalise_filters(self, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not filters:
            return {}
        prepared: Dict[str, Any] = {}
        for key, value in filters.items():
            if isinstance(value, dict):
                prepared[key] = value
                continue
            if isinstance(value, (list, tuple, set)):
                items = [item for item in value if item is not None]
                if not items:
                    continue
                prepared[key] = {"lookup": "in", "value": items}
                continue
            prepared[key] = value
        return prepared

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

        if isinstance(model, OrderItem):
            product = getattr(model, "product", None)
            if product is not None:
                data.setdefault("product_id", getattr(model, "product_id", None))
                data["product_name"] = getattr(product, "product_name_en", None)
                data.setdefault("unit", DataAccessTool._normalise_value(getattr(product, "unit", None)))
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

    async def _resolve_order_id_for_items(
        self,
        payload: Dict[str, Any],
    ) -> Tuple[Optional[Any], Optional[Dict[str, Any]], List[str]]:
        notes: List[str] = []

        explicit_source: Optional[str] = None
        explicit_order_id: Optional[UUID] = None
        explicit_raw: Optional[Any] = None
        explicit_invalid = False
        for key in ("order_id", "order_reference", "transaction_id"):
            candidate = payload.get(key)
            if candidate:
                explicit_source = key
                explicit_raw = candidate
                explicit_order_id = self._normalise_order_id(candidate)
                if explicit_order_id is None:
                    explicit_invalid = True
                break

        user_id_value = (
            payload.get("user_id")
            or payload.get("customer_id")
            or payload.get("buyer_id")
        )
        user_id = self._normalise_user_id(user_id_value)
        context_user_name: Optional[str] = None
        if user_id is None:
            context_user_id, context_user_name = self._extract_context_user()
            if context_user_id is not None:
                user_id = context_user_id
                note_label = (
                    f"Linked order to {context_user_name} from session context."
                    if context_user_name
                    else "Linked order to the signed-in customer from session context."
                )
                notes.append(note_label)

        if user_id is None:
            return None, {
                "error": "order_id_required",
                "message": "order_id is required when creating order items. Provide order_id explicitly, include user_id, or authenticate first so a recent order can be inferred.",
            }, notes

        filters: Dict[str, Any] = {"user_id": user_id}

        status_hint = payload.get("status") or payload.get("order_status")
        status_enum = None
        if status_hint is not None:
            status_enum = self._normalise_transaction_status(status_hint)
            if status_enum is None:
                notes.append("Ignoring unrecognised status hint while locating order_id.")
        if status_enum is None:
            status_enum = TransactionStatus.PENDING
        filters["status"] = status_enum

        delivery_raw = payload.get("delivery_date")
        parsed_delivery: Optional[date] = None
        if delivery_raw is not None:
            parsed_delivery = self._parse_date(delivery_raw)
            if parsed_delivery is not None:
                filters["delivery_date"] = parsed_delivery
            else:
                notes.append("delivery_date could not be parsed when resolving order_id; searching without it.")

        if explicit_order_id is not None:
            existing_tx = await TransactionRepository.get_transaction_by_id(explicit_order_id)
            if existing_tx:
                notes.append(f"Using provided order reference {existing_tx.order_id}.")
                return existing_tx.order_id, None, notes
            auto_transaction, auto_error = await self._auto_create_transaction_for_items(
                payload,
                user_id,
                status_enum,
                parsed_delivery,
                override_order_id=explicit_order_id,
            )
            if auto_error:
                return None, auto_error, notes
            if auto_transaction is not None:
                notes.append(
                    f"Created new order {auto_transaction.order_id} from the supplied {explicit_source or 'order reference'}."
                )
                return auto_transaction.order_id, None, notes
        elif explicit_raw is not None and explicit_invalid:
            notes.append("Ignored invalid order reference while creating the order; generated a fresh transaction instead.")

        transactions = await self._fetch_repository_records(TransactionRepository, filters, limit=5)

        # Fallback without status if nothing matched.
        if not transactions:
            fallback_filters = dict(filters)
            fallback_filters.pop("status", None)
            transactions = await self._fetch_repository_records(TransactionRepository, fallback_filters, limit=5)
            if transactions:
                notes.append("No pending orders matched; selected the most recent order instead.")

        if not transactions:
            auto_transaction, auto_error = await self._auto_create_transaction_for_items(
                payload,
                user_id,
                status_enum,
                parsed_delivery,
            )
            if auto_error:
                return None, auto_error, notes
            if auto_transaction is not None:
                transactions = [auto_transaction]
                notes.append(
                    f"Created new order {auto_transaction.order_id} for user {user_id}."
                )

        if not transactions:
            return None, {
                "error": "order_id_required",
                "message": "No matching order found for the provided details. Specify order_id explicitly or create a transaction first.",
            }, notes

        def _get_attr(record: Any, attr: str, default: Any = None) -> Any:
            if isinstance(record, dict):
                return record.get(attr, default)
            return getattr(record, attr, default)

        def _score(record: Any) -> datetime:
            created = _get_attr(record, "created_at")
            if isinstance(created, datetime):
                return created
            record_date = _get_attr(record, "date")
            if isinstance(record_date, date):
                return datetime.combine(record_date, datetime.min.time())
            return datetime.min

        selected = max(transactions, key=_score)
        order_id_value = _get_attr(selected, "order_id") or _get_attr(selected, "id")
        if not order_id_value:
            return None, {
                "error": "order_id_required",
                "message": "Unable to determine order_id from recent transactions. Provide order_id explicitly.",
            }, notes

        notes.append(f"Linked order items to order {order_id_value} for user {user_id}.")
        return order_id_value, None, notes

    async def _auto_create_transaction_for_items(
        self,
        payload: Dict[str, Any],
        user_id: Any,
        status: TransactionStatus,
        delivery_date: Optional[date],
        override_order_id: Optional[UUID] = None,
    ) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
        if user_id is None:
            return None, {
                "error": "order_id_required",
                "message": "Unable to create an order without a recognised customer reference.",
            }

        user_id_normalised = self._normalise_user_id(user_id)
        if user_id_normalised is None:
            return None, {
                "error": "order_id_required",
                "message": "Customer reference was not recognised; provide a valid user_id before creating order items.",
            }

        total_hint = (
            payload.get("total_price")
            or payload.get("total_amount")
            or payload.get("subtotal")
            or payload.get("line_total")
        )
        total_val = self._parse_float(total_hint) if total_hint is not None else None
        if total_val is None or total_val < 0:
            total_val = 0.0

        payment_hint = payload.get("payment_method") or payload.get("payment_type")
        payment_method = self._normalise_payment_method(payment_hint) or PaymentMethod.COD

        order_date_hint = payload.get("order_date") or payload.get("date")
        order_date = self._parse_date(order_date_hint) or date.today()

        create_kwargs: Dict[str, Any] = {
            "user_id": user_id_normalised,
            "date": order_date,
            "delivery_date": delivery_date,
            "total_price": float(total_val),
            "payment_method": payment_method,
            "status": status,
        }
        if override_order_id is not None:
            create_kwargs["order_id"] = override_order_id

        try:
            transaction = await TransactionRepository.create_transaction(**create_kwargs)
        except IntegrityError as exc:
            return None, {
                "error": "transaction_creation_failed",
                "message": f"Unable to create transaction automatically: {exc}",
            }
        except Exception as exc:  # pragma: no cover - defensive
            return None, {
                "error": "transaction_creation_failed",
                "message": f"Unexpected error while creating transaction: {exc}",
            }

        return transaction, None

    async def _prepare_order_item_payload(
        self,
        data: Dict[str, Any],
        *,
        default_order_id: Optional[Any] = None,
        default_supplier_id: Optional[int] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[str]]:
        notes: List[str] = []
        payload = dict(data)

        supplier_product: Optional[SupplierProduct] = None
        supplier_seed_id: Optional[int] = None
        supplier_product_id = (
            payload.get("supplier_product_id")
            or payload.get("inventory_id")
            or payload.get("supplier_inventory_id")
        )
        if supplier_product_id:
            try:
                supplier_product = await SupplierProductRepository.get_supplier_product_by_id(supplier_product_id)
            except Exception as exc:  # pragma: no cover - defensive
                notes.append(f"Unable to load supplier product {supplier_product_id}: {exc}")
            else:
                if not supplier_product:
                    notes.append(f"Supplier product {supplier_product_id} was not found; continuing without it.")
                else:
                    supplier_seed_id = getattr(supplier_product, "supplier_id", None)
                    product_id_from_inventory = getattr(supplier_product, "product_id", None)
                    if product_id_from_inventory and not payload.get("product_id"):
                        payload["product_id"] = product_id_from_inventory
                    product_obj = getattr(supplier_product, "product", None)
                    if not payload.get("product_name") and product_obj is not None:
                        inferred_name = (
                            getattr(product_obj, "product_name_en", None)
                            or getattr(product_obj, "product_name", None)
                        )
                        if inferred_name:
                            payload["product_name"] = inferred_name
                    if payload.get("unit") in (None, ""):
                        unit_from_inventory = getattr(supplier_product, "unit", None)
                        if unit_from_inventory:
                            payload["unit"] = unit_from_inventory
                    if all(payload.get(alias) is None for alias in ("price_per_unit", "unit_price", "unit_price_etb")):
                        price_from_inventory = getattr(supplier_product, "unit_price_etb", None)
                        if price_from_inventory is not None:
                            payload["price_per_unit"] = price_from_inventory

        order_id = payload.get("order_id") or default_order_id
        if not order_id:
            resolved_order_id, resolve_error, auto_notes = await self._resolve_order_id_for_items(payload)
            notes.extend(auto_notes)
            if resolve_error:
                return None, resolve_error, notes
            order_id = resolved_order_id
        payload["order_id"] = order_id

        ignored_helper_fields = (
            "user_id",
            "customer_id",
            "buyer_id",
            "order_reference",
            "transaction_id",
            "delivery_date",
            "status",
            "order_status",
        )
        for helper_field in ignored_helper_fields:
            payload.pop(helper_field, None)

        supplier_name_hint = payload.pop("supplier_name", None)
        supplier_source = payload.get("supplier_id")

        def _apply_supplier_reference(
            ref: Any,
            current_id: Optional[int],
            current_name: Optional[str],
        ) -> Tuple[Optional[int], Optional[str]]:
            if ref is None:
                return current_id, current_name
            if isinstance(ref, (int, float)):
                if current_id is None:
                    return int(ref), current_name
                return current_id, current_name
            if isinstance(ref, str):
                candidate = ref.strip()
                if not candidate:
                    return current_id, current_name
                if candidate.isdigit():
                    if current_id is None:
                        return int(candidate), current_name
                    return current_id, current_name
                return current_id, current_name or candidate
            if isinstance(ref, dict):
                value = ref.get("value") or ref.get("id")
                name_val = ref.get("name") or ref.get("label")
                if isinstance(name_val, str) and name_val.strip():
                    current_name = current_name or name_val.strip()
                if isinstance(value, (int, float)):
                    if current_id is None:
                        return int(value), current_name
                    return current_id, current_name
                if isinstance(value, str):
                    value_str = value.strip()
                    if value_str.isdigit():
                        if current_id is None:
                            return int(value_str), current_name
                        return current_id, current_name
                    current_name = current_name or value_str
                return current_id, current_name
            return current_id, current_name

        supplier_id: Optional[int] = None
        supplier_id, supplier_name_hint = _apply_supplier_reference(supplier_seed_id, supplier_id, supplier_name_hint)
        supplier_id, supplier_name_hint = _apply_supplier_reference(default_supplier_id, supplier_id, supplier_name_hint)
        supplier_id, supplier_name_hint = _apply_supplier_reference(supplier_source, supplier_id, supplier_name_hint)
        supplier_id, supplier_name_hint = _apply_supplier_reference(payload.get("supplier"), supplier_id, supplier_name_hint)

        payload.pop("supplier", None)
        payload.pop("supplier_id", None)
        payload.pop("supplier_product_id", None)
        payload.pop("inventory_id", None)
        payload.pop("supplier_inventory_id", None)

        if supplier_name_hint and supplier_id is None:
            supplier = await UserRepository.get_user_by_name(supplier_name_hint)
            if not supplier or getattr(supplier, "role", None) != UserRole.SUPPLIER:
                return None, {
                    "error": "supplier_not_found",
                    "message": f"Supplier '{supplier_name_hint}' was not recognised.",
                }, notes
            supplier_id = supplier.user_id
            notes.append(f"Linked supplier {supplier.name} to order item.")

        if supplier_id is not None:
            try:
                payload["supplier_id"] = int(supplier_id)
            except (TypeError, ValueError):
                payload["supplier_id"] = supplier_id

        quantity_val = self._parse_float(payload.get("quantity"))
        if quantity_val is None or quantity_val <= 0:
            return None, {"error": "quantity must be a positive number."}, notes
        payload["quantity"] = quantity_val

        price_raw = (
            payload.get("price_per_unit")
            or payload.get("unit_price")
            or payload.get("unit_price_etb")
        )
        price_val = self._parse_float(price_raw)
        if price_val is None or price_val < 0:
            return None, {"error": "price_per_unit must be a non-negative number."}, notes
        payload["price_per_unit"] = price_val

        subtotal_raw = (
            payload.get("subtotal")
            or payload.get("total_price")
            or payload.get("line_total")
        )
        subtotal_val = self._parse_float(subtotal_raw) if subtotal_raw is not None else None
        if subtotal_val is None:
            subtotal_val = round(quantity_val * price_val, 2)
        payload["subtotal"] = subtotal_val

        product = None
        product_id = payload.get("product_id") or payload.get("product")
        product_name = payload.get("product_name") or payload.get("name")
        if product_id:
            product = await ProductRepository.get_product_by_id(product_id)
            if not product:
                return None, {"error": f"Product {product_id} not found for order item."}, notes
        elif product_name:
            product = await ProductRepository.find_product_by_any_name(product_name)
            if not product:
                return None, {"error": f"Product '{product_name}' not found for order item."}, notes
        else:
            return None, {"error": "product_id or product_name is required for order items."}, notes
        payload["product_id"] = product.product_id

        unit_value = payload.get("unit")
        unit_enum = self._normalise_unit(unit_value)
        if unit_enum is None and product is not None:
            unit_enum = product.unit
        if unit_enum is None:
            return None, {"error": "unit is required for order items."}, notes
        payload["unit"] = unit_enum

        for alias in (
            "unit_price",
            "unit_price_etb",
            "product_name",
            "name",
            "product",
            "total_price",
            "line_total",
        ):
            payload.pop(alias, None)

        payload["quantity"] = float(payload["quantity"])
        payload["price_per_unit"] = float(payload["price_per_unit"])
        payload["subtotal"] = float(payload["subtotal"])
        return payload, None, notes

    async def _create_order_items_batch(self, data: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(data)
        items = payload.get("items")
        if not isinstance(items, list) or not items:
            return {
                "error": "items_required",
                "message": "Provide a non-empty 'items' list to create multiple order items.",
            }
        notes: List[str] = []
        order_id = payload.get("order_id")
        if not order_id:
            resolved_order_id, resolve_error, auto_notes = await self._resolve_order_id_for_items(payload)
            notes.extend(auto_notes)
            if resolve_error:
                return resolve_error
            order_id = resolved_order_id
            payload["order_id"] = order_id
        supplier_id = payload.get("supplier_id") or payload.get("supplier")
        if payload.get("supplier_name") and not supplier_id:
            supplier_id = payload.get("supplier_name")
        extraneous_fields: List[str] = []
        total_amount_raw = payload.get("total_amount")
        for field in (
            "customer_id",
            "buyer_id",
            "user_id",
            "delivery_location",
            "delivery_date",
            "order_reference",
            "transaction_id",
            "order_status",
            "status",
            "total_amount",
        ):
            if field in payload:
                extraneous_fields.append(field)
                payload.pop(field, None)

        normalised_items: List[Dict[str, Any]] = []
        accum_subtotal = Decimal("0")
        for item in items:
            normalised, error, item_notes = await self._prepare_order_item_payload(
                item,
                default_order_id=order_id,
                default_supplier_id=supplier_id,
            )
            if error:
                return error
            notes.extend(item_notes)
            try:
                accum_subtotal += Decimal(str(normalised.get("subtotal")))
            except (InvalidOperation, TypeError):
                pass
            normalised_items.append(normalised)

        created_records: List[Dict[str, Any]] = []
        try:
            async with in_transaction() as connection:
                for normalised in normalised_items:
                    instance = await OrderItemRepository.create_order_item(using_db=connection, **normalised)
                    await instance.fetch_related("product")
                    record = await self._serialize_model(instance)
                    created_records.append(record)
        except IntegrityError as exc:
            return {"error": f"Order item batch creation failed: {exc}"}

        if created_records:
            await self._recalculate_transaction_total(order_id)

        result: Dict[str, Any] = {
            "message": "Order items created successfully.",
            "records": created_records,
        }
        if extraneous_fields:
            notes.append(
                "Ignored unsupported fields during order item creation: "
                + ", ".join(sorted(extraneous_fields))
            )
        if total_amount_raw is not None:
            try:
                total_amount_val = Decimal(str(total_amount_raw))
                if accum_subtotal and abs(accum_subtotal - total_amount_val) > Decimal("0.05"):
                    notes.append(
                        f"Provided total_amount {total_amount_val} differs from computed subtotal {accum_subtotal}."
                    )
            except (InvalidOperation, TypeError):
                pass
        if notes:
            result["notes"] = notes
        return result

    async def _recalculate_transaction_total(self, order_id: Any) -> None:
        if not order_id:
            return
        try:
            items = await OrderItemRepository.list_order_items({"order_id": order_id})
        except Exception:
            return
        total = Decimal("0")
        for item in items or []:
            subtotal = getattr(item, "subtotal", None)
            try:
                total += Decimal(str(subtotal)) if subtotal is not None else Decimal("0")
            except (InvalidOperation, TypeError):
                continue
        await TransactionRepository.update_transaction(order_id, total_price=float(total))

    async def _create_entity(self, repository, entity: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch create requests with entity-specific normalisation."""
        if not data:
            return {"error": "Data payload is required for 'create' operation."}

        creation_notes: List[str] = []

        if entity == "supplier_products":
            return await self._create_supplier_product(data)
        if entity == "products":
            return await self._create_product(data)
        if entity == "flash_sales":
            return await self._create_flash_sale(data)

        if entity == "users":
            payload = dict(data)
            role_value = payload.get("role")
            normalised_role = self._normalise_role(role_value) if role_value is not None else UserRole.SUPPLIER
            if normalised_role == UserRole.CUSTOMER and not payload.get("default_location"):
                return {
                    "error": "default_location_required",
                    "message": "Customer accounts need a default_location (e.g. kebele, district, or neighborhood). Suppliers do not require one.",
                }
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

        if entity == "transactions":
            payload = dict(data)

            alias_map = {
                "customer_id": "user_id",
                "buyer_id": "user_id",
                "total_amount": "total_price",
                "total": "total_price",
            }
            for source, target in alias_map.items():
                if source in payload and target not in payload:
                    payload[target] = payload.pop(source)

            ignored_fields = []
            for extraneous in ("delivery_location", "preferred_delivery_date"):
                if extraneous in payload:
                    payload.pop(extraneous)
                    ignored_fields.append(extraneous)
            if ignored_fields:
                creation_notes.append(
                    "Ignored unsupported fields during transaction creation: " + ", ".join(ignored_fields)
                )

            if "total_price" in payload:
                total_price = self._parse_float(payload["total_price"])
                if total_price is None or total_price < 0:
                    return {"error": "total_price must be a non-negative number."}
                payload["total_price"] = total_price

            if "payment_method" in payload:
                payment_method = self._normalise_payment_method(payload["payment_method"])
                if payment_method is None:
                    return {"error": "payment_method is not recognised for transactions."}
                payload["payment_method"] = payment_method
            else:
                payload["payment_method"] = PaymentMethod.COD

            if "status" in payload:
                status = self._normalise_transaction_status(payload["status"])
                if status is None:
                    return {"error": "status is not recognised for transactions."}
                payload["status"] = status
            else:
                payload["status"] = TransactionStatus.PENDING

            for field in ("date", "delivery_date"):
                if field in payload:
                    parsed_date = self._parse_date(payload[field])
                    if parsed_date is None:
                        return {"error": f"{field} must be a valid ISO date or a keyword like 'today'."}
                    payload[field] = parsed_date

            if "date" not in payload:
                payload["date"] = date.today()

            data = payload

        if entity == "order_items":
            if isinstance(data, dict) and "items" in data:
                return await self._create_order_items_batch(data)

            payload, error, notes = await self._prepare_order_item_payload(data)
            if error:
                return error
            data = payload
            creation_notes.extend(notes)

        method = self._get_repo_method(entity, "create")
        if method is None:
            return {"error": f"'create' operation is not available for entity: {entity}."}

        try:
            instance = await method(**data)
        except IntegrityError as exc:
            return {"error": f"Create failed due to integrity error: {exc}"}

        result = {
            "message": "Record created successfully.",
            "record": await self._serialize_model(instance),
        }
        if entity == "order_items":
            await self._recalculate_transaction_total(data.get("order_id"))
        if creation_notes:
            result["notes"] = creation_notes
        return result

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

    async def _update_order_delivery(self, request: DataAccessRequest) -> Dict[str, Any]:
        filters = request.filters or {}
        delivery_raw = request.data.get("delivery_date") or request.data.get("preferred_delivery_date")
        delivery_value = None
        if isinstance(delivery_raw, dict):
            delivery_value = delivery_raw.get("value")
            if delivery_value is None and delivery_raw.get("lookup") in delivery_raw:
                delivery_value = delivery_raw.get(delivery_raw.get("lookup"))
        else:
            delivery_value = delivery_raw
        if delivery_value is None:
            return {"error": "delivery_date is required to update an order delivery."}

        notes: List[str] = []
        explicit_year = self._contains_explicit_year(delivery_value)
        delivery_date, schedule_notes, reference_date = await self._resolve_delivery_date_with_schedule(
            delivery_value
        )
        if schedule_notes:
            notes.extend(schedule_notes)
        if delivery_date is None:
            delivery_date = self._parse_date(delivery_value)
        if delivery_date is None:
            error_payload: Dict[str, Any] = {
                "error": "delivery_date must be a valid ISO date or a keyword like 'today'."
            }
            if notes:
                error_payload["notes"] = notes
            return error_payload

        reference_date = max(reference_date, date.today())
        if delivery_date < reference_date:
            if not explicit_year and delivery_date.year < reference_date.year:
                adjusted_date = self._roll_forward_delivery_date(delivery_date, reference_date)
                if adjusted_date >= reference_date:
                    notes.append(
                        f"Adjusted delivery date to {adjusted_date.isoformat()} to keep it in the future."
                    )
                    delivery_date = adjusted_date
                else:
                    notes.append(
                        "Unable to adjust the delivery date to a future value; please provide a full future date."
                    )
                    return {"error": "delivery_date must be in the future.", "notes": notes}
            else:
                notes.append(
                    f"Resolved delivery date {delivery_date.isoformat()} is before {reference_date.isoformat()}."
                )
                notes.append("Please provide a future delivery date.")
                return {"error": "delivery_date must be in the future.", "notes": notes}

        # Try to resolve relevant order IDs based on the provided filters
        candidate_order_ids: set[str] = set()
        try:
            order_item_records = await self._fetch_repository_records(
                OrderItemRepository,
                filters,
                limit=25,
            )
            for record in order_item_records:
                order_id_value = getattr(record, "order_id", None)
                if order_id_value:
                    candidate_order_ids.add(str(order_id_value))
        except Exception:
            # Continue even if order item lookup fails; we can still rely on broader filters
            pass

        tx_filters: Dict[str, Any] = {}
        order_reference = filters.get("order_id") or filters.get("order_reference")
        if order_reference is not None:
            tx_filters["order_id"] = order_reference
        user_filter = filters.get("user_id")
        if user_filter is not None:
            tx_filters["user_id"] = user_filter

        status_filter = filters.get("status")
        status_enum = None
        if status_filter is not None:
            status_enum = self._normalise_transaction_status(status_filter)
            if status_enum is None:
                return {"error": "status filter is not recognised for delivery updates."}
            tx_filters["status"] = status_enum

        normalised_filters = await self._maybe_expand_special_filters(
            TransactionRepository,
            tx_filters,
        )
        if normalised_filters is None:
            normalised_filters = {}

        if candidate_order_ids:
            if "order_id" in normalised_filters and isinstance(normalised_filters["order_id"], dict):
                existing = normalised_filters["order_id"].get("value")
                if isinstance(existing, list):
                    merged_set = set(existing)
                    merged_set.update(candidate_order_ids)
                    normalised_filters["order_id"] = {
                        "lookup": normalised_filters["order_id"].get("lookup", "in"),
                        "value": list(merged_set),
                    }
            elif "order_id" in normalised_filters:
                merged_set = set(candidate_order_ids)
                merged_set.add(str(normalised_filters["order_id"]))
                normalised_filters["order_id"] = {"lookup": "in", "value": list(merged_set)}
            else:
                normalised_filters["order_id"] = {"lookup": "in", "value": list(candidate_order_ids)}

        transactions = await TransactionRepository.list_transactions(normalised_filters)

        if candidate_order_ids:
            filtered_transactions = [
                tx for tx in transactions if str(tx.order_id) in candidate_order_ids
            ]
            if filtered_transactions:
                transactions = filtered_transactions
            else:
                notes.append(
                    "No direct match for the provided order filters; falling back to the most recent pending order."
                )

        if not transactions and status_enum is not None:
            fallback_filters = dict(normalised_filters)
            fallback_filters.pop("status", None)
            transactions = await TransactionRepository.list_transactions(fallback_filters)
            if transactions:
                notes.append(
                    "No orders matched the requested status; updated the most recent matching order instead."
                )

        if not transactions and user_filter is not None and not normalised_filters:
            transactions = await TransactionRepository.list_transactions({"user_id": user_filter})

        if not transactions:
            return {"error": "No matching order found to update the delivery date."}

        selected = max(
            transactions,
            key=lambda tx: getattr(tx, "created_at", datetime.min),
        )
        if len(transactions) > 1:
            notes.append(
                "Multiple orders matched the filters; updated the most recent one."
            )

        updated = await TransactionRepository.update_transaction(
            selected.order_id,
            delivery_date=delivery_date,
        )
        if not updated:
            return {"error": "Unable to update the delivery date for the selected order."}

        try:
            await updated.refresh_from_db()
        except Exception:
            pass

        result = {
            "message": "Delivery date updated successfully.",
            "record": await self._serialize_model(updated),
        }
        if notes:
            result["notes"] = notes
        return result

    async def _update_entity(
        self,
        entity: str,
        entity_id: Any,
        data: Dict[str, Any],
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Update records across repositories with light normalisation."""
        if not data:
            return {"error": "Data payload is required for 'update' operation."}

        repository = self.repositories.get(entity)
        if repository is None:
            return {"error": f"Unknown entity: {entity}."}

        resolved_id = entity_id
        batch_matches: Optional[List[Any]] = None
        matched_record: Optional[Any] = None
        if not resolved_id:
            if filters:
                matches = await self._fetch_repository_records(repository, filters)
                if not matches:
                    return {"error": "No records matched the provided filters for update."}
                if len(matches) > 1:
                    if entity == "order_items":
                        batch_matches = matches
                    else:
                        matched_ids = [
                            str(self._extract_pk(record))
                            for record in matches
                            if self._extract_pk(record) is not None
                        ]
                        return {
                            "error": "Multiple records matched update filters; provide an explicit id.",
                            "matched_ids": matched_ids,
                        }
                else:
                    matched_record = matches[0]
                    resolved_id = self._extract_pk(matched_record)
                    if resolved_id is None:
                        return {"error": "Unable to determine record id for update."}
            else:
                return {"error": "ID is required for 'update' operation."}

        update_payload = dict(data)
        order_status_update: Optional[TransactionStatus] = None

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

        elif entity == "order_items":
            normalised_payload: Dict[str, Any] = {}

            if "quantity" in update_payload:
                quantity_val = self._parse_float(update_payload["quantity"])
                if quantity_val is None or quantity_val <= 0:
                    return {"error": "quantity must be a positive number."}
                normalised_payload["quantity"] = quantity_val

            if any(key in update_payload for key in ("price_per_unit", "unit_price", "unit_price_etb")):
                price_val = self._parse_float(
                    update_payload.get("price_per_unit")
                    or update_payload.get("unit_price")
                    or update_payload.get("unit_price_etb")
                )
                if price_val is None or price_val < 0:
                    return {"error": "price_per_unit must be a non-negative number."}
                normalised_payload["price_per_unit"] = price_val

            if "subtotal" in update_payload or "total_price" in update_payload:
                subtotal_val = self._parse_float(
                    update_payload.get("subtotal")
                    or update_payload.get("total_price")
                )
                if subtotal_val is None or subtotal_val < 0:
                    return {"error": "subtotal must be a non-negative number."}
                normalised_payload["subtotal"] = subtotal_val

            if "unit" in update_payload:
                unit_enum = self._normalise_unit(update_payload["unit"])
                if unit_enum is None:
                    return {"error": "unit is not recognised for order items."}
                normalised_payload["unit"] = unit_enum

            product_override = (
                update_payload.get("product_id")
                or update_payload.get("product")
            )
            if product_override:
                product = await ProductRepository.get_product_by_id(product_override)
                if not product:
                    return {"error": "Product not found for order item update."}
                normalised_payload["product_id"] = product.product_id
            elif update_payload.get("product_name"):
                product = await ProductRepository.find_product_by_any_name(update_payload["product_name"])
                if not product:
                    return {"error": f"Product '{update_payload['product_name']}' not found for order item update."}
                normalised_payload["product_id"] = product.product_id

            supplier_hint = update_payload.get("supplier_id") or update_payload.get("supplier")
            if supplier_hint is not None:
                supplier_id_value: Optional[int] = None
                if isinstance(supplier_hint, (int, float)):
                    supplier_id_value = int(supplier_hint)
                elif isinstance(supplier_hint, str) and supplier_hint.strip().isdigit():
                    supplier_id_value = int(supplier_hint.strip())
                if supplier_id_value is None:
                    supplier = await UserRepository.get_user_by_name(str(supplier_hint)) if isinstance(supplier_hint, str) else None
                    supplier_id_value = getattr(supplier, "user_id", None)
                if supplier_id_value is None:
                    return {"error": "supplier reference for order item update was not recognised."}
                normalised_payload["supplier_id"] = supplier_id_value

            if "status" in update_payload:
                status_enum = self._normalise_transaction_status(update_payload["status"])
                if status_enum is None:
                    return {
                        "error": "status is not recognised for order confirmation.",
                        "hint": "Use Pending, Confirmed, Delivered, or Cancelled.",
                    }
                order_status_update = status_enum

            update_payload = normalised_payload
            if not update_payload and order_status_update is None:
                return {"error": "No valid order item fields supplied for update."}

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

        order_ids_to_update: List[Any] = []

        if batch_matches is not None:
            updated_records: List[Dict[str, Any]] = []
            for match in batch_matches:
                match_id = self._extract_pk(match)
                if match_id is None:
                    continue
                instance = await method(match_id, **update_payload)
                if not instance:
                    continue
                order_ids_to_update.append(getattr(instance, "order_id", None))
                updated_records.append(await self._serialize_model(instance))

            if not updated_records and order_status_update is None:
                return {"error": "No matching order_items were updated."}

            result: Dict[str, Any] = {
                "message": f"{len(updated_records) or len(batch_matches)} order item(s) updated successfully.",
            }
            if updated_records:
                result["records"] = updated_records

        else:
            instance = await method(resolved_id, **update_payload)
            if not instance:
                return {"error": f"{entity} with ID {resolved_id} not found."}
            order_ids_to_update.append(getattr(instance, "order_id", None))
            result = {
                "message": "Record updated successfully.",
                "record": await self._serialize_model(instance),
            }

        if order_status_update is not None:
            status_notes: List[str] = []
            for order_id in order_ids_to_update:
                if not order_id:
                    continue
                updated_tx = await TransactionRepository.update_transaction(order_id, status=order_status_update)
                if updated_tx:
                    status_notes.append(
                        f"Order {order_id} status set to {order_status_update.value}."
                    )
            if status_notes:
                result.setdefault("notes", []).extend(status_notes)

        return result

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

    def _extract_context_user(self) -> Tuple[Optional[int], Optional[str]]:
        context = getattr(self, "_active_context", None)
        if not isinstance(context, dict):
            return None, None
        user_info = context.get("user")
        if not isinstance(user_info, dict):
            return None, None
        user_id = self._normalise_user_id(
            user_info.get("id") or user_info.get("user_id")
        )
        name = user_info.get("name") if isinstance(user_info.get("name"), str) else None
        if name:
            name = name.strip() or None
        return user_id, name

    @staticmethod
    def _normalise_user_id(value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            candidate = value.strip()
            if candidate.isdigit():
                return int(candidate)
        return None

    @staticmethod
    def _normalise_order_id(value: Any) -> Optional[UUID]:
        if value is None:
            return None
        if isinstance(value, UUID):
            return value
        if isinstance(value, str):
            candidate = value.strip()
            if not candidate:
                return None
            try:
                return UUID(candidate)
            except ValueError:
                return None
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
            stripped = value.strip().lower()
            if stripped in {"today", "now"}:
                return date.today()
            if stripped == "tomorrow":
                return date.today() + timedelta(days=1)
            if stripped == "yesterday":
                return date.today() - timedelta(days=1)
            try:
                return date.fromisoformat(stripped.split("T")[0])
            except ValueError:
                return None
        return None

    @staticmethod
    def _contains_explicit_year(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, (date, datetime)):
            return True
        return bool(re.search(r"\b(19|20)\d{2}\b", str(value)))

    @staticmethod
    def _roll_forward_delivery_date(candidate: date, reference: date) -> date:
        if candidate >= reference:
            return candidate

        def _replace_year(day: date, year: int) -> date:
            try:
                return day.replace(year=year)
            except ValueError:
                max_day = calendar.monthrange(year, day.month)[1]
                return day.replace(year=year, day=min(day.day, max_day))

        year = max(candidate.year, reference.year)
        adjusted = _replace_year(candidate, year)
        while adjusted < reference:
            year = adjusted.year + 1
            adjusted = _replace_year(candidate, year)
        return adjusted

    async def _resolve_delivery_date_with_schedule(self, value: Any) -> Tuple[Optional[date], List[str], date]:
        notes: List[str] = []
        reference_date = date.today()
        if value is None:
            return None, notes, reference_date
        if isinstance(value, date):
            return value, notes, reference_date
        phrase = str(value).strip()
        if not phrase:
            return None, notes, reference_date
        try:
            result = await self._schedule_tool.run(
                {
                    "operation": "expiry_date",
                    "phrase": phrase,
                },
                context={
                    "current_date": reference_date.isoformat(),
                    "reference_date": reference_date.isoformat(),
                },
            )
        except Exception as exc:  # pragma: no cover - defensive
            notes.append(f"schedule_helper error: {exc}")
            return None, notes, reference_date

        if not isinstance(result, dict):
            notes.append("schedule_helper returned an unexpected payload format.")
            return None, notes, reference_date

        if result.get("error"):
            error_detail = result.get("error")
            if isinstance(error_detail, str):
                notes.append(error_detail)
            else:
                notes.append("schedule_helper reported an error.")
            return None, notes, reference_date

        resolved = result.get("resolved_date")
        if resolved:
            try:
                resolved_date = date.fromisoformat(str(resolved).split("T")[0])
            except ValueError:
                notes.append("schedule_helper produced an invalid date value.")
                resolved_date = None
        else:
            resolved_date = None

        reference_raw = result.get("reference_date")
        if reference_raw:
            try:
                schedule_reference = date.fromisoformat(str(reference_raw).split("T")[0])
                reference_date = max(reference_date, schedule_reference)
            except ValueError:
                notes.append("schedule_helper provided an invalid reference date.")

        if resolved_date and resolved_date < reference_date:
            notes.append(
                f"schedule_helper resolved {resolved_date.isoformat()} before reference {reference_date.isoformat()}."
            )

        schedule_notes = result.get("notes")
        if isinstance(schedule_notes, list):
            notes.extend(str(item) for item in schedule_notes if item is not None)

        return resolved_date, notes, reference_date

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
    def _normalise_payment_method(value: Any) -> Optional[PaymentMethod]:
        if value is None:
            return None
        if isinstance(value, PaymentMethod):
            return value
        if isinstance(value, str):
            normalised = value.strip().lower()
            for method in PaymentMethod:
                if method.value.lower() == normalised:
                    return method
        return None

    @staticmethod
    def _normalise_transaction_status(value: Any) -> Optional[TransactionStatus]:
        if value is None:
            return None
        if isinstance(value, TransactionStatus):
            return value
        if isinstance(value, str):
            normalised = value.strip().lower()
            for status in TransactionStatus:
                if status.value.lower() == normalised:
                    return status
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
