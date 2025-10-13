from typing import Any, Dict, List, Optional
import json
from app.tools.base import ToolBase
from app.db.repository.user_repository import UserRepository
from app.db.repository.product_repository import ProductRepository
from app.db.repository.supplier_product_repository import SupplierProductRepository
from app.db.repository.competitor_price_repository import CompetitorPriceRepository
from app.db.repository.transaction_repository import TransactionRepository
from app.db.repository.order_item_repository import OrderItemRepository
from pydantic import BaseModel, Field, ValidationError, field_validator


class DataAccessRequest(BaseModel):
    entity: str = Field(description="Target entity name, e.g. products")
    operation: str = Field(default="list", description="Operation to execute: list|get|search")
    id: Optional[Any] = Field(default=None, description="Identifier for get operations")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Filter criteria")
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
    user_id: Optional[int] = None
    supplier_id: Optional[int] = None
    filters: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("operation", mode="before")
    @classmethod
    def _lower_operation(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("operation must be a string")
        return value.strip().lower()


class DataAccessTool(ToolBase):
    """
    Tool for AI agent to access and query data from various repositories.
    Supports listing, searching, and retrieving data from users, products, 
    supplier products, competitor prices, transactions, and order items.
    """

    def __init__(self):
        super().__init__(
            name="data_access",
            description=(
                "Access and query data from the database. "
                "Supports operations: 'list', 'get', 'search' on entities: "
                "'users', 'products', 'supplier_products', 'competitor_prices', "
                "'transactions', 'order_items'. "
                "Example input: {'entity': 'products', 'operation': 'list', 'filters': {'category': 'Vegetable'}}"
            )
        )
        self.repositories = {
            'users': UserRepository,
            'products': ProductRepository,
            'supplier_products': SupplierProductRepository,
            'competitor_prices': CompetitorPriceRepository,
            'transactions': TransactionRepository,
            'order_items': OrderItemRepository
        }

    async def run(self, input: Dict[str, Any], context: Dict[str, Any] = None) -> Any:
        """Execute validated data access operations and surface friendly errors."""
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
            return {
                "error": f"Unknown operation: {operation}.",
                "valid_operations": ["list", "get", "search"],
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
        """Convert Tortoise ORM model to dictionary."""
        if model is None:
            return None
        
        # Get model data
        data = {}
        for field_name in model._meta.fields:
            value = getattr(model, field_name, None)
            
            # Handle special types
            if hasattr(value, 'isoformat'):  # datetime/date objects
                data[field_name] = value.isoformat()
            elif hasattr(value, 'value'):  # Enum objects
                data[field_name] = value.value
            else:
                data[field_name] = str(value) if value is not None else None
        
        return data


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
                "'price_comparison', 'supplier_inventory'. "
                "Example input: {'operation': 'product_stats', 'product_id': 'uuid-here'}"
            )
        )

    async def run(self, input: Dict[str, Any], context: Dict[str, Any] = None) -> Any:
        """Execute validated analytics operations."""
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
                return await self._get_price_comparison(request.product_id)
            if operation == 'supplier_inventory':
                return await self._get_supplier_inventory(request.supplier_id)
            return {
                "error": f"Unknown operation: {operation}.",
                "valid_operations": [
                    'product_stats',
                    'user_stats',
                    'transaction_stats',
                    'price_comparison',
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

    async def _get_price_comparison(self, product_id: str) -> Dict[str, Any]:
        """Compare prices across suppliers and competitors."""
        if not product_id:
            return {"error": "product_id is required"}
        
        product = await ProductRepository.get_product_by_id(product_id)
        if not product:
            return {"error": f"Product {product_id} not found"}
        
        # Get supplier prices
        supplier_products = await SupplierProductRepository.list_supplier_products(
            {'product_id': product_id}
        )
        
        # Get competitor prices
        competitor_prices = await CompetitorPriceRepository.list_competitor_prices(
            {'product_id': product_id}
        )
        
        supplier_prices = [
            {
                "supplier_id": sp.supplier_id,
                "price": sp.unit_price_etb,
                "quantity_available": sp.quantity_available,
                "status": sp.status.value
            }
            for sp in supplier_products
        ]
        
        competitor_price_data = [
            {
                "tier": cp.tier.value,
                "price": cp.price_etb_per_kg,
                "location": cp.source_location,
                "date": cp.date.isoformat()
            }
            for cp in competitor_prices
        ]
        
        return {
            "product_id": str(product_id),
            "product_name": product.product_name_en,
            "base_price": product.base_price_etb,
            "supplier_prices": supplier_prices,
            "competitor_prices": competitor_price_data,
            "lowest_supplier_price": min((sp.unit_price_etb for sp in supplier_products), default=None),
            "highest_supplier_price": max((sp.unit_price_etb for sp in supplier_products), default=None),
            "lowest_competitor_price": min((cp.price_etb_per_kg for cp in competitor_prices), default=None),
            "highest_competitor_price": max((cp.price_etb_per_kg for cp in competitor_prices), default=None)
        }

    async def _get_supplier_inventory(self, supplier_id: int) -> Dict[str, Any]:
        """Get inventory details for a specific supplier."""
        if not supplier_id:
            return {"error": "supplier_id is required"}
        
        supplier = await UserRepository.get_user_by_id(supplier_id)
        if not supplier:
            return {"error": f"Supplier {supplier_id} not found"}
        
        if supplier.role.value != 'supplier':
            return {"error": f"User {supplier_id} is not a supplier"}
        
        # Get all supplier products
        supplier_products = await SupplierProductRepository.list_supplier_products(
            {'supplier_id': supplier_id}
        )
        
        total_products = len(supplier_products)
        total_quantity = sum(sp.quantity_available for sp in supplier_products)
        total_value = sum(sp.quantity_available * sp.unit_price_etb for sp in supplier_products)
        
        # Group by status
        by_status = {}
        for sp in supplier_products:
            status = sp.status.value
            if status not in by_status:
                by_status[status] = {"count": 0, "quantity": 0}
            by_status[status]["count"] += 1
            by_status[status]["quantity"] += sp.quantity_available
        
        inventory_items = [
            {
                "inventory_id": str(sp.inventory_id),
                "product_id": str(sp.product_id),
                "quantity_available": sp.quantity_available,
                "unit": sp.unit.value,
                "unit_price": sp.unit_price_etb,
                "status": sp.status.value,
                "expiry_date": sp.expiry_date.isoformat() if sp.expiry_date else None
            }
            for sp in supplier_products
        ]
        
        return {
            "supplier_id": supplier_id,
            "supplier_name": supplier.name,
            "total_products": total_products,
            "total_quantity": total_quantity,
            "total_inventory_value": round(total_value, 2),
            "breakdown_by_status": by_status,
            "inventory": inventory_items
        }
