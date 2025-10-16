from typing import Optional

from tortoise.backends.base.client import BaseDBAsyncClient
from tortoise.exceptions import DoesNotExist

from app.db.models import OrderItem

class OrderItemRepository:
    @staticmethod
    async def create_order_item(*, using_db: Optional[BaseDBAsyncClient] = None, **kwargs):
        return await OrderItem.create(using_db=using_db, **kwargs)

    @staticmethod
    async def get_order_item_by_id(id):
        try:
            return await OrderItem.get(id=id).fetch_related('order', 'product', 'supplier')
        except DoesNotExist:
            return None

    @staticmethod
    async def update_order_item(id, **kwargs):
        order_item = await OrderItemRepository.get_order_item_by_id(id)
        if order_item:
            for key, value in kwargs.items():
                setattr(order_item, key, value)
            await order_item.save()
        return order_item

    @staticmethod
    async def delete_order_item(id):
        order_item = await OrderItemRepository.get_order_item_by_id(id)
        if order_item:
            await order_item.delete()
            return True
        return False

    @staticmethod
    async def list_order_items(filters=None):
        query = OrderItem.all().prefetch_related('order', 'product', 'supplier')
        if filters:
            for key, value in filters.items():
                if isinstance(value, dict):
                    query = query.filter(**{f"{key}__{value['lookup']}": value['value']})
                else:
                    query = query.filter(**{key: value})
        return await query
