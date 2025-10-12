from app.db.models import SupplierProduct
from tortoise.exceptions import DoesNotExist

class SupplierProductRepository:
    @staticmethod
    async def create_supplier_product(**kwargs):
        return await SupplierProduct.create(**kwargs)

    @staticmethod
    async def get_supplier_product_by_id(inventory_id):
        try:
            return await SupplierProduct.get(inventory_id=inventory_id)
        except DoesNotExist:
            return None

    @staticmethod
    async def update_supplier_product(inventory_id, **kwargs):
        supplier_product = await SupplierProductRepository.get_supplier_product_by_id(inventory_id)
        if supplier_product:
            for key, value in kwargs.items():
                setattr(supplier_product, key, value)
            await supplier_product.save()
        return supplier_product

    @staticmethod
    async def delete_supplier_product(inventory_id):
        supplier_product = await SupplierProductRepository.get_supplier_product_by_id(inventory_id)
        if supplier_product:
            await supplier_product.delete()
            return True
        return False

    @staticmethod
    async def list_supplier_products(filters=None):
        query = SupplierProduct.all()
        if filters:
            for key, value in filters.items():
                if isinstance(value, dict):
                    query = query.filter(**{f"{key}__{value['lookup']}": value['value']})
                else:
                    query = query.filter(**{key: value})
        return await query
