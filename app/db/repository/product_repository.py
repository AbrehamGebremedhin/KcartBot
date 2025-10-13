from app.db.models import Product
from tortoise.exceptions import DoesNotExist

class ProductRepository:
    @staticmethod
    async def create_product(**kwargs):
        return await Product.create(**kwargs)

    @staticmethod
    async def get_product_by_name(name: str):
        if not name:
            return None
        return await Product.filter(product_name_en__iexact=name).first()

    @staticmethod
    async def get_product_by_id(product_id):
        try:
            return await Product.get(product_id=product_id)
        except DoesNotExist:
            return None

    @staticmethod
    async def update_product(product_id, **kwargs):
        product = await ProductRepository.get_product_by_id(product_id)
        if product:
            for key, value in kwargs.items():
                setattr(product, key, value)
            await product.save()
        return product

    @staticmethod
    async def delete_product(product_id):
        product = await ProductRepository.get_product_by_id(product_id)
        if product:
            await product.delete()
            return True
        return False

    @staticmethod
    async def list_products(filters=None):
        query = Product.all()
        if filters:
            for key, value in filters.items():
                if isinstance(value, dict):
                    query = query.filter(**{f"{key}__{value['lookup']}": value['value']})
                else:
                    query = query.filter(**{key: value})
        return await query
