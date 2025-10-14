import difflib
import re

from app.db.models import Product
from tortoise.exceptions import DoesNotExist

class ProductRepository:
    @staticmethod
    def _normalise_text(value: str) -> str:
        return re.sub(r"\s+", " ", value.strip().lower())

    @staticmethod
    async def create_product(**kwargs):
        return await Product.create(**kwargs)

    @staticmethod
    async def get_product_by_name(name: str):
        if not name:
            return None
        return await Product.filter(product_name_en__iexact=name).first()

    @classmethod
    async def find_product_by_any_name(cls, name: str):
        """Resolve a product by matching across English, Amharic, and phonetic variants.

        The lookup tolerates small spelling mistakes by falling back to a simple
        fuzzy match across all known product name variants.
        """

        if not name:
            return None

        raw = name.strip()
        if not raw:
            return None

        # Try exact matches across each stored variant.
        for field in ("product_name_en", "product_name_am", "product_name_am_latin"):
            product = await Product.filter(**{f"{field}__iexact": raw}).first()
            if product:
                return product

        # Try case-insensitive substring matches for simple typos/spacing variations.
        substring_candidates = {raw, raw.lower(), cls._normalise_text(raw)}
        for candidate in substring_candidates:
            for field in ("product_name_en", "product_name_am", "product_name_am_latin"):
                product = await Product.filter(**{f"{field}__icontains": candidate}).first()
                if product:
                    return product

        # Fuzzy match against all known variants to handle small spelling mistakes.
        products = await Product.all()
        if not products:
            return None

        target = cls._normalise_text(raw)
        variant_map = {}
        for product in products:
            for variant in (
                getattr(product, "product_name_en", None),
                getattr(product, "product_name_am", None),
                getattr(product, "product_name_am_latin", None),
            ):
                if not variant:
                    continue
                key = cls._normalise_text(str(variant))
                variant_map.setdefault(key, product)

        close = difflib.get_close_matches(target, variant_map.keys(), n=1, cutoff=0.7)
        if close:
            return variant_map[close[0]]

        return None

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
