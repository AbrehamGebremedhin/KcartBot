from datetime import datetime, timedelta, time
from typing import List

from tortoise.exceptions import DoesNotExist

from app.db.models import SupplierProduct, SupplierProductStatus
from app.db.repository.flash_sale_repository import FlashSaleRepository

class SupplierProductRepository:
    @staticmethod
    async def create_supplier_product(**kwargs):
        supplier_product = await SupplierProduct.create(**kwargs)
        # Fetch related objects for flash sale creation
        await supplier_product.fetch_related('supplier', 'product')
        
        # Post-save: create a proposed flash sale if product is near expiry
        expiry_date = getattr(supplier_product, 'expiry_date', None)
        status = getattr(supplier_product, 'status', None)
        current_date = datetime.utcnow().date()
        if (
            expiry_date
            and status in {SupplierProductStatus.ON_SALE, SupplierProductStatus.ACTIVE}
            and current_date <= expiry_date <= (current_date + timedelta(days=3))
        ):
            end_date = datetime.combine(expiry_date, time.max)
            await FlashSaleRepository.create_or_get_proposal(
                supplier_product,
                discount_percent=20.0,
                start_date=datetime.utcnow(),
                end_date=end_date,
            )
        return supplier_product

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
        query = SupplierProduct.all().prefetch_related('product')
        if filters:
            for key, value in filters.items():
                if key in {"product_name", "product_name_en"}:
                    if isinstance(value, dict):
                        lookup = value.get("lookup", "exact")
                        target = value.get("value")
                        if target is None:
                            continue
                        lookup_suffix = f"__{lookup}" if lookup and lookup != "exact" else ""
                        query = query.filter(**{f"product__product_name_en{lookup_suffix}": target})
                    else:
                        query = query.filter(product__product_name_en=value)
                    continue
                if key in {"product_label"}:
                    query = query.filter(product__product_name_en=value)
                    continue
                if isinstance(value, dict):
                    query = query.filter(**{f"{key}__{value['lookup']}": value['value']})
                else:
                    query = query.filter(**{key: value})
        return await query

    @staticmethod
    async def get_expiring_products(supplier_id: int, within_days: int = 3) -> List[SupplierProduct]:
        """Return supplier products that will expire within the given horizon."""
        horizon = datetime.utcnow().date() + timedelta(days=within_days)
        current_date = datetime.utcnow().date()
        query = SupplierProduct.filter(
            supplier_id=supplier_id,
            expiry_date__isnull=False,
            expiry_date__lte=horizon,
            expiry_date__gte=current_date,
            quantity_available__gt=0,
            status__in=[SupplierProductStatus.ACTIVE, SupplierProductStatus.ON_SALE],
        )
        return await query

    @staticmethod
    async def generate_flash_sale_proposals(
        supplier_id: int,
        *,
        within_days: int = 3,
        default_discount: float = 25.0,
    ) -> List[SupplierProduct]:
        """Ensure flash sale proposals exist for soon-to-expire products."""
        expiring_products = await SupplierProductRepository.get_expiring_products(
            supplier_id,
            within_days=within_days,
        )
        now = datetime.utcnow()
        for product in expiring_products:
            await product.fetch_related('supplier', 'product')
            end_of_day = datetime.combine(product.expiry_date, time.max)
            await FlashSaleRepository.create_or_get_proposal(
                product,
                discount_percent=default_discount,
                start_date=now,
                end_date=end_of_day,
            )
        return expiring_products
