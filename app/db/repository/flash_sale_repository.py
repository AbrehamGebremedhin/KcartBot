"""Repository helpers for flash sale workflow."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from tortoise.exceptions import DoesNotExist

from app.db.models import FlashSale, FlashSaleStatus, SupplierProduct, SupplierProductStatus


class FlashSaleRepository:
    @staticmethod
    async def create_flash_sale(
        supplier_product,
        supplier,
        product,
        start_date,
        end_date,
        discount_percent,
        *,
        status: Optional[FlashSaleStatus] = None,
        auto_generated: bool = False,
    ) -> FlashSale:
        """Persist a flash sale entry with derived status fallback."""
        resolved_status = status or FlashSaleRepository._derive_status(start_date, end_date)
        return await FlashSale.create(
            supplier_product=supplier_product,
            supplier=supplier,
            product=product,
            start_date=start_date,
            end_date=end_date,
            discount_percent=discount_percent,
            status=resolved_status,
            auto_generated=auto_generated,
        )

    @staticmethod
    async def create_or_get_proposal(
        supplier_product: SupplierProduct,
        *,
        discount_percent: float,
        start_date,
        end_date,
    ) -> Tuple[Optional[FlashSale], bool]:
        """Ensure there is a proposed flash sale for the given supplier product."""
        await supplier_product.fetch_related("supplier", "product")
        current_date = datetime.utcnow().date()
        allowed_statuses = {SupplierProductStatus.ACTIVE, SupplierProductStatus.ON_SALE}
        expiry_date = getattr(supplier_product, "expiry_date", None)
        if (
            getattr(supplier_product, "status", None) not in allowed_statuses
            or not expiry_date
            or expiry_date < current_date
        ):
            return None, False
        existing = await FlashSale.filter(
            supplier_product=supplier_product,
            status__in=[
                FlashSaleStatus.PROPOSED,
                FlashSaleStatus.SCHEDULED,
                FlashSaleStatus.ACTIVE,
            ],
        ).order_by("-created_at").first()
        if existing:
            await existing.fetch_related("supplier_product", "supplier", "product")
            sp_status = getattr(getattr(existing, "supplier_product", None), "status", None)
            sp_expiry = getattr(getattr(existing, "supplier_product", None), "expiry_date", None)
            if (
                sp_status not in allowed_statuses
                or (sp_expiry and sp_expiry < current_date)
            ):
                await FlashSaleRepository.cancel_flash_sale(existing.id)
                return None, False
            return existing, False
        created = await FlashSaleRepository.create_flash_sale(
            supplier_product=supplier_product,
            supplier=supplier_product.supplier,
            product=supplier_product.product,
            start_date=start_date,
            end_date=end_date,
            discount_percent=discount_percent,
            status=FlashSaleStatus.PROPOSED,
            auto_generated=True,
        )
        return created, True

    @staticmethod
    async def update_flash_sale_status(flash_sale_id):
        try:
            flash_sale = await FlashSale.get(id=flash_sale_id)
        except DoesNotExist:
            return None
        if flash_sale.status in (FlashSaleStatus.CANCELLED, FlashSaleStatus.PROPOSED):
            return flash_sale
        flash_sale.status = FlashSaleRepository._derive_status(
            flash_sale.start_date,
            flash_sale.end_date,
        )
        await flash_sale.save()
        return flash_sale

    @staticmethod
    async def cancel_flash_sale(flash_sale_id):
        try:
            flash_sale = await FlashSale.get(id=flash_sale_id)
        except DoesNotExist:
            return None
        flash_sale.status = FlashSaleStatus.CANCELLED
        await flash_sale.save()
        return flash_sale

    @staticmethod
    async def accept_flash_sale(flash_sale_id):
        """Convert a proposed flash sale into an active/scheduled one."""
        try:
            flash_sale = await FlashSale.get(id=flash_sale_id)
        except DoesNotExist:
            return None
        if flash_sale.status != FlashSaleStatus.PROPOSED:
            return flash_sale
        flash_sale.status = FlashSaleRepository._derive_status(
            flash_sale.start_date,
            flash_sale.end_date,
        )
        await flash_sale.save()
        return flash_sale

    @staticmethod
    async def update_flash_sale(flash_sale_id, **kwargs):
        flash_sale = await FlashSaleRepository.get_flash_sale_by_id(flash_sale_id)
        if not flash_sale:
            return None
        for key, value in kwargs.items():
            setattr(flash_sale, key, value)
        await flash_sale.save()
        return flash_sale

    @staticmethod
    async def delete_flash_sale(flash_sale_id):
        flash_sale = await FlashSaleRepository.get_flash_sale_by_id(flash_sale_id)
        if not flash_sale:
            return False
        await flash_sale.delete()
        return True

    @staticmethod
    async def get_flash_sale_by_id(flash_sale_id):
        try:
            return await FlashSale.get(id=flash_sale_id).fetch_related('supplier_product', 'supplier', 'product')
        except DoesNotExist:
            return None

    @staticmethod
    async def list_flash_sales(filters: Optional[Dict[str, Any]] = None):
        query = FlashSale.all().prefetch_related('supplier_product', 'supplier', 'product')
        if filters:
            for key, value in filters.items():
                if isinstance(value, dict):
                    query = query.filter(**{f"{key}__{value['lookup']}": value["value"]})
                else:
                    query = query.filter(**{key: value})
        return await query

    @staticmethod
    def _derive_status(start_date, end_date, *, now: Optional[datetime] = None) -> FlashSaleStatus:
        """Determine appropriate status based on timing."""
        now = now or datetime.utcnow()
        if start_date > now:
            return FlashSaleStatus.SCHEDULED
        if start_date <= now < end_date:
            return FlashSaleStatus.ACTIVE
        return FlashSaleStatus.EXPIRED
