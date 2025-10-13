"""Tooling for supplier flash sale proposals."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.db.models import FlashSaleStatus
from app.db.repository.flash_sale_repository import FlashSaleRepository
from app.db.repository.supplier_product_repository import SupplierProductRepository
from app.tools.base import ToolBase


class FlashSaleToolPayload(BaseModel):
    action: str = Field(description="Operation to perform: list_proposals|accept|decline")
    flash_sale_id: Optional[int] = Field(default=None, description="Target flash sale identifier")
    supplier_id: Optional[int] = Field(default=None, description="Supplier identifier (defaults from session)")
    within_days: Optional[int] = Field(default=3, ge=1, le=14, description="Days horizon for proposals")

    @field_validator("action", mode="before")
    @classmethod
    def _normalise_action(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise TypeError("action must be a string")
        return value.strip().lower()


class FlashSaleTool(ToolBase):
    """Manage flash sale proposals for suppliers."""

    def __init__(self) -> None:
        super().__init__(
            name="flash_sale_manager",
            description=(
                "Handle supplier flash sale workflow: list proposals, accept, or decline them. "
                "Example input: {'action': 'accept', 'flash_sale_id': 10}."
            ),
        )

    async def run(self, input: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            payload = FlashSaleToolPayload.model_validate(input)
        except ValidationError as exc:
            return {"error": "Invalid flash sale payload", "details": exc.errors()}

        supplier_id = payload.supplier_id or self._extract_supplier_id(context)
        if payload.action == "list_proposals":
            if not supplier_id:
                return {"error": "supplier_id required to list proposals"}
            await SupplierProductRepository.generate_flash_sale_proposals(
                supplier_id,
                within_days=payload.within_days or 3,
            )
            proposals = await FlashSaleRepository.list_flash_sales(
                {
                    "supplier_id": supplier_id,
                    "status": FlashSaleStatus.PROPOSED,
                }
            )
            items = [await self._serialise_proposal(proposal) for proposal in proposals]
            return {"count": len(items), "proposals": items}

        if payload.action in {"accept", "decline"}:
            if not payload.flash_sale_id:
                return {"error": "flash_sale_id is required for accept/decline"}
            if payload.action == "accept":
                updated = await FlashSaleRepository.accept_flash_sale(payload.flash_sale_id)
            else:
                updated = await FlashSaleRepository.cancel_flash_sale(payload.flash_sale_id)
            if not updated:
                return {"error": f"Flash sale {payload.flash_sale_id} not found"}
            return await self._serialise_proposal(updated)

        return {"error": f"Unsupported action: {payload.action}"}

    async def _serialise_proposal(self, proposal) -> Dict[str, Any]:
        await proposal.fetch_related("product", "supplier_product")
        return {
            "id": proposal.id,
            "product": getattr(getattr(proposal, "product", None), "product_name_en", None),
            "supplier_product_id": getattr(proposal.supplier_product, "inventory_id", None),
            "start_date": proposal.start_date.isoformat() if proposal.start_date else None,
            "end_date": proposal.end_date.isoformat() if proposal.end_date else None,
            "discount_percent": proposal.discount_percent,
            "status": proposal.status.value if proposal.status else None,
        }

    @staticmethod
    def _extract_supplier_id(context: Optional[Dict[str, Any]]) -> Optional[int]:
        if not context:
            return None
        user = context.get("user") if isinstance(context, dict) else None
        if not isinstance(user, dict):
            return None
        return user.get("id") if user.get("role") == "supplier" else None
