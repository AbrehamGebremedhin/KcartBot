"""Root API routes and router composition."""

from fastapi import APIRouter

from app.api.v1.routes import router as v1_router


router = APIRouter()


@router.get("/health", tags=["health"])
async def health_check() -> dict:
	"""Lightweight readiness probe."""
	return {"status": "ok"}


router.include_router(v1_router, prefix="/v1")


__all__ = ["router"]
