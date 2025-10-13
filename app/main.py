"""Application entry point for KcartBot."""

from fastapi import FastAPI

from app.api.routes import router as api_router


APP_DESCRIPTION = (
	"Conversational commerce copilot orchestrating a Gemini LLM, Milvus RAG layer, and "
	"PostgreSQL-backed operations for both customers and suppliers."
)

TAGS_METADATA = [
	{
		"name": "meta",
		"description": "Service-level metadata and documentation helpers.",
	},
	{
		"name": "health",
		"description": "Lightweight readiness and liveness checks.",
	},
	{
		"name": "chat",
		"description": "Primary conversational API for KcartBot sessions.",
	},
]


app = FastAPI(
	title="KcartBot",
	description=APP_DESCRIPTION,
	version="1.0.0",
	docs_url="/docs",
	redoc_url="/redoc",
	openapi_tags=TAGS_METADATA,
)
app.include_router(api_router, prefix="/api")


@app.get("/", tags=["meta"], summary="KcartBot overview")
async def service_overview() -> dict:
	"""Return a high-level description and helpful linkage for the platform."""
	return {
		"service": "KcartBot",
		"version": app.version,
		"summary": APP_DESCRIPTION,
		"components": {
			"llm": "Gemini via custom LLMService wrapper",
			"vector_store": "Milvus collection for RAG context",
			"database": "PostgreSQL managed through Tortoise ORM",
			"api": "FastAPI with LangChain agent orchestration",
		},
		"links": {
			"docs": "/docs",
			"redoc": "/redoc",
			"health": "/api/health",
			"chat": "/api/v1/chat",
		},
	}


__all__ = ["app"]
