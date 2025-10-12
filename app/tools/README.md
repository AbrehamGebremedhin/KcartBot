# KcartBot AI Agent Tools

This directory contains custom tools for AI agents to interact with the KcartBot system, including data access, analytics, image generation, and vector search for Retrieval-Augmented Generation (RAG).

## Available Tools

### 1. ToolBase

Abstract base class for all tools. Inherit from this to implement custom async tools for LangChain or other AI agent frameworks.

---

### 2. DataAccessTool

Provides access to database entities via repository pattern. Supports listing, searching, and retrieving users, products, supplier products, competitor prices, transactions, and order items.

**Example Usage:**

```python
from app.tools import DataAccessTool
result = await DataAccessTool().run({
    'entity': 'products',
    'operation': 'list',
    'filters': {'category': 'Vegetable'},
    'limit': 10
})
```

---

### 3. AnalyticsDataTool

Performs analytical queries and provides insights, statistics, and cross-entity analysis (e.g., product stats, user stats, transaction stats, price comparison, supplier inventory).

**Example Usage:**

```python
from app.tools import AnalyticsDataTool
result = await AnalyticsDataTool().run({
    'operation': 'product_stats',
    'product_id': 'uuid-here'
})
```

---

### 4. ImageGenerator

Generates images using AI models (e.g., for product visualization or marketing). Accepts prompts and returns generated images.

**Example Usage:**

```python
from app.tools import ImageGenerator
result = await ImageGenerator().run({
    'prompt': 'A basket of fresh vegetables',
    'size': '512x512'
})
```

---

### 5. VectorSearchTool (RAG)

Semantic vector search tool for Retrieval-Augmented Generation. Uses Milvus and Gemini embeddings to find relevant context for product knowledge, documentation, and more.

**Example Usage:**

```python
from app.tools.search_context import VectorSearchTool
result = await VectorSearchTool().run({
    'query': 'What vegetables are available?',
    'top_k': 5,
    'min_score': 0.5
})
```

**Convenience Method:**

```python
context = await VectorSearchTool().get_context_for_query(
    query="What products can I buy?",
    top_k=3,
    min_score=0.5
)
```

---

## How to Add a New Tool

1. Inherit from `ToolBase`.
2. Implement the async `run` method.
3. Add your tool to `__init__.py` and update this README.

**Example:**

```python
from app.tools.base import ToolBase
class MyCustomTool(ToolBase):
    def __init__(self):
        super().__init__(name="my_custom_tool", description="Does something cool.")
    async def run(self, input, context=None):
        return {"result": "success"}
```

---

## Error Handling

All tools return errors in a consistent format:

```json
{ "error": "Description of what went wrong" }
```

---

## Requirements

- Database and ORM (Tortoise) must be initialized
- Milvus server must be running for vector search
- Gemini API key must be set in `.env` for embeddings
- Required packages: `pymilvus`, `google-genai`, etc.

---

## Integration

These tools are designed for use with LangChain or similar agent frameworks. They support async operation and flexible input formats (dict or JSON string).
