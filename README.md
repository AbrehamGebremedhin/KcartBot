# KcartBot

Conversational commerce copilot that helps customers shop smarter and suppliers run pricing, inventory, and flash-sale workflows. The bot orchestrates a Gemini LLM, Milvus vector search, and a PostgreSQL operational store behind a FastAPI service.

## Architecture

```
User (web / chat client)
		  │   HTTP (JSON)
		  ▼
┌──────────────────────┐
│ FastAPI `/api/v1`    │  Rate-limited REST edge
└──────────┬───────────┘
			  │ calls
			  ▼
┌──────────────────────┐
│ ChatService          │  Session state, login flow, context
└──────────┬───────────┘
			  │ ainvoke
			  ▼
┌──────────────────────┐
│ LangChain Agent      │  Gemini LLM + tool calling
└┬───────┬───────┬─────┘
 │       │       │
 │       │       └──────────────┐
 │       │                      ▼
 │       │              ┌────────────────┐
 │       │              │ Image Tool     │→ Gemini Image API (marketing visuals)
 │       │
 │       └──────────────┐
 │                      ▼
 │              ┌────────────────┐
 │              │ FlashSaleTool  │→ PostgreSQL flash_sale tables
 │
 └──────────────┐
					 ▼
		  ┌──────────────┬───────────────┐
		  │ VectorSearch │ Data/Analytics │
		  │ Tool         │ Tools          │
		  └──────┬───────┴──────┬────────┘
					│              │
		Milvus (RAG)    PostgreSQL via Tortoise ORM
```

Why this layout?

- **Single agent brain with explicit tools** keeps Gemini focused on reasoning while LangChain enforces intent-first routing and structured tool use.
- **Milvus RAG layer** grounds advisory answers (product education, policy reminders) with curated PDF knowledge, avoiding hallucinations.
- **PostgreSQL OLTP store** models transactions, suppliers, inventory, and pricing history so supplier and customer actions mutate durable records.
- **Tools as thin, typed adapters** (analytics, flash sales, vector search, image generation) isolate external systems, making it simple to swap implementations (e.g., different vector DB or BI backend) later.

## Prerequisites

- Python 3.13 (recommended) or 3.11+
- [uv](https://docs.astral.sh/uv/) package/dependency manager (`pip install uv`)
- Running services:
  - PostgreSQL 15+ (`postgres://` connection string)
  - Milvus standalone 2.4.x (vector search)
- Google Gemini API key with access to `text-embedding-004` and `gemini-2.5-flash-image`

## Local Setup

1. **Clone & enter project**

   ```powershell
   git clone https://github.com/AbrehamGebremedhin/KcartBot.git
   cd KcartBot
   ```

2. **Create `.env`** (adjust values to your environment):

   ```dotenv
   GEMINI_API_KEY=your-google-genai-key
   DATABASE_URL=postgres://kcartbot:kcartbot@localhost:5432/kcartbot
   ```

3. **Provision databases** (example Docker commands):

   ```powershell
   docker run --name kcartbot-postgres -e POSTGRES_DB=kcartbot -e POSTGRES_USER=kcartbot -e POSTGRES_PASSWORD=kcartbot -p 5432:5432 -d postgres:16

   docker run --name kcartbot-milvus -p 19530:19530 -p 9091:9091 -d milvusdb/milvus:v2.4.3-20240813-74fbddb7
   ```

   > Tip: Use Docker volumes for persistent data in non-demo environments.

4. **Install project dependencies**

   ```powershell
   uv sync
   ```

5. **Generate seed data and embeddings** (ensures schema, populates PostgreSQL + Milvus):

   ```powershell
   uv run -m app.utils.generate_data insert
   ```

   - Verifies PostgreSQL database, creates tables when empty.
   - Inserts mock products, users, supplier inventory, competitor pricing, transactions.
   - Splits `data/Context.pdf` into embeddings and loads them into Milvus (`KCartBot` collection).

   Re-run with `uv run -m app.utils.generate_data check` to inspect counts without re-inserting.

6. **Start the API**

   ```powershell
   uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

   `http://localhost:8000/docs` exposes interactive Swagger for the chat endpoint.

## Usage Script (End-to-End Test)

All interactions hit `POST /api/v1/chat` with a stable `session_id`. Suggested flow to exercise every capability:

1. **Kick-off + role capture** (login funnel)

   - Send: `{"session_id":"demo-1","message":"Hi, I'm a supplier"}`
   - Expect prompt about existing account → respond with `"Yes, I have an account"`, then supply name/phone from seeded data (e.g., `"Name: Selam Agro"`, `"Phone: +251911000111"`).
   - Confirms session context and supplier summary (tests state machine + PostgreSQL lookups).

2. **Flash sale proposals**

   - Ask: `"What flash sale proposals do I have this week?"`
   - Bot should call `flash_sale_manager` (list) and summarise proposals.
   - Confirm tool execution by inspecting `trace.tool_calls` in response.
   - Accept one: `"Accept proposal 12"` → verifies write path + status transitions.

3. **Inventory analytics**

   - Prompt: `"Show me inventory stats for my account"` → triggers `analytics_data` with `supplier_inventory`.
   - Follow-up: `"Any items close to expiry I should discount?"` → should reuse context from prior tool output.

4. **RAG knowledge retrieval (Milvus)**

   - Ask: `"What packaging guidelines do we follow for leafy greens?"`
   - Response should cite knowledge chunk loaded from `Context.pdf`; check `trace.tool_calls` for `vector_search` and confirm retrieved snippet.

5. **Customer flow & ordering**

   - Start new session `demo-2`: `"Hi, I'm shopping as a new customer"`.
   - Bot gathers account info → respond `"I'm new"` to skip account lookup.
   - Ask: `"Recommend a fruit combo with current best prices"` → expects `analytics_data` or `data_access` usage for pricing insights.
   - Request order: `"Place an order for 5kg of Valencia oranges"` → agent should confirm required details (tests intent classifier + order workflow prompts).

6. **Competitor price intelligence**

   - Prompt (supplier session): `"Compare competitor prices for Product-0005"` (use actual UUID from data generation output via `SupplierProductRepository` logs or `check` command).
   - Ensures `analytics_data` handles `price_comparison` branch.

7. **Image generation**

   - Command: `"Create a marketing image for our guava flash sale"`.
   - Agent should call `image_generator`, save PNG to `data/images/guava.png`, and reply with the file path.

8. **Vector sanity check**
   - Ask odd fact: `"Summarise the sourcing policy"` → expect RAG response; follow with `"Where did that info come from?"` to see if agent references retrieved metadata.

Each step exercises classifier-first routing, relational reads/writes, analytics, flash sale tooling, Milvus RAG, and Gemini image generation. Inspect logs or the `trace` payload to confirm intended tool usage.

## Operational Notes

- The FastAPI layer applies a sliding-window rate limiter (60 req/min per session/IP). Handle `429` responses by backing off.
- Milvus host/port default to `localhost:19530`; adjust via environment if running remotely.
- Data generation script calls `Tortoise.generate_schemas()` for clean installs. For production, manage migrations with Aerich (`uv run aerich upgrade`).
- Vector ingestion expects `data/Context.pdf`; add your own knowledge base and rerun `generate_data insert` with `overwrite=True` behaviour.

## Future Improvements

- **Cloud native deployment**: Containerise FastAPI + background workers, deploy to Kubernetes with managed PostgreSQL (Cloud SQL/Aurora) and Milvus (Zilliz Cloud) for horizontal scaling.
- **Event-driven toolchain**: Offload long-running image generation and flash sale updates to message queues (e.g., Pub/Sub) with callback webhooks.
- **Session scalability**: Persist session state/history in Redis or DynamoDB to serve thousands of concurrent chats without exhausting API memory.
- **Observability**: Add OpenTelemetry tracing around tool calls and vector searches; feed logs into dashboards for quality monitoring.
- **Fine-grained access**: Introduce OAuth for supplier/customer authentication and enforce role-based data access on repos.

## Support

- Check existing data: `uv run -m app.utils.generate_data check`
- Reset Milvus collection: rerun `insert` after deleting the container volume.
- File issues or feature ideas via GitHub Issues in this repository.
