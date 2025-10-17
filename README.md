# KcartBot

Conversational commerce copilot that helps customers shop smarter and suppliers run pricing, inventory, and flash-sale workflows. The bot orchestrates a Gemini LLM, Milvus vector search, and a PostgreSQL operational store behind a FastAPI service.

## Quick Start

Ensure you have Python 3.11+, PostgreSQL, Milvus running, and API keys set up as described in Prerequisites.

### Using uv (Recommended)

1. Clone the repository and navigate to the project directory:

   ```powershell
   git clone https://github.com/AbrehamGebremedhin/KcartBot.git
   cd KcartBot
   ```

2. Install dependencies:

   ```powershell
   uv sync
   ```

3. Set up environment variables in `.env`:

   ```dotenv
   DEEPSEEK_API_KEY=your-deepseek-key
   GEMINI_API_KEY=your-google-genai-key
   DATABASE_URL=postgres://kcartbot:kcartbot@localhost:5432/kcartbot
   ```

   The chat orchestration now uses DeepSeek (`DEEPSEEK_API_KEY`), while Gemini remains available for image generation and data utilities.

4. Provision databases:

   ```powershell
   docker run --name kcartbot-postgres -e POSTGRES_DB=kcartbot -e POSTGRES_USER=kcartbot -e POSTGRES_PASSWORD=kcartbot -p 5432:5432 -d postgres:16

   docker run --name kcartbot-milvus -p 19530:19530 -p 9091:9091 -d milvusdb/milvus:v2.4.3-20240813-74fbddb7
   ```

5. Generate seed data:

   ```powershell
   uv run -m app.utils.generate_data insert
   ```

6. Start the application:
   ```powershell
   uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

### Using pip

1. Clone the repository and navigate to the project directory:

   ```powershell
   git clone https://github.com/AbrehamGebremedhin/KcartBot.git
   cd KcartBot
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Set up environment variables in `.env`:

   ```dotenv
   DEEPSEEK_API_KEY=your-deepseek-key
   GEMINI_API_KEY=your-google-genai-key
   DATABASE_URL=postgres://kcartbot:kcartbot@localhost:5432/kcartbot
   ```

   The chat orchestration now uses DeepSeek (`DEEPSEEK_API_KEY`), while Gemini remains available for image generation and data utilities.

4. Provision databases:

   ```powershell
   docker run --name kcartbot-postgres -e POSTGRES_DB=kcartbot -e POSTGRES_USER=kcartbot -e POSTGRES_PASSWORD=kcartbot -p 5432:5432 -d postgres:16

   docker run --name kcartbot-milvus -p 19530:19530 -p 9091:9091 -d milvusdb/milvus:v2.4.3-20240813-74fbddb7
   ```

5. Generate seed data:

   ```powershell
   python -m app.utils.generate_data insert
   ```

6. Start the application:
   ```powershell
   python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

## Prerequisites

- Python 3.13 (recommended) or 3.11+
- [uv](https://docs.astral.sh/uv/) package/dependency manager (`pip install uv`)
- Running services:
  - PostgreSQL 15+ (`postgres://` connection string)
  - Milvus standalone 2.4.x (vector search)
- Google Gemini API key with access to `text-embedding-004` and `gemini-2.5-flash-image`

## Architecture

![KcartBot architecture diagram](architecture.png)

_Figure: KcartBot system architecture — FastAPI edge, LangChain agent + Gemini LLM, Milvus RAG, PostgreSQL operational store, and typed tool adapters._

Why this layout?

- **Single agent brain with explicit tools** keeps Gemini focused on reasoning while LangChain enforces intent-first routing and structured tool use.
- **Milvus RAG layer** grounds advisory answers (product education, policy reminders) with curated PDF knowledge, avoiding hallucinations.
- **PostgreSQL OLTP store** models transactions, suppliers, inventory, and pricing history so supplier and customer actions mutate durable records.
- **Tools as thin, typed adapters** (analytics, flash sales, vector search, image generation) isolate external systems, making it simple to swap implementations (e.g., different vector DB or BI backend) later.

## Data Flow

The system processes user interactions through a structured data flow designed for conversational commerce:

1. **User Interaction**: Users send messages via HTTP POST requests to `/api/v1/chat`, including a `session_id` and `message` payload.

2. **API Gateway**: FastAPI serves as the rate-limited REST edge, validating requests and routing them to the ChatService.

3. **Session Management**: ChatService handles session state, user authentication flows, and maintains conversational context across interactions.

4. **Agent Orchestration**: ChatService asynchronously invokes the LangChain Agent, which uses the DeepSeek LLM for intent classification and reasoning.

5. **Tool Execution**: The Agent calls specialized tools based on user intent:

   - **Image Generation**: Image Tool invokes Gemini API to create marketing visuals, saving outputs to `data/images/`.
   - **Flash Sales**: FlashSaleTool interacts with PostgreSQL flash_sale tables for proposals and updates.
   - **Knowledge Retrieval**: VectorSearch Tool queries Milvus for RAG-based answers from embedded knowledge.
   - **Analytics & Data**: Data/Analytics Tools perform queries on PostgreSQL via Tortoise ORM for pricing, inventory, and competitor data.

6. **Response Synthesis**: The Agent combines tool outputs with LLM-generated responses to provide coherent, context-aware replies.

7. **Data Persistence**:

   - Operational data (users, products, transactions, inventory) is stored in PostgreSQL.
   - Vector embeddings for RAG are maintained in Milvus.
   - Session state is managed in-memory (consider Redis for scalability).

8. **Response Delivery**: Final responses are streamed back through the API to the user client.

This flow ensures secure, efficient handling of conversational commerce tasks while isolating concerns across layers.

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

## Example Conversations

For real-world examples of customer and supplier interactions, see the `customer.json` and `supplier.json` files in the project root. These contain full conversation flows demonstrating multilingual support, intent classification, tool execution, and seamless dialogue handling.

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
