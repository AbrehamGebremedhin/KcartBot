# KCartBot Data Generation Guide

## Overview

The `generate_data.py` script is a comprehensive data population tool that sets up all required data for the KCartBot application, including both relational database (PostgreSQL) and vector database (Milvus).

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  generate_data.py                           │
│                 (Master Orchestrator)                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Repository Pattern Layer                        │
│  (ProductRepository, UserRepository, etc.)                   │
└─────────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
┌───────────────────────┐       ┌──────────────────────┐
│   PostgreSQL Database │       │  Milvus Vector DB    │
│   (Relational Data)   │       │  (Embeddings)        │
└───────────────────────┘       └──────────────────────┘
```

## Data Generation Flow

### Step 0: Products (Foundation)

- **Count**: 20 products
- **Categories**: Fruits, Vegetables, Dairy
- **Purpose**: Core product catalog that all other data references
- **Examples**: Avocado, Tomato, Milk, Cheese, etc.

### Step 1: Users

- **Count**: 45 users (30 customers + 15 suppliers)
- **Data**: Ethiopian names, phone numbers, locations in Addis Ababa
- **Purpose**: User base for transactions and supplier inventory

### Step 2: Supplier Products

- **Purpose**: Inventory data linking suppliers to products
- **Features**: Quantities, prices, expiry dates, delivery schedules
- **Auto-generated**: Flash sales for expiring products

### Step 3: Competitor Prices

- **Purpose**: Market price intelligence
- **Data**: Historical pricing from different competitor tiers
- **Tiers**: Local Shop, Supermarket, Distribution Center

### Step 4: Transactions & Order Items

- **Purpose**: Sales history and order data
- **Features**:
  - Multi-item orders
  - Seasonal purchase patterns
  - Price variations over time

### Step 5: Milvus Vector Database

- **Source**: `data/Context.pdf`
- **Purpose**: Context embeddings for RAG (Retrieval Augmented Generation)
- **Process**:
  1. Extract text from PDF
  2. Split into semantic chunks
  3. Generate embeddings using Google Gemini
  4. Store in Milvus for similarity search
- **Collection**: `KCartBot`

## Usage

### Check Existing Data

```bash
uv run -m app.utils.generate_data check
```

**Output Example:**

```
================================================================================
CHECKING EXISTING DATA
================================================================================

PostgreSQL Database:
  Products: 20
  Users: 45
  Supplier Products: 150
  Competitor Prices: 2400
  Transactions: 500
  Order Items: 1200

Milvus Vector Database:
  Collection 'KCartBot': 156 embeddings

✓ Found 20 products in database
================================================================================
```

### Generate All Data

```bash
uv run -m app.utils.generate_data insert
```

This will:

1. ✅ Populate all PostgreSQL tables
2. ✅ Create Milvus vector embeddings
3. ✅ Set up complete test environment

## Key Features

### 1. Repository Pattern

- All database operations go through repository layer
- No direct model access in generation script
- Clean separation of concerns
- Easy to test and maintain

### 2. Automatic Database Setup

- **Automatically checks if database exists** - creates it if missing
- **Auto-generates schema** - creates tables if they don't exist
- **No manual migration needed** - handles schema setup automatically
- **Idempotent** - safe to run multiple times

### 3. Dependency Management

- Products created first (foundation)
- Users created before transactions
- Proper foreign key relationships maintained
- No orphaned records

### 4. Error Handling

- Graceful degradation if Milvus unavailable
- Clear error messages
- Traceback for debugging
- Transaction rollback on failures

### 5. Flexibility

- Milvus population is optional (continues if Milvus unavailable)
- Can check data without modifying
- Overwrite option for Milvus collection
- Configurable parameters

## Requirements

### Database Requirements

- ✅ PostgreSQL server running (database will be auto-created if it doesn't exist)
- ✅ Milvus vector database running (optional)

### File Requirements

- ✅ `data/Context.pdf` must exist for Milvus population

### Configuration

All configuration comes from `app.core.config.get_settings()`:

- `DATABASE_URL`: PostgreSQL connection (e.g., `postgresql://user:pass@localhost:5432/kcartbot`)
- `gemini_api_key`: For embedding generation

**Note:** The script will automatically:

1. Create the database if it doesn't exist
2. Generate the schema (tables) if they're missing
3. No manual `aerich` commands needed!

## Data Statistics

| Data Type         | Count    | Purpose        |
| ----------------- | -------- | -------------- |
| Products          | 20       | Core catalog   |
| Customers         | 30       | Buyers         |
| Suppliers         | 15       | Sellers        |
| Supplier Products | ~150     | Inventory      |
| Competitor Prices | ~2,400   | Market data    |
| Transactions      | Variable | Sales history  |
| Order Items       | Variable | Order details  |
| Vector Embeddings | Variable | Context search |

## Integration with Other Tools

### Used By

- `app/main.py` - Main application startup
- `app/services/lllm_service.py` - Uses Milvus for context retrieval
- `app/tools/access_data.py` - Queries generated data
- `app/tools/search_context.py` - Searches Milvus embeddings

### Uses

- `mock_product_data_generate.py` - Product generation
- `mock_user_data_generate.py` - User generation
- `mock_supplier_product_data_generate.py` - Supplier inventory
- `mock_competitor_price_data_generate.py` - Price history
- `mock_transaction_data_generate.py` - Transaction history
- `dataloader.py` - Milvus population

## Troubleshooting

### PostgreSQL Connection Error

```
Error: Could not connect to PostgreSQL
```

**Solution**: Ensure PostgreSQL is running and `DATABASE_URL` is correct in config.

### Milvus Connection Error

```
Warning: Failed to load context to Milvus
```

**Solution**: Script continues without Milvus. Start Milvus server or load manually later:

```bash
uv run -m app.utils.dataloader
```

### Context.pdf Not Found

```
⚠️  Context PDF not found at: d:\Projects\KcartBot\data\Context.pdf
```

**Solution**: Add `Context.pdf` to the `data/` directory and run again.

### Duplicate Key Errors

```
Error: duplicate key value violates unique constraint
```

**Solution**: Clear existing data first or use repository methods that handle duplicates.

## Best Practices

1. **Run in Clean Environment**: Start with empty databases for consistent results
2. **Check Before Insert**: Always run `check` command first
3. **Monitor Output**: Watch for warnings or errors during generation
4. **Verify Milvus**: If using RAG features, ensure Milvus data loaded successfully
5. **Backup Data**: Before regenerating, backup if you have custom data

## Future Enhancements

- [ ] Add command-line arguments for customization (e.g., number of users)
- [ ] Support multiple PDF files for Milvus
- [ ] Add data validation after generation
- [ ] Create sample queries to test generated data
- [ ] Add option to generate data incrementally
- [ ] Support for different embedding models
