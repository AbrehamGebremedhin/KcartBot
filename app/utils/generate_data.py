# Master script to generate all mock data in the correct order
# This ensures data dependencies are maintained (products -> users -> supplier products -> competitor prices -> transactions)
import asyncio
from tortoise import Tortoise
from app.core.config import get_settings
from app.core.tortoise_config import TORTOISE_ORM

# Import all generator modules
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.utils.mock_product_data_generate import insert_mock_products
from app.utils.mock_user_data_generate import insert_mock_users
from app.utils.mock_supplier_product_data_generate import insert_mock_supplier_products
from app.utils.mock_competitor_price_data_generate import insert_mock_competitor_prices
from app.utils.mock_transaction_data_generate import insert_mock_transactions_and_orders

# Import dataloader for Milvus population
from app.utils.dataloader import load_context_pdf

# Import repositories for data checking
from app.db.repository.product_repository import ProductRepository
from app.db.repository.user_repository import UserRepository
from app.db.repository.supplier_product_repository import SupplierProductRepository
from app.db.repository.competitor_price_repository import CompetitorPriceRepository
from app.db.repository.transaction_repository import TransactionRepository
from app.db.repository.order_item_repository import OrderItemRepository


async def ensure_database_setup():
    """Ensure database exists and schema is up to date"""
    import asyncpg
    from urllib.parse import urlparse
    
    settings = get_settings()
    db_url = settings.DATABASE_URL
    
    # Convert postgres:// to postgresql:// for asyncpg compatibility
    asyncpg_url = db_url.replace('postgres://', 'postgresql://', 1)
    
    # Parse database URL
    parsed = urlparse(asyncpg_url)
    db_name = parsed.path.lstrip('/')
    
    # Connection URL without database name (connect to default 'postgres' db)
    admin_url = f"{parsed.scheme}://{parsed.netloc}/postgres"
    
    print("\n" + "="*80)
    print("DATABASE SETUP CHECK")
    print("="*80)
    
    try:
        # Connect to postgres database to check if our database exists
        conn = await asyncpg.connect(admin_url)
        
        # Check if database exists
        db_exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", db_name
        )
        
        if not db_exists:
            print(f"\n⚠️  Database '{db_name}' does not exist. Creating...")
            await conn.execute(f'CREATE DATABASE "{db_name}"')
            print(f"✓ Database '{db_name}' created successfully!")
        else:
            print(f"\n✓ Database '{db_name}' already exists")
        
        await conn.close()
        
        # Now check if tables exist (schema)
        test_conn = await asyncpg.connect(asyncpg_url)
        tables_exist = await test_conn.fetchval(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'"
        )
        await test_conn.close()
        
        if tables_exist == 0:
            print(f"\n⚠️  No tables found. Running Aerich migrations...")
            print("   This will initialize the database schema...")
            
            # Run aerich init-db using Tortoise generate_schemas
            from tortoise import Tortoise
            await Tortoise.init(config=TORTOISE_ORM)
            await Tortoise.generate_schemas()
            await Tortoise.close_connections()
            
            print("✓ Database schema created successfully!")
        else:
            print(f"✓ Database schema exists ({tables_exist} tables found)")
        
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error during database setup: {e}")
        print("   Please ensure PostgreSQL is running and credentials are correct.")
        raise


async def generate_all_mock_data():
    """Generate all mock data in the correct dependency order"""
    
    print("="*80)
    print("KCARTBOT MOCK DATA GENERATION")
    print("="*80)
    
    print("\nThis will generate mock data for:")
    print("  0. Products (20 products - fruits, vegetables, dairy)")
    print("  1. Users (customers and suppliers)")
    print("  2. Supplier Products (inventory)")
    print("  3. Competitor Prices (market data)")
    print("  4. Transactions and Order Items (sales history)")
    print("  5. Vector Database (Milvus context embeddings)")
    print("="*80)
    
    # Step -1: Ensure database exists and schema is set up
    await ensure_database_setup()
    
    # Initialize database connection
    print("\nInitializing database connection...")
    await Tortoise.init(config=TORTOISE_ORM)
    
    try:
        # Step 0: Generate Products (FIRST - all other data depends on this)
        print("\n" + "="*80)
        print("STEP 0: Generating Products (Foundation Data)")
        print("="*80)
        await insert_mock_products()
        
        # Step 1: Generate Users
        print("\n" + "="*80)
        print("STEP 1: Generating Users")
        print("="*80)
        await insert_mock_users()
        
        # Step 2: Generate Supplier Products
        print("\n" + "="*80)
        print("STEP 2: Generating Supplier Product Inventory")
        print("="*80)
        await insert_mock_supplier_products()
        
        # Step 3: Generate Competitor Prices
        print("\n" + "="*80)
        print("STEP 3: Generating Competitor Price History")
        print("="*80)
        await insert_mock_competitor_prices()
        
        # Step 4: Generate Transactions and Order Items
        print("\n" + "="*80)
        print("STEP 4: Generating Transaction History and Order Items")
        print("="*80)
        await insert_mock_transactions_and_orders()
        
        # Step 5: Load Context PDF to Milvus Vector Database
        print("\n" + "="*80)
        print("STEP 5: Loading Context Data to Milvus Vector Database")
        print("="*80)
        
        # Get the path to Context.pdf
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        context_pdf_path = os.path.join(project_root, "data", "Context.pdf")
        
        if os.path.exists(context_pdf_path):
            print(f"Loading PDF: {context_pdf_path}")
            try:
                milvus_stats = await load_context_pdf(
                    pdf_path=context_pdf_path,
                    collection_name="KCartBot",
                    overwrite=True,  # Overwrite existing collection
                )
                print(f"\n✓ Successfully loaded context to Milvus:")
                print(f"  - Chunks created: {milvus_stats['num_chunks']}")
                print(f"  - Total characters: {milvus_stats['total_characters']}")
                print(f"  - Embedding dimension: {milvus_stats['embedding_dimension']}")
                print(f"  - Collection: {milvus_stats['collection_name']}")
            except Exception as e:
                print(f"\n⚠️  Warning: Failed to load context to Milvus: {e}")
                print("   You can load it manually later using: uv run -m app.utils.dataloader")
        else:
            print(f"\n⚠️  Context PDF not found at: {context_pdf_path}")
            print("   Skipping Milvus population. Add Context.pdf to data/ directory and run again.")
        
        print("\n" + "="*80)
        print("ALL MOCK DATA GENERATED SUCCESSFULLY!")
        print("="*80)
        print("\nYou can now use this data to:")
        print("  - Analyze sales patterns by product and season")
        print("  - Compare pricing across different time periods")
        print("  - Identify best-selling products")
        print("  - Test recommendation algorithms")
        print("  - Generate reports and visualizations")
        
    except Exception as e:
        print(f"\n❌ Error during mock data generation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await Tortoise.close_connections()


async def check_existing_data():
    """Check what data already exists in the database using repositories"""
    
    # Ensure database exists first
    try:
        await ensure_database_setup()
    except Exception as e:
        print(f"\n❌ Cannot check data - database setup failed: {e}")
        return
    
    await Tortoise.init(config=TORTOISE_ORM)
    
    print("\n" + "="*80)
    print("CHECKING EXISTING DATA")
    print("="*80)
    
    # Use repositories to count data
    products = await ProductRepository.list_products()
    product_count = len(products)
    
    users = await UserRepository.list_users()
    user_count = len(users)
    
    supplier_products = await SupplierProductRepository.list_supplier_products()
    supplier_product_count = len(supplier_products)
    
    competitor_prices = await CompetitorPriceRepository.list_competitor_prices()
    competitor_price_count = len(competitor_prices)
    
    transactions = await TransactionRepository.list_transactions()
    transaction_count = len(transactions)
    
    order_items = await OrderItemRepository.list_order_items()
    order_item_count = len(order_items)
    
    print(f"\nPostgreSQL Database:")
    print(f"  Products: {product_count}")
    print(f"  Users: {user_count}")
    print(f"  Supplier Products: {supplier_product_count}")
    print(f"  Competitor Prices: {competitor_price_count}")
    print(f"  Transactions: {transaction_count}")
    print(f"  Order Items: {order_item_count}")
    
    # Check Milvus
    print(f"\nMilvus Vector Database:")
    try:
        from app.db.milvus_handler import MilvusHandler
        milvus = MilvusHandler()
        await milvus.connect()
        
        collection_name = "KCartBot"
        if milvus.collection_exists(collection_name):
            # Get collection info
            from pymilvus import Collection
            collection = Collection(collection_name)
            num_entities = collection.num_entities
            print(f"  Collection '{collection_name}': {num_entities} embeddings")
        else:
            print(f"  Collection '{collection_name}': Not found")
        
        milvus.disconnect()
    except Exception as e:
        print(f"  Status: Unable to connect ({str(e)[:50]}...)")
    
    if product_count == 0:
        print("\n⚠️  No products found - will be created during data generation.")
    else:
        print(f"\n✓ Found {product_count} products in database")
    
    print("="*80)
    
    await Tortoise.close_connections()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        # Just check existing data
        asyncio.run(check_existing_data())
    elif len(sys.argv) > 1 and sys.argv[1] == "insert":
        # Generate all mock data (including products)
        asyncio.run(generate_all_mock_data())
    else:
        print("Usage:")
        print("  Check existing data: uv run -m app.utils.generate_data check")
        print("  Generate all data:   uv run -m app.utils.generate_data insert")
        print("\nNote: This will generate:")
        print("  - 20 Products (fruits, vegetables, dairy)")
        print("  - 30 Customers + 15 Suppliers")
        print("  - Supplier inventory data")
        print("  - Competitor price history")
        print("  - Transaction and order history")
        print("  - Milvus vector database (Context.pdf embeddings)")
        print("\nRequirements:")
        print("  - PostgreSQL database running")
        print("  - Milvus vector database running (optional)")
        print("  - data/Context.pdf file present (for Milvus)")
