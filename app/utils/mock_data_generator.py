# Master script to generate all mock data in the correct order
# This ensures data dependencies are maintained (users before transactions, etc.)
import asyncio
from tortoise import Tortoise
from app.core.config import get_settings

# Import all generator modules
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.utils.mock_user_data_generate import insert_mock_users
from app.utils.mock_supplier_product_data_generate import insert_mock_supplier_products
from app.utils.mock_competitor_price_data_generate import insert_mock_competitor_prices
from app.utils.mock_transaction_data_generate import insert_mock_transactions_and_orders


async def generate_all_mock_data():
    """Generate all mock data in the correct dependency order"""
    
    print("="*80)
    print("KCARTBOT MOCK DATA GENERATION")
    print("="*80)
    
    print("\nThis will generate mock data for:")
    print("  1. Users (customers and suppliers)")
    print("  2. Supplier Products (inventory)")
    print("  3. Competitor Prices (market data)")
    print("  4. Transactions and Order Items (sales history)")
    print("\nNote: Product data should already exist (20 products)")
    print("="*80)
    
    # Initialize database connection
    print("\nInitializing database connection...")
    await Tortoise.init(config={
        "connections": {
            "default": get_settings().DATABASE_URL
        },
        "apps": {
            "models": {
                "models": ["app.db.models"],
                "default_connection": "default"
            }
        }
    })
    
    try:
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
    """Check what data already exists in the database"""
    from app.db.models import Product, User, SupplierProduct, CompetitorPrice, Transaction, OrderItem
    
    await Tortoise.init(config={
        "connections": {
            "default": get_settings().DATABASE_URL
        },
        "apps": {
            "models": {
                "models": ["app.db.models"],
                "default_connection": "default"
            }
        }
    })
    
    print("\n" + "="*80)
    print("CHECKING EXISTING DATA")
    print("="*80)
    
    product_count = await Product.all().count()
    user_count = await User.all().count()
    supplier_product_count = await SupplierProduct.all().count()
    competitor_price_count = await CompetitorPrice.all().count()
    transaction_count = await Transaction.all().count()
    order_item_count = await OrderItem.all().count()
    
    print(f"\nCurrent database state:")
    print(f"  Products: {product_count}")
    print(f"  Users: {user_count}")
    print(f"  Supplier Products: {supplier_product_count}")
    print(f"  Competitor Prices: {competitor_price_count}")
    print(f"  Transactions: {transaction_count}")
    print(f"  Order Items: {order_item_count}")
    
    if product_count == 0:
        print("\n⚠️  WARNING: No products found!")
        print("   Please run: uv run -m app.utils.mock_product_data_generate insert")
        print("   before running this script.")
    
    print("="*80)
    
    await Tortoise.close_connections()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        # Just check existing data
        asyncio.run(check_existing_data())
    elif len(sys.argv) > 1 and sys.argv[1] == "insert":
        # Generate all mock data
        asyncio.run(generate_all_mock_data())
    else:
        print("Usage:")
        print("  Check existing data: uv run -m app.utils.mock_data_generator check")
        print("  Generate all data:   uv run -m app.utils.mock_data_generator insert")
