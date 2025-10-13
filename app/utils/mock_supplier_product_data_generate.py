# Mock data generator for supplier_products table in KCartBot database
import random
from datetime import datetime, timedelta
import uuid
import asyncio
from tortoise import Tortoise, run_async
from app.db.models import User, Product, SupplierProduct, UnitType, SupplierProductStatus, UserRole
from app.core.config import get_settings


def get_price_with_markup(base_price, season_factor=1.0):
    """Calculate supplier price based on base price and season"""
    # Suppliers typically price 10-30% below base retail price
    markup = random.uniform(0.70, 0.90)
    price = base_price * markup * season_factor
    return round(price, 2)


def is_in_season(month, in_season_start, in_season_end):
    """Check if a given month is within the product's season"""
    months = ["January", "February", "March", "April", "May", "June", 
              "July", "August", "September", "October", "November", "December"]
    
    start_idx = months.index(in_season_start)
    end_idx = months.index(in_season_end)
    month_idx = month - 1  # Convert to 0-indexed
    
    if start_idx <= end_idx:
        return start_idx <= month_idx <= end_idx
    else:  # Season wraps around year end
        return month_idx >= start_idx or month_idx <= end_idx


async def generate_mock_supplier_products():
    """Generate mock supplier product inventory data"""
    # Get all suppliers
    suppliers = await User.filter(role=UserRole.SUPPLIER).all()
    
    # Get all products
    products = await Product.all()
    
    if not suppliers:
        print("No suppliers found. Please run mock_user_data_generate.py first.")
        return []
    
    if not products:
        print("No products found. Please run mock_product_data_generate.py first.")
        return []
    
    print(f"Found {len(suppliers)} suppliers and {len(products)} products")
    
    mock_data = []
    current_date = datetime.now().date()
    current_month = datetime.now().month
    
    # Each supplier carries 8-15 products
    for supplier in suppliers:
        num_products = random.randint(8, 15)
        supplier_products = random.sample(products, num_products)
        
        for product in supplier_products:
            # Determine if product is in season
            in_season = is_in_season(current_month, product.in_season_start, product.in_season_end)
            season_factor = 1.0 if in_season else random.uniform(1.2, 1.5)
            
            # Calculate pricing
            unit_price = get_price_with_markup(product.base_price_etb, season_factor)
            
            # Determine quantity available
            if in_season:
                quantity_available = round(random.uniform(50, 500), 2)
            else:
                quantity_available = round(random.uniform(10, 100), 2)
            
            # Determine status
            status_weights = {
                "active": 0.75,
                "on_sale": 0.15,
                "expired": 0.10
            }
            
            if not in_season:
                # Out of season products less likely to be on sale or active
                status_weights = {
                    "active": 0.60,
                    "on_sale": 0.10,
                    "expired": 0.30
                }
            
            status = random.choices(
                list(status_weights.keys()),
                weights=list(status_weights.values())
            )[0]
            
            # Set expiry date for perishables
            expiry_date = None
            if product.category in ["Fruit", "Vegetable", "Dairy"]:
                days_until_expiry = random.randint(3, 30)
                expiry_date = current_date + timedelta(days=days_until_expiry)
                
                # If expired status, set expiry in the past
                if status == "expired":
                    expiry_date = current_date - timedelta(days=random.randint(1, 10))
            
            # Delivery days (e.g., "Mon,Wed,Fri" or "Daily")
            delivery_patterns = [
                "Mon,Wed,Fri",
                "Tue,Thu,Sat",
                "Daily",
                "Mon,Tue,Wed,Thu,Fri",
                "Mon,Wed,Fri,Sat"
            ]
            available_delivery_days = random.choice(delivery_patterns)
            
            mock_data.append({
                "inventory_id": str(uuid.uuid4()),
                "supplier_id": supplier.user_id,
                "product_id": product.product_id,
                "quantity_available": quantity_available,
                "unit": product.unit.value,
                "unit_price_etb": unit_price,
                "expiry_date": expiry_date,
                "available_delivery_days": available_delivery_days,
                "status": status
            })
    
    return mock_data


async def insert_mock_supplier_products():
    """Insert mock supplier products into the database"""
    from app.db.repository.supplier_product_repository import SupplierProductRepository
    data = await generate_mock_supplier_products()
    count = 0
    for item in data:
        await SupplierProductRepository.create_supplier_product(
            inventory_id=item["inventory_id"],
            supplier_id=item["supplier_id"],
            product_id=item["product_id"],
            quantity_available=item["quantity_available"],
            unit=UnitType(item["unit"]),
            unit_price_etb=item["unit_price_etb"],
            expiry_date=item["expiry_date"],
            available_delivery_days=item["available_delivery_days"],
            status=SupplierProductStatus(item["status"])
        )
        count += 1
    print(f"Successfully inserted {count} supplier product entries")


async def print_mock_supplier_products():
    """Print mock supplier products as JSON"""
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
    
    import json
    data = await generate_mock_supplier_products()
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
    await Tortoise.close_connections()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "insert":
        run_async(insert_mock_supplier_products())
    else:
        run_async(print_mock_supplier_products())
