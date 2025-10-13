# Mock data generator for transactions and order_items tables in KCartBot database
# This generates realistic past order data showing sales patterns by product, price, and season
import random
from datetime import datetime, timedelta
import uuid
import asyncio
from tortoise import Tortoise, run_async
from app.db.models import (
    User, Product, SupplierProduct, Transaction, OrderItem,
    UserRole, UnitType, PaymentMethod, TransactionStatus
)
from app.core.config import get_settings


def is_in_season(date, in_season_start, in_season_end):
    """Check if a given date is within the product's season"""
    months = ["January", "February", "March", "April", "May", "June", 
              "July", "August", "September", "October", "November", "December"]
    
    month = date.month
    start_idx = months.index(in_season_start)
    end_idx = months.index(in_season_end)
    month_idx = month - 1  # Convert to 0-indexed
    
    if start_idx <= end_idx:
        return start_idx <= month_idx <= end_idx
    else:  # Season wraps around year end
        return month_idx >= start_idx or month_idx <= end_idx


def get_seasonal_demand_multiplier(date, product):
    """Get demand multiplier based on season and product type"""
    in_season = is_in_season(date, product.in_season_start, product.in_season_end)
    
    if in_season:
        # In season: higher demand, 2-4x normal
        return random.uniform(2.0, 4.0)
    else:
        # Out of season: lower demand, 0.3-0.8x normal
        return random.uniform(0.3, 0.8)


def get_day_of_week_multiplier(date):
    """Get demand multiplier based on day of week"""
    # 0=Monday, 6=Sunday
    day = date.weekday()
    
    if day in [4, 5]:  # Friday, Saturday - high shopping days
        return random.uniform(1.5, 2.0)
    elif day == 6:  # Sunday - medium shopping day
        return random.uniform(1.0, 1.3)
    else:  # Weekdays - normal
        return random.uniform(0.8, 1.1)


def get_holiday_multiplier(date):
    """Get demand multiplier for Ethiopian holidays"""
    # Simplified - check for major Ethiopian holidays (approximate dates)
    month, day = date.month, date.day
    
    # Ethiopian New Year (Meskel) - September 11
    if month == 9 and 5 <= day <= 15:
        return random.uniform(2.0, 3.0)
    
    # Ethiopian Christmas - January 7
    if month == 1 and 1 <= day <= 10:
        return random.uniform(1.8, 2.5)
    
    # Ethiopian Easter - variable, roughly April
    if month == 4 and 15 <= day <= 25:
        return random.uniform(1.8, 2.5)
    
    # Timkat - January 19
    if month == 1 and 15 <= day <= 22:
        return random.uniform(1.5, 2.0)
    
    return 1.0


async def generate_mock_transactions_and_orders():
    """Generate realistic transaction and order data over the past year"""
    # Get all customers
    customers = await User.filter(role=UserRole.CUSTOMER).all()
    
    # Get all products with their details
    products = await Product.all()
    
    # Get all suppliers
    suppliers = await User.filter(role=UserRole.SUPPLIER).all()
    
    if not customers or not products or not suppliers:
        print("Missing required data. Please run user and product generators first.")
        return [], []
    
    print(f"Generating orders for {len(customers)} customers, {len(products)} products")
    
    transactions = []
    order_items = []
    
    current_date = datetime.now().date()
    start_date = current_date - timedelta(days=365)  # Past year
    
    # Track product sales statistics for reporting
    product_sales_stats = {p.product_id: {
        'total_quantity': 0,
        'total_revenue': 0,
        'order_count': 0,
        'best_season': None,
        'avg_price': []
    } for p in products}
    
    # Generate orders for each customer based on their join date
    for customer in customers:
        # Customer order frequency: varies by customer loyalty
        # New customers: 1-2 orders per month
        # Regular customers: 2-4 orders per month
        
        customer_start = max(customer.joined_date, start_date)
        days_active = (current_date - customer_start).days
        
        if days_active < 30:
            orders_per_month = random.uniform(0.5, 1.5)  # New customers
        elif days_active < 90:
            orders_per_month = random.uniform(1.0, 2.5)  # Learning customers
        else:
            orders_per_month = random.uniform(2.0, 4.5)  # Regular customers
        
        num_orders = int((days_active / 30) * orders_per_month)
        
        # Generate orders for this customer
        for _ in range(num_orders):
            # Random order date within customer's active period
            days_offset = random.randint(0, days_active)
            order_date = customer_start + timedelta(days=days_offset)
            
            # Skip future dates
            if order_date > current_date:
                continue
            
            # Delivery date: 1-3 days after order
            delivery_date = order_date + timedelta(days=random.randint(1, 3))
            
            # Order status based on date
            days_since_order = (current_date - order_date).days
            if days_since_order > 7:
                # Old orders: mostly delivered, some cancelled
                status = random.choices(
                    ["Delivered", "Cancelled"],
                    weights=[0.92, 0.08]
                )[0]
            elif days_since_order > 3:
                # Recent orders: mix of statuses
                status = random.choices(
                    ["Delivered", "Confirmed", "Cancelled"],
                    weights=[0.70, 0.25, 0.05]
                )[0]
            else:
                # Very recent orders: pending or confirmed
                status = random.choices(
                    ["Pending", "Confirmed"],
                    weights=[0.40, 0.60]
                )[0]
            
            # Generate order items
            # Each order has 2-8 products
            num_items = random.randint(2, 8)
            selected_products = random.sample(products, num_items)
            
            items_for_order = []
            order_total = 0.0
            
            for product in selected_products:
                # Get demand multipliers
                seasonal_mult = get_seasonal_demand_multiplier(order_date, product)
                dow_mult = get_day_of_week_multiplier(order_date)
                holiday_mult = get_holiday_multiplier(order_date)
                
                # Combined demand affects quantity ordered
                demand_mult = seasonal_mult * dow_mult * holiday_mult
                
                # Base quantity varies by product type
                if product.category == "Dairy":
                    base_qty = random.uniform(1, 5)
                elif product.category == "Fruit":
                    base_qty = random.uniform(2, 10)
                else:  # Vegetable
                    base_qty = random.uniform(2, 8)
                
                quantity = round(base_qty * random.uniform(0.8, 1.2), 2)
                
                # Price calculation with seasonal variation
                in_season = is_in_season(order_date, product.in_season_start, product.in_season_end)
                
                if in_season:
                    # In season: lower prices
                    price_mult = random.uniform(0.85, 1.0)
                else:
                    # Out of season: higher prices
                    price_mult = random.uniform(1.1, 1.4)
                
                price_per_unit = round(product.base_price_etb * price_mult, 2)
                subtotal = round(quantity * price_per_unit, 2)
                order_total += subtotal
                
                # Select a random supplier for this product
                supplier = random.choice(suppliers)
                
                # Track statistics
                product_sales_stats[product.product_id]['total_quantity'] += quantity
                product_sales_stats[product.product_id]['total_revenue'] += subtotal
                product_sales_stats[product.product_id]['order_count'] += 1
                product_sales_stats[product.product_id]['avg_price'].append(price_per_unit)
                
                items_for_order.append({
                    "id": str(uuid.uuid4()),
                    "product_id": product.product_id,
                    "supplier_id": supplier.user_id,
                    "quantity": quantity,
                    "unit": product.unit.value,
                    "price_per_unit": price_per_unit,
                    "subtotal": subtotal
                })
            
            # Create transaction
            order_id = str(uuid.uuid4())
            transactions.append({
                "order_id": order_id,
                "user_id": customer.user_id,
                "date": order_date,
                "delivery_date": delivery_date,
                "total_price": round(order_total, 2),
                "payment_method": "COD",  # Currently only COD
                "status": status
            })
            
            # Add order items with reference to transaction
            for item in items_for_order:
                item["order_id"] = order_id
                order_items.append(item)
    
    # Print sales statistics
    print("\n=== SALES STATISTICS ===")
    print(f"Total Transactions: {len(transactions)}")
    print(f"Total Order Items: {len(order_items)}")
    
    # Find top selling products
    sorted_products = sorted(
        product_sales_stats.items(),
        key=lambda x: x[1]['order_count'],
        reverse=True
    )
    
    print("\nTop 10 Best Selling Products:")
    for i, (product_id, stats) in enumerate(sorted_products[:10], 1):
        product = next(p for p in products if p.product_id == product_id)
        avg_price = sum(stats['avg_price']) / len(stats['avg_price']) if stats['avg_price'] else 0
        print(f"{i}. {product.product_name_en}: "
              f"{stats['order_count']} orders, "
              f"{stats['total_quantity']:.2f} {product.unit.value} sold, "
              f"Revenue: {stats['total_revenue']:.2f} ETB, "
              f"Avg Price: {avg_price:.2f} ETB")
    
    return transactions, order_items


async def insert_mock_transactions_and_orders():
    """Insert mock transactions and order items into the database"""
    from app.db.repository.transaction_repository import TransactionRepository
    from app.db.repository.order_item_repository import OrderItemRepository
    transactions, order_items = await generate_mock_transactions_and_orders()
    transaction_count = 0
    for item in transactions:
        await TransactionRepository.create_transaction(
            order_id=item["order_id"],
            user_id=item["user_id"],
            date=item["date"],
            delivery_date=item["delivery_date"],
            total_price=item["total_price"],
            payment_method=PaymentMethod(item["payment_method"]),
            status=TransactionStatus(item["status"])
        )
        transaction_count += 1
    print(f"\nSuccessfully inserted {transaction_count} transactions")
    order_item_count = 0
    for item in order_items:
        await OrderItemRepository.create_order_item(
            id=item["id"],
            order_id=item["order_id"],
            product_id=item["product_id"],
            supplier_id=item["supplier_id"],
            quantity=item["quantity"],
            unit=UnitType(item["unit"]),
            price_per_unit=item["price_per_unit"],
            subtotal=item["subtotal"]
        )
        order_item_count += 1
    print(f"Successfully inserted {order_item_count} order items")


async def print_mock_transactions_and_orders():
    """Print sample mock transactions and order items as JSON"""
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
    transactions, order_items = await generate_mock_transactions_and_orders()
    
    # Print first 5 transactions with their items
    print("Sample Transactions:")
    for i, trans in enumerate(transactions[:5]):
        print(f"\nTransaction {i+1}:")
        print(json.dumps(trans, ensure_ascii=False, indent=2, default=str))
        
        # Find items for this transaction
        trans_items = [item for item in order_items if item["order_id"] == trans["order_id"]]
        print(f"  Items ({len(trans_items)}):")
        for item in trans_items:
            print(json.dumps(item, ensure_ascii=False, indent=4, default=str))
    
    print(f"\n... Generated {len(transactions)} total transactions with {len(order_items)} total items")
    
    await Tortoise.close_connections()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "insert":
        run_async(insert_mock_transactions_and_orders())
    else:
        run_async(print_mock_transactions_and_orders())
