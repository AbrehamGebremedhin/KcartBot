# Analytics script to analyze the generated mock data
# Provides insights into sales patterns, seasonal trends, and product performance

import asyncio
from datetime import datetime, timedelta
from collections import defaultdict
from tortoise import Tortoise
from app.core.config import get_settings
from app.db.models import (
    Product, User, Transaction, OrderItem, CompetitorPrice,
    SupplierProduct, UserRole, TransactionStatus
)


async def analyze_sales_by_season():
    """Analyze sales performance by season for each product"""
    print("\n" + "="*80)
    print("SALES ANALYSIS BY SEASON")
    print("="*80)
    
    products = await Product.all()
    
    for product in products:
        # Get all order items for this product
        order_items = await OrderItem.filter(product_id=product.product_id).all()
        
        if not order_items:
            continue
        
        # Get associated transactions
        order_ids = [item.order_id for item in order_items]
        transactions = await Transaction.filter(order_id__in=order_ids).all()
        
        # Create transaction lookup
        trans_lookup = {t.order_id: t for t in transactions}
        
        # Separate in-season vs out-of-season sales
        in_season_sales = []
        out_season_sales = []
        
        for item in order_items:
            trans = trans_lookup.get(item.order_id)
            if not trans:
                continue
            
            month = trans.date.month
            is_in = is_in_season(month, product.in_season_start, product.in_season_end)
            
            if is_in:
                in_season_sales.append(item)
            else:
                out_season_sales.append(item)
        
        if in_season_sales or out_season_sales:
            print(f"\n{product.product_name_en} ({product.in_season_start} - {product.in_season_end})")
            
            if in_season_sales:
                in_qty = sum(item.quantity for item in in_season_sales)
                in_rev = sum(item.subtotal for item in in_season_sales)
                in_avg_price = sum(item.price_per_unit for item in in_season_sales) / len(in_season_sales)
                print(f"  In-Season:  {len(in_season_sales):3d} orders, {in_qty:7.2f} {product.unit.value}, "
                      f"{in_rev:10.2f} ETB, Avg: {in_avg_price:.2f} ETB/{product.unit.value}")
            
            if out_season_sales:
                out_qty = sum(item.quantity for item in out_season_sales)
                out_rev = sum(item.subtotal for item in out_season_sales)
                out_avg_price = sum(item.price_per_unit for item in out_season_sales) / len(out_season_sales)
                print(f"  Out-Season: {len(out_season_sales):3d} orders, {out_qty:7.2f} {product.unit.value}, "
                      f"{out_rev:10.2f} ETB, Avg: {out_avg_price:.2f} ETB/{product.unit.value}")
            
            if in_season_sales and out_season_sales:
                in_avg = sum(item.price_per_unit for item in in_season_sales) / len(in_season_sales)
                out_avg = sum(item.price_per_unit for item in out_season_sales) / len(out_season_sales)
                price_diff = ((out_avg - in_avg) / in_avg) * 100
                demand_diff = ((len(in_season_sales) - len(out_season_sales)) / len(out_season_sales)) * 100
                print(f"  Analysis: Out-season prices {price_diff:+.1f}% vs in-season, "
                      f"demand {demand_diff:+.1f}% higher in-season")


async def analyze_monthly_trends():
    """Analyze sales trends by month"""
    print("\n" + "="*80)
    print("MONTHLY SALES TRENDS")
    print("="*80)
    
    transactions = await Transaction.all()
    
    # Group by month
    monthly_data = defaultdict(lambda: {'orders': 0, 'revenue': 0, 'items': 0})
    
    for trans in transactions:
        month_key = trans.date.strftime("%Y-%m")
        monthly_data[month_key]['orders'] += 1
        monthly_data[month_key]['revenue'] += trans.total_price
    
    # Get order items per month
    for month_key in monthly_data:
        year, month = map(int, month_key.split('-'))
        start_date = datetime(year, month, 1).date()
        if month == 12:
            end_date = datetime(year + 1, 1, 1).date()
        else:
            end_date = datetime(year, month + 1, 1).date()
        
        trans_in_month = [t for t in transactions if start_date <= t.date < end_date]
        order_ids = [t.order_id for t in trans_in_month]
        item_count = await OrderItem.filter(order_id__in=order_ids).count()
        monthly_data[month_key]['items'] = item_count
    
    # Print sorted by month
    print(f"\n{'Month':<10} {'Orders':>8} {'Items':>8} {'Revenue':>12} {'Avg Order':>10}")
    print("-" * 60)
    
    for month_key in sorted(monthly_data.keys()):
        data = monthly_data[month_key]
        avg_order = data['revenue'] / data['orders'] if data['orders'] > 0 else 0
        print(f"{month_key:<10} {data['orders']:>8} {data['items']:>8} "
              f"{data['revenue']:>12.2f} {avg_order:>10.2f}")
    
    total_orders = sum(d['orders'] for d in monthly_data.values())
    total_revenue = sum(d['revenue'] for d in monthly_data.values())
    total_items = sum(d['items'] for d in monthly_data.values())
    
    print("-" * 60)
    print(f"{'TOTAL':<10} {total_orders:>8} {total_items:>8} {total_revenue:>12.2f} "
          f"{total_revenue/total_orders:>10.2f}")


async def analyze_customer_segments():
    """Analyze customer behavior and segmentation"""
    print("\n" + "="*80)
    print("CUSTOMER SEGMENTATION ANALYSIS")
    print("="*80)
    
    customers = await User.filter(role=UserRole.CUSTOMER).all()
    
    customer_stats = []
    
    for customer in customers:
        transactions = await Transaction.filter(user_id=customer.user_id).all()
        
        if not transactions:
            segment = "Inactive"
            total_spent = 0
            order_count = 0
            avg_order = 0
        else:
            order_count = len(transactions)
            total_spent = sum(t.total_price for t in transactions)
            avg_order = total_spent / order_count
            
            # Segment customers
            if order_count >= 15:
                segment = "VIP"
            elif order_count >= 8:
                segment = "Regular"
            elif order_count >= 3:
                segment = "Occasional"
            else:
                segment = "New"
        
        customer_stats.append({
            'name': customer.name,
            'segment': segment,
            'orders': order_count,
            'total_spent': total_spent,
            'avg_order': avg_order,
            'location': customer.default_location
        })
    
    # Group by segment
    segments = defaultdict(list)
    for stat in customer_stats:
        segments[stat['segment']].append(stat)
    
    # Print summary by segment
    for segment in ["VIP", "Regular", "Occasional", "New", "Inactive"]:
        if segment not in segments:
            continue
        
        customers_in_seg = segments[segment]
        count = len(customers_in_seg)
        total_orders = sum(c['orders'] for c in customers_in_seg)
        total_revenue = sum(c['total_spent'] for c in customers_in_seg)
        avg_revenue = total_revenue / count if count > 0 else 0
        
        print(f"\n{segment} Customers ({count})")
        print(f"  Total Orders: {total_orders}")
        print(f"  Total Revenue: {total_revenue:.2f} ETB")
        print(f"  Avg Revenue/Customer: {avg_revenue:.2f} ETB")
        
        # Top 3 customers in segment
        if customers_in_seg:
            top_3 = sorted(customers_in_seg, key=lambda x: x['total_spent'], reverse=True)[:3]
            print(f"  Top Customers:")
            for i, cust in enumerate(top_3, 1):
                print(f"    {i}. {cust['name']}: {cust['orders']} orders, {cust['total_spent']:.2f} ETB")


async def analyze_price_competitiveness():
    """Compare our prices with competitor prices"""
    print("\n" + "="*80)
    print("PRICE COMPETITIVENESS ANALYSIS")
    print("="*80)
    
    products = await Product.all()
    
    current_date = datetime.now().date()
    week_ago = current_date - timedelta(days=7)
    
    print(f"\nComparing base prices with recent competitor prices (past 7 days)\n")
    print(f"{'Product':<20} {'Our Base':>10} {'Local Shop':>12} {'Supermarket':>12} {'Distrib Ctr':>12}")
    print("-" * 70)
    
    for product in products[:10]:  # Limit to 10 for readability
        # Get recent competitor prices
        comp_prices = await CompetitorPrice.filter(
            product_id=product.product_id,
            date__gte=week_ago
        ).all()
        
        if not comp_prices:
            continue
        
        # Average by tier
        tier_prices = defaultdict(list)
        for cp in comp_prices:
            tier_prices[cp.tier.value].append(cp.price_etb_per_kg)
        
        local_avg = sum(tier_prices.get('Local_Shop', [0])) / len(tier_prices.get('Local_Shop', [1])) if tier_prices.get('Local_Shop') else 0
        super_avg = sum(tier_prices.get('Supermarket', [0])) / len(tier_prices.get('Supermarket', [1])) if tier_prices.get('Supermarket') else 0
        distrib_avg = sum(tier_prices.get('Distribution_Center', [0])) / len(tier_prices.get('Distribution_Center', [1])) if tier_prices.get('Distribution_Center') else 0
        
        print(f"{product.product_name_en:<20} {product.base_price_etb:>10.2f} "
              f"{local_avg:>12.2f} {super_avg:>12.2f} {distrib_avg:>12.2f}")


def is_in_season(month, in_season_start, in_season_end):
    """Check if a given month is within the product's season"""
    months = ["January", "February", "March", "April", "May", "June", 
              "July", "August", "September", "October", "November", "December"]
    
    start_idx = months.index(in_season_start)
    end_idx = months.index(in_season_end)
    month_idx = month - 1
    
    if start_idx <= end_idx:
        return start_idx <= month_idx <= end_idx
    else:
        return month_idx >= start_idx or month_idx <= end_idx


async def run_all_analytics():
    """Run all analytics reports"""
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
    print("KCARTBOT MOCK DATA ANALYTICS")
    print("="*80)
    
    try:
        await analyze_monthly_trends()
        await analyze_customer_segments()
        await analyze_sales_by_season()
        await analyze_price_competitiveness()
        
        print("\n" + "="*80)
        print("ANALYTICS COMPLETE")
        print("="*80)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await Tortoise.close_connections()


if __name__ == "__main__":
    asyncio.run(run_all_analytics())
