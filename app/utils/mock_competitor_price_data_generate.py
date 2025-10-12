# Mock data generator for competitor_prices table in KCartBot database
import random
from datetime import datetime, timedelta
import uuid
import asyncio
from tortoise import Tortoise, run_async
from app.db.models import Product, CompetitorPrice, CompetitorTier
from app.core.config import get_settings

# Competitor locations in Addis Ababa
COMPETITOR_LOCATIONS = {
    "Local_Shop": [
        "Bole Local Market", "Piassa Corner Shop", "Merkato Small Vendor",
        "CMC Neighborhood Store", "Gerji Mini Market", "Mexico Local Shop",
        "Kazanchis Corner Store", "Megenagna Small Market"
    ],
    "Supermarket": [
        "Shoa Supermarket", "Bambis Supermarket", "Betelhem Supermarket",
        "Wegagen Supermarket", "Salit Supermarket", "Alpha Supermarket"
    ],
    "Distribution_Center": [
        "Merkato Wholesale Center", "Piassa Distribution Hub",
        "Addis Ketema Wholesale", "Akaki Distribution Center"
    ]
}


def get_competitor_price_factor(tier):
    """Get price multiplier based on competitor tier"""
    factors = {
        "Local_Shop": (1.15, 1.35),  # Local shops charge 15-35% more
        "Supermarket": (1.05, 1.20),  # Supermarkets charge 5-20% more
        "Distribution_Center": (0.85, 1.00)  # Distribution centers charge 15% less to same as base
    }
    return random.uniform(*factors[tier])


async def generate_mock_competitor_prices():
    """Generate historical competitor price data over the past year"""
    products = await Product.all()
    
    if not products:
        print("No products found. Please run mock_product_data_generate.py first.")
        return []
    
    print(f"Generating competitor prices for {len(products)} products")
    
    mock_data = []
    current_date = datetime.now().date()
    
    # Generate price data for the past 365 days
    # Sample prices weekly (52 weeks) for each product and tier combination
    for weeks_ago in range(52):
        price_date = current_date - timedelta(weeks=weeks_ago)
        month = price_date.month
        
        for product in products:
            # Determine if product was in season at that time
            in_season = is_in_season(month, product.in_season_start, product.in_season_end)
            
            # Base seasonal adjustment
            if in_season:
                season_factor = random.uniform(0.90, 1.0)  # Slightly cheaper in season
            else:
                season_factor = random.uniform(1.1, 1.4)  # More expensive out of season
            
            # Generate prices for each competitor tier
            for tier in ["Local_Shop", "Supermarket", "Distribution_Center"]:
                # Not all tiers report prices every week
                if random.random() < 0.7:  # 70% chance of having data for this week
                    tier_factor = get_competitor_price_factor(tier)
                    price_per_kg = round(product.base_price_etb * season_factor * tier_factor, 2)
                    
                    # Add some random variation (±5%)
                    price_per_kg *= random.uniform(0.95, 1.05)
                    price_per_kg = round(price_per_kg, 2)
                    
                    location = random.choice(COMPETITOR_LOCATIONS[tier])
                    
                    mock_data.append({
                        "id": str(uuid.uuid4()),
                        "product_id": product.product_id,
                        "tier": tier,
                        "date": price_date,
                        "price_etb_per_kg": price_per_kg,
                        "source_location": location
                    })
    
    return mock_data


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


async def insert_mock_competitor_prices():
    """Insert mock competitor prices into the database"""
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
    
    data = await generate_mock_competitor_prices()
    count = 0
    
    for item in data:
        await CompetitorPrice.create(
            id=item["id"],
            product_id=item["product_id"],
            tier=CompetitorTier(item["tier"]),
            date=item["date"],
            price_etb_per_kg=item["price_etb_per_kg"],
            source_location=item["source_location"]
        )
        count += 1
    
    print(f"Successfully inserted {count} competitor price records")
    await Tortoise.close_connections()


async def print_mock_competitor_prices():
    """Print mock competitor prices as JSON (limited to first 100 for readability)"""
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
    data = await generate_mock_competitor_prices()
    # Limit output for readability
    print(json.dumps(data[:100], ensure_ascii=False, indent=2, default=str))
    print(f"\n... and {len(data) - 100} more records")
    await Tortoise.close_connections()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "insert":
        run_async(insert_mock_competitor_prices())
    else:
        run_async(print_mock_competitor_prices())
