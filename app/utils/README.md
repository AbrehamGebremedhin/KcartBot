# Mock Data Generators for KcartBot

This directory contains mock data generation scripts for populating the KcartBot database with realistic test data.

## Overview

The mock data generators create realistic Ethiopian market data including:

- **Users**: Customers and suppliers with Ethiopian names and phone numbers
- **Supplier Products**: Inventory data with seasonal availability and pricing
- **Competitor Prices**: Historical price data from different market tiers (local shops, supermarkets, distribution centers)
- **Transactions & Order Items**: Realistic purchase history showing seasonal sales patterns and popular products

## Files

### Individual Generators

1. **`mock_product_data_generate.py`** (Already run - 20 products exist)

   - Generates product catalog with Ethiopian produce names
   - Includes English, Amharic (UTF-8), and Amharic (Latin) names
   - Sets seasonal availability and base pricing

2. **`mock_user_data_generate.py`**

   - Creates 30 customers and 15 suppliers
   - Uses realistic Ethiopian names and phone numbers
   - Assigns locations in Addis Ababa neighborhoods
   - Sets join dates over the past 2 years

3. **`mock_supplier_product_data_generate.py`**

   - Generates inventory for each supplier
   - Each supplier carries 8-15 products
   - Pricing varies by season (in-season vs out-of-season)
   - Includes expiry dates, delivery schedules, and status

4. **`mock_competitor_price_data_generate.py`**

   - Generates 1 year of historical price data
   - Covers 3 competitor tiers: Local Shops, Supermarkets, Distribution Centers
   - Weekly price sampling for all products
   - Prices reflect seasonal variations

5. **`mock_transaction_data_generate.py`**
   - Creates realistic purchase history for the past year
   - Customer order frequency varies by loyalty (new vs regular customers)
   - Order quantities and prices reflect:
     - Seasonal demand (higher in-season)
     - Day of week patterns (higher on weekends)
     - Ethiopian holidays (increased demand)
   - Generates detailed order items with supplier assignments
   - Includes comprehensive sales statistics

### Master Generator

**`mock_data_generator.py`**

- Runs all generators in the correct dependency order
- Provides progress reporting and statistics
- Includes data validation checks

## Usage

### Check Existing Data

Before generating new data, check what already exists:

```bash
uv run -m app.utils.mock_data_generator check
```

### Generate All Mock Data (Recommended)

Run all generators in the correct order:

```bash
uv run -m app.utils.mock_data_generator insert
```

This will:

1. Create 45 users (30 customers + 15 suppliers)
2. Generate ~150 supplier product entries
3. Create ~7,000+ competitor price records
4. Generate 200-400 transactions with 600-2,000+ order items

### Run Individual Generators

If you need to regenerate specific data:

```bash
# Users only
uv run -m app.utils.mock_user_data_generate insert

# Supplier products only
uv run -m app.utils.mock_supplier_product_data_generate insert

# Competitor prices only
uv run -m app.utils.mock_competitor_price_data_generate insert

# Transactions and orders only
uv run -m app.utils.mock_transaction_data_generate insert
```

### Preview Data (JSON output)

To see what data will be generated without inserting:

```bash
# Preview any generator without 'insert' flag
uv run -m app.utils.mock_user_data_generate
uv run -m app.utils.mock_transaction_data_generate
```

## Data Characteristics

### Seasonal Patterns

Products show realistic seasonal behavior:

- **In-season**: Lower prices (10-15% below base), higher demand (2-4x normal)
- **Out-of-season**: Higher prices (10-40% above base), lower demand (30-80% of normal)

### Sales Patterns

Transaction data reflects:

- **Weekly patterns**: Higher sales Friday-Saturday
- **Holiday spikes**: Ethiopian New Year, Christmas, Easter, Timkat
- **Customer loyalty**: Regular customers order 2-4x per month, new customers 1-2x per month

### Price Variations

Competitor prices vary by tier:

- **Local Shops**: 15-35% above base price
- **Supermarkets**: 5-20% above base price
- **Distribution Centers**: 0-15% below base price

### Top Selling Products (Example Output)

The transaction generator shows which products sold best:

```
Top 10 Best Selling Products:
1. Tomato: 245 orders, 1,234.50 kg sold, Revenue: 45,678.90 ETB
2. Onion: 238 orders, 1,156.30 kg sold, Revenue: 38,456.20 ETB
3. Potato: 221 orders, 1,089.40 kg sold, Revenue: 35,234.10 ETB
...
```

## Dependencies

All generators require:

- Products table populated (20 products from `mock_product_data_generate.py`)
- Tortoise ORM configured
- Database connection settings in `.env`

## Database Schema

Generators populate these tables:

- `users` - Customer and supplier accounts
- `products` - Product catalog (already populated)
- `supplier_products` - Current inventory
- `competitor_prices` - Market price intelligence
- `transactions` - Order headers
- `order_items` - Order line items

## Notes

- All generators use the existing 20 products in the system
- Dates span the past year (365 days) for historical analysis
- Ethiopian calendar holidays are approximated to Gregorian dates
- Phone numbers use valid Ethiopian mobile prefixes (091x, 092x)
- Locations are actual Addis Ababa neighborhoods
- All prices in Ethiopian Birr (ETB)

## Analytics Use Cases

This mock data enables:

1. **Seasonal Analysis**: Identify which products sell best in which months
2. **Price Optimization**: Compare your prices against competitors
3. **Demand Forecasting**: Predict future sales based on patterns
4. **Supplier Performance**: Evaluate which suppliers are used most
5. **Customer Segmentation**: Analyze customer loyalty and behavior
6. **Revenue Analysis**: Track sales trends over time
7. **Inventory Planning**: Determine optimal stock levels by season

## Analytics Script

An analytics script is included to analyze the generated data:

```bash
# Run comprehensive analytics
uv run -m app.utils.analyze_mock_data
```

This provides:

- **Monthly Sales Trends**: Orders, items, and revenue by month
- **Customer Segmentation**: VIP, Regular, Occasional, New, Inactive customers
- **Seasonal Sales Analysis**: In-season vs out-of-season performance for each product
- **Price Competitiveness**: Compare your prices with competitors across market tiers

The analytics reveal insights such as:

- Out-of-season prices typically 33-37% higher than in-season
- In-season demand typically 2-4x higher than out-of-season
- VIP customers (15+ orders) generate ~96% of total revenue
- Friday-Saturday are peak shopping days

## Troubleshooting

**Error: No products found**

- Run: `uv run -m app.utils.mock_product_data_generate insert`

**Error: Database connection failed**

- Check `.env` file has valid `DATABASE_URL`
- Ensure PostgreSQL is running

**Error: Unique constraint violation**

- Data already exists; you may need to clear tables first
- Or modify generators to check for existing records

## Future Enhancements

Potential additions:

- Customer reviews and ratings
- Product recommendations based on purchase history
- Supplier ratings and reliability metrics
- Delivery tracking and logistics data
- Payment transaction details
- Discount and promotion campaigns
