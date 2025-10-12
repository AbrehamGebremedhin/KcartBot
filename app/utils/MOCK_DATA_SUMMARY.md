# Mock Data Generation Summary

## Successfully Generated Mock Data for KcartBot

### Overview

Complete historical sales and market data has been generated for the KcartBot system, simulating one year of operations with realistic Ethiopian market patterns.

### Data Generated

#### 1. Users (45 total)

- **30 Customers** with Ethiopian names and phone numbers
  - Joined over the past 2 years
  - Located across Addis Ababa neighborhoods
  - Language preferences: English and Amharic
- **15 Suppliers** representing local businesses
  - Farm suppliers, wholesale centers, and produce companies
  - Joined 1-3 years ago
  - Serving various Addis Ababa locations

#### 2. Supplier Products (171 entries)

- Each supplier carries 8-15 products from the 20-product catalog
- Pricing reflects seasonal availability:
  - In-season products: 10-30% below retail base price
  - Out-of-season products: 20-50% markup
- Includes:
  - Current inventory quantities
  - Expiry dates for perishables
  - Delivery schedules (daily, weekly patterns)
  - Status (active, on_sale, expired)

#### 3. Competitor Prices (2,196 records)

- **52 weeks** of historical price data
- **3 market tiers**:
  - Local Shops (15-35% above base)
  - Supermarkets (5-20% above base)
  - Distribution Centers (0-15% below base)
- Covers all 20 products
- Reflects seasonal price variations

#### 4. Transaction History (840 orders)

- **One year** of customer purchase history
- Order distribution:
  - New customers: 1-2 orders/month
  - Regular customers: 2-4 orders/month
- **4,304 order items** across all transactions
- Order statuses:
  - 92% Delivered
  - 5% Confirmed/Pending
  - 3% Cancelled

### Sales Insights from Generated Data

#### Top 10 Best-Selling Products

| Rank | Product      | Orders | Quantity Sold | Revenue (ETB) | Avg Price |
| ---- | ------------ | ------ | ------------- | ------------- | --------- |
| 1    | Beetroot     | 238    | 1,185.39 kg   | 54,762.53     | 46.25     |
| 2    | Milk         | 237    | 690.55 L      | 55,800.73     | 80.76     |
| 3    | Potato       | 231    | 1,122.98 kg   | 54,079.85     | 48.63     |
| 4    | Green Pepper | 230    | 1,442.35 kg   | 181,195.56    | 125.64    |
| 5    | Cabbage      | 229    | 1,115.06 kg   | 60,689.35     | 54.63     |
| 6    | Pumpkin      | 222    | 1,061.67 kg   | 47,802.85     | 45.16     |
| 7    | Tomato       | 222    | 1,115.91 kg   | 33,559.69     | 30.17     |
| 8    | Orange       | 221    | 1,349.43 kg   | 159,406.45    | 118.67    |
| 9    | Banana       | 219    | 1,289.98 kg   | 108,217.88    | 83.35     |
| 10   | Avocado      | 218    | 1,300.48 kg   | 59,545.86     | 45.95     |

### Key Features of the Data

#### Realistic Patterns

1. **Seasonal Demand**

   - Products show 2-4x higher demand during their season
   - Out-of-season demand drops to 30-80% of normal

2. **Weekly Patterns**

   - Friday-Saturday: Peak shopping days (1.5-2x normal)
   - Sunday: Moderate activity (1.0-1.3x normal)
   - Weekdays: Baseline activity

3. **Holiday Spikes**

   - Ethiopian New Year (September): 2-3x demand
   - Christmas (January 7): 1.8-2.5x demand
   - Easter (April): 1.8-2.5x demand
   - Timkat (January 19): 1.5-2x demand

4. **Price Variation**
   - Seasonal pricing (in-season vs out-of-season)
   - Market tier differences (local shops more expensive)
   - Weekly price fluctuations (±5%)

### Use Cases Enabled

This data enables comprehensive analytics:

1. **Sales Analysis**

   - Identify trending products
   - Analyze revenue by product category
   - Track order frequency and customer loyalty

2. **Seasonal Intelligence**

   - Determine optimal pricing by season
   - Forecast demand for inventory planning
   - Identify best times to promote products

3. **Competitive Analysis**

   - Compare prices across market tiers
   - Track competitor pricing trends
   - Optimize pricing strategy

4. **Customer Insights**

   - Segment customers by purchase behavior
   - Identify high-value customers
   - Predict customer needs

5. **Supplier Performance**
   - Evaluate supplier reliability
   - Analyze supplier price competitiveness
   - Optimize supplier selection

### Database Statistics

```
Current database state:
  Products: 20
  Users: 45 (30 customers + 15 suppliers)
  Supplier Products: 171
  Competitor Prices: 2,196
  Transactions: 840
  Order Items: 4,304
```

### Files Created

All generator scripts are in `app/utils/`:

1. `mock_user_data_generate.py` - User/customer data
2. `mock_supplier_product_data_generate.py` - Inventory data
3. `mock_competitor_price_data_generate.py` - Market price data
4. `mock_transaction_data_generate.py` - Sales history
5. `mock_data_generator.py` - Master generator script
6. `README.md` - Complete documentation

### Next Steps

With this data, you can now:

1. **Build Analytics Dashboards**

   - Sales trends over time
   - Product performance comparisons
   - Revenue forecasting

2. **Develop Recommendation Systems**

   - Based on purchase history
   - Seasonal product suggestions
   - Price-based recommendations

3. **Test AI Features**

   - Demand prediction models
   - Price optimization algorithms
   - Customer behavior analysis

4. **Create Reports**
   - Monthly sales reports
   - Supplier performance reviews
   - Market analysis reports

### Regenerating Data

To regenerate or add more data:

```bash
# Check current state
uv run -m app.utils.mock_data_generator check

# Regenerate all (may need to clear tables first)
uv run -m app.utils.mock_data_generator insert

# Regenerate specific datasets
uv run -m app.utils.mock_user_data_generate insert
uv run -m app.utils.mock_transaction_data_generate insert
```

---

**Generated**: October 12, 2025  
**Data Period**: October 12, 2024 - October 12, 2025 (1 year)  
**Total Records**: 7,576 records across all tables
