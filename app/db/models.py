from enum import Enum
# Month Enum

# Month Enum with names
class Month(str, Enum):
	JANUARY = "January"
	FEBRUARY = "February"
	MARCH = "March"
	APRIL = "April"
	MAY = "May"
	JUNE = "June"
	JULY = "July"
	AUGUST = "August"
	SEPTEMBER = "September"
	OCTOBER = "October"
	NOVEMBER = "November"
	DECEMBER = "December"
# Tortoise ORM models for KcartBot

from tortoise import fields, models

# Enums for fields
class PreferredLanguage(str, Enum):
	ENGLISH = "English"
	AMHARIC = "Amharic"

class UserRole(str, Enum):
	CUSTOMER = "customer"
	SUPPLIER = "supplier"

class ProductCategory(str, Enum):
	VEGETABLE = "Vegetable"
	FRUIT = "Fruit"
	DAIRY = "Dairy"

class UnitType(str, Enum):
	KG = "kg"
	LITER = "liter"

class SupplierProductStatus(str, Enum):
	ACTIVE = "active"
	EXPIRED = "expired"
	ON_SALE = "on_sale"

class CompetitorTier(str, Enum):
	LOCAL_SHOP = "Local_Shop"
	SUPERMARKET = "Supermarket"
	DISTRIBUTION_CENTER = "Distribution_Center"

class PaymentMethod(str, Enum):
	COD = "COD"

class TransactionStatus(str, Enum):
	PENDING = "Pending"
	CONFIRMED = "Confirmed"
	DELIVERED = "Delivered"
	CANCELLED = "Cancelled"

class User(models.Model):
	user_id = fields.IntField(pk=True)
	name = fields.CharField(max_length=100)
	phone = fields.CharField(max_length=20, unique=True)
	default_location = fields.CharField(max_length=100)
	preferred_language = fields.CharEnumField(enum_type=PreferredLanguage)
	role = fields.CharEnumField(enum_type=UserRole)
	joined_date = fields.DateField(auto_now_add=True)
	created_at = fields.DatetimeField(auto_now_add=True)

	class Meta:
		table = "users"

class Product(models.Model):
	product_id = fields.UUIDField(pk=True)
	product_name_en = fields.CharField(max_length=100)
	product_name_am = fields.CharField(max_length=100)
	product_name_am_latin = fields.CharField(max_length=100)
	category = fields.CharEnumField(enum_type=ProductCategory)
	unit = fields.CharEnumField(enum_type=UnitType)
	base_price_etb = fields.FloatField()
	in_season_start = fields.CharEnumField(enum_type=Month)
	in_season_end = fields.CharEnumField(enum_type=Month)
	image_url = fields.CharField(max_length=255, null=True)
	created_at = fields.DatetimeField(auto_now_add=True)

	class Meta:
		table = "products"

class SupplierProduct(models.Model):
	inventory_id = fields.UUIDField(pk=True)
	supplier = fields.ForeignKeyField("models.User", related_name="supplier_products")
	product = fields.ForeignKeyField("models.Product", related_name="supplier_products")
	quantity_available = fields.FloatField()
	unit = fields.CharEnumField(enum_type=UnitType)
	unit_price_etb = fields.FloatField()
	expiry_date = fields.DateField(null=True)
	available_delivery_days = fields.CharField(max_length=50, null=True)
	last_updated = fields.DatetimeField(auto_now=True)
	status = fields.CharEnumField(enum_type=SupplierProductStatus)

	class Meta:
		table = "supplier_products"

class CompetitorPrice(models.Model):
	id = fields.UUIDField(pk=True)
	product = fields.ForeignKeyField("models.Product", related_name="competitor_prices")
	tier = fields.CharEnumField(enum_type=CompetitorTier)
	date = fields.DateField()
	price_etb_per_kg = fields.FloatField()
	source_location = fields.CharField(max_length=100)
	created_at = fields.DatetimeField(auto_now_add=True)

	class Meta:
		table = "competitor_prices"

class Transaction(models.Model):
	order_id = fields.UUIDField(pk=True)
	user = fields.ForeignKeyField("models.User", related_name="transactions")
	date = fields.DateField()
	delivery_date = fields.DateField(null=True)
	total_price = fields.FloatField()
	payment_method = fields.CharEnumField(enum_type=PaymentMethod)
	status = fields.CharEnumField(enum_type=TransactionStatus)
	created_at = fields.DatetimeField(auto_now_add=True)

	class Meta:
		table = "transactions"


# Restore OrderItem Model
class OrderItem(models.Model):
	id = fields.UUIDField(pk=True)
	order = fields.ForeignKeyField("models.Transaction", related_name="order_items")
	product = fields.ForeignKeyField("models.Product", related_name="order_items")
	supplier = fields.ForeignKeyField("models.User", related_name="order_items", null=True)
	quantity = fields.FloatField()
	unit = fields.CharEnumField(enum_type=UnitType)
	price_per_unit = fields.FloatField()
	subtotal = fields.FloatField()

	class Meta:
		table = "order_items"

# FlashSaleStatus Enum
class FlashSaleStatus(str, Enum):
	PROPOSED = "proposed"
	SCHEDULED = "scheduled"
	ACTIVE = "active"
	EXPIRED = "expired"
	CANCELLED = "cancelled"

# FlashSale Model
class FlashSale(models.Model):
	id = fields.IntField(pk=True)
	supplier_product = fields.ForeignKeyField("models.SupplierProduct", related_name="flash_sales", null=True)
	supplier = fields.ForeignKeyField("models.User", related_name="flash_sales")
	product = fields.ForeignKeyField("models.Product", related_name="flash_sales")
	start_date = fields.DatetimeField()
	end_date = fields.DatetimeField()
	discount_percent = fields.FloatField()
	status = fields.CharEnumField(enum_type=FlashSaleStatus)
	auto_generated = fields.BooleanField(default=False)
	created_at = fields.DatetimeField(auto_now_add=True)
	updated_at = fields.DatetimeField(auto_now=True)

	class Meta:
		table = "flash_sales"
