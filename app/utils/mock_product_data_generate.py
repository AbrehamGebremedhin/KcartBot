# Mock data generator for the product table in KCartBot Postgres database
import random
from datetime import datetime
import uuid
import asyncio
from tortoise import Tortoise, run_async
from app.db.models import Product, ProductCategory, UnitType, Month
from app.core.config import get_settings

PRODUCTS = [
	{"en": "Avocado", "am": "አቮካዶ", "latin": "Avokado", "category": "Fruit", "unit": "kg"},
	{"en": "Prickly pear", "am": "በለስ", "latin": "Beles", "category": "Fruit", "unit": "kg"},
	{"en": "Orange", "am": "ብርቱካን", "latin": "Birtukan", "category": "Fruit", "unit": "kg"},
	{"en": "Broccoli", "am": "ብሮኮሊ", "latin": "Broccoli", "category": "Vegetable", "unit": "kg"},
	{"en": "Cheese", "am": "አይብ", "latin": "Ayib", "category": "Dairy", "unit": "kg"},
	{"en": "Pumpkin", "am": "ዱባ", "latin": "Duba", "category": "Vegetable", "unit": "kg"},
	{"en": "Egg", "am": "እንቁላል", "latin": "Enqulal", "category": "Dairy", "unit": "kg"},
	{"en": "Yogurt", "am": "እርጎ", "latin": "Ergo", "category": "Dairy", "unit": "liter"},
	{"en": "Watermelon", "am": "ሃብሃብ", "latin": "Habhab", "category": "Fruit", "unit": "kg"},
	{"en": "Green Pepper", "am": "ካሪያ", "latin": "Kariya", "category": "Fruit", "unit": "kg"},
	{"en": "Onion", "am": "ቀይ ሽንኩርት", "latin": "Key Shinkurt", "category": "Vegetable", "unit": "kg"},
	{"en": "Beetroot", "am": "ቀይ ስር", "latin": "Keysir", "category": "Vegetable", "unit": "kg"},
	{"en": "Clarified butter / Ghee", "am": "ቂቤ", "latin": "Kibe", "category": "Dairy", "unit": "kg"},
	{"en": "Mango", "am": "ማንጎ", "latin": "Mango", "category": "Fruit", "unit": "kg"},
	{"en": "Milk", "am": "ወተት", "latin": "Wetet", "category": "Dairy", "unit": "liter"},
	{"en": "Banana", "am": "ሙዝ", "latin": "Muz", "category": "Fruit", "unit": "kg"},
	{"en": "Potato", "am": "ድንች", "latin": "Dinich", "category": "Vegetable", "unit": "kg"},
	{"en": "Cabbage", "am": "ጥቅል ጎመን", "latin": "Tekle Gomen", "category": "Vegetable", "unit": "kg"},
	{"en": "Kale", "am": "ጥቁር ጎመን", "latin": "Tekur Gomen", "category": "Vegetable", "unit": "kg"},
	{"en": "Tomato", "am": "ቲማቲም", "latin": "Timatim", "category": "Vegetable", "unit": "kg"},
]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
UNITS = {"Fruit": "kg", "Vegetable": "kg", "Dairy": "liter"}

# Cheese uses kg, not liter
SPECIAL_UNIT = {"cheese": "kg", "Egg": "kg", "Clarified butter / Ghee": "kg"}
import os

IMAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'images')
IMAGE_MAP = {
	"Avocado": "avocado.png",
	"Prickly pear": "beles.png",
	"Orange": "birtukan.png",
	"Broccoli": "broccoli.png",
	"Cheese": "ayib.png",
	"Pumpkin": "duba.png",
	"Egg": "enqulale.png",
	"Yogurt": "Ergo.png",
	"Watermelon": "habhab.png",
	"Green Pepper": "kariya.png",
	"Onion": "key_shnikurte.png",
	"Beetroot": "keysir.jpg",
	"Clarified butter / Ghee": "kibe.png",
	"Mango": "mango.png",
	"Milk": "milk.png",
	"Banana": "muze.png",
	"Potato": "potato.jpg",
	"Cabbage": "teklegomen.png",
	"Kale": "tekurgomen.png",
	"Tomato": "tomato.jpg",
}

def generate_image_url(product_en):
	fname = IMAGE_MAP.get(product_en, None)
	if fname:
		return f"data/images/{fname}"
	return "data/images/placeholder.png"

def generate_base_price(category):
	# Simple price ranges for mock data
	if category == "Fruit":
		return round(random.uniform(30, 120), 2)
	elif category == "Vegetable":
		return round(random.uniform(15, 60), 2)
	elif category == "Dairy":
		return round(random.uniform(40, 150), 2)
	else:
		return 50.0

def generate_season():
	start = random.choice(MONTHS)
	end_idx = (MONTHS.index(start) + random.randint(2, 5)) % 12
	end = MONTHS[end_idx]
	return start, end

def generate_mock_products():
	mock_data = []
	for prod in PRODUCTS:
		unit = prod["unit"]
		base_price = generate_base_price(prod["category"])
		in_season_start, in_season_end = generate_season()
		image_url = generate_image_url(prod["en"])
		created_at = datetime.now().isoformat(sep=' ', timespec='seconds')
		mock_data.append({
			"product_id": str(uuid.uuid4()),
			"product_name_en": prod["en"],
			"product_name_am": prod["am"],
			"product_name_am_latin": prod["latin"],
			"category": prod["category"],
			"unit": unit,
			"base_price_etb": base_price,
			"in_season_start": in_season_start,
			"in_season_end": in_season_end,
			"image_url": image_url,
			"created_at": created_at
		})
	return mock_data


async def insert_mock_products():
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
	await Tortoise.generate_schemas()
	data = generate_mock_products()
	for item in data:
		await Product.create(
			product_id=item["product_id"],
			product_name_en=item["product_name_en"],
			product_name_am=item["product_name_am"],
			product_name_am_latin=item["product_name_am_latin"],
			category=ProductCategory(item["category"]),
			unit=UnitType(item["unit"]),
			base_price_etb=item["base_price_etb"],
			in_season_start=Month(item["in_season_start"]),
			in_season_end=Month(item["in_season_end"]),
			image_url=item["image_url"],
			created_at=datetime.strptime(item["created_at"], "%Y-%m-%d %H:%M:%S")
		)
	await Tortoise.close_connections()

def print_mock_products():
	import json
	data = generate_mock_products()
	print(json.dumps(data, ensure_ascii=False, indent=2))

if __name__ == "__main__":
	import sys
	if len(sys.argv) > 1 and sys.argv[1] == "insert":
		run_async(insert_mock_products())
	else:
		print_mock_products()
