# Mock data generator for the users table in KCartBot database
import random
from datetime import datetime, timedelta
import asyncio
from tortoise import Tortoise, run_async
from app.db.models import User, PreferredLanguage, UserRole
from app.core.config import get_settings

# Ethiopian phone numbers and common names
ETHIOPIAN_FIRST_NAMES = [
    "Abebe", "Almaz", "Bekele", "Berhane", "Chaltu", "Dawit", "Eleni", "Fikre",
    "Girma", "Hanna", "Tadesse", "Kebede", "Lemlem", "Mekdes", "Negash", "Sara",
    "Tesfaye", "Tigist", "Yohannes", "Zeritu", "Mulugeta", "Alem", "Seble", "Haile"
]

ETHIOPIAN_LAST_NAMES = [
    "Alemu", "Bekele", "Gebre", "Haile", "Kebede", "Lemma", "Mengistu", "Negash",
    "Tadesse", "Tesfaye", "Wolde", "Yohannes", "Desta", "Getachew", "Mulugeta", "Abera"
]

ADDIS_LOCATIONS = [
    "Bole", "Piassa", "Merkato", "4 Kilo", "6 Kilo", "CMC", "Megenagna",
    "Gerji", "Arat Kilo", "Kazanchis", "Mexico", "Lideta", "Nifas Silk",
    "Kolfe", "Yeka", "Arada", "Addis Ketema", "Kirkos", "Akaki Kality"
]

def generate_ethiopian_phone():
    """Generate a valid Ethiopian phone number"""
    prefixes = ["0911", "0912", "0913", "0914", "0921", "0923", "0924", "0925"]
    return f"{random.choice(prefixes)}{random.randint(100000, 999999)}"

def generate_mock_users(num_customers=30, num_suppliers=15):
    """Generate mock user data with both customers and suppliers"""
    mock_data = []
    used_phones = set()
    
    # Generate customers
    for i in range(num_customers):
        while True:
            phone = generate_ethiopian_phone()
            if phone not in used_phones:
                used_phones.add(phone)
                break
        
        first_name = random.choice(ETHIOPIAN_FIRST_NAMES)
        last_name = random.choice(ETHIOPIAN_LAST_NAMES)
        name = f"{first_name} {last_name}"
        
        # Customers joined over the past 2 years
        days_ago = random.randint(1, 730)
        joined_date = (datetime.now() - timedelta(days=days_ago)).date()
        
        mock_data.append({
            "name": name,
            "phone": phone,
            "default_location": random.choice(ADDIS_LOCATIONS),
            "preferred_language": random.choice(["English", "Amharic"]),
            "role": "customer",
            "joined_date": joined_date,
        })
    
    # Generate suppliers
    supplier_business_names = [
        "Fresh Farm Supplies", "Green Valley Products", "Highland Produce Co.",
        "Sheger Fresh Foods", "Entoto Organic Farm", "Rift Valley Harvest",
        "Awash Dairy Products", "Tana Fresh Fruits", "Merkato Wholesale",
        "Addis Fresh Market", "Bole Produce Hub", "Unity Farm Supplies",
        "Golden Harvest Co.", "Nature's Best Ethiopia", "Prime Agro Suppliers"
    ]
    
    for i in range(num_suppliers):
        while True:
            phone = generate_ethiopian_phone()
            if phone not in used_phones:
                used_phones.add(phone)
                break
        
        # Suppliers typically joined earlier (1-3 years ago)
        days_ago = random.randint(365, 1095)
        joined_date = (datetime.now() - timedelta(days=days_ago)).date()
        
        mock_data.append({
            "name": supplier_business_names[i] if i < len(supplier_business_names) else f"Supplier {i+1}",
            "phone": phone,
            "default_location": random.choice(ADDIS_LOCATIONS),
            "preferred_language": random.choice(["English", "Amharic"]),
            "role": "supplier",
            "joined_date": joined_date,
        })
    
    return mock_data


async def insert_mock_users():
    """Insert mock users into the database"""
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
    
    data = generate_mock_users()
    count = 0
    for item in data:
        await User.create(
            name=item["name"],
            phone=item["phone"],
            default_location=item["default_location"],
            preferred_language=PreferredLanguage(item["preferred_language"]),
            role=UserRole(item["role"]),
            joined_date=item["joined_date"]
        )
        count += 1
    
    print(f"Successfully inserted {count} users ({sum(1 for u in data if u['role'] == 'customer')} customers, {sum(1 for u in data if u['role'] == 'supplier')} suppliers)")
    await Tortoise.close_connections()


def print_mock_users():
    """Print mock users as JSON"""
    import json
    data = generate_mock_users()
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "insert":
        run_async(insert_mock_users())
    else:
        print_mock_users()
