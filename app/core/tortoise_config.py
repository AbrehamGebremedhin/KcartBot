import os
from tortoise import Tortoise
from app.core.config import get_settings

TORTOISE_ORM = {
    "connections": {
        "default": os.getenv(
            "DATABASE_URL",
            get_settings().DATABASE_URL
        )
    },
    "apps": {
        "models": {
            "models": [
                "app.db.models",
                "aerich.models"
            ],
            "default_connection": "default",
        }
    }
}


async def init_db():
    """Initialize Tortoise ORM with database connection"""
    await Tortoise.init(config=TORTOISE_ORM)
    await Tortoise.generate_schemas()


async def close_db():
    """Close all database connections"""
    await Tortoise.close_connections()
