from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from typing import List


class Settings(BaseSettings):
    """Application settings loaded from environment or .env file.

    Uses pydantic v2 style `model_config` so pydantic-settings will pick up
    the `.env` file and case sensitivity correctly.
    """

    gemini_api_key: str = Field(..., alias='GEMINI_API_KEY', description="API key for Gemini services")
    DATABASE_URL: str = Field(..., alias='DATABASE_URL', description="Database connection URL")

    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
    }

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    try:
        return Settings()
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Configuration error: {str(e)}")
        logger.error("Please check your .env file and ensure all required variables are set.")
        raise