"""
Configuration management for the Ad Headline Optimizer.
Handles environment variables and application settings.
"""

import os
from typing import Optional
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field
except ImportError:
    from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    app_name: str = Field(default="Ad Headline Optimizer", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", env="OPENAI_MODEL")
    openai_temperature: float = Field(default=0.7, env="OPENAI_TEMPERATURE")
    openai_max_tokens: int = Field(default=1000, env="OPENAI_MAX_TOKENS")
    
    # Anthropic Configuration (Fallback)
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-3-haiku-20240307", env="ANTHROPIC_MODEL")
    
    # Gemini Configuration (Google)
    gemini_api_key: Optional[str] = Field(default=None, env="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-pro", env="GEMINI_MODEL")
    gemini_temperature: float = Field(default=0.7, env="GEMINI_TEMPERATURE")
    gemini_max_tokens: int = Field(default=1000, env="GEMINI_MAX_TOKENS")
    
    # Database Configuration
    database_url: str = Field(
        default="sqlite:///./headline_optimizer.db", 
        env="DATABASE_URL"
    )
    sqlite_url: str = Field(
        default="sqlite:///./headline_optimizer.db", 
        env="SQLITE_URL"
    )
    
    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    redis_ttl: int = Field(default=86400, env="REDIS_TTL")
    
    # Rate Limiting
    rate_limit_per_hour: int = Field(default=100, env="RATE_LIMIT_PER_HOUR")
    rate_limit_burst: int = Field(default=10, env="RATE_LIMIT_BURST")
    
    # Model Configuration
    model_path: str = Field(default="./models/ctr_model.pkl", env="MODEL_PATH")
    feature_cache_ttl: int = Field(default=3600, env="FEATURE_CACHE_TTL")
    
    # Training Configuration
    training_samples: int = Field(default=10000, env="TRAINING_SAMPLES")
    optimization_trials: int = Field(default=20, env="OPTIMIZATION_TRIALS")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    
    # A/B Testing Configuration
    default_impressions: int = Field(default=10000, env="DEFAULT_IMPRESSIONS")
    significance_level: float = Field(default=0.05, env="SIGNIFICANCE_LEVEL")
    min_sample_size: int = Field(default=1000, env="MIN_SAMPLE_SIZE")
    
    # Cost Tracking
    cost_per_token_openai: float = Field(default=0.00003, env="COST_PER_TOKEN_OPENAI")
    cost_per_token_anthropic: float = Field(default=0.000015, env="COST_PER_TOKEN_ANTHROPIC")
    max_cost_per_request: float = Field(default=0.10, env="MAX_COST_PER_REQUEST")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_database_url() -> str:
    """Get the appropriate database URL based on environment."""
    if settings.database_url.startswith("postgresql://"):
        return settings.database_url
    return settings.sqlite_url


def validate_api_keys() -> bool:
    """Validate that at least one LLM API key is configured."""
    return bool(settings.openai_api_key or settings.anthropic_api_key or settings.gemini_api_key)
