"""
Arena Configuration Settings

This module loads and validates all Arena configuration from environment variables.
It provides a centralized, type-safe way to access configuration throughout the Arena system.

The settings are loaded from environment variables, with sensible defaults provided.
For production use, ensure all required settings (especially API keys) are properly configured.

Author: Homunculus Team
"""

from typing import Literal, Optional, Dict, Any
from pydantic import Field, validator
from pydantic_settings import BaseSettings
import os
from pathlib import Path


class ArenaSettings(BaseSettings):
    """
    Arena configuration settings loaded from environment variables.
    
    This class uses Pydantic for validation and type safety. Environment variables
    are automatically loaded, with the prefix ARENA_ being optional.
    
    Attributes are grouped by functional area for better organization.
    """
    
    # =========================================================================
    # Message Bus Settings (Kafka)
    # =========================================================================
    kafka_bootstrap_servers: str = Field(
        default="localhost:9092",
        description="Kafka bootstrap servers for message bus connection"
    )
    kafka_topic_prefix: str = Field(
        default="arena",
        description="Prefix for all Arena Kafka topics"
    )
    kafka_consumer_group: str = Field(
        default="arena-orchestrator",
        description="Consumer group ID for the orchestrator"
    )
    kafka_max_message_size: int = Field(
        default=1048576,  # 1MB
        description="Maximum message size in bytes"
    )
    
    # =========================================================================
    # Database Settings (PostgreSQL)
    # =========================================================================
    postgres_host: str = Field(
        default="localhost",
        description="PostgreSQL host address"
    )
    postgres_port: int = Field(
        default=5432,
        description="PostgreSQL port number"
    )
    postgres_db: str = Field(
        default="arena_db",
        description="PostgreSQL database name"
    )
    postgres_user: str = Field(
        default="arena_user",
        description="PostgreSQL username"
    )
    postgres_password: str = Field(
        default="arena_pass",
        description="PostgreSQL password"
    )
    postgres_max_connections: int = Field(
        default=20,
        description="Maximum number of database connections"
    )
    postgres_connection_timeout: int = Field(
        default=30,
        description="Database connection timeout in seconds"
    )
    
    @property
    def postgres_url(self) -> str:
        """Construct PostgreSQL connection URL."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    
    # =========================================================================
    # Cache Settings (Redis)
    # =========================================================================
    redis_host: str = Field(
        default="localhost",
        description="Redis host address"
    )
    redis_port: int = Field(
        default=6379,
        description="Redis port number"
    )
    redis_db: int = Field(
        default=1,
        ge=0,
        le=15,
        description="Redis database number (0-15)"
    )
    redis_password: Optional[str] = Field(
        default=None,
        description="Redis password if authentication is enabled"
    )
    redis_state_ttl: int = Field(
        default=3600,  # 1 hour
        description="TTL for game state in seconds"
    )
    redis_score_ttl: int = Field(
        default=300,  # 5 minutes
        description="TTL for score cache in seconds"
    )
    redis_message_ttl: int = Field(
        default=7200,  # 2 hours
        description="TTL for message cache in seconds"
    )
    
    @property
    def redis_url(self) -> str:
        """Construct Redis connection URL."""
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    # =========================================================================
    # LLM Settings
    # =========================================================================
    default_llm_provider: Literal["anthropic", "openai", "ollama"] = Field(
        default="anthropic",
        description="Default LLM provider to use"
    )
    
    # Anthropic Settings
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key for Claude"
    )
    anthropic_model: str = Field(
        default="claude-3-sonnet-20240229",
        description="Anthropic model to use"
    )
    anthropic_max_tokens: int = Field(
        default=4096,
        description="Maximum tokens for Anthropic responses"
    )
    anthropic_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for Anthropic model"
    )
    
    # OpenAI Settings (backup)
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key"
    )
    openai_model: str = Field(
        default="gpt-4-turbo-preview",
        description="OpenAI model to use"
    )
    openai_max_tokens: int = Field(
        default=4096,
        description="Maximum tokens for OpenAI responses"
    )
    openai_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for OpenAI model"
    )
    
    # Ollama Settings (local)
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama base URL"
    )
    ollama_model: str = Field(
        default="llama3.3:70b",
        description="Ollama model to use"
    )
    ollama_timeout: int = Field(
        default=120,
        description="Timeout for Ollama requests in seconds"
    )
    
    # LLM Rate Limiting
    llm_rate_limit_requests: int = Field(
        default=60,
        description="Maximum LLM requests per period"
    )
    llm_rate_limit_period: int = Field(
        default=60,
        description="Rate limit period in seconds"
    )
    
    # =========================================================================
    # Arena Game Settings
    # =========================================================================
    min_agents: int = Field(
        default=2,
        ge=2,
        description="Minimum number of agents in a game"
    )
    max_agents: int = Field(
        default=8,
        le=20,
        description="Maximum number of agents in a game"
    )
    default_agent_count: int = Field(
        default=4,
        description="Default number of agents if not specified"
    )
    
    # Turn Settings
    default_max_turns: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Default maximum turns per game"
    )
    min_turn_time: int = Field(
        default=5,
        description="Minimum time for a turn in seconds"
    )
    max_turn_time: int = Field(
        default=30,
        description="Maximum time for a turn in seconds"
    )
    turn_timeout: int = Field(
        default=60,
        description="Timeout before forcing next turn in seconds"
    )
    
    # Elimination Settings
    default_elimination_threshold: float = Field(
        default=-10.0,
        description="Default score threshold for elimination"
    )
    elimination_check_interval: int = Field(
        default=3,
        ge=1,
        description="Check for elimination every N turns"
    )
    min_agents_for_elimination: int = Field(
        default=3,
        ge=2,
        description="Minimum agents required to allow elimination"
    )
    
    # Scoring Weights (must sum to 1.0)
    score_weight_novelty: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Weight for novelty in scoring"
    )
    score_weight_builds: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Weight for building on others"
    )
    score_weight_solves: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Weight for solving subproblems"
    )
    score_weight_radical: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Weight for radical ideas"
    )
    score_weight_manipulation: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Weight for successful manipulation"
    )
    
    @validator('score_weight_manipulation')
    def validate_scoring_weights(cls, v, values):
        """Ensure scoring weights sum to approximately 1.0."""
        total = (
            values.get('score_weight_novelty', 0) +
            values.get('score_weight_builds', 0) +
            values.get('score_weight_solves', 0) +
            values.get('score_weight_radical', 0) +
            v
        )
        if abs(total - 1.0) > 0.01:  # Allow small floating point errors
            raise ValueError(f"Scoring weights must sum to 1.0, got {total}")
        return v
    
    @property
    def scoring_weights(self) -> Dict[str, float]:
        """Return scoring weights as a dictionary."""
        return {
            "novelty": self.score_weight_novelty,
            "builds_on_others": self.score_weight_builds,
            "solves_subproblem": self.score_weight_solves,
            "radical_idea": self.score_weight_radical,
            "manipulation": self.score_weight_manipulation,
        }
    
    # Game Theory Settings
    default_game_theory_mode: Literal["adversarial", "collaborative", "neutral"] = Field(
        default="adversarial",
        description="Default game theory mode"
    )
    chaos_factor: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Chaos factor for unpredictability (0.0-1.0)"
    )
    fairness_weight: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Weight for preventing agent domination"
    )
    
    # Accusation Settings
    accusation_penalty_false: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Score multiplier penalty for false accusations"
    )
    accusation_proof_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for proven accusations"
    )
    max_accusations_per_agent: int = Field(
        default=3,
        ge=1,
        description="Maximum accusations per agent per game"
    )
    
    # =========================================================================
    # Champion Settings
    # =========================================================================
    champion_memory_boost: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Score bonus for returning champions"
    )
    champion_experience_weight: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Weight of past experience in champion decisions"
    )
    max_champion_rounds: int = Field(
        default=10,
        ge=1,
        description="Maximum rounds a champion can participate"
    )
    
    # =========================================================================
    # Logging and Monitoring
    # =========================================================================
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level"
    )
    log_file: Path = Field(
        default=Path("./data/arena/logs/arena.log"),
        description="Path to log file"
    )
    log_max_size: int = Field(
        default=10485760,  # 10MB
        description="Maximum log file size in bytes"
    )
    log_backup_count: int = Field(
        default=5,
        description="Number of log file backups to keep"
    )
    log_format: Literal["json", "text"] = Field(
        default="json",
        description="Log output format"
    )
    
    # Monitoring
    enable_metrics: bool = Field(
        default=True,
        description="Enable metrics collection"
    )
    metrics_port: int = Field(
        default=9090,
        description="Port for metrics endpoint"
    )
    metrics_update_interval: int = Field(
        default=10,
        description="Metrics update interval in seconds"
    )
    
    # Tracing
    enable_tracing: bool = Field(
        default=False,
        description="Enable distributed tracing"
    )
    jaeger_endpoint: Optional[str] = Field(
        default=None,
        description="Jaeger endpoint for tracing"
    )
    
    # =========================================================================
    # Development Settings
    # =========================================================================
    dev_mode: bool = Field(
        default=False,
        description="Enable development mode"
    )
    dev_auto_reload: bool = Field(
        default=True,
        description="Auto-reload on code changes in dev mode"
    )
    dev_debug_messages: bool = Field(
        default=False,
        description="Enable debug message logging"
    )
    dev_skip_llm: bool = Field(
        default=False,
        description="Skip LLM calls and use mock responses"
    )
    dev_fast_elimination: bool = Field(
        default=False,
        description="Eliminate an agent every turn for testing"
    )
    
    # Testing
    test_mode: bool = Field(
        default=False,
        description="Enable test mode"
    )
    test_seed: int = Field(
        default=42,
        description="Random seed for reproducible tests"
    )
    
    # =========================================================================
    # Security Settings
    # =========================================================================
    api_key_required: bool = Field(
        default=False,
        description="Require API key for Arena API access"
    )
    api_key_header: str = Field(
        default="X-Arena-API-Key",
        description="Header name for API key"
    )
    api_rate_limit: int = Field(
        default=100,
        description="API rate limit per minute"
    )
    
    # Message Validation
    validate_message_schema: bool = Field(
        default=True,
        description="Validate message schemas"
    )
    max_message_length: int = Field(
        default=10000,
        description="Maximum message length in characters"
    )
    
    # Prompt Injection Detection
    detect_prompt_injection: bool = Field(
        default=True,
        description="Enable prompt injection detection"
    )
    prompt_injection_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for prompt injection detection"
    )
    
    # =========================================================================
    # Performance Tuning
    # =========================================================================
    async_batch_size: int = Field(
        default=10,
        ge=1,
        description="Batch size for async operations"
    )
    async_workers: int = Field(
        default=4,
        ge=1,
        description="Number of async workers"
    )
    
    # Timeouts
    database_query_timeout: int = Field(
        default=30,
        description="Database query timeout in seconds"
    )
    cache_operation_timeout: int = Field(
        default=5,
        description="Cache operation timeout in seconds"
    )
    message_bus_timeout: int = Field(
        default=10,
        description="Message bus operation timeout in seconds"
    )
    
    # Resource Limits
    max_memory_per_game: int = Field(
        default=512,  # MB
        description="Maximum memory per game in MB"
    )
    max_cpu_per_game: int = Field(
        default=2,
        description="Maximum CPU cores per game"
    )
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env.arena"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    def __init__(self, **kwargs):
        """Initialize settings and ensure required directories exist."""
        super().__init__(**kwargs)
        # Ensure log directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def validate_llm_keys(self) -> bool:
        """
        Validate that required LLM API keys are configured.
        
        Returns:
            bool: True if valid keys are configured, False otherwise
        """
        if self.dev_skip_llm or self.test_mode:
            return True
            
        if self.default_llm_provider == "anthropic":
            return bool(self.anthropic_api_key)
        elif self.default_llm_provider == "openai":
            return bool(self.openai_api_key)
        elif self.default_llm_provider == "ollama":
            return True  # Ollama doesn't need API key
        return False
    
    def get_llm_config(self) -> Dict[str, Any]:
        """
        Get LLM configuration for the default provider.
        
        Returns:
            Dictionary with LLM configuration
        """
        if self.default_llm_provider == "anthropic":
            return {
                "provider": "anthropic",
                "api_key": self.anthropic_api_key,
                "model": self.anthropic_model,
                "max_tokens": self.anthropic_max_tokens,
                "temperature": self.anthropic_temperature,
            }
        elif self.default_llm_provider == "openai":
            return {
                "provider": "openai",
                "api_key": self.openai_api_key,
                "model": self.openai_model,
                "max_tokens": self.openai_max_tokens,
                "temperature": self.openai_temperature,
            }
        elif self.default_llm_provider == "ollama":
            return {
                "provider": "ollama",
                "base_url": self.ollama_base_url,
                "model": self.ollama_model,
                "timeout": self.ollama_timeout,
            }
        else:
            raise ValueError(f"Unknown LLM provider: {self.default_llm_provider}")


# Global settings instance
# This will be imported throughout the Arena codebase
settings = ArenaSettings()

# Validate on import if not in test mode
if not settings.test_mode and not settings.validate_llm_keys():
    import warnings
    warnings.warn(
        f"No API key configured for {settings.default_llm_provider}. "
        "Please set the appropriate API key in your .env.arena file or "
        "enable dev_skip_llm for testing without LLM calls."
    )