"""Settings management using Pydantic and environment variables."""

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings
from pydantic import Field, ConfigDict
from typing import Optional, Annotated
import os
from pathlib import Path


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env file.
    
    Uses Pydantic BaseSettings for automatic validation and type conversion.
    """
    
    # Environment and debug
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    
    # Ollama LLM Configuration
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.3:70b", env="OLLAMA_MODEL")
    
    # ChromaDB Configuration
    chroma_persist_directory: str = Field(
        default="./data/chroma_db",
        env="CHROMA_PERSIST_DIRECTORY"
    )
    
    # Neo4j Configuration
    neo4j_uri: str = Field(default="bolt://localhost:7687", env="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", env="NEO4J_USER")
    neo4j_password: str = Field(default="your_password_here", env="NEO4J_PASSWORD")
    
    # Redis Configuration
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    # Tavily Web Search API
    tavily_api_key: Optional[str] = Field(default=None, env="TAVILY_API_KEY")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="./data/logs/homunculus.log", env="LOG_FILE")
    
    # Character and Memory Settings
    max_conversation_history: int = Field(default=20, env="MAX_CONVERSATION_HISTORY")
    max_web_search_history: int = Field(default=20, env="MAX_WEB_SEARCH_HISTORY")
    max_knowledge_updates: int = Field(default=50, env="MAX_KNOWLEDGE_UPDATES")
    default_memory_retrieval_limit: int = Field(default=5, env="DEFAULT_MEMORY_RETRIEVAL_LIMIT")
    
    # LLM Generation Settings
    default_temperature: float = Field(default=0.7, env="DEFAULT_TEMPERATURE")
    default_max_tokens: int = Field(default=500, env="DEFAULT_MAX_TOKENS")
    agent_consultation_max_tokens: int = Field(default=200, env="AGENT_CONSULTATION_MAX_TOKENS")
    
    # Neurochemical System Settings
    hormone_decay_interval_seconds: int = Field(default=30, env="HORMONE_DECAY_INTERVAL_SECONDS")
    enable_hormone_decay: bool = Field(default=True, env="ENABLE_HORMONE_DECAY")
    
    # Web Search Settings
    max_web_search_results: int = Field(default=5, env="MAX_WEB_SEARCH_RESULTS")
    web_search_enabled: bool = Field(default=True, env="WEB_SEARCH_ENABLED")
    web_search_cache_ttl_hours: int = Field(default=24, env="WEB_SEARCH_CACHE_TTL_HOURS")
    
    # File Paths
    character_schemas_dir: str = Field(default="schemas/characters", env="CHARACTER_SCHEMAS_DIR")
    data_dir: str = Field(default="./data", env="DATA_DIR")
    saves_dir: str = Field(default="./data/saves", env="SAVES_DIR")
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # Ignore extra fields from .env
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure data directories exist
        self._create_directories()
    
    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.data_dir,
            self.saves_dir,
            os.path.dirname(self.log_file),
            os.path.dirname(self.chroma_persist_directory) if not os.path.isabs(self.chroma_persist_directory) else None
        ]
        
        for directory in directories:
            if directory:
                try:
                    Path(directory).mkdir(parents=True, exist_ok=True)
                except PermissionError:
                    # Skip directories we can't create (like system directories)
                    pass
    
    @property
    def is_tavily_enabled(self) -> bool:
        """Check if Tavily web search is properly configured."""
        return self.web_search_enabled and self.tavily_api_key is not None
    
    @property
    def character_config_files(self) -> list[str]:
        """Get list of available character configuration files."""
        schemas_path = Path(self.character_schemas_dir)
        if not schemas_path.exists():
            return []
        
        yaml_files = list(schemas_path.glob("*.yaml"))
        return [str(f) for f in yaml_files]
    
    def get_character_config_path(self, character_file: str) -> str:
        """Get full path to a character configuration file."""
        if not character_file.endswith('.yaml'):
            character_file += '.yaml'
        
        return os.path.join(self.character_schemas_dir, character_file)
    
    def validate_database_connections(self) -> dict[str, bool]:
        """
        Validate that database connection settings are reasonable.
        Returns dict of validation results (doesn't actually test connections).
        """
        validations = {
            "ollama_url_format": self.ollama_base_url.startswith(("http://", "https://")),
            "neo4j_uri_format": self.neo4j_uri.startswith(("bolt://", "neo4j://", "neo4j+s://", "bolt+s://")),
            "redis_port_range": 1 <= self.redis_port <= 65535,
            "log_file_writable": os.access(os.path.dirname(self.log_file), os.W_OK) if os.path.exists(os.path.dirname(self.log_file)) else True,
            "chroma_directory_writable": os.access(os.path.dirname(self.chroma_persist_directory), os.W_OK) if os.path.exists(os.path.dirname(self.chroma_persist_directory)) else True
        }
        
        return validations
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL."""
        auth_part = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth_part}{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    def get_neo4j_auth(self) -> tuple[str, str]:
        """Get Neo4j authentication tuple."""
        return (self.neo4j_user, self.neo4j_password)


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def reload_settings() -> Settings:
    """Reload settings from environment (useful for testing)."""
    global settings
    settings = Settings()
    return settings