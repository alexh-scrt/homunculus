"""Centralized logging configuration for the Homunculus system."""

import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from .settings import get_settings


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    force_setup: bool = False
) -> None:
    """
    Set up comprehensive logging for the entire application.
    
    Args:
        log_level: Override log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Override log file path
        force_setup: Force reconfiguration even if already set up
    """
    # Prevent duplicate setup unless forced
    if hasattr(setup_logging, '_configured') and not force_setup:
        return
    
    settings = get_settings()
    
    # Use provided values or fall back to settings
    log_level = log_level or settings.log_level
    log_file = log_file or settings.log_file
    
    # Ensure log directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter with detailed information
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)-30s | %(funcName)-20s:%(lineno)-4d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create and configure file handler with immediate flushing
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    
    # Enable immediate flushing for real-time log viewing
    class FlushFileHandler(logging.FileHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()
    
    # Replace with flushing handler
    flush_file_handler = FlushFileHandler(log_file, mode='a', encoding='utf-8')
    flush_file_handler.setLevel(numeric_level)
    flush_file_handler.setFormatter(formatter)
    
    # Create console handler for important messages (WARNING and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter(
        '%(levelname)-8s | %(name)-20s | %(message)s'
    ))
    
    # Add handlers to root logger
    root_logger.addHandler(flush_file_handler)
    root_logger.addHandler(console_handler)
    
    # Configure third-party loggers to reduce noise
    logging.getLogger('chromadb').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('neo4j').setLevel(logging.ERROR)
    logging.getLogger('ollama').setLevel(logging.WARNING)
    
    # Additional noisy loggers
    logging.getLogger('integration.agent_orchestrator').setLevel(logging.ERROR)
    logging.getLogger('websockets').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    
    # Log initial startup message
    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info(f"Homunculus Logging System Initialized - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log Level: {log_level.upper()}")
    logger.info(f"Log File: {log_file}")
    logger.info(f"Real-time viewing: tail -f {log_file}")
    logger.info("="*80)
    
    # Mark as configured
    setup_logging._configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name, ensuring logging is set up.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    # Ensure logging is set up
    setup_logging()
    return logging.getLogger(name)


def log_system_info():
    """Log system information for debugging."""
    logger = get_logger(__name__)
    settings = get_settings()
    
    logger.info("System Configuration:")
    logger.info(f"  - Character Schemas Dir: {settings.character_schemas_dir}")
    logger.info(f"  - Data Directory: {settings.data_dir}")
    logger.info(f"  - ChromaDB Path: {settings.chroma_persist_directory}")
    logger.info(f"  - Neo4j URI: {settings.neo4j_uri}")
    logger.info(f"  - Ollama URL: {settings.ollama_base_url}")
    logger.info(f"  - Ollama Model: {settings.ollama_model}")
    logger.info(f"  - Web Search Enabled: {settings.web_search_enabled}")
    logger.info(f"  - Max Conversation History: {settings.max_conversation_history}")


def log_character_session_start(character_id: str, character_name: str):
    """Log the start of a character session."""
    logger = get_logger(__name__)
    logger.info("+"*80)
    logger.info(f"CHARACTER SESSION STARTED")
    logger.info(f"Character ID: {character_id}")
    logger.info(f"Character Name: {character_name}")
    logger.info(f"Session Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("+"*80)


def log_character_session_end(character_id: str, character_name: str):
    """Log the end of a character session."""
    logger = get_logger(__name__)
    logger.info("+"*80)
    logger.info(f"CHARACTER SESSION ENDED")
    logger.info(f"Character ID: {character_id}")
    logger.info(f"Character Name: {character_name}")
    logger.info(f"Session End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("+"*80)


def set_debug_level():
    """Convenience function to set all loggers to DEBUG level."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    for handler in root_logger.handlers:
        handler.setLevel(logging.DEBUG)
    
    logger = get_logger(__name__)
    logger.debug("All loggers set to DEBUG level")


def tail_command():
    """Print the tail command for easy copy-paste."""
    settings = get_settings()
    logger = get_logger(__name__)
    logger.info(f"To monitor logs in real-time, run: tail -f {settings.log_file}")
    print(f"\n📋 Copy this command to monitor logs:")
    print(f"tail -f {settings.log_file}\n")