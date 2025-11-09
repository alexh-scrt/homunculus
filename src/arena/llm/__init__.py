"""
LLM Integration for Arena

This module provides LLM clients and utilities for Arena agents,
based on the proven approach from the talks project.
"""

from .llm_client import (
    ArenaLLMClient,
    strip_reasoning
)

__all__ = [
    "ArenaLLMClient",
    "strip_reasoning"
]