"""
Arena CLI Module

Command-line interface for managing Arena games, tournaments,
agents, and analytics.

Features:
- Game management (start, stop, save, load)
- Agent configuration and management
- Tournament organization
- Replay viewing and analysis
- Statistics and reporting
- Configuration management

Author: Homunculus Team
"""

from .main import main, ArenaCLI
from .commands import (
    GameCommands,
    AgentCommands,
    TournamentCommands,
    ReplayCommands,
    StatsCommands,
    ConfigCommands
)

__all__ = [
    "main",
    "ArenaCLI",
    "GameCommands",
    "AgentCommands", 
    "TournamentCommands",
    "ReplayCommands",
    "StatsCommands",
    "ConfigCommands"
]