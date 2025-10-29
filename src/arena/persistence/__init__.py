"""
Arena Persistence Module

This module provides comprehensive persistence functionality for Arena games,
including database storage, champion memory, replay systems, and analytics.

Components:
- Database models and schema
- Game storage and retrieval
- Champion memory management
- Replay system
- Multi-round tournaments
- Analytics and statistics
- Data export

Author: Homunculus Team
"""

from .database import (
    DatabaseManager,
    GameRecord,
    TurnRecord,
    AgentRecord,
    MessageRecord,
    ScoreRecord
)

from .champion_memory import (
    ChampionMemory,
    ChampionProfile,
    MemoryBank,
    ExperienceReplay
)

from .game_storage import (
    GameStorage,
    SaveGame,
    LoadGame,
    GameArchive,
    StorageFormat
)

from .replay_system import (
    ReplayManager,
    ReplayViewer,
    ReplayFrame,
    ReplaySpeed,
    ReplayAnalyzer
)

from .tournament import (
    TournamentManager,
    TournamentRound,
    TournamentBracket,
    TournamentResults,
    SeasonManager,
    TournamentFormat
)

from .analytics import (
    AnalyticsEngine,
    GameMetrics,
    AgentPerformance,
    StatisticsAggregator,
    TrendAnalyzer
)

from .export import (
    DataExporter,
    ExportFormat,
    ExportConfig,
    ReportGenerator
)

__all__ = [
    # Database
    "DatabaseManager",
    "GameRecord",
    "TurnRecord",
    "AgentRecord",
    "MessageRecord",
    "ScoreRecord",
    
    # Champion Memory
    "ChampionMemory",
    "ChampionProfile",
    "MemoryBank",
    "ExperienceReplay",
    
    # Game Storage
    "GameStorage",
    "SaveGame",
    "LoadGame",
    "GameArchive",
    "StorageFormat",
    
    # Replay
    "ReplayManager",
    "ReplayViewer",
    "ReplayFrame",
    "ReplaySpeed",
    "ReplayAnalyzer",
    
    # Tournament
    "TournamentManager",
    "TournamentRound",
    "TournamentBracket",
    "TournamentResults",
    "SeasonManager",
    "TournamentFormat",
    
    # Analytics
    "AnalyticsEngine",
    "GameMetrics",
    "AgentPerformance",
    "StatisticsAggregator",
    "TrendAnalyzer",
    
    # Export
    "DataExporter",
    "ExportFormat",
    "ExportConfig",
    "ReportGenerator"
]