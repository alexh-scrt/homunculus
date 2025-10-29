"""
Arena Game Theory Module

This module contains game theory algorithms, scoring systems,
and strategic mechanics for the Arena competitive environment.

Components:
- Scoring algorithms with multi-dimensional metrics
- Elimination mechanics with fairness guarantees
- Coalition detection and prevention
- Reputation and trust systems
- Strategic equilibrium analysis

Author: Homunculus Team
"""

from .scoring_engine import (
    ScoringEngine,
    ScoringStrategy,
    MultidimensionalScorer,
    WeightedScorer,
    AdaptiveScorer
)

from .elimination_mechanics import (
    EliminationEngine,
    EliminationStrategy,
    FairElimination,
    PerformanceBasedElimination,
    AccusationBasedElimination
)

from .coalition_detection import (
    CoalitionDetector,
    CollaborationPattern,
    AllianceTracker,
    ManipulationDetector
)

from .reputation_system import (
    ReputationEngine,
    TrustMetric,
    CredibilityScore,
    ReputationDecay
)

from .game_strategies import (
    GameStrategy,
    TitForTat,
    AlwaysCooperate,
    AlwaysDefect,
    AdaptiveStrategy,
    PavlovStrategy
)

from .leaderboard import (
    Leaderboard,
    RankingSystem,
    EloRating,
    PerformanceMetrics
)

__all__ = [
    # Scoring
    "ScoringEngine",
    "ScoringStrategy",
    "MultidimensionalScorer",
    "WeightedScorer",
    "AdaptiveScorer",
    
    # Elimination
    "EliminationEngine",
    "EliminationStrategy",
    "FairElimination",
    "PerformanceBasedElimination",
    "AccusationBasedElimination",
    
    # Coalition
    "CoalitionDetector",
    "CollaborationPattern",
    "AllianceTracker",
    "ManipulationDetector",
    
    # Reputation
    "ReputationEngine",
    "TrustMetric",
    "CredibilityScore",
    "ReputationDecay",
    
    # Strategies
    "GameStrategy",
    "TitForTat",
    "AlwaysCooperate",
    "AlwaysDefect",
    "AdaptiveStrategy",
    "PavlovStrategy",
    
    # Leaderboard
    "Leaderboard",
    "RankingSystem",
    "EloRating",
    "PerformanceMetrics"
]