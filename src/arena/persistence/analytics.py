"""
Analytics Engine for Arena

This module provides comprehensive analytics and statistics tracking
for game performance, agent behavior, and strategic patterns.

Features:
- Real-time metrics collection
- Performance analysis
- Trend detection
- Pattern recognition
- Statistical aggregation

Author: Homunculus Team
"""

import logging
import statistics
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from enum import Enum
import math

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics tracked."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    RATE = "rate"
    PERCENTILE = "percentile"


@dataclass
class GameMetrics:
    """Metrics for a single game."""
    game_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Core metrics
    total_turns: int = 0
    total_messages: int = 0
    total_events: int = 0
    total_eliminations: int = 0
    
    # Performance metrics
    avg_turn_duration: float = 0.0
    avg_message_length: float = 0.0
    avg_score_per_turn: float = 0.0
    
    # Participation metrics
    agent_participation: Dict[str, int] = field(default_factory=dict)
    speaker_distribution: Dict[str, int] = field(default_factory=dict)
    elimination_timeline: List[Tuple[int, str]] = field(default_factory=list)
    
    # Score metrics
    score_progression: Dict[str, List[float]] = field(default_factory=dict)
    final_scores: Dict[str, float] = field(default_factory=dict)
    score_volatility: Dict[str, float] = field(default_factory=dict)
    
    # Interaction metrics
    interaction_matrix: Dict[Tuple[str, str], int] = field(default_factory=dict)
    accusation_counts: Dict[str, int] = field(default_factory=dict)
    coalition_formations: List[Set[str]] = field(default_factory=list)
    
    def calculate_derived_metrics(self) -> None:
        """Calculate derived metrics from raw data."""
        # Calculate score volatility
        for agent, scores in self.score_progression.items():
            if len(scores) > 1:
                self.score_volatility[agent] = statistics.stdev(scores)
            else:
                self.score_volatility[agent] = 0.0
        
        # Calculate average score per turn
        if self.total_turns > 0 and self.final_scores:
            self.avg_score_per_turn = sum(self.final_scores.values()) / (
                self.total_turns * len(self.final_scores)
            )
    
    def get_winner(self) -> Optional[str]:
        """Get game winner."""
        if not self.final_scores:
            return None
        return max(self.final_scores, key=self.final_scores.get)
    
    def get_elimination_rate(self) -> float:
        """Get elimination rate per turn."""
        if self.total_turns > 0:
            return self.total_eliminations / self.total_turns
        return 0.0


@dataclass
class AgentPerformance:
    """Performance metrics for a single agent."""
    agent_id: str
    agent_name: str
    
    # Game statistics
    games_played: int = 0
    games_won: int = 0
    win_rate: float = 0.0
    
    # Survival metrics
    avg_survival_turns: float = 0.0
    total_eliminations_survived: int = 0
    survival_rate: float = 0.0
    
    # Score metrics
    avg_final_score: float = 0.0
    highest_score: float = 0.0
    lowest_score: float = 0.0
    score_consistency: float = 0.0
    
    # Contribution metrics
    total_contributions: int = 0
    avg_contributions_per_game: float = 0.0
    contribution_quality: float = 0.0
    
    # Interaction metrics
    total_messages_sent: int = 0
    avg_message_length: float = 0.0
    response_rate: float = 0.0
    
    # Strategy metrics
    preferred_strategies: List[str] = field(default_factory=list)
    strategy_success_rates: Dict[str, float] = field(default_factory=dict)
    adaptability_score: float = 0.0
    
    # Social metrics
    alliance_formation_rate: float = 0.0
    betrayal_rate: float = 0.0
    reputation_score: float = 0.0
    
    def update_from_game(self, metrics: GameMetrics) -> None:
        """Update performance from game metrics."""
        self.games_played += 1
        
        # Update win rate
        if metrics.get_winner() == self.agent_id:
            self.games_won += 1
        self.win_rate = self.games_won / self.games_played if self.games_played > 0 else 0
        
        # Update score metrics
        if self.agent_id in metrics.final_scores:
            score = metrics.final_scores[self.agent_id]
            self.avg_final_score = (
                (self.avg_final_score * (self.games_played - 1) + score) / self.games_played
            )
            self.highest_score = max(self.highest_score, score)
            self.lowest_score = min(self.lowest_score, score) if self.lowest_score > 0 else score
        
        # Update contribution metrics
        if self.agent_id in metrics.agent_participation:
            contributions = metrics.agent_participation[self.agent_id]
            self.total_contributions += contributions
            self.avg_contributions_per_game = self.total_contributions / self.games_played
        
        # Update message metrics
        if self.agent_id in metrics.speaker_distribution:
            self.total_messages_sent += metrics.speaker_distribution[self.agent_id]
    
    def calculate_performance_score(self) -> float:
        """Calculate overall performance score."""
        # Weighted combination of metrics
        weights = {
            "win_rate": 0.3,
            "survival_rate": 0.2,
            "contribution_quality": 0.2,
            "score_consistency": 0.15,
            "adaptability": 0.15
        }
        
        score = (
            weights["win_rate"] * self.win_rate +
            weights["survival_rate"] * self.survival_rate +
            weights["contribution_quality"] * self.contribution_quality +
            weights["score_consistency"] * (1.0 - min(self.score_consistency, 1.0)) +
            weights["adaptability"] * self.adaptability_score
        )
        
        return score


class StatisticsAggregator:
    """
    Aggregates statistics across multiple games and agents.
    """
    
    def __init__(self):
        """Initialize aggregator."""
        self.game_metrics: Dict[str, GameMetrics] = {}
        self.agent_performances: Dict[str, AgentPerformance] = {}
        self.global_stats: Dict[str, Any] = {}
        
    def add_game_metrics(self, metrics: GameMetrics) -> None:
        """Add game metrics."""
        self.game_metrics[metrics.game_id] = metrics
        
        # Update agent performances
        for agent_id in metrics.agent_participation:
            if agent_id not in self.agent_performances:
                self.agent_performances[agent_id] = AgentPerformance(
                    agent_id=agent_id,
                    agent_name=agent_id  # Default to ID
                )
            self.agent_performances[agent_id].update_from_game(metrics)
        
        # Update global statistics
        self._update_global_stats()
    
    def _update_global_stats(self) -> None:
        """Update global statistics."""
        if not self.game_metrics:
            return
        
        total_games = len(self.game_metrics)
        
        # Calculate averages
        avg_turns = statistics.mean(
            m.total_turns for m in self.game_metrics.values()
        )
        avg_messages = statistics.mean(
            m.total_messages for m in self.game_metrics.values()
        )
        avg_eliminations = statistics.mean(
            m.total_eliminations for m in self.game_metrics.values()
        )
        
        # Calculate game duration statistics
        durations = []
        for metrics in self.game_metrics.values():
            if metrics.end_time and metrics.start_time:
                duration = (metrics.end_time - metrics.start_time).total_seconds()
                durations.append(duration)
        
        avg_duration = statistics.mean(durations) if durations else 0
        
        # Winner statistics
        winner_counts = Counter()
        for metrics in self.game_metrics.values():
            winner = metrics.get_winner()
            if winner:
                winner_counts[winner] += 1
        
        self.global_stats = {
            "total_games": total_games,
            "average_turns": avg_turns,
            "average_messages": avg_messages,
            "average_eliminations": avg_eliminations,
            "average_duration_seconds": avg_duration,
            "most_wins": winner_counts.most_common(5),
            "total_unique_agents": len(self.agent_performances)
        }
    
    def get_agent_rankings(
        self,
        metric: str = "performance_score"
    ) -> List[Tuple[str, float]]:
        """
        Get agent rankings by metric.
        
        Args:
            metric: Metric to rank by
            
        Returns:
            Ranked list of (agent_id, score)
        """
        rankings = []
        
        for agent_id, performance in self.agent_performances.items():
            if metric == "performance_score":
                score = performance.calculate_performance_score()
            elif hasattr(performance, metric):
                score = getattr(performance, metric)
            else:
                score = 0
            
            rankings.append((agent_id, score))
        
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings
    
    def get_statistics_summary(self) -> Dict[str, Any]:
        """Get comprehensive statistics summary."""
        return {
            "global": self.global_stats,
            "top_performers": self.get_agent_rankings()[:10],
            "games_analyzed": len(self.game_metrics),
            "agents_tracked": len(self.agent_performances)
        }


class TrendAnalyzer:
    """
    Analyzes trends and patterns in game data.
    """
    
    def __init__(self, window_size: int = 10):
        """
        Initialize trend analyzer.
        
        Args:
            window_size: Size of sliding window for trend analysis
        """
        self.window_size = window_size
        self.time_series_data: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        self.pattern_counts: Counter = Counter()
        
    def add_data_point(
        self,
        metric_name: str,
        timestamp: datetime,
        value: float
    ) -> None:
        """Add a data point for trend analysis."""
        self.time_series_data[metric_name].append((timestamp, value))
        
        # Keep only recent data
        if len(self.time_series_data[metric_name]) > self.window_size * 10:
            self.time_series_data[metric_name] = self.time_series_data[metric_name][-self.window_size * 10:]
    
    def detect_trend(self, metric_name: str) -> str:
        """
        Detect trend for a metric.
        
        Args:
            metric_name: Metric to analyze
            
        Returns:
            Trend direction (increasing, decreasing, stable)
        """
        if metric_name not in self.time_series_data:
            return "unknown"
        
        data = self.time_series_data[metric_name]
        if len(data) < self.window_size:
            return "insufficient_data"
        
        # Get recent window
        recent_window = data[-self.window_size:]
        values = [v for _, v in recent_window]
        
        # Calculate linear regression slope
        n = len(values)
        x = list(range(n))
        
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return "stable"
        
        slope = numerator / denominator
        
        # Determine trend based on slope
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"
    
    def find_patterns(self, game_metrics: List[GameMetrics]) -> Dict[str, int]:
        """
        Find common patterns in games.
        
        Args:
            game_metrics: List of game metrics
            
        Returns:
            Pattern counts
        """
        patterns = Counter()
        
        for metrics in game_metrics:
            # Early elimination pattern
            early_elims = sum(1 for turn, _ in metrics.elimination_timeline if turn < 10)
            if early_elims > 2:
                patterns["early_elimination_heavy"] += 1
            
            # Comeback pattern
            for agent, scores in metrics.score_progression.items():
                if len(scores) > 10:
                    early_avg = statistics.mean(scores[:5])
                    late_avg = statistics.mean(scores[-5:])
                    if late_avg > early_avg * 1.5:
                        patterns["comeback"] += 1
                        break
            
            # Dominant winner pattern
            if metrics.final_scores:
                winner_score = max(metrics.final_scores.values())
                avg_score = statistics.mean(metrics.final_scores.values())
                if winner_score > avg_score * 1.5:
                    patterns["dominant_winner"] += 1
            
            # Coalition pattern
            if len(metrics.coalition_formations) > 0:
                patterns["coalition_game"] += 1
            
            # High interaction pattern
            if metrics.total_messages > metrics.total_turns * 5:
                patterns["high_interaction"] += 1
        
        return dict(patterns)
    
    def calculate_volatility(
        self,
        metric_name: str,
        period: int = 10
    ) -> float:
        """
        Calculate volatility for a metric.
        
        Args:
            metric_name: Metric name
            period: Period for calculation
            
        Returns:
            Volatility score
        """
        if metric_name not in self.time_series_data:
            return 0.0
        
        data = self.time_series_data[metric_name]
        if len(data) < period:
            return 0.0
        
        recent_values = [v for _, v in data[-period:]]
        
        if len(recent_values) < 2:
            return 0.0
        
        return statistics.stdev(recent_values)
    
    def predict_next_value(
        self,
        metric_name: str,
        steps: int = 1
    ) -> Optional[float]:
        """
        Predict next value using simple linear regression.
        
        Args:
            metric_name: Metric name
            steps: Steps ahead to predict
            
        Returns:
            Predicted value
        """
        if metric_name not in self.time_series_data:
            return None
        
        data = self.time_series_data[metric_name]
        if len(data) < 3:
            return None
        
        recent_values = [v for _, v in data[-self.window_size:]]
        
        # Simple moving average prediction
        if len(recent_values) < 3:
            return None
        
        return statistics.mean(recent_values[-3:])


class AnalyticsEngine:
    """
    Main analytics engine for Arena.
    """
    
    def __init__(self):
        """Initialize analytics engine."""
        self.aggregator = StatisticsAggregator()
        self.trend_analyzer = TrendAnalyzer()
        
        # Real-time metrics
        self.current_game_metrics: Optional[GameMetrics] = None
        self.metrics_buffer: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.performance_history: Dict[str, List[float]] = defaultdict(list)
        
    def start_game_tracking(self, game_id: str) -> GameMetrics:
        """
        Start tracking a new game.
        
        Args:
            game_id: Game ID
            
        Returns:
            Game metrics object
        """
        self.current_game_metrics = GameMetrics(
            game_id=game_id,
            start_time=datetime.utcnow()
        )
        return self.current_game_metrics
    
    def record_turn(
        self,
        turn_number: int,
        turn_data: Dict[str, Any]
    ) -> None:
        """
        Record turn data.
        
        Args:
            turn_number: Turn number
            turn_data: Turn data
        """
        if not self.current_game_metrics:
            return
        
        metrics = self.current_game_metrics
        metrics.total_turns = max(metrics.total_turns, turn_number)
        
        # Record messages
        messages = turn_data.get("messages", [])
        metrics.total_messages += len(messages)
        
        # Record speaker distribution
        for msg in messages:
            sender = msg.get("sender_id")
            if sender:
                metrics.speaker_distribution[sender] = metrics.speaker_distribution.get(sender, 0) + 1
        
        # Record eliminations
        eliminated = turn_data.get("eliminated", [])
        for agent in eliminated:
            metrics.elimination_timeline.append((turn_number, agent))
            metrics.total_eliminations += 1
        
        # Record scores
        scores = turn_data.get("scores", {})
        for agent, score in scores.items():
            if agent not in metrics.score_progression:
                metrics.score_progression[agent] = []
            metrics.score_progression[agent].append(score)
        
        # Record events
        events = turn_data.get("events", [])
        metrics.total_events += len(events)
        
        # Update trend analyzer
        self.trend_analyzer.add_data_point(
            "turn_messages",
            datetime.utcnow(),
            len(messages)
        )
        self.trend_analyzer.add_data_point(
            "active_agents",
            datetime.utcnow(),
            len(scores)
        )
    
    def record_interaction(
        self,
        agent1: str,
        agent2: str,
        interaction_type: str = "message"
    ) -> None:
        """
        Record interaction between agents.
        
        Args:
            agent1: First agent
            agent2: Second agent
            interaction_type: Type of interaction
        """
        if not self.current_game_metrics:
            return
        
        key = tuple(sorted([agent1, agent2]))
        self.current_game_metrics.interaction_matrix[key] = (
            self.current_game_metrics.interaction_matrix.get(key, 0) + 1
        )
    
    def finalize_game(
        self,
        final_scores: Dict[str, float],
        winner: Optional[str] = None
    ) -> GameMetrics:
        """
        Finalize game tracking.
        
        Args:
            final_scores: Final scores
            winner: Winner agent ID
            
        Returns:
            Final game metrics
        """
        if not self.current_game_metrics:
            return None
        
        metrics = self.current_game_metrics
        metrics.end_time = datetime.utcnow()
        metrics.final_scores = final_scores
        
        # Calculate derived metrics
        metrics.calculate_derived_metrics()
        
        # Add to aggregator
        self.aggregator.add_game_metrics(metrics)
        
        # Update performance history
        for agent, score in final_scores.items():
            self.performance_history[agent].append(score)
        
        # Clear current metrics
        completed_metrics = self.current_game_metrics
        self.current_game_metrics = None
        
        return completed_metrics
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time metrics for current game."""
        if not self.current_game_metrics:
            return {}
        
        metrics = self.current_game_metrics
        
        return {
            "game_id": metrics.game_id,
            "current_turn": metrics.total_turns,
            "active_agents": len(metrics.score_progression),
            "total_messages": metrics.total_messages,
            "elimination_rate": metrics.get_elimination_rate(),
            "message_rate": metrics.total_messages / max(metrics.total_turns, 1),
            "current_leader": max(metrics.score_progression, 
                                key=lambda a: metrics.score_progression[a][-1])
                                if metrics.score_progression else None
        }
    
    def get_agent_analytics(self, agent_id: str) -> Dict[str, Any]:
        """
        Get analytics for specific agent.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Agent analytics
        """
        performance = self.aggregator.agent_performances.get(agent_id)
        if not performance:
            return {}
        
        return {
            "performance": asdict(performance),
            "trend": self.trend_analyzer.detect_trend(f"agent_{agent_id}_score"),
            "recent_scores": self.performance_history.get(agent_id, [])[-10:],
            "ranking": next(
                (i + 1 for i, (aid, _) in enumerate(self.aggregator.get_agent_rankings())
                 if aid == agent_id),
                None
            )
        }
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive analytics report."""
        patterns = self.trend_analyzer.find_patterns(
            list(self.aggregator.game_metrics.values())
        )
        
        return {
            "summary": self.aggregator.get_statistics_summary(),
            "patterns": patterns,
            "trends": {
                metric: self.trend_analyzer.detect_trend(metric)
                for metric in self.trend_analyzer.time_series_data
            },
            "top_performers": self.aggregator.get_agent_rankings()[:10],
            "games_analyzed": len(self.aggregator.game_metrics),
            "report_generated": datetime.utcnow().isoformat()
        }