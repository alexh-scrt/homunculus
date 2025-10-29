"""
Scoring Engine for Arena

This module implements sophisticated scoring algorithms that evaluate
agent contributions across multiple dimensions with adaptive weighting.

Features:
- Multi-dimensional scoring metrics
- Adaptive weight adjustment
- Normalization and scaling
- Outlier detection
- Temporal decay
- Contextual scoring

Author: Homunculus Team
"""

import logging
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np

from ..models import ScoringMetrics, AgentState, Message

logger = logging.getLogger(__name__)


@dataclass
class ScoringContext:
    """Context for scoring decisions."""
    game_phase: str  # early, mid, late, final
    turn_number: int
    total_agents: int
    eliminated_agents: int
    recent_scores: List[float]
    problem_complexity: float
    time_remaining: Optional[timedelta] = None
    
    @property
    def elimination_pressure(self) -> float:
        """Calculate elimination pressure."""
        if self.total_agents == 0:
            return 0.0
        return self.eliminated_agents / self.total_agents
    
    @property
    def game_progress(self) -> float:
        """Estimate game progress (0.0 to 1.0)."""
        # Could be more sophisticated
        phase_weights = {
            "early": 0.2,
            "mid": 0.5,
            "late": 0.8,
            "final": 0.95
        }
        return phase_weights.get(self.game_phase, 0.5)


class ScoringStrategy(ABC):
    """Abstract base class for scoring strategies."""
    
    @abstractmethod
    def score(
        self,
        contribution: Message,
        context: ScoringContext,
        history: List[ScoringMetrics]
    ) -> ScoringMetrics:
        """Score a contribution."""
        pass
    
    @abstractmethod
    def get_weights(self, context: ScoringContext) -> Dict[str, float]:
        """Get scoring weights for context."""
        pass


class MultidimensionalScorer(ScoringStrategy):
    """
    Scores contributions across multiple dimensions.
    
    Dimensions:
    - Novelty: How original is the idea?
    - Building: Does it build on others?
    - Solution: Does it solve the problem?
    - Radical: Is it paradigm-shifting?
    - Manipulation: Is it gaming the system?
    """
    
    def __init__(self):
        """Initialize the multidimensional scorer."""
        self.base_weights = {
            "novelty": 0.25,
            "builds_on_others": 0.20,
            "solves_subproblem": 0.25,
            "radical_idea": 0.15,
            "manipulation": -0.15  # Negative weight
        }
        
        # Track patterns for better scoring
        self.agent_patterns: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
    
    def score(
        self,
        contribution: Message,
        context: ScoringContext,
        history: List[ScoringMetrics]
    ) -> ScoringMetrics:
        """
        Score a contribution across dimensions.
        
        Args:
            contribution: The contribution to score
            context: Current game context
            history: Scoring history
            
        Returns:
            Scoring metrics
        """
        agent_id = contribution.sender_id
        
        # Calculate base scores
        scores = self._calculate_base_scores(contribution, context, history)
        
        # Apply contextual adjustments
        scores = self._apply_context_adjustments(scores, context)
        
        # Detect manipulation
        manipulation_score = self._detect_manipulation(
            contribution, 
            history,
            agent_id
        )
        scores["manipulation"] = manipulation_score
        
        # Create metrics
        metrics = ScoringMetrics(
            agent_id=agent_id,
            message_id=contribution.message_id,
            turn_number=context.turn_number,
            novelty=scores["novelty"],
            builds_on_others=scores["builds_on_others"],
            solves_subproblem=scores["solves_subproblem"],
            radical_idea=scores["radical_idea"],
            manipulation=scores["manipulation"]
        )
        
        # Calculate weighted score
        weights = self.get_weights(context)
        metrics.calculate_weighted_score(weights)
        
        # Track patterns
        self._track_patterns(agent_id, scores)
        
        return metrics
    
    def get_weights(self, context: ScoringContext) -> Dict[str, float]:
        """
        Get adaptive weights based on context.
        
        Args:
            context: Current game context
            
        Returns:
            Scoring weights
        """
        weights = self.base_weights.copy()
        
        # Adjust for game phase
        if context.game_phase == "early":
            weights["novelty"] *= 1.2
            weights["radical_idea"] *= 1.3
        elif context.game_phase == "late":
            weights["solves_subproblem"] *= 1.4
            weights["builds_on_others"] *= 1.2
        elif context.game_phase == "final":
            weights["solves_subproblem"] *= 1.5
            weights["manipulation"] *= 1.5  # Stricter
        
        # Adjust for elimination pressure
        if context.elimination_pressure > 0.5:
            weights["manipulation"] *= 1.3  # More scrutiny
        
        # Normalize weights to sum to 1.0 (model requirement)
        # Handle negative weight for manipulation specially
        manipulation_weight = weights.get("manipulation", 0)
        positive_weights = {k: v for k, v in weights.items() if k != "manipulation"}
        
        # Scale positive weights to sum to 1.0 - abs(manipulation_weight)
        pos_total = sum(positive_weights.values())
        target_sum = 1.0 - abs(manipulation_weight)
        
        if pos_total > 0:
            scale = target_sum / pos_total
            for k in positive_weights:
                weights[k] = positive_weights[k] * scale
        
        # Keep manipulation weight as is (negative)
        
        return weights
    
    def _calculate_base_scores(
        self,
        contribution: Message,
        context: ScoringContext,
        history: List[ScoringMetrics]
    ) -> Dict[str, float]:
        """Calculate base scores for each dimension."""
        content = contribution.content
        content_length = len(content)
        
        # Novelty: Check similarity to recent contributions
        novelty = self._calculate_novelty(content, history)
        
        # Builds on others: Check references to other agents
        builds_on = self._calculate_building_score(contribution, history)
        
        # Solves subproblem: Heuristic based on keywords and structure
        solves = self._calculate_solution_score(content, context)
        
        # Radical idea: Detect paradigm shifts
        radical = self._calculate_radical_score(content, history)
        
        return {
            "novelty": novelty,
            "builds_on_others": builds_on,
            "solves_subproblem": solves,
            "radical_idea": radical,
            "manipulation": 0.0  # Calculated separately
        }
    
    def _calculate_novelty(
        self,
        content: str,
        history: List[ScoringMetrics]
    ) -> float:
        """Calculate novelty score."""
        if not history:
            return 0.8  # First contribution is novel
        
        # Simple heuristic: longer unique content = more novel
        # In production, use embeddings or similarity metrics
        recent_contents = [h.metadata.get("content", "") for h in history[-10:]]
        
        # Check for repetition
        for past in recent_contents:
            if content in past or past in content:
                return 0.2  # Low novelty if repeating
        
        # Length-based novelty (simple heuristic)
        if len(content) > 500:
            return 0.7
        elif len(content) > 200:
            return 0.6
        else:
            return 0.5
    
    def _calculate_building_score(
        self,
        contribution: Message,
        history: List[ScoringMetrics]
    ) -> float:
        """Calculate building-on-others score."""
        content = contribution.content.lower()
        
        # Check for references to other agents
        references = 0
        for metric in history[-5:]:
            agent_name = metric.metadata.get("agent_name", "")
            if agent_name and agent_name.lower() in content:
                references += 1
        
        # Check for building keywords
        building_keywords = [
            "building on", "as mentioned", "following up",
            "expanding on", "adding to", "furthermore"
        ]
        
        keyword_score = sum(1 for kw in building_keywords if kw in content)
        
        # Combine scores
        reference_score = min(1.0, references * 0.3)
        keyword_score = min(1.0, keyword_score * 0.2)
        
        return (reference_score + keyword_score) / 2
    
    def _calculate_solution_score(
        self,
        content: str,
        context: ScoringContext
    ) -> float:
        """Calculate problem-solving score."""
        content_lower = content.lower()
        
        # Solution keywords
        solution_keywords = [
            "solution", "solve", "approach", "method",
            "algorithm", "implement", "strategy", "plan"
        ]
        
        # Technical indicators
        technical_indicators = [
            "step", "first", "second", "then", "finally",
            "1.", "2.", "3.", "a)", "b)", "c)"
        ]
        
        keyword_count = sum(1 for kw in solution_keywords if kw in content_lower)
        technical_count = sum(1 for ind in technical_indicators if ind in content_lower)
        
        # Base score
        base_score = min(1.0, (keyword_count * 0.15 + technical_count * 0.1))
        
        # Adjust for problem complexity
        complexity_multiplier = 0.5 + (context.problem_complexity * 0.5)
        
        return base_score * complexity_multiplier
    
    def _calculate_radical_score(
        self,
        content: str,
        history: List[ScoringMetrics]
    ) -> float:
        """Calculate radical/paradigm-shift score."""
        content_lower = content.lower()
        
        # Radical indicators
        radical_keywords = [
            "completely different", "paradigm", "revolutionary",
            "what if", "contrary to", "instead of", "rethink",
            "fundamental", "breakthrough"
        ]
        
        keyword_score = sum(1 for kw in radical_keywords if kw in content_lower)
        
        # Check if very different from recent contributions
        if history:
            avg_length = np.mean([len(h.metadata.get("content", "")) 
                                 for h in history[-5:]])
            length_deviation = abs(len(content) - avg_length) / (avg_length + 1)
            
            if length_deviation > 2.0:  # Very different length
                keyword_score += 0.3
        
        return min(1.0, keyword_score * 0.25)
    
    def _detect_manipulation(
        self,
        contribution: Message,
        history: List[ScoringMetrics],
        agent_id: str
    ) -> float:
        """
        Detect manipulation attempts.
        
        Returns:
            Manipulation score (0.0 = none, 1.0 = blatant)
        """
        manipulation_score = 0.0
        content = contribution.content
        
        # Check for gaming patterns
        patterns = self.agent_patterns[agent_id]
        
        # Repetitive behavior
        if patterns["content_lengths"]:
            recent_lengths = patterns["content_lengths"][-5:]
            if len(set(recent_lengths)) == 1:  # All same length
                manipulation_score += 0.2
        
        # Keyword stuffing
        keyword_density = self._calculate_keyword_density(content)
        if keyword_density > 0.3:  # Too many keywords
            manipulation_score += 0.3
        
        # Copy-paste detection
        if self._detect_copy_paste(content, history):
            manipulation_score += 0.4
        
        # Rapid-fire contributions
        if self._detect_rapid_fire(agent_id, history):
            manipulation_score += 0.2
        
        return min(1.0, manipulation_score)
    
    def _calculate_keyword_density(self, content: str) -> float:
        """Calculate keyword stuffing density."""
        keywords = [
            "novel", "innovative", "creative", "solution",
            "brilliant", "paradigm", "revolutionary"
        ]
        
        word_count = len(content.split())
        if word_count == 0:
            return 0.0
        
        keyword_count = sum(content.lower().count(kw) for kw in keywords)
        return keyword_count / word_count
    
    def _detect_copy_paste(
        self,
        content: str,
        history: List[ScoringMetrics]
    ) -> bool:
        """Detect copy-paste behavior."""
        # Check for exact matches in recent history
        for metric in history[-10:]:
            past_content = metric.metadata.get("content", "")
            if past_content and (content in past_content or past_content in content):
                if len(content) > 50:  # Significant overlap
                    return True
        return False
    
    def _detect_rapid_fire(
        self,
        agent_id: str,
        history: List[ScoringMetrics]
    ) -> bool:
        """Detect rapid-fire contribution pattern."""
        recent_from_agent = [
            m for m in history[-10:]
            if m.agent_id == agent_id
        ]
        
        # Too many recent contributions
        return len(recent_from_agent) > 3
    
    def _apply_context_adjustments(
        self,
        scores: Dict[str, float],
        context: ScoringContext
    ) -> Dict[str, float]:
        """Apply contextual adjustments to scores."""
        adjusted = scores.copy()
        
        # Late game boost for solutions
        if context.game_phase in ["late", "final"]:
            adjusted["solves_subproblem"] *= 1.2
        
        # High pressure reduces radical ideas
        if context.elimination_pressure > 0.6:
            adjusted["radical_idea"] *= 0.8
        
        # Normalize to [0, 1]
        for key in adjusted:
            adjusted[key] = max(0.0, min(1.0, adjusted[key]))
        
        return adjusted
    
    def _track_patterns(self, agent_id: str, scores: Dict[str, float]) -> None:
        """Track agent patterns for manipulation detection."""
        patterns = self.agent_patterns[agent_id]
        
        # Track score patterns
        for dim, score in scores.items():
            patterns[f"{dim}_scores"].append(score)
            # Keep bounded history
            if len(patterns[f"{dim}_scores"]) > 20:
                patterns[f"{dim}_scores"].pop(0)


class WeightedScorer(ScoringStrategy):
    """Simple weighted scorer with fixed weights."""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """Initialize with custom weights."""
        self.weights = weights or {
            "novelty": 0.3,
            "builds_on_others": 0.2,
            "solves_subproblem": 0.3,
            "radical_idea": 0.1,
            "manipulation": -0.1
        }
    
    def score(
        self,
        contribution: Message,
        context: ScoringContext,
        history: List[ScoringMetrics]
    ) -> ScoringMetrics:
        """Score with fixed weights."""
        # Simple random scores for demo
        metrics = ScoringMetrics(
            agent_id=contribution.sender_id,
            message_id=contribution.message_id,
            turn_number=context.turn_number,
            novelty=np.random.uniform(0.3, 0.9),
            builds_on_others=np.random.uniform(0.2, 0.8),
            solves_subproblem=np.random.uniform(0.3, 0.85),
            radical_idea=np.random.uniform(0.0, 0.5),
            manipulation=np.random.uniform(0.0, 0.2)
        )
        
        metrics.calculate_weighted_score(self.weights)
        return metrics
    
    def get_weights(self, context: ScoringContext) -> Dict[str, float]:
        """Return fixed weights."""
        return self.weights


class AdaptiveScorer(ScoringStrategy):
    """
    Adaptive scorer that learns from game outcomes.
    
    This scorer adjusts its weights based on which scoring
    dimensions best predict game winners.
    """
    
    def __init__(self, learning_rate: float = 0.1):
        """Initialize adaptive scorer."""
        self.learning_rate = learning_rate
        
        # Start with balanced weights
        self.weights = {
            "novelty": 0.2,
            "builds_on_others": 0.2,
            "solves_subproblem": 0.2,
            "radical_idea": 0.2,
            "manipulation": -0.2
        }
        
        # Track weight performance
        self.weight_performance: Dict[str, List[float]] = defaultdict(list)
        
        # Use multidimensional scorer as base
        self.base_scorer = MultidimensionalScorer()
    
    def score(
        self,
        contribution: Message,
        context: ScoringContext,
        history: List[ScoringMetrics]
    ) -> ScoringMetrics:
        """Score with adaptive weights."""
        # Use base scorer for calculation
        metrics = self.base_scorer.score(contribution, context, history)
        
        # Apply adaptive weights
        metrics.calculate_weighted_score(self.weights)
        
        return metrics
    
    def get_weights(self, context: ScoringContext) -> Dict[str, float]:
        """Get current adaptive weights."""
        return self.weights.copy()
    
    def update_weights(
        self,
        winner_metrics: List[ScoringMetrics],
        loser_metrics: List[ScoringMetrics]
    ) -> None:
        """
        Update weights based on game outcome.
        
        Args:
            winner_metrics: Metrics from winning agents
            loser_metrics: Metrics from losing agents
        """
        if not winner_metrics or not loser_metrics:
            return
        
        # Calculate average scores for each dimension
        winner_avgs = self._calculate_averages(winner_metrics)
        loser_avgs = self._calculate_averages(loser_metrics)
        
        # Update weights based on discriminative power
        for dim in self.weights:
            if dim == "manipulation":
                # Manipulation should be negative
                diff = loser_avgs[dim] - winner_avgs[dim]
            else:
                diff = winner_avgs[dim] - loser_avgs[dim]
            
            # Update weight
            self.weights[dim] += self.learning_rate * diff
            
            # Track performance
            self.weight_performance[dim].append(diff)
        
        # Normalize weights
        self._normalize_weights()
    
    def _calculate_averages(
        self,
        metrics: List[ScoringMetrics]
    ) -> Dict[str, float]:
        """Calculate average scores across metrics."""
        avgs = defaultdict(float)
        
        for m in metrics:
            avgs["novelty"] += m.novelty
            avgs["builds_on_others"] += m.builds_on_others
            avgs["solves_subproblem"] += m.solves_subproblem
            avgs["radical_idea"] += m.radical_idea
            avgs["manipulation"] += m.manipulation
        
        n = len(metrics)
        return {k: v/n for k, v in avgs.items()}
    
    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1."""
        total = sum(abs(w) for w in self.weights.values())
        if total > 0:
            self.weights = {k: v/total for k, v in self.weights.items()}


class ScoringEngine:
    """
    Main scoring engine that orchestrates scoring strategies.
    
    Features:
    - Strategy selection
    - Score aggregation
    - Historical analysis
    - Fairness monitoring
    """
    
    def __init__(
        self,
        strategy: Optional[ScoringStrategy] = None,
        enable_fairness: bool = True
    ):
        """
        Initialize scoring engine.
        
        Args:
            strategy: Scoring strategy to use
            enable_fairness: Enable fairness monitoring
        """
        self.strategy = strategy or MultidimensionalScorer()
        self.enable_fairness = enable_fairness
        
        # Track all scores
        self.score_history: List[ScoringMetrics] = []
        self.agent_scores: Dict[str, List[float]] = defaultdict(list)
        
        # Fairness monitoring
        self.fairness_violations: List[Dict[str, Any]] = []
    
    def score_contribution(
        self,
        contribution: Message,
        context: ScoringContext
    ) -> ScoringMetrics:
        """
        Score a contribution.
        
        Args:
            contribution: Contribution to score
            context: Current game context
            
        Returns:
            Scoring metrics
        """
        # Score using strategy
        metrics = self.strategy.score(
            contribution,
            context,
            self.score_history
        )
        
        # Check fairness
        if self.enable_fairness:
            self._check_fairness(metrics, contribution.sender_id)
        
        # Track history
        self.score_history.append(metrics)
        self.agent_scores[contribution.sender_id].append(metrics.weighted_score)
        
        return metrics
    
    def get_agent_statistics(self, agent_id: str) -> Dict[str, Any]:
        """
        Get scoring statistics for an agent.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Statistics dictionary
        """
        scores = self.agent_scores.get(agent_id, [])
        
        if not scores:
            return {
                "agent_id": agent_id,
                "total_contributions": 0,
                "average_score": 0.0,
                "max_score": 0.0,
                "min_score": 0.0,
                "std_dev": 0.0,
                "trend": "stable"
            }
        
        return {
            "agent_id": agent_id,
            "total_contributions": len(scores),
            "average_score": np.mean(scores),
            "max_score": max(scores),
            "min_score": min(scores),
            "std_dev": np.std(scores),
            "trend": self._calculate_trend(scores)
        }
    
    def _check_fairness(self, metrics: ScoringMetrics, agent_id: str) -> None:
        """Check for scoring fairness violations."""
        if not self.agent_scores[agent_id]:
            return
        
        recent_scores = self.agent_scores[agent_id][-5:]
        
        # Check for consistent low scores (potential bias)
        if len(recent_scores) >= 5 and max(recent_scores) < 0.3:
            self.fairness_violations.append({
                "type": "consistent_low_scores",
                "agent_id": agent_id,
                "scores": recent_scores,
                "timestamp": datetime.utcnow()
            })
        
        # Check for score volatility (potential randomness)
        if len(recent_scores) >= 3:
            volatility = np.std(recent_scores)
            if volatility > 0.4:
                self.fairness_violations.append({
                    "type": "high_volatility",
                    "agent_id": agent_id,
                    "volatility": volatility,
                    "timestamp": datetime.utcnow()
                })
    
    def _calculate_trend(self, scores: List[float]) -> str:
        """Calculate score trend."""
        if len(scores) < 3:
            return "insufficient_data"
        
        recent = scores[-3:]
        older = scores[-6:-3] if len(scores) >= 6 else scores[:-3]
        
        recent_avg = np.mean(recent)
        older_avg = np.mean(older)
        
        diff = recent_avg - older_avg
        
        if diff > 0.1:
            return "improving"
        elif diff < -0.1:
            return "declining"
        else:
            return "stable"
    
    def get_leaderboard(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Get current leaderboard.
        
        Args:
            top_n: Number of top agents to return
            
        Returns:
            List of (agent_id, average_score) tuples
        """
        agent_avgs = []
        
        for agent_id, scores in self.agent_scores.items():
            if scores:
                avg_score = np.mean(scores)
                agent_avgs.append((agent_id, avg_score))
        
        # Sort by average score
        agent_avgs.sort(key=lambda x: x[1], reverse=True)
        
        return agent_avgs[:top_n]