"""
Elimination Mechanics for Arena

This module implements fair and strategic elimination mechanics
for the Arena competitive environment.

Features:
- Multiple elimination strategies
- Fairness guarantees
- Protection periods
- Appeal mechanisms
- Comeback opportunities

Author: Homunculus Team
"""

import logging
import random
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np

from ..models import AgentState, Message, AccusationOutcome

logger = logging.getLogger(__name__)


@dataclass
class EliminationContext:
    """Context for elimination decisions."""
    turn_number: int
    total_agents: int
    active_agents: int
    elimination_round: int
    scores: Dict[str, float]  # agent_id -> current score
    accusations: Dict[str, List[str]]  # agent_id -> list of accusers
    protections: Dict[str, bool]  # agent_id -> protected status
    
    @property
    def elimination_rate(self) -> float:
        """Calculate current elimination rate."""
        if self.total_agents == 0:
            return 0.0
        eliminated = self.total_agents - self.active_agents
        return eliminated / self.total_agents
    
    @property
    def should_eliminate(self) -> bool:
        """Determine if elimination should occur."""
        # Eliminate every 10 turns after turn 20
        if self.turn_number < 20:
            return False
        return self.turn_number % 10 == 0


@dataclass
class EliminationCandidate:
    """Candidate for elimination."""
    agent_id: str
    score: float
    elimination_score: float  # Combined score for elimination
    reasons: List[str]
    is_protected: bool = False
    appeal_count: int = 0
    
    def can_be_eliminated(self) -> bool:
        """Check if candidate can be eliminated."""
        return not self.is_protected and self.appeal_count < 2


class EliminationStrategy(ABC):
    """Abstract base class for elimination strategies."""
    
    @abstractmethod
    def select_for_elimination(
        self,
        context: EliminationContext,
        candidates: List[EliminationCandidate]
    ) -> List[str]:
        """Select agents for elimination."""
        pass
    
    @abstractmethod
    def get_elimination_count(self, context: EliminationContext) -> int:
        """Determine how many agents to eliminate."""
        pass


class FairElimination(EliminationStrategy):
    """
    Fair elimination strategy with multiple safeguards.
    
    Features:
    - Grace period for new agents
    - Protection for top performers
    - Random tiebreaking
    - Appeal consideration
    """
    
    def __init__(
        self,
        grace_period: int = 5,
        protect_top_percent: float = 0.2,
        min_score_threshold: float = 0.3
    ):
        """
        Initialize fair elimination.
        
        Args:
            grace_period: Turns before agent can be eliminated
            protect_top_percent: Percentage of top agents to protect
            min_score_threshold: Minimum score to avoid elimination
        """
        self.grace_period = grace_period
        self.protect_top_percent = protect_top_percent
        self.min_score_threshold = min_score_threshold
        
        # Track agent history
        self.agent_history: Dict[str, Dict[str, Any]] = defaultdict(dict)
    
    def select_for_elimination(
        self,
        context: EliminationContext,
        candidates: List[EliminationCandidate]
    ) -> List[str]:
        """
        Select agents for elimination fairly.
        
        Args:
            context: Elimination context
            candidates: List of candidates
            
        Returns:
            List of agent IDs to eliminate
        """
        if not candidates:
            return []
        
        # Apply protections
        eligible_candidates = self._apply_protections(candidates, context)
        
        if not eligible_candidates:
            logger.info("No eligible candidates for elimination")
            return []
        
        # Sort by elimination score
        eligible_candidates.sort(key=lambda c: c.elimination_score)
        
        # Get elimination count
        count = self.get_elimination_count(context)
        count = min(count, len(eligible_candidates))
        
        # Select bottom performers with fairness
        selected = []
        for candidate in eligible_candidates[:count * 2]:  # Consider more candidates
            if len(selected) >= count:
                break
            
            # Apply fairness checks
            if self._should_eliminate(candidate, context):
                selected.append(candidate.agent_id)
                logger.info(f"Selected {candidate.agent_id} for elimination: {candidate.reasons}")
        
        # If not enough selected, fill with worst performers
        if len(selected) < count:
            remaining = [c.agent_id for c in eligible_candidates 
                        if c.agent_id not in selected][:count - len(selected)]
            selected.extend(remaining)
        
        return selected
    
    def get_elimination_count(self, context: EliminationContext) -> int:
        """
        Determine elimination count based on game phase.
        
        Args:
            context: Elimination context
            
        Returns:
            Number of agents to eliminate
        """
        active = context.active_agents
        
        # Never eliminate more than 20% at once
        max_eliminate = max(1, int(active * 0.2))
        
        # Scale with game progress
        if context.elimination_round < 3:
            # Early game: eliminate fewer
            count = max(1, min(2, max_eliminate))
        elif context.elimination_round < 6:
            # Mid game: standard elimination
            count = max(1, min(3, max_eliminate))
        else:
            # Late game: accelerate
            count = max(1, min(4, max_eliminate))
        
        # Ensure at least 3 agents remain
        count = min(count, active - 3)
        
        return max(0, count)
    
    def _apply_protections(
        self,
        candidates: List[EliminationCandidate],
        context: EliminationContext
    ) -> List[EliminationCandidate]:
        """Apply protection rules."""
        eligible = []
        
        # Sort by score to identify top performers
        sorted_candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
        
        # Protect top performers
        protect_count = max(1, int(len(candidates) * self.protect_top_percent))
        protected_ids = {c.agent_id for c in sorted_candidates[:protect_count]}
        
        for candidate in candidates:
            # Check if protected
            if candidate.agent_id in protected_ids:
                candidate.is_protected = True
                candidate.reasons.append("Protected: top performer")
                continue
            
            # Check grace period
            history = self.agent_history.get(candidate.agent_id, {})
            turns_active = history.get("turns_active", 0)
            
            if turns_active < self.grace_period:
                candidate.is_protected = True
                candidate.reasons.append(f"Protected: grace period ({turns_active}/{self.grace_period})")
                continue
            
            # Check minimum score threshold
            if candidate.score >= self.min_score_threshold:
                # Give them a chance
                if random.random() < 0.5:  # 50% protection
                    candidate.is_protected = True
                    candidate.reasons.append("Protected: met minimum threshold")
                    continue
            
            # Check if already protected this round
            if context.protections.get(candidate.agent_id, False):
                candidate.is_protected = True
                candidate.reasons.append("Protected: special protection active")
                continue
            
            eligible.append(candidate)
        
        return eligible
    
    def _should_eliminate(
        self,
        candidate: EliminationCandidate,
        context: EliminationContext
    ) -> bool:
        """
        Final fairness check for elimination.
        
        Args:
            candidate: Elimination candidate
            context: Elimination context
            
        Returns:
            True if should eliminate
        """
        # Check appeals
        if candidate.appeal_count > 0:
            # Give them another chance
            if random.random() < 0.3 * candidate.appeal_count:
                return False
        
        # Check if they've been accused
        accusations = context.accusations.get(candidate.agent_id, [])
        if len(accusations) > 2:
            # Multiple accusations increase elimination chance
            return True
        
        # Random factor for fairness
        elimination_probability = min(0.9, candidate.elimination_score + 0.1)
        return random.random() < elimination_probability


class PerformanceBasedElimination(EliminationStrategy):
    """
    Elimination based purely on performance metrics.
    
    Simple strategy that eliminates lowest performers.
    """
    
    def __init__(self, score_weight: float = 0.8, recency_weight: float = 0.2):
        """
        Initialize performance-based elimination.
        
        Args:
            score_weight: Weight for overall score
            recency_weight: Weight for recent performance
        """
        self.score_weight = score_weight
        self.recency_weight = recency_weight
        
        # Track recent scores
        self.recent_scores: Dict[str, List[float]] = defaultdict(list)
    
    def select_for_elimination(
        self,
        context: EliminationContext,
        candidates: List[EliminationCandidate]
    ) -> List[str]:
        """Select lowest performers for elimination."""
        if not candidates:
            return []
        
        # Calculate combined scores
        for candidate in candidates:
            # Get recent performance
            recent = self.recent_scores.get(candidate.agent_id, [candidate.score])
            recent_avg = np.mean(recent[-5:]) if recent else candidate.score
            
            # Combine overall and recent scores
            combined = (
                self.score_weight * candidate.score +
                self.recency_weight * recent_avg
            )
            candidate.elimination_score = 1.0 - combined  # Invert for elimination
        
        # Sort by elimination score (worst first)
        candidates.sort(key=lambda c: c.elimination_score, reverse=True)
        
        # Get count and select
        count = self.get_elimination_count(context)
        selected = [c.agent_id for c in candidates[:count] 
                   if c.can_be_eliminated()]
        
        return selected
    
    def get_elimination_count(self, context: EliminationContext) -> int:
        """Standard elimination count."""
        # Eliminate 1-2 agents per round
        if context.active_agents > 10:
            return 2
        elif context.active_agents > 5:
            return 1
        else:
            return 0  # Don't eliminate if too few remain
    
    def update_recent_scores(self, agent_id: str, score: float) -> None:
        """Update recent score tracking."""
        self.recent_scores[agent_id].append(score)
        # Keep bounded history
        if len(self.recent_scores[agent_id]) > 10:
            self.recent_scores[agent_id].pop(0)


class AccusationBasedElimination(EliminationStrategy):
    """
    Elimination based on proven cheating accusations.
    
    Immediate elimination for proven cheaters.
    """
    
    def __init__(self, false_accusation_penalty: float = 0.2):
        """
        Initialize accusation-based elimination.
        
        Args:
            false_accusation_penalty: Score penalty for false accusations
        """
        self.false_accusation_penalty = false_accusation_penalty
        
        # Track accusations
        self.proven_cheaters: Set[str] = set()
        self.false_accusers: Dict[str, int] = defaultdict(int)
    
    def select_for_elimination(
        self,
        context: EliminationContext,
        candidates: List[EliminationCandidate]
    ) -> List[str]:
        """Select proven cheaters for elimination."""
        eliminated = []
        
        for candidate in candidates:
            if candidate.agent_id in self.proven_cheaters:
                eliminated.append(candidate.agent_id)
                candidate.reasons.append("Eliminated: proven cheating")
        
        return eliminated
    
    def get_elimination_count(self, context: EliminationContext) -> int:
        """No fixed count - eliminate all proven cheaters."""
        return len(self.proven_cheaters)
    
    def process_accusation_verdict(
        self,
        accuser: str,
        accused: str,
        outcome: AccusationOutcome
    ) -> None:
        """
        Process an accusation verdict.
        
        Args:
            accuser: Agent who made accusation
            accused: Agent who was accused
            outcome: Verdict outcome
        """
        if outcome == AccusationOutcome.PROVEN:
            self.proven_cheaters.add(accused)
            logger.info(f"Marked {accused} as proven cheater")
        elif outcome == AccusationOutcome.FALSE:
            self.false_accusers[accuser] += 1
            logger.info(f"Recorded false accusation by {accuser}")


class EliminationEngine:
    """
    Main elimination engine that orchestrates elimination strategies.
    
    Features:
    - Strategy composition
    - Protection management
    - Appeal system
    - Comeback mechanics
    """
    
    def __init__(
        self,
        strategies: Optional[List[EliminationStrategy]] = None,
        enable_protection: bool = True,
        enable_appeals: bool = True,
        enable_comebacks: bool = True
    ):
        """
        Initialize elimination engine.
        
        Args:
            strategies: List of elimination strategies
            enable_protection: Enable protection mechanics
            enable_appeals: Enable appeal system
            enable_comebacks: Enable comeback opportunities
        """
        self.strategies = strategies or [
            FairElimination(),
            PerformanceBasedElimination(),
            AccusationBasedElimination()
        ]
        
        self.enable_protection = enable_protection
        self.enable_appeals = enable_appeals
        self.enable_comebacks = enable_comebacks
        
        # Track eliminations
        self.elimination_history: List[Dict[str, Any]] = []
        self.eliminated_agents: Dict[str, Dict[str, Any]] = {}
        
        # Protection system
        self.protected_agents: Set[str] = set()
        self.protection_tokens: Dict[str, int] = defaultdict(int)
        
        # Appeal system
        self.pending_appeals: Dict[str, Dict[str, Any]] = {}
        
        # Comeback system
        self.comeback_candidates: List[str] = []
    
    def process_elimination_round(
        self,
        context: EliminationContext,
        agent_states: Dict[str, AgentState]
    ) -> List[str]:
        """
        Process an elimination round.
        
        Args:
            context: Elimination context
            agent_states: Current agent states
            
        Returns:
            List of eliminated agent IDs
        """
        if not context.should_eliminate:
            return []
        
        logger.info(f"Processing elimination round {context.elimination_round}")
        
        # Create candidates
        candidates = self._create_candidates(context, agent_states)
        
        # Apply protection tokens
        if self.enable_protection:
            self._apply_protection_tokens(candidates)
        
        # Collect elimination selections from all strategies
        all_selections = []
        strategy_weights = [0.5, 0.3, 0.2]  # Weight each strategy
        
        for strategy, weight in zip(self.strategies, strategy_weights):
            selected = strategy.select_for_elimination(context, candidates)
            all_selections.extend([(agent_id, weight) for agent_id in selected])
        
        # Aggregate selections
        elimination_scores = defaultdict(float)
        for agent_id, weight in all_selections:
            elimination_scores[agent_id] += weight
        
        # Sort by aggregate score and select top candidates
        sorted_agents = sorted(
            elimination_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Determine final count
        base_count = self.strategies[0].get_elimination_count(context)
        
        # Select agents for elimination
        eliminated = []
        for agent_id, score in sorted_agents:
            if len(eliminated) >= base_count:
                break
            
            # Check for pending appeals
            if self.enable_appeals and agent_id in self.pending_appeals:
                if self._process_appeal(agent_id):
                    continue  # Appeal successful
            
            eliminated.append(agent_id)
            self._record_elimination(agent_id, context, score)
        
        # Check for comeback opportunities
        if self.enable_comebacks and len(eliminated) > 0:
            comeback = self._check_comeback_opportunity(context)
            if comeback:
                eliminated.remove(comeback)
                self._process_comeback(comeback)
        
        logger.info(f"Eliminated agents: {eliminated}")
        return eliminated
    
    def _create_candidates(
        self,
        context: EliminationContext,
        agent_states: Dict[str, AgentState]
    ) -> List[EliminationCandidate]:
        """Create elimination candidates from agent states."""
        candidates = []
        
        for agent_id, state in agent_states.items():
            if not state.is_active:
                continue
            
            score = context.scores.get(agent_id, 0.0)
            
            candidate = EliminationCandidate(
                agent_id=agent_id,
                score=score,
                elimination_score=1.0 - score,  # Initial score
                reasons=[],
                is_protected=agent_id in self.protected_agents
            )
            
            # Add contextual reasons
            if score < 0.3:
                candidate.reasons.append("Low performance score")
            
            accusations = context.accusations.get(agent_id, [])
            if accusations:
                candidate.reasons.append(f"Accused by {len(accusations)} agents")
            
            candidates.append(candidate)
        
        return candidates
    
    def _apply_protection_tokens(
        self,
        candidates: List[EliminationCandidate]
    ) -> None:
        """Apply protection tokens to candidates."""
        for candidate in candidates:
            tokens = self.protection_tokens.get(candidate.agent_id, 0)
            if tokens > 0:
                candidate.is_protected = True
                candidate.reasons.append(f"Used protection token ({tokens} remaining)")
                self.protection_tokens[candidate.agent_id] -= 1
    
    def _process_appeal(self, agent_id: str) -> bool:
        """
        Process an appeal.
        
        Args:
            agent_id: Agent making appeal
            
        Returns:
            True if appeal successful
        """
        appeal = self.pending_appeals.get(agent_id, {})
        
        # Simple appeal logic - could be more sophisticated
        appeal_strength = appeal.get("strength", 0.5)
        
        # Random chance based on appeal strength
        success = random.random() < appeal_strength
        
        if success:
            logger.info(f"Appeal successful for {agent_id}")
            self.protected_agents.add(agent_id)
        
        # Remove processed appeal
        self.pending_appeals.pop(agent_id, None)
        
        return success
    
    def _check_comeback_opportunity(
        self,
        context: EliminationContext
    ) -> Optional[str]:
        """
        Check for comeback opportunity.
        
        Args:
            context: Elimination context
            
        Returns:
            Agent ID to bring back, or None
        """
        if not self.comeback_candidates:
            return None
        
        # Simple comeback: bring back highest scoring eliminated agent
        # if current lowest active score is very low
        min_active_score = min(context.scores.values()) if context.scores else 0
        
        if min_active_score < 0.2:  # Very poor performance
            # Bring back best eliminated agent
            best_eliminated = self.comeback_candidates[0]
            return best_eliminated
        
        return None
    
    def _process_comeback(self, agent_id: str) -> None:
        """Process agent comeback."""
        logger.info(f"Comeback for {agent_id}")
        
        # Remove from eliminated
        self.eliminated_agents.pop(agent_id, None)
        self.comeback_candidates.remove(agent_id)
        
        # Add protection
        self.protected_agents.add(agent_id)
        self.protection_tokens[agent_id] = 1
    
    def _record_elimination(
        self,
        agent_id: str,
        context: EliminationContext,
        score: float
    ) -> None:
        """Record an elimination."""
        elimination_record = {
            "agent_id": agent_id,
            "turn": context.turn_number,
            "round": context.elimination_round,
            "score": context.scores.get(agent_id, 0),
            "elimination_score": score,
            "timestamp": datetime.utcnow()
        }
        
        self.elimination_history.append(elimination_record)
        self.eliminated_agents[agent_id] = elimination_record
        
        # Add to comeback candidates (sorted by score)
        self.comeback_candidates.append(agent_id)
        self.comeback_candidates.sort(
            key=lambda x: self.eliminated_agents[x]["score"],
            reverse=True
        )
        
        # Keep only top 3 comeback candidates
        self.comeback_candidates = self.comeback_candidates[:3]
    
    def grant_protection(self, agent_id: str, duration: int = 1) -> None:
        """
        Grant protection to an agent.
        
        Args:
            agent_id: Agent to protect
            duration: Number of rounds to protect
        """
        self.protection_tokens[agent_id] += duration
        logger.info(f"Granted {duration} protection tokens to {agent_id}")
    
    def submit_appeal(
        self,
        agent_id: str,
        reason: str,
        evidence: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Submit an elimination appeal.
        
        Args:
            agent_id: Agent submitting appeal
            reason: Appeal reason
            evidence: Supporting evidence
        """
        appeal_strength = 0.3  # Base strength
        
        # Increase strength based on evidence
        if evidence:
            if "high_recent_score" in evidence:
                appeal_strength += 0.2
            if "unique_contribution" in evidence:
                appeal_strength += 0.2
            if "unfair_accusation" in evidence:
                appeal_strength += 0.3
        
        self.pending_appeals[agent_id] = {
            "reason": reason,
            "evidence": evidence,
            "strength": min(0.9, appeal_strength),
            "timestamp": datetime.utcnow()
        }
        
        logger.info(f"Appeal submitted by {agent_id}: {reason}")
    
    def get_elimination_statistics(self) -> Dict[str, Any]:
        """Get elimination statistics."""
        return {
            "total_eliminated": len(self.eliminated_agents),
            "elimination_rounds": len(set(e["round"] for e in self.elimination_history)),
            "protected_agents": len(self.protected_agents),
            "pending_appeals": len(self.pending_appeals),
            "comeback_candidates": len(self.comeback_candidates),
            "elimination_history": self.elimination_history[-10:]  # Last 10
        }