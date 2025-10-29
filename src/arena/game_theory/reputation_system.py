"""
Reputation System for Arena

This module implements a comprehensive reputation and trust
system for tracking agent behavior and credibility.

Features:
- Multi-factor reputation scoring
- Trust networks
- Credibility assessment
- Reputation decay over time
- Behavioral tracking

Author: Homunculus Team
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum
import numpy as np

from ..models import Message, AgentState, AccusationOutcome

logger = logging.getLogger(__name__)


class ReputationFactor(Enum):
    """Factors that influence reputation."""
    CONTRIBUTION_QUALITY = "contribution_quality"
    ACCUSATION_ACCURACY = "accusation_accuracy"
    COLLABORATION = "collaboration"
    CONSISTENCY = "consistency"
    FAIRNESS = "fairness"
    INNOVATION = "innovation"
    HELPFULNESS = "helpfulness"


@dataclass
class TrustMetric:
    """Trust relationship between two agents."""
    trustor: str  # Agent giving trust
    trustee: str  # Agent receiving trust
    trust_level: float  # 0.0 to 1.0
    confidence: float  # Confidence in trust assessment
    last_updated: datetime
    interaction_count: int
    positive_interactions: int
    negative_interactions: int
    
    @property
    def trust_score(self) -> float:
        """Calculate weighted trust score."""
        return self.trust_level * self.confidence
    
    def update_interaction(self, positive: bool) -> None:
        """Update based on interaction outcome."""
        self.interaction_count += 1
        
        if positive:
            self.positive_interactions += 1
        else:
            self.negative_interactions += 1
        
        # Update trust level
        if self.interaction_count > 0:
            self.trust_level = self.positive_interactions / self.interaction_count
        
        # Update confidence (more interactions = higher confidence)
        self.confidence = min(1.0, self.interaction_count / 10)
        
        self.last_updated = datetime.utcnow()


@dataclass
class CredibilityScore:
    """Agent credibility assessment."""
    agent_id: str
    overall_score: float  # 0.0 to 1.0
    factors: Dict[ReputationFactor, float]
    evidence: List[str]
    last_calculated: datetime
    
    def get_factor_score(self, factor: ReputationFactor) -> float:
        """Get score for specific factor."""
        return self.factors.get(factor, 0.5)
    
    def is_credible(self, threshold: float = 0.6) -> bool:
        """Check if agent is credible."""
        return self.overall_score >= threshold


@dataclass
class ReputationEvent:
    """Event that affects reputation."""
    agent_id: str
    event_type: str
    impact: float  # -1.0 to 1.0
    factor: ReputationFactor
    description: str
    timestamp: datetime
    evidence: Optional[Dict[str, Any]] = None


class ReputationDecay:
    """
    Handles reputation decay over time.
    
    Recent events have more weight than older ones.
    """
    
    def __init__(
        self,
        half_life_days: float = 7.0,
        min_weight: float = 0.1
    ):
        """
        Initialize reputation decay.
        
        Args:
            half_life_days: Days for reputation to decay by half
            min_weight: Minimum weight for old events
        """
        self.half_life = timedelta(days=half_life_days)
        self.min_weight = min_weight
    
    def calculate_weight(self, event_time: datetime) -> float:
        """
        Calculate weight based on time decay.
        
        Args:
            event_time: Time of event
            
        Returns:
            Weight between min_weight and 1.0
        """
        now = datetime.utcnow()
        age = now - event_time
        
        # Exponential decay
        half_lives_passed = age.total_seconds() / self.half_life.total_seconds()
        weight = math.pow(0.5, half_lives_passed)
        
        return max(self.min_weight, weight)
    
    def apply_decay(
        self,
        events: List[ReputationEvent]
    ) -> List[Tuple[ReputationEvent, float]]:
        """
        Apply decay to list of events.
        
        Args:
            events: List of reputation events
            
        Returns:
            List of (event, weight) tuples
        """
        weighted_events = []
        
        for event in events:
            weight = self.calculate_weight(event.timestamp)
            weighted_events.append((event, weight))
        
        return weighted_events


class ReputationEngine:
    """
    Main reputation engine for tracking and calculating agent reputation.
    """
    
    def __init__(
        self,
        initial_reputation: float = 0.5,
        decay_enabled: bool = True
    ):
        """
        Initialize reputation engine.
        
        Args:
            initial_reputation: Starting reputation for new agents
            decay_enabled: Enable time-based decay
        """
        self.initial_reputation = initial_reputation
        self.decay_enabled = decay_enabled
        
        # Reputation tracking
        self.reputations: Dict[str, float] = defaultdict(lambda: initial_reputation)
        self.reputation_history: Dict[str, List[ReputationEvent]] = defaultdict(list)
        
        # Trust network
        self.trust_network: Dict[Tuple[str, str], TrustMetric] = {}
        
        # Credibility scores
        self.credibility_scores: Dict[str, CredibilityScore] = {}
        
        # Decay handler
        self.decay_handler = ReputationDecay()
        
        # Factor weights
        self.factor_weights = {
            ReputationFactor.CONTRIBUTION_QUALITY: 0.25,
            ReputationFactor.ACCUSATION_ACCURACY: 0.20,
            ReputationFactor.COLLABORATION: 0.15,
            ReputationFactor.CONSISTENCY: 0.15,
            ReputationFactor.FAIRNESS: 0.10,
            ReputationFactor.INNOVATION: 0.10,
            ReputationFactor.HELPFULNESS: 0.05
        }
    
    def update_reputation(
        self,
        agent_id: str,
        event: ReputationEvent
    ) -> float:
        """
        Update agent reputation based on event.
        
        Args:
            agent_id: Agent ID
            event: Reputation event
            
        Returns:
            New reputation score
        """
        # Record event
        self.reputation_history[agent_id].append(event)
        
        # Calculate new reputation
        new_reputation = self._calculate_reputation(agent_id)
        self.reputations[agent_id] = new_reputation
        
        # Update credibility
        self._update_credibility(agent_id)
        
        logger.info(
            f"Updated reputation for {agent_id}: {new_reputation:.3f} "
            f"(event: {event.event_type}, impact: {event.impact:+.3f})"
        )
        
        return new_reputation
    
    def _calculate_reputation(self, agent_id: str) -> float:
        """
        Calculate overall reputation score.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Reputation score (0.0 to 1.0)
        """
        events = self.reputation_history[agent_id]
        
        if not events:
            return self.initial_reputation
        
        # Apply decay if enabled
        if self.decay_enabled:
            weighted_events = self.decay_handler.apply_decay(events)
        else:
            weighted_events = [(e, 1.0) for e in events]
        
        # Calculate factor scores
        factor_scores = defaultdict(list)
        
        for event, weight in weighted_events:
            factor_scores[event.factor].append(event.impact * weight)
        
        # Average each factor
        factor_averages = {}
        for factor, scores in factor_scores.items():
            if scores:
                factor_averages[factor] = np.mean(scores)
            else:
                factor_averages[factor] = 0.0
        
        # Apply factor weights
        weighted_score = 0.0
        total_weight = 0.0
        
        for factor, weight in self.factor_weights.items():
            if factor in factor_averages:
                weighted_score += factor_averages[factor] * weight
                total_weight += weight
        
        if total_weight > 0:
            final_score = weighted_score / total_weight
        else:
            final_score = self.initial_reputation
        
        # Normalize to [0, 1]
        return max(0.0, min(1.0, (final_score + 1.0) / 2.0))
    
    def update_trust(
        self,
        trustor: str,
        trustee: str,
        interaction_positive: bool
    ) -> TrustMetric:
        """
        Update trust between agents.
        
        Args:
            trustor: Agent giving trust
            trustee: Agent receiving trust
            interaction_positive: Whether interaction was positive
            
        Returns:
            Updated trust metric
        """
        key = (trustor, trustee)
        
        if key not in self.trust_network:
            # Create new trust relationship
            self.trust_network[key] = TrustMetric(
                trustor=trustor,
                trustee=trustee,
                trust_level=0.5,
                confidence=0.1,
                last_updated=datetime.utcnow(),
                interaction_count=0,
                positive_interactions=0,
                negative_interactions=0
            )
        
        trust_metric = self.trust_network[key]
        trust_metric.update_interaction(interaction_positive)
        
        return trust_metric
    
    def get_trust_level(
        self,
        trustor: str,
        trustee: str
    ) -> float:
        """
        Get trust level between agents.
        
        Args:
            trustor: Agent giving trust
            trustee: Agent receiving trust
            
        Returns:
            Trust level (0.0 to 1.0)
        """
        key = (trustor, trustee)
        
        if key in self.trust_network:
            return self.trust_network[key].trust_level
        
        # Default trust based on reputation
        return self.reputations.get(trustee, self.initial_reputation)
    
    def calculate_credibility(
        self,
        agent_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> CredibilityScore:
        """
        Calculate agent credibility.
        
        Args:
            agent_id: Agent ID
            context: Optional context for calculation
            
        Returns:
            Credibility score
        """
        # Get reputation events
        events = self.reputation_history[agent_id]
        
        # Calculate factor scores
        factor_scores = {}
        evidence = []
        
        # Contribution quality
        quality_events = [e for e in events 
                         if e.factor == ReputationFactor.CONTRIBUTION_QUALITY]
        if quality_events:
            avg_quality = np.mean([e.impact for e in quality_events[-10:]])
            factor_scores[ReputationFactor.CONTRIBUTION_QUALITY] = (avg_quality + 1) / 2
            if avg_quality > 0.5:
                evidence.append("High quality contributions")
        else:
            factor_scores[ReputationFactor.CONTRIBUTION_QUALITY] = 0.5
        
        # Accusation accuracy
        accuracy_events = [e for e in events 
                          if e.factor == ReputationFactor.ACCUSATION_ACCURACY]
        if accuracy_events:
            avg_accuracy = np.mean([e.impact for e in accuracy_events])
            factor_scores[ReputationFactor.ACCUSATION_ACCURACY] = (avg_accuracy + 1) / 2
            if avg_accuracy > 0.3:
                evidence.append("Accurate accusations")
            elif avg_accuracy < -0.3:
                evidence.append("History of false accusations")
        else:
            factor_scores[ReputationFactor.ACCUSATION_ACCURACY] = 0.5
        
        # Collaboration
        collab_events = [e for e in events 
                        if e.factor == ReputationFactor.COLLABORATION]
        if collab_events:
            avg_collab = np.mean([e.impact for e in collab_events[-10:]])
            factor_scores[ReputationFactor.COLLABORATION] = (avg_collab + 1) / 2
            if avg_collab > 0.4:
                evidence.append("Good collaborator")
        else:
            factor_scores[ReputationFactor.COLLABORATION] = 0.5
        
        # Calculate overall credibility
        overall_score = self.reputations.get(agent_id, self.initial_reputation)
        
        credibility = CredibilityScore(
            agent_id=agent_id,
            overall_score=overall_score,
            factors=factor_scores,
            evidence=evidence,
            last_calculated=datetime.utcnow()
        )
        
        # Cache result
        self.credibility_scores[agent_id] = credibility
        
        return credibility
    
    def _update_credibility(self, agent_id: str) -> None:
        """Update cached credibility score."""
        self.calculate_credibility(agent_id)
    
    def process_contribution(
        self,
        agent_id: str,
        quality_score: float,
        innovative: bool = False
    ) -> None:
        """
        Process a contribution for reputation impact.
        
        Args:
            agent_id: Contributing agent
            quality_score: Quality of contribution (0.0 to 1.0)
            innovative: Whether contribution was innovative
        """
        # Map quality to impact
        impact = (quality_score - 0.5) * 2  # Convert to [-1, 1]
        
        # Quality event
        quality_event = ReputationEvent(
            agent_id=agent_id,
            event_type="contribution",
            impact=impact,
            factor=ReputationFactor.CONTRIBUTION_QUALITY,
            description=f"Contribution with quality {quality_score:.2f}",
            timestamp=datetime.utcnow()
        )
        
        self.update_reputation(agent_id, quality_event)
        
        # Innovation bonus
        if innovative:
            innovation_event = ReputationEvent(
                agent_id=agent_id,
                event_type="innovation",
                impact=0.5,
                factor=ReputationFactor.INNOVATION,
                description="Innovative contribution",
                timestamp=datetime.utcnow()
            )
            self.update_reputation(agent_id, innovation_event)
    
    def process_accusation_outcome(
        self,
        accuser: str,
        accused: str,
        outcome: AccusationOutcome
    ) -> None:
        """
        Process accusation outcome for reputation impact.
        
        Args:
            accuser: Agent who made accusation
            accused: Agent who was accused
            outcome: Accusation outcome
        """
        if outcome == AccusationOutcome.PROVEN:
            # Positive impact for accurate accusation
            accuser_event = ReputationEvent(
                agent_id=accuser,
                event_type="accurate_accusation",
                impact=0.7,
                factor=ReputationFactor.ACCUSATION_ACCURACY,
                description="Made accurate cheating accusation",
                timestamp=datetime.utcnow()
            )
            self.update_reputation(accuser, accuser_event)
            
            # Negative impact for cheating
            accused_event = ReputationEvent(
                agent_id=accused,
                event_type="proven_cheating",
                impact=-1.0,
                factor=ReputationFactor.FAIRNESS,
                description="Proven cheating",
                timestamp=datetime.utcnow()
            )
            self.update_reputation(accused, accused_event)
            
        elif outcome == AccusationOutcome.FALSE:
            # Negative impact for false accusation
            accuser_event = ReputationEvent(
                agent_id=accuser,
                event_type="false_accusation",
                impact=-0.5,
                factor=ReputationFactor.ACCUSATION_ACCURACY,
                description="Made false accusation",
                timestamp=datetime.utcnow()
            )
            self.update_reputation(accuser, accuser_event)
            
            # Small positive for being falsely accused
            accused_event = ReputationEvent(
                agent_id=accused,
                event_type="false_accusation_target",
                impact=0.2,
                factor=ReputationFactor.FAIRNESS,
                description="Falsely accused",
                timestamp=datetime.utcnow()
            )
            self.update_reputation(accused, accused_event)
    
    def process_collaboration(
        self,
        agent1: str,
        agent2: str,
        quality: float
    ) -> None:
        """
        Process collaboration between agents.
        
        Args:
            agent1: First collaborator
            agent2: Second collaborator
            quality: Quality of collaboration (0.0 to 1.0)
        """
        impact = (quality - 0.5) * 2  # Convert to [-1, 1]
        
        for agent_id in [agent1, agent2]:
            event = ReputationEvent(
                agent_id=agent_id,
                event_type="collaboration",
                impact=impact,
                factor=ReputationFactor.COLLABORATION,
                description=f"Collaboration with quality {quality:.2f}",
                timestamp=datetime.utcnow()
            )
            self.update_reputation(agent_id, event)
        
        # Update trust if collaboration was positive
        if impact > 0:
            self.update_trust(agent1, agent2, True)
            self.update_trust(agent2, agent1, True)
        elif impact < 0:
            self.update_trust(agent1, agent2, False)
            self.update_trust(agent2, agent1, False)
    
    def get_reputation_report(
        self,
        agent_id: str
    ) -> Dict[str, Any]:
        """
        Get comprehensive reputation report for agent.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Reputation report
        """
        reputation = self.reputations.get(agent_id, self.initial_reputation)
        credibility = self.credibility_scores.get(agent_id)
        
        # Get trust relationships
        trust_given = {}
        trust_received = {}
        
        for (trustor, trustee), metric in self.trust_network.items():
            if trustor == agent_id:
                trust_given[trustee] = metric.trust_level
            elif trustee == agent_id:
                trust_received[trustor] = metric.trust_level
        
        # Get recent events
        recent_events = self.reputation_history[agent_id][-5:]
        
        return {
            "agent_id": agent_id,
            "reputation_score": reputation,
            "credibility": credibility.overall_score if credibility else 0.5,
            "credibility_factors": credibility.factors if credibility else {},
            "trust_given": trust_given,
            "trust_received": trust_received,
            "recent_events": [
                {
                    "type": e.event_type,
                    "impact": e.impact,
                    "factor": e.factor.value,
                    "description": e.description
                }
                for e in recent_events
            ],
            "total_events": len(self.reputation_history[agent_id])
        }
    
    def get_trust_network_summary(self) -> Dict[str, Any]:
        """Get summary of trust network."""
        total_relationships = len(self.trust_network)
        
        if total_relationships == 0:
            return {
                "total_relationships": 0,
                "average_trust": 0.0,
                "highest_trust": None,
                "lowest_trust": None
            }
        
        trust_levels = [m.trust_level for m in self.trust_network.values()]
        
        # Find extremes
        highest_trust = max(self.trust_network.items(), 
                          key=lambda x: x[1].trust_level)
        lowest_trust = min(self.trust_network.items(), 
                         key=lambda x: x[1].trust_level)
        
        return {
            "total_relationships": total_relationships,
            "average_trust": np.mean(trust_levels),
            "trust_std_dev": np.std(trust_levels),
            "highest_trust": {
                "trustor": highest_trust[0][0],
                "trustee": highest_trust[0][1],
                "level": highest_trust[1].trust_level
            },
            "lowest_trust": {
                "trustor": lowest_trust[0][0],
                "trustee": lowest_trust[0][1],
                "level": lowest_trust[1].trust_level
            }
        }