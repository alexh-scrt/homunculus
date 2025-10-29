"""
Phase Controller for Arena

This module manages game phase transitions and phase-specific logic.

Features:
- Phase transition conditions
- Phase-specific rules
- Transition validation
- Phase metrics tracking

Author: Homunculus Team
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from .game_state import GamePhase

logger = logging.getLogger(__name__)


@dataclass
class TransitionCondition:
    """Condition for phase transition."""
    name: str
    check_function: Callable[[Dict[str, Any]], bool]
    priority: int = 0
    required: bool = False
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate the condition."""
        try:
            return self.check_function(context)
        except Exception as e:
            logger.error(f"Condition evaluation failed: {e}")
            return not self.required


@dataclass
class PhaseTransition:
    """Phase transition definition."""
    from_phase: GamePhase
    to_phase: GamePhase
    conditions: List[TransitionCondition]
    automatic: bool = True
    min_turns: int = 0
    max_turns: Optional[int] = None
    
    def can_transition(self, context: Dict[str, Any]) -> bool:
        """Check if transition is allowed."""
        # Check turn constraints
        turn = context.get("turn", 0)
        if turn < self.min_turns:
            return False
        if self.max_turns and turn > self.max_turns:
            return False
        
        # Check required conditions
        for condition in self.conditions:
            if condition.required and not condition.evaluate(context):
                return False
        
        # Check if any non-required condition is met
        for condition in self.conditions:
            if not condition.required and condition.evaluate(context):
                return True
        
        # If no non-required conditions, check all passed
        return all(c.evaluate(context) for c in self.conditions)


@dataclass
class PhaseMetrics:
    """Metrics for a game phase."""
    phase: GamePhase
    start_turn: int
    end_turn: Optional[int] = None
    duration: Optional[timedelta] = None
    agents_at_start: int = 0
    agents_at_end: Optional[int] = None
    events: List[Dict[str, Any]] = field(default_factory=list)
    
    def complete(self, end_turn: int, agents_remaining: int) -> None:
        """Mark phase as complete."""
        self.end_turn = end_turn
        self.agents_at_end = agents_remaining
        self.duration = timedelta(turns=end_turn - self.start_turn)


class PhaseController:
    """
    Controls game phase transitions and phase-specific logic.
    """
    
    def __init__(self):
        """Initialize phase controller."""
        self.current_phase = GamePhase.EARLY
        self.transitions: List[PhaseTransition] = []
        self.phase_metrics: Dict[GamePhase, PhaseMetrics] = {}
        self.phase_callbacks: Dict[GamePhase, List[Callable]] = {
            phase: [] for phase in GamePhase
        }
        
        # Initialize default transitions
        self._setup_default_transitions()
        
        # Track phase history
        self.phase_history: List[Tuple[GamePhase, int]] = [
            (GamePhase.EARLY, 0)
        ]
    
    def _setup_default_transitions(self) -> None:
        """Set up default phase transitions."""
        # Early to Mid
        self.add_transition(
            PhaseTransition(
                from_phase=GamePhase.EARLY,
                to_phase=GamePhase.MID,
                conditions=[
                    TransitionCondition(
                        "min_turns_20",
                        lambda ctx: ctx.get("turn", 0) >= 20
                    ),
                    TransitionCondition(
                        "agents_established",
                        lambda ctx: ctx.get("contributions_made", 0) > 10
                    )
                ],
                min_turns=15
            )
        )
        
        # Mid to Late
        self.add_transition(
            PhaseTransition(
                from_phase=GamePhase.MID,
                to_phase=GamePhase.LATE,
                conditions=[
                    TransitionCondition(
                        "min_turns_50",
                        lambda ctx: ctx.get("turn", 0) >= 50
                    ),
                    TransitionCondition(
                        "elimination_started",
                        lambda ctx: ctx.get("elimination_rate", 0) > 0.2,
                        priority=1
                    )
                ],
                min_turns=40
            )
        )
        
        # Late to Final
        self.add_transition(
            PhaseTransition(
                from_phase=GamePhase.LATE,
                to_phase=GamePhase.FINAL,
                conditions=[
                    TransitionCondition(
                        "min_turns_80",
                        lambda ctx: ctx.get("turn", 0) >= 80
                    ),
                    TransitionCondition(
                        "high_elimination",
                        lambda ctx: ctx.get("elimination_rate", 0) > 0.5,
                        priority=2
                    ),
                    TransitionCondition(
                        "few_agents",
                        lambda ctx: ctx.get("active_agents", 10) <= 5,
                        priority=3
                    )
                ],
                min_turns=60
            )
        )
        
        # Final to Ended
        self.add_transition(
            PhaseTransition(
                from_phase=GamePhase.FINAL,
                to_phase=GamePhase.ENDED,
                conditions=[
                    TransitionCondition(
                        "winner_determined",
                        lambda ctx: ctx.get("active_agents", 2) <= 1,
                        required=True
                    )
                ],
                automatic=True
            )
        )
    
    def add_transition(self, transition: PhaseTransition) -> None:
        """Add a phase transition."""
        self.transitions.append(transition)
        # Sort by priority
        self.transitions.sort(
            key=lambda t: max(c.priority for c in t.conditions),
            reverse=True
        )
    
    def check_transition(self, context: Dict[str, Any]) -> Optional[GamePhase]:
        """
        Check if phase should transition.
        
        Args:
            context: Current game context
            
        Returns:
            New phase if transition should occur
        """
        # Find applicable transitions
        for transition in self.transitions:
            if transition.from_phase != self.current_phase:
                continue
            
            if transition.can_transition(context):
                logger.info(
                    f"Phase transition available: "
                    f"{transition.from_phase.value} -> {transition.to_phase.value}"
                )
                
                if transition.automatic:
                    return transition.to_phase
                else:
                    # Manual transition, just log
                    logger.info("Manual transition required")
        
        return None
    
    def transition_to(
        self,
        new_phase: GamePhase,
        turn: int,
        agents_remaining: int
    ) -> bool:
        """
        Transition to a new phase.
        
        Args:
            new_phase: Target phase
            turn: Current turn number
            agents_remaining: Number of active agents
            
        Returns:
            Success status
        """
        # Validate transition
        valid_next = {
            GamePhase.EARLY: [GamePhase.MID],
            GamePhase.MID: [GamePhase.LATE],
            GamePhase.LATE: [GamePhase.FINAL],
            GamePhase.FINAL: [GamePhase.ENDED],
            GamePhase.ENDED: []
        }
        
        if new_phase not in valid_next.get(self.current_phase, []):
            logger.error(
                f"Invalid phase transition: "
                f"{self.current_phase.value} -> {new_phase.value}"
            )
            return False
        
        # Complete current phase metrics
        if self.current_phase in self.phase_metrics:
            metrics = self.phase_metrics[self.current_phase]
            metrics.complete(turn, agents_remaining)
        
        # Start new phase metrics
        self.phase_metrics[new_phase] = PhaseMetrics(
            phase=new_phase,
            start_turn=turn,
            agents_at_start=agents_remaining
        )
        
        # Update phase
        old_phase = self.current_phase
        self.current_phase = new_phase
        self.phase_history.append((new_phase, turn))
        
        # Trigger callbacks
        self._trigger_phase_callbacks(old_phase, new_phase)
        
        logger.info(
            f"Phase transitioned: {old_phase.value} -> {new_phase.value} "
            f"at turn {turn}"
        )
        
        return True
    
    def register_phase_callback(
        self,
        phase: GamePhase,
        callback: Callable[[GamePhase, GamePhase], None]
    ) -> None:
        """Register callback for phase entry."""
        self.phase_callbacks[phase].append(callback)
    
    def _trigger_phase_callbacks(
        self,
        old_phase: GamePhase,
        new_phase: GamePhase
    ) -> None:
        """Trigger phase transition callbacks."""
        for callback in self.phase_callbacks[new_phase]:
            try:
                callback(old_phase, new_phase)
            except Exception as e:
                logger.error(f"Phase callback error: {e}")
    
    def get_phase_rules(self, phase: GamePhase) -> Dict[str, Any]:
        """
        Get rules for a specific phase.
        
        Args:
            phase: Game phase
            
        Returns:
            Phase-specific rules
        """
        rules = {
            GamePhase.EARLY: {
                "elimination_enabled": False,
                "accusation_enabled": False,
                "contribution_weight": 1.0,
                "exploration_bonus": 0.2,
                "cooperation_incentive": 0.3
            },
            GamePhase.MID: {
                "elimination_enabled": True,
                "accusation_enabled": True,
                "contribution_weight": 1.0,
                "exploration_bonus": 0.1,
                "cooperation_incentive": 0.2,
                "elimination_rate": 0.1
            },
            GamePhase.LATE: {
                "elimination_enabled": True,
                "accusation_enabled": True,
                "contribution_weight": 1.2,
                "exploration_bonus": 0.0,
                "cooperation_incentive": 0.1,
                "elimination_rate": 0.2,
                "competition_bonus": 0.2
            },
            GamePhase.FINAL: {
                "elimination_enabled": True,
                "accusation_enabled": True,
                "contribution_weight": 1.5,
                "exploration_bonus": 0.0,
                "cooperation_incentive": 0.0,
                "elimination_rate": 0.3,
                "competition_bonus": 0.4,
                "winner_take_all": True
            },
            GamePhase.ENDED: {
                "elimination_enabled": False,
                "accusation_enabled": False,
                "contribution_weight": 0.0
            }
        }
        
        return rules.get(phase, {})
    
    def get_phase_duration(self, phase: GamePhase) -> Optional[int]:
        """
        Get duration of a phase in turns.
        
        Args:
            phase: Game phase
            
        Returns:
            Duration in turns or None
        """
        if phase in self.phase_metrics:
            metrics = self.phase_metrics[phase]
            if metrics.end_turn:
                return metrics.end_turn - metrics.start_turn
        
        # Check history
        phase_entries = [
            (p, t) for p, t in self.phase_history if p == phase
        ]
        
        if len(phase_entries) >= 2:
            start = phase_entries[0][1]
            end = phase_entries[-1][1]
            return end - start
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get phase statistics."""
        stats = {
            "current_phase": self.current_phase.value,
            "phase_history": [
                {"phase": p.value, "turn": t}
                for p, t in self.phase_history
            ],
            "phase_durations": {}
        }
        
        # Calculate phase durations
        for phase in GamePhase:
            duration = self.get_phase_duration(phase)
            if duration is not None:
                stats["phase_durations"][phase.value] = duration
        
        # Add metrics
        stats["phase_metrics"] = {}
        for phase, metrics in self.phase_metrics.items():
            stats["phase_metrics"][phase.value] = {
                "start_turn": metrics.start_turn,
                "end_turn": metrics.end_turn,
                "agents_at_start": metrics.agents_at_start,
                "agents_at_end": metrics.agents_at_end
            }
        
        return stats