"""
Turn Management for Arena

This module manages turn-based gameplay including turn flow,
agent ordering, and turn validation.

Features:
- Turn sequencing
- Agent turn order management
- Turn timeout handling
- Turn validation
- Turn result aggregation

Author: Homunculus Team
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from ..models import Message, AgentState
from ..agents import BaseAgent

logger = logging.getLogger(__name__)


class TurnPhase(Enum):
    """Phases within a turn."""
    SETUP = "setup"
    SELECTION = "selection"
    ACTION = "action"
    RESPONSE = "response"
    EVALUATION = "evaluation"
    CLEANUP = "cleanup"


@dataclass
class TurnContext:
    """Context for a turn."""
    turn_number: int
    game_phase: str
    active_agents: List[str]
    eliminated_agents: List[str]
    current_speaker: Optional[str]
    scores: Dict[str, float]
    recent_messages: List[Message]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_available_agents(self) -> List[str]:
        """Get agents available for selection."""
        return [a for a in self.active_agents if a != self.current_speaker]
    
    def is_agent_active(self, agent_id: str) -> bool:
        """Check if agent is active."""
        return agent_id in self.active_agents


@dataclass
class TurnResult:
    """Result of a turn."""
    turn_number: int
    speaker: Optional[str]
    messages: List[Message]
    scores_changed: Dict[str, float]
    eliminations: List[str]
    phase_changed: bool
    duration: timedelta
    success: bool
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "turn_number": self.turn_number,
            "speaker": self.speaker,
            "message_count": len(self.messages),
            "scores_changed": self.scores_changed,
            "eliminations": self.eliminations,
            "phase_changed": self.phase_changed,
            "duration_seconds": self.duration.total_seconds(),
            "success": self.success,
            "errors": self.errors
        }


class TurnFlow:
    """
    Manages the flow of a single turn.
    """
    
    def __init__(
        self,
        turn_number: int,
        context: TurnContext,
        timeout: int = 60
    ):
        """
        Initialize turn flow.
        
        Args:
            turn_number: Current turn number
            context: Turn context
            timeout: Turn timeout in seconds
        """
        self.turn_number = turn_number
        self.context = context
        self.timeout = timeout
        
        self.current_phase = TurnPhase.SETUP
        self.messages: List[Message] = []
        self.errors: List[str] = []
        
        self.start_time = datetime.utcnow()
    
    async def execute_phase(self, phase: TurnPhase) -> bool:
        """
        Execute a turn phase.
        
        Args:
            phase: Phase to execute
            
        Returns:
            Success status
        """
        self.current_phase = phase
        logger.debug(f"Turn {self.turn_number}: Executing phase {phase.value}")
        
        try:
            if phase == TurnPhase.SETUP:
                return await self._setup_phase()
            elif phase == TurnPhase.SELECTION:
                return await self._selection_phase()
            elif phase == TurnPhase.ACTION:
                return await self._action_phase()
            elif phase == TurnPhase.RESPONSE:
                return await self._response_phase()
            elif phase == TurnPhase.EVALUATION:
                return await self._evaluation_phase()
            elif phase == TurnPhase.CLEANUP:
                return await self._cleanup_phase()
            else:
                return False
        
        except asyncio.TimeoutError:
            self.errors.append(f"Phase {phase.value} timed out")
            return False
        except Exception as e:
            self.errors.append(f"Phase {phase.value} error: {e}")
            logger.error(f"Turn phase error: {e}")
            return False
    
    async def _setup_phase(self) -> bool:
        """Set up the turn."""
        # Clear previous turn data
        self.messages.clear()
        self.context.recent_messages.clear()
        
        logger.info(f"Turn {self.turn_number} setup complete")
        return True
    
    async def _selection_phase(self) -> bool:
        """Select agent for action."""
        # This would be handled by TurnSelector agent
        # For now, handled by orchestrator
        return True
    
    async def _action_phase(self) -> bool:
        """Execute agent action."""
        if not self.context.current_speaker:
            self.errors.append("No speaker selected")
            return False
        
        # Action execution handled by orchestrator
        return True
    
    async def _response_phase(self) -> bool:
        """Process responses from other agents."""
        # Response processing handled by orchestrator
        return True
    
    async def _evaluation_phase(self) -> bool:
        """Evaluate turn results."""
        # Scoring and evaluation handled by orchestrator
        return True
    
    async def _cleanup_phase(self) -> bool:
        """Clean up after turn."""
        # Archive messages
        self.context.metadata["turn_messages"] = len(self.messages)
        
        logger.info(f"Turn {self.turn_number} cleanup complete")
        return True
    
    def get_result(self) -> TurnResult:
        """Get turn result."""
        duration = datetime.utcnow() - self.start_time
        
        return TurnResult(
            turn_number=self.turn_number,
            speaker=self.context.current_speaker,
            messages=self.messages.copy(),
            scores_changed={},  # Would be calculated
            eliminations=[],  # Would be determined
            phase_changed=False,  # Would be detected
            duration=duration,
            success=len(self.errors) == 0,
            errors=self.errors.copy()
        )


class TurnManager:
    """
    Main turn manager for Arena games.
    """
    
    def __init__(
        self,
        game_id: str,
        agents: Dict[str, BaseAgent],
        turn_timeout: int = 60,
        max_messages_per_turn: int = 10
    ):
        """
        Initialize turn manager.
        
        Args:
            game_id: Game ID
            agents: Dictionary of agents
            turn_timeout: Timeout per turn in seconds
            max_messages_per_turn: Maximum messages per turn
        """
        self.game_id = game_id
        self.agents = agents
        self.turn_timeout = turn_timeout
        self.max_messages_per_turn = max_messages_per_turn
        
        # Turn tracking
        self.current_turn = 0
        self.turn_history: List[TurnResult] = []
        self.turn_order: List[str] = []
        self.speaker_history: List[str] = []
        
        # Turn validation
        self.validation_rules: List[Callable] = []
    
    async def execute_turn(
        self,
        context: TurnContext
    ) -> TurnResult:
        """
        Execute a complete turn.
        
        Args:
            context: Turn context
            
        Returns:
            Turn result
        """
        self.current_turn += 1
        logger.info(f"Executing turn {self.current_turn}")
        
        # Create turn flow
        flow = TurnFlow(self.current_turn, context, self.turn_timeout)
        
        # Execute phases in order
        phases = [
            TurnPhase.SETUP,
            TurnPhase.SELECTION,
            TurnPhase.ACTION,
            TurnPhase.RESPONSE,
            TurnPhase.EVALUATION,
            TurnPhase.CLEANUP
        ]
        
        for phase in phases:
            try:
                success = await asyncio.wait_for(
                    flow.execute_phase(phase),
                    timeout=self.turn_timeout / len(phases)
                )
                
                if not success:
                    logger.warning(f"Phase {phase.value} failed")
                    break
            
            except asyncio.TimeoutError:
                logger.error(f"Phase {phase.value} timed out")
                flow.errors.append(f"Phase {phase.value} timeout")
                break
        
        # Get result
        result = flow.get_result()
        
        # Track history
        self.turn_history.append(result)
        if context.current_speaker:
            self.speaker_history.append(context.current_speaker)
        
        # Validate turn
        self._validate_turn(result)
        
        return result
    
    def _validate_turn(self, result: TurnResult) -> None:
        """
        Validate turn result.
        
        Args:
            result: Turn result to validate
        """
        for rule in self.validation_rules:
            try:
                rule(result)
            except Exception as e:
                logger.warning(f"Turn validation failed: {e}")
                result.errors.append(f"Validation: {e}")
    
    def get_turn_order(
        self,
        active_agents: List[str],
        strategy: str = "round_robin"
    ) -> List[str]:
        """
        Get turn order for agents.
        
        Args:
            active_agents: List of active agents
            strategy: Ordering strategy
            
        Returns:
            Ordered list of agent IDs
        """
        if strategy == "round_robin":
            # Simple round-robin
            if not self.turn_order:
                self.turn_order = active_agents.copy()
            
            # Rotate order
            if self.turn_order:
                self.turn_order = self.turn_order[1:] + [self.turn_order[0]]
            
            return self.turn_order
        
        elif strategy == "random":
            # Random order each turn
            import random
            order = active_agents.copy()
            random.shuffle(order)
            return order
        
        elif strategy == "performance":
            # Order by performance (would need scores)
            return sorted(active_agents)
        
        else:
            return active_agents
    
    def get_next_speaker(
        self,
        active_agents: List[str],
        excluded: Optional[Set[str]] = None
    ) -> Optional[str]:
        """
        Get next speaker.
        
        Args:
            active_agents: Active agents
            excluded: Agents to exclude
            
        Returns:
            Next speaker ID or None
        """
        excluded = excluded or set()
        
        # Get agents who haven't spoken recently
        candidates = [
            agent for agent in active_agents
            if agent not in excluded
        ]
        
        if not candidates:
            return None
        
        # Prefer agents who haven't spoken recently
        recent_speakers = set(self.speaker_history[-3:])
        preferred = [a for a in candidates if a not in recent_speakers]
        
        if preferred:
            return preferred[0]
        return candidates[0]
    
    def add_validation_rule(self, rule: Callable[[TurnResult], None]) -> None:
        """
        Add turn validation rule.
        
        Args:
            rule: Validation function that raises on failure
        """
        self.validation_rules.append(rule)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get turn statistics.
        
        Returns:
            Statistics dictionary
        """
        if not self.turn_history:
            return {
                "total_turns": 0,
                "average_duration": 0,
                "success_rate": 0
            }
        
        successful = sum(1 for t in self.turn_history if t.success)
        total_duration = sum(
            t.duration.total_seconds() for t in self.turn_history
        )
        
        # Speaker distribution
        speaker_counts = {}
        for speaker in self.speaker_history:
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        
        return {
            "total_turns": len(self.turn_history),
            "successful_turns": successful,
            "success_rate": successful / len(self.turn_history),
            "average_duration": total_duration / len(self.turn_history),
            "total_messages": sum(len(t.messages) for t in self.turn_history),
            "total_eliminations": sum(len(t.eliminations) for t in self.turn_history),
            "speaker_distribution": speaker_counts,
            "unique_speakers": len(set(self.speaker_history))
        }
    
    def reset(self) -> None:
        """Reset turn manager state."""
        self.current_turn = 0
        self.turn_history.clear()
        self.turn_order.clear()
        self.speaker_history.clear()
        logger.info("Turn manager reset")