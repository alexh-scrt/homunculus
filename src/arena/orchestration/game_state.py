"""
Game State Management for Arena

This module provides comprehensive state management including
checkpointing, recovery, and state transitions.

Features:
- State snapshots and restoration
- Checkpoint management
- State validation
- History tracking
- Efficient state updates

Author: Homunculus Team
"""

import logging
import json
import pickle
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import hashlib

from ..models import AgentState, ArenaState, Message

logger = logging.getLogger(__name__)


class GamePhase(Enum):
    """Game phase enumeration."""
    EARLY = "early"
    MID = "mid"
    LATE = "late"
    FINAL = "final"
    ENDED = "ended"
    
    def next_phase(self) -> 'GamePhase':
        """Get the next phase."""
        transitions = {
            GamePhase.EARLY: GamePhase.MID,
            GamePhase.MID: GamePhase.LATE,
            GamePhase.LATE: GamePhase.FINAL,
            GamePhase.FINAL: GamePhase.ENDED,
            GamePhase.ENDED: GamePhase.ENDED
        }
        return transitions[self]


@dataclass
class TurnState:
    """State for a single turn."""
    turn_number: int
    phase: GamePhase
    active_agents: List[str]
    eliminated_agents: List[str]
    current_speaker: Optional[str]
    messages: List[Message] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "turn_number": self.turn_number,
            "phase": self.phase.value,
            "active_agents": self.active_agents,
            "eliminated_agents": self.eliminated_agents,
            "current_speaker": self.current_speaker,
            "messages": [msg.to_dict() for msg in self.messages],
            "scores": self.scores,
            "events": self.events,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TurnState':
        """Create from dictionary."""
        return cls(
            turn_number=data["turn_number"],
            phase=GamePhase(data["phase"]),
            active_agents=data["active_agents"],
            eliminated_agents=data["eliminated_agents"],
            current_speaker=data.get("current_speaker"),
            messages=[Message.from_dict(m) for m in data.get("messages", [])],
            scores=data.get("scores", {}),
            events=data.get("events", []),
            timestamp=datetime.fromisoformat(data["timestamp"])
        )


@dataclass
class StateSnapshot:
    """Complete game state snapshot."""
    game_id: str
    turn: int
    phase: GamePhase
    arena_state: ArenaState
    agent_states: Dict[str, AgentState]
    turn_history: List[TurnState]
    metadata: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.utcnow)
    checksum: Optional[str] = None
    
    def calculate_checksum(self) -> str:
        """Calculate checksum for state integrity."""
        # Create deterministic string representation
        state_str = json.dumps({
            "game_id": self.game_id,
            "turn": self.turn,
            "phase": self.phase.value,
            "agent_states": {
                k: v.to_dict() for k, v in self.agent_states.items()
            }
        }, sort_keys=True)
        
        # Calculate SHA256
        return hashlib.sha256(state_str.encode()).hexdigest()
    
    def validate(self) -> bool:
        """Validate state integrity."""
        if not self.checksum:
            return True
        
        calculated = self.calculate_checksum()
        return calculated == self.checksum
    
    def to_bytes(self) -> bytes:
        """Serialize to bytes."""
        return pickle.dumps(self)
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'StateSnapshot':
        """Deserialize from bytes."""
        return pickle.loads(data)


class CheckpointManager:
    """
    Manages game checkpoints for recovery.
    """
    
    def __init__(
        self,
        checkpoint_dir: Optional[Path] = None,
        max_checkpoints: int = 10
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for checkpoints
            max_checkpoints: Maximum checkpoints to keep
        """
        self.checkpoint_dir = checkpoint_dir or Path("/tmp/arena_checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        
        self.checkpoints: Dict[str, StateSnapshot] = {}
        self.checkpoint_order: List[str] = []
    
    def create_checkpoint(
        self,
        game_id: str,
        state: StateSnapshot
    ) -> str:
        """
        Create a new checkpoint.
        
        Args:
            game_id: Game ID
            state: State snapshot
            
        Returns:
            Checkpoint ID
        """
        # Calculate checksum
        state.checksum = state.calculate_checksum()
        
        # Create checkpoint ID
        checkpoint_id = f"{game_id}_turn{state.turn}_{datetime.utcnow().timestamp()}"
        
        # Save to memory
        self.checkpoints[checkpoint_id] = state
        self.checkpoint_order.append(checkpoint_id)
        
        # Save to disk
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.ckpt"
        with open(checkpoint_path, 'wb') as f:
            f.write(state.to_bytes())
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        logger.info(f"Created checkpoint {checkpoint_id}")
        return checkpoint_id
    
    def restore_checkpoint(self, checkpoint_id: str) -> Optional[StateSnapshot]:
        """
        Restore from checkpoint.
        
        Args:
            checkpoint_id: Checkpoint ID
            
        Returns:
            Restored state or None
        """
        # Try memory first
        if checkpoint_id in self.checkpoints:
            state = self.checkpoints[checkpoint_id]
            if state.validate():
                logger.info(f"Restored from memory: {checkpoint_id}")
                return state
        
        # Try disk
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.ckpt"
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'rb') as f:
                    state = StateSnapshot.from_bytes(f.read())
                
                if state.validate():
                    logger.info(f"Restored from disk: {checkpoint_id}")
                    self.checkpoints[checkpoint_id] = state
                    return state
                else:
                    logger.error(f"Checkpoint validation failed: {checkpoint_id}")
            
            except Exception as e:
                logger.error(f"Failed to restore checkpoint: {e}")
        
        return None
    
    def get_latest_checkpoint(self, game_id: str) -> Optional[StateSnapshot]:
        """
        Get the latest checkpoint for a game.
        
        Args:
            game_id: Game ID
            
        Returns:
            Latest checkpoint or None
        """
        # Find checkpoints for this game
        game_checkpoints = [
            cid for cid in self.checkpoint_order
            if cid.startswith(f"{game_id}_")
        ]
        
        if game_checkpoints:
            latest = game_checkpoints[-1]
            return self.restore_checkpoint(latest)
        
        return None
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints beyond max limit."""
        while len(self.checkpoint_order) > self.max_checkpoints:
            old_id = self.checkpoint_order.pop(0)
            
            # Remove from memory
            self.checkpoints.pop(old_id, None)
            
            # Remove from disk
            checkpoint_path = self.checkpoint_dir / f"{old_id}.ckpt"
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            
            logger.debug(f"Removed old checkpoint: {old_id}")


class GameStateManager:
    """
    Main state manager for Arena games.
    """
    
    def __init__(
        self,
        game_id: str,
        enable_history: bool = True
    ):
        """
        Initialize state manager.
        
        Args:
            game_id: Game ID
            enable_history: Enable turn history tracking
        """
        self.game_id = game_id
        self.enable_history = enable_history
        
        # Current state
        self.current_turn = 0
        self.current_phase = GamePhase.EARLY
        self.arena_state: Optional[ArenaState] = None
        self.agent_states: Dict[str, AgentState] = {}
        
        # History
        self.turn_history: List[TurnState] = []
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager()
        
        # State change callbacks
        self.state_change_callbacks: List[Callable] = []
    
    def initialize_state(
        self,
        active_agents: List[str],
        phase: GamePhase = GamePhase.EARLY
    ) -> None:
        """
        Initialize game state.
        
        Args:
            active_agents: List of active agent IDs
            phase: Starting phase
        """
        # Create arena state
        self.arena_state = ArenaState(
            game_id=self.game_id,
            current_turn=0,
            active_agent_ids=active_agents,
            eliminated_agent_ids=[],
            current_phase=phase.value,
            is_active=True
        )
        
        # Create agent states
        for agent_id in active_agents:
            self.agent_states[agent_id] = AgentState(
                agent_id=agent_id,
                is_active=True,
                score=0.0,
                total_contributions=0,
                total_accusations=0,
                times_accused=0,
                elimination_votes_received=0,
                last_contribution_turn=None
            )
        
        logger.info(f"Initialized state for game {self.game_id}")
    
    def update_turn(
        self,
        turn_number: int,
        active_agents: List[str],
        scores: Dict[str, float]
    ) -> None:
        """
        Update state for new turn.
        
        Args:
            turn_number: Current turn number
            active_agents: Active agent IDs
            scores: Current scores
        """
        self.current_turn = turn_number
        
        # Update arena state
        if self.arena_state:
            self.arena_state.current_turn = turn_number
            self.arena_state.active_agent_ids = active_agents
            
            # Update eliminated
            all_agents = set(self.agent_states.keys())
            active_set = set(active_agents)
            eliminated = all_agents - active_set
            self.arena_state.eliminated_agent_ids = list(eliminated)
        
        # Update agent scores
        for agent_id, score in scores.items():
            if agent_id in self.agent_states:
                self.agent_states[agent_id].score = score
        
        # Check phase transition
        self._check_phase_transition()
        
        # Save turn state if history enabled
        if self.enable_history:
            turn_state = TurnState(
                turn_number=turn_number,
                phase=self.current_phase,
                active_agents=active_agents,
                eliminated_agents=list(eliminated) if self.arena_state else [],
                current_speaker=None,
                scores=scores
            )
            self.turn_history.append(turn_state)
        
        # Trigger callbacks
        self._trigger_state_change()
    
    def _check_phase_transition(self) -> None:
        """Check and update game phase."""
        if not self.arena_state:
            return
        
        total_agents = len(self.agent_states)
        eliminated = len(self.arena_state.eliminated_agent_ids)
        elimination_rate = eliminated / total_agents if total_agents > 0 else 0
        
        # Phase based on turn and elimination
        if self.current_turn < 20:
            new_phase = GamePhase.EARLY
        elif self.current_turn < 50:
            new_phase = GamePhase.MID
        elif self.current_turn < 80:
            new_phase = GamePhase.LATE
        else:
            new_phase = GamePhase.FINAL
        
        # Override based on elimination
        if elimination_rate > 0.5:
            new_phase = GamePhase.LATE
        if elimination_rate > 0.75:
            new_phase = GamePhase.FINAL
        
        if new_phase != self.current_phase:
            logger.info(f"Phase transition: {self.current_phase.value} -> {new_phase.value}")
            self.current_phase = new_phase
            self.arena_state.current_phase = new_phase.value
    
    def create_snapshot(self) -> StateSnapshot:
        """
        Create a state snapshot.
        
        Returns:
            State snapshot
        """
        snapshot = StateSnapshot(
            game_id=self.game_id,
            turn=self.current_turn,
            phase=self.current_phase,
            arena_state=self.arena_state,
            agent_states=self.agent_states.copy(),
            turn_history=self.turn_history.copy() if self.enable_history else [],
            metadata={
                "timestamp": datetime.utcnow().isoformat(),
                "history_size": len(self.turn_history)
            }
        )
        
        return snapshot
    
    def restore_snapshot(self, snapshot: StateSnapshot) -> None:
        """
        Restore from snapshot.
        
        Args:
            snapshot: State snapshot to restore
        """
        if not snapshot.validate():
            raise ValueError("Invalid snapshot checksum")
        
        self.game_id = snapshot.game_id
        self.current_turn = snapshot.turn
        self.current_phase = snapshot.phase
        self.arena_state = snapshot.arena_state
        self.agent_states = snapshot.agent_states.copy()
        
        if self.enable_history:
            self.turn_history = snapshot.turn_history.copy()
        
        logger.info(f"Restored state to turn {self.current_turn}")
        self._trigger_state_change()
    
    def checkpoint(self) -> str:
        """
        Create a checkpoint.
        
        Returns:
            Checkpoint ID
        """
        snapshot = self.create_snapshot()
        checkpoint_id = self.checkpoint_manager.create_checkpoint(
            self.game_id, snapshot
        )
        return checkpoint_id
    
    def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Restore from checkpoint.
        
        Args:
            checkpoint_id: Checkpoint ID
            
        Returns:
            Success status
        """
        snapshot = self.checkpoint_manager.restore_checkpoint(checkpoint_id)
        if snapshot:
            self.restore_snapshot(snapshot)
            return True
        return False
    
    def register_state_change_callback(self, callback: Callable) -> None:
        """Register callback for state changes."""
        self.state_change_callbacks.append(callback)
    
    def _trigger_state_change(self) -> None:
        """Trigger state change callbacks."""
        for callback in self.state_change_callbacks:
            try:
                callback(self)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get state statistics.
        
        Returns:
            Statistics dictionary
        """
        if not self.arena_state:
            return {}
        
        return {
            "game_id": self.game_id,
            "current_turn": self.current_turn,
            "current_phase": self.current_phase.value,
            "active_agents": len(self.arena_state.active_agent_ids),
            "eliminated_agents": len(self.arena_state.eliminated_agent_ids),
            "history_size": len(self.turn_history),
            "top_score": max(
                (s.score for s in self.agent_states.values()),
                default=0
            ),
            "average_score": sum(
                s.score for s in self.agent_states.values()
            ) / len(self.agent_states) if self.agent_states else 0
        }