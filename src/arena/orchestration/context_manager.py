"""
Context Management for Arena Orchestration

Inspired by the talks project's rich context system for turn-based interactions.
Provides structured context building and management for agent actions.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

from ..models import Message, AgentState, ArenaState
from .game_state import GamePhase


@dataclass
class TurnContext:
    """Rich context for agent turn execution"""
    
    # Basic turn info
    turn_number: int
    phase: GamePhase
    current_speaker: str
    
    # Game state
    active_agents: List[str]
    eliminated_agents: List[str]
    scores: Dict[str, float]
    
    # Message history
    recent_messages: List[Message]
    full_message_history: List[Message]
    
    # Speaker-specific context
    speaker_history: List[Message]
    addressed_to_speaker: List[Message]
    
    # Meta context
    narrator_context: Optional[str] = None
    strategic_context: Optional[Dict[str, Any]] = None
    progression_context: Optional[Dict[str, Any]] = None
    
    # Environmental factors
    pressure_level: float = 0.0  # 0.0 = calm, 1.0 = high pressure
    elimination_threat: bool = False
    alliance_opportunities: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for agent consumption"""
        return {
            "turn": self.turn_number,
            "phase": self.phase.value if isinstance(self.phase, GamePhase) else self.phase,
            "speaker": self.current_speaker,
            "active_agents": self.active_agents,
            "eliminated_agents": self.eliminated_agents,
            "scores": self.scores,
            "recent_messages": [msg.to_dict() for msg in self.recent_messages],
            "speaker_history": [msg.to_dict() for msg in self.speaker_history],
            "addressed_to_me": [msg.to_dict() for msg in self.addressed_to_speaker],
            "narrator_context": self.narrator_context,
            "strategic_context": self.strategic_context,
            "progression_context": self.progression_context,
            "pressure_level": self.pressure_level,
            "elimination_threat": self.elimination_threat,
            "alliance_opportunities": self.alliance_opportunities or []
        }


class ContextManager:
    """Manages rich context building for Arena turns, inspired by talks project"""
    
    def __init__(self, config):
        self.config = config
        self.context_window = 5  # Number of recent messages to include
        
    def build_turn_context(
        self,
        current_speaker: str,
        arena_state: ArenaState,
        message_history: List[Message],
        turn_number: int,
        phase: GamePhase,
        **kwargs
    ) -> TurnContext:
        """Build comprehensive context for an agent's turn"""
        
        # Recent messages (last N)
        recent_messages = message_history[-self.context_window:] if message_history else []
        
        # Speaker-specific history
        speaker_history = [
            msg for msg in message_history 
            if msg.sender_id == current_speaker
        ][-3:]  # Last 3 messages from this speaker
        
        # Messages addressed to this speaker
        addressed_to_speaker = [
            msg for msg in message_history
            if hasattr(msg, 'recipient_id') and msg.recipient_id == current_speaker
        ][-3:]  # Last 3 messages to this speaker
        
        # Calculate pressure level based on game state
        pressure_level = self._calculate_pressure_level(
            current_speaker, arena_state, turn_number
        )
        
        # Check elimination threat
        elimination_threat = self._check_elimination_threat(
            current_speaker, arena_state, turn_number
        )
        
        # Find alliance opportunities
        alliance_opportunities = self._find_alliance_opportunities(
            current_speaker, arena_state, message_history
        )
        
        return TurnContext(
            turn_number=turn_number,
            phase=phase,
            current_speaker=current_speaker,
            active_agents=arena_state.get_active_agent_ids(),
            eliminated_agents=arena_state.eliminated_agent_ids or [],
            scores=arena_state.get_agent_scores(),
            recent_messages=recent_messages,
            full_message_history=message_history,
            speaker_history=speaker_history,
            addressed_to_speaker=addressed_to_speaker,
            narrator_context=kwargs.get('narrator_context'),
            strategic_context=kwargs.get('strategic_context'),
            progression_context=kwargs.get('progression_context'),
            pressure_level=pressure_level,
            elimination_threat=elimination_threat,
            alliance_opportunities=alliance_opportunities
        )
    
    def _calculate_pressure_level(
        self, 
        agent_id: str, 
        arena_state: ArenaState, 
        turn_number: int
    ) -> float:
        """Calculate pressure level for an agent (0.0 = calm, 1.0 = extreme pressure)"""
        
        pressure = 0.0
        
        # Time pressure (later in game = more pressure)
        max_turns = getattr(self.config, 'max_turns', 100)
        time_pressure = min(1.0, turn_number / max_turns)
        pressure += time_pressure * 0.3
        
        # Score pressure (low score relative to others)
        scores = arena_state.get_agent_scores()
        if agent_id in scores and scores:
            agent_score = scores[agent_id]
            avg_score = sum(scores.values()) / len(scores)
            max_score = max(scores.values()) if scores.values() else 0
            
            if max_score > 0:
                score_ratio = agent_score / max_score
                score_pressure = max(0.0, 1.0 - score_ratio)
                pressure += score_pressure * 0.4
        
        # Elimination pressure (fewer agents = more pressure)
        total_agents = len(arena_state.get_active_agent_ids())
        if total_agents <= 3:
            pressure += 0.3
        elif total_agents <= 5:
            pressure += 0.15
        
        return min(1.0, pressure)
    
    def _check_elimination_threat(
        self,
        agent_id: str,
        arena_state: ArenaState, 
        turn_number: int
    ) -> bool:
        """Check if agent is under immediate elimination threat"""
        
        # Check if elimination is possible this turn
        if turn_number < 20:  # No elimination before turn 20
            return False
        
        # Check if agent has lowest score
        scores = arena_state.get_agent_scores()
        if agent_id in scores and scores:
            agent_score = scores[agent_id]
            min_score = min(scores.values())
            return agent_score == min_score and len(scores) > 2
        
        return False
    
    def _find_alliance_opportunities(
        self,
        agent_id: str,
        arena_state: ArenaState,
        message_history: List[Message]
    ) -> List[str]:
        """Find potential alliance partners based on recent interactions"""
        
        alliance_candidates = []
        active_agents = arena_state.get_active_agent_ids()
        
        # Look for agents who recently supported this agent
        for msg in message_history[-10:]:  # Last 10 messages
            if (hasattr(msg, 'message_type') and 
                msg.message_type == 'support' and
                hasattr(msg, 'recipient_id') and
                msg.recipient_id == agent_id and
                msg.sender_id in active_agents and
                msg.sender_id != agent_id):
                
                if msg.sender_id not in alliance_candidates:
                    alliance_candidates.append(msg.sender_id)
        
        return alliance_candidates[:3]  # Max 3 alliance opportunities
    
    def add_intervention_context(
        self,
        context: TurnContext,
        intervention_type: str,
        intervention_data: Dict[str, Any]
    ) -> TurnContext:
        """Add intervention context (inspired by talks fractional turns)"""
        
        # Update progression context with intervention
        if context.progression_context is None:
            context.progression_context = {}
        
        context.progression_context['intervention'] = {
            'type': intervention_type,
            'data': intervention_data,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return context
    
    def build_narrator_context(
        self,
        arena_state: ArenaState,
        message_history: List[Message],
        event_type: str = "turn_transition"
    ) -> str:
        """Build context for narrator/moderator interventions"""
        
        if not message_history:
            return ""
        
        last_message = message_history[-1]
        recent_context = " ".join([
            f"{msg.sender_id}: {msg.content[:100]}..."
            for msg in message_history[-3:]
        ])
        
        narrator_context = f"""
        Current situation: {event_type}
        Last speaker: {last_message.sender_id}
        Recent exchanges: {recent_context}
        Active agents: {len(arena_state.get_active_agent_ids())}
        Current phase: {getattr(arena_state, 'current_phase', 'unknown')}
        """.strip()
        
        return narrator_context