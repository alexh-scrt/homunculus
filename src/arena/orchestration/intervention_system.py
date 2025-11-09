"""
Intervention System for Arena Orchestration

Inspired by the talks project's intervention system with fractional turns.
Provides moderator interventions, consequence testing, and narrative pivots.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

from ..models import Message, AgentState, ArenaState

logger = logging.getLogger(__name__)


class InterventionType(Enum):
    """Types of interventions that can occur during games"""
    CONSEQUENCE_TEST = "consequence_test"      # Test philosophical consequences 
    STRATEGIC_PIVOT = "strategic_pivot"       # Force strategic direction change
    ALLIANCE_BREAK = "alliance_break"         # Break up overpowered alliances
    NARRATIVE_TWIST = "narrative_twist"       # Introduce plot twists
    ELIMINATION_WARNING = "elimination_warning" # Warn about impending elimination
    MODERATOR_CLARIFICATION = "clarification" # Clarify rules or situation
    TIME_PRESSURE = "time_pressure"           # Announce time constraints


@dataclass
class Intervention:
    """Represents a moderator intervention"""
    intervention_type: InterventionType
    turn: float  # Fractional turn (e.g., 5.5 for between turns 5 and 6)
    speaker: str  # Usually "Moderator" or "Game Master"
    content: str
    target_agents: List[str] = None
    metadata: Dict[str, Any] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.target_agents is None:
            self.target_agents = []
        if self.metadata is None:
            self.metadata = {}


class InterventionSystem:
    """Manages dynamic interventions during Arena games"""
    
    def __init__(self, config):
        self.config = config
        self.interventions_history: List[Intervention] = []
        self.intervention_triggers = {
            # Trigger conditions for different intervention types
            InterventionType.CONSEQUENCE_TEST: self._should_trigger_consequence_test,
            InterventionType.STRATEGIC_PIVOT: self._should_trigger_strategic_pivot,
            InterventionType.ALLIANCE_BREAK: self._should_trigger_alliance_break,
            InterventionType.ELIMINATION_WARNING: self._should_trigger_elimination_warning,
            InterventionType.TIME_PRESSURE: self._should_trigger_time_pressure,
        }
        
        # Templates for intervention content
        self.intervention_templates = {
            InterventionType.CONSEQUENCE_TEST: [
                "Let's explore the practical implications: If {scenario}, what would be the real-world consequences?",
                "Consider this test case: In a situation where {scenario}, how would your approach actually work?",
                "Challenge time: Can your strategy handle this edge case: {scenario}?"
            ],
            InterventionType.STRATEGIC_PIVOT: [
                "The game dynamics are shifting. New information reveals: {revelation}",
                "Plot twist: The situation has evolved. {new_context}",
                "Strategic update: Conditions have changed. {strategic_shift}"
            ],
            InterventionType.ALLIANCE_BREAK: [
                "Attention all players: New intelligence suggests {alliance_threat}",
                "The balance of power is shifting. Evidence shows: {power_shift}",
                "Breaking news that affects current alliances: {disruptive_info}"
            ],
            InterventionType.ELIMINATION_WARNING: [
                "Reminder: The next elimination round is approaching. Current standings matter.",
                "Attention: Performance in the coming turns will be critical for survival.",
                "Warning: The elimination threshold is approaching. Step up your game."
            ],
            InterventionType.TIME_PRESSURE: [
                "Time check: We're at turn {turn} of {max_turns}. Decisions become more critical.",
                "Urgent: Only {remaining} turns remain. Make every move count.",
                "Clock update: Game approaching critical phase with {remaining} turns left."
            ]
        }
    
    def check_for_interventions(
        self,
        arena_state: ArenaState,
        message_history: List[Message],
        turn_number: int,
        current_phase: str
    ) -> List[Intervention]:
        """Check if any interventions should be triggered"""
        
        interventions = []
        
        for intervention_type, trigger_func in self.intervention_triggers.items():
            if trigger_func(arena_state, message_history, turn_number, current_phase):
                intervention = self._create_intervention(
                    intervention_type,
                    arena_state, 
                    message_history,
                    turn_number,
                    current_phase
                )
                if intervention:
                    interventions.append(intervention)
                    logger.info(f"🎭 Triggered intervention: {intervention_type.value}")
        
        return interventions
    
    def _create_intervention(
        self,
        intervention_type: InterventionType,
        arena_state: ArenaState,
        message_history: List[Message],
        turn_number: int,
        current_phase: str
    ) -> Optional[Intervention]:
        """Create a specific intervention"""
        
        try:
            # Get appropriate template
            templates = self.intervention_templates.get(intervention_type, [])
            if not templates:
                return None
            
            # Select template and fill with context
            import random
            template = random.choice(templates)
            
            # Fill template based on intervention type
            content = self._fill_template(
                template, intervention_type, arena_state, message_history, turn_number
            )
            
            # Determine fractional turn (insert between current and next)
            fractional_turn = turn_number + 0.5
            
            # Determine targets
            target_agents = self._get_intervention_targets(
                intervention_type, arena_state, message_history
            )
            
            return Intervention(
                intervention_type=intervention_type,
                turn=fractional_turn,
                speaker="Game Master",
                content=content,
                target_agents=target_agents,
                metadata={
                    "trigger_turn": turn_number,
                    "trigger_phase": current_phase,
                    "arena_state_snapshot": {
                        "active_agents": len(arena_state.get_active_agent_ids()),
                        "eliminated_agents": len(arena_state.eliminated_agent_ids or [])
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to create intervention {intervention_type}: {e}")
            return None
    
    def _fill_template(
        self,
        template: str,
        intervention_type: InterventionType,
        arena_state: ArenaState,
        message_history: List[Message],
        turn_number: int
    ) -> str:
        """Fill template with contextual information"""
        
        max_turns = getattr(self.config, 'max_turns', 100)
        remaining_turns = max_turns - turn_number
        
        # Common variables
        variables = {
            'turn': turn_number,
            'max_turns': max_turns,
            'remaining': remaining_turns,
            'active_count': len(arena_state.get_active_agent_ids())
        }
        
        # Specific variables by intervention type
        if intervention_type == InterventionType.CONSEQUENCE_TEST:
            # Create scenario based on recent game state
            variables['scenario'] = self._generate_consequence_scenario(
                arena_state, message_history
            )
        
        elif intervention_type == InterventionType.STRATEGIC_PIVOT:
            variables['revelation'] = self._generate_strategic_revelation(
                arena_state, message_history
            )
            variables['new_context'] = self._generate_new_context(
                arena_state, message_history
            )
            variables['strategic_shift'] = self._generate_strategic_shift(
                arena_state, message_history
            )
        
        elif intervention_type == InterventionType.ALLIANCE_BREAK:
            variables['alliance_threat'] = self._detect_alliance_threat(
                arena_state, message_history
            )
            variables['power_shift'] = self._describe_power_shift(
                arena_state, message_history
            )
            variables['disruptive_info'] = self._generate_disruptive_info(
                arena_state, message_history
            )
        
        # Fill template
        try:
            return template.format(**variables)
        except KeyError as e:
            logger.warning(f"Missing template variable {e}, using basic version")
            return template
    
    def _should_trigger_consequence_test(
        self, arena_state, message_history, turn_number, current_phase
    ) -> bool:
        """Check if consequence test should be triggered"""
        
        # Trigger every 8-12 turns (with randomness)
        import random
        base_interval = 10
        if turn_number > 0 and turn_number % base_interval == random.randint(0, 2):
            
            # Only if there's been enough discussion
            if len(message_history) >= 5:
                # Don't repeat too frequently
                recent_interventions = [
                    i for i in self.interventions_history[-3:]
                    if i.intervention_type == InterventionType.CONSEQUENCE_TEST
                ]
                return len(recent_interventions) == 0
        
        return False
    
    def _should_trigger_strategic_pivot(
        self, arena_state, message_history, turn_number, current_phase
    ) -> bool:
        """Check if strategic pivot should be triggered"""
        
        # Trigger in mid-to-late game when things get stagnant
        if turn_number > 15 and len(message_history) > 10:
            # Check for repetitive patterns
            recent_messages = message_history[-5:]
            if len(recent_messages) >= 5:
                # Simple repetition check (could be more sophisticated)
                unique_senders = set(msg.sender_id for msg in recent_messages)
                if len(unique_senders) <= 2:  # Only 2 agents dominating
                    return True
        
        return False
    
    def _should_trigger_alliance_break(
        self, arena_state, message_history, turn_number, current_phase
    ) -> bool:
        """Check if alliance breaking intervention should be triggered"""
        
        # Look for patterns of cooperation that might be too powerful
        if turn_number > 10 and len(message_history) > 8:
            # Simplified alliance detection (could be more sophisticated)
            scores = arena_state.get_agent_scores()
            if scores:
                score_values = list(scores.values())
                if len(score_values) > 2:
                    max_score = max(score_values)
                    avg_score = sum(score_values) / len(score_values)
                    # If someone is way ahead, might indicate alliance
                    return max_score > avg_score * 1.5
        
        return False
    
    def _should_trigger_elimination_warning(
        self, arena_state, message_history, turn_number, current_phase
    ) -> bool:
        """Check if elimination warning should be triggered"""
        
        # Warn before elimination rounds
        elimination_turns = [20, 30, 40, 50]  # Typical elimination checkpoints
        return turn_number in [t - 2 for t in elimination_turns]  # Warn 2 turns before
    
    def _should_trigger_time_pressure(
        self, arena_state, message_history, turn_number, current_phase
    ) -> bool:
        """Check if time pressure announcement should be triggered"""
        
        max_turns = getattr(self.config, 'max_turns', 100)
        remaining_ratio = (max_turns - turn_number) / max_turns
        
        # Trigger at 75%, 50%, 25%, and 10% remaining
        pressure_points = [0.75, 0.50, 0.25, 0.10]
        return any(abs(remaining_ratio - point) < 0.02 for point in pressure_points)
    
    def _generate_consequence_scenario(self, arena_state, message_history) -> str:
        """Generate a consequence testing scenario"""
        scenarios = [
            "your proposed strategy was implemented system-wide",
            "everyone adopted your approach simultaneously", 
            "your solution had to scale to millions of users",
            "your method was tested under extreme conditions",
            "your proposal faced determined opposition"
        ]
        import random
        return random.choice(scenarios)
    
    def _generate_strategic_revelation(self, arena_state, message_history) -> str:
        """Generate strategic revelation content"""
        revelations = [
            "hidden information has changed the strategic landscape",
            "new players have entered the field with different objectives",
            "the victory conditions have evolved",
            "external pressures are influencing the competition",
            "previously unknown constraints now apply"
        ]
        import random
        return random.choice(revelations)
    
    def _generate_new_context(self, arena_state, message_history) -> str:
        """Generate new context for strategic pivots"""
        contexts = [
            "The competitive environment has shifted significantly",
            "New objectives have been introduced that change optimal strategies",
            "Resource constraints have been modified",
            "Time pressure has increased due to external factors",
            "The evaluation criteria have been updated"
        ]
        import random
        return random.choice(contexts)
    
    def _generate_strategic_shift(self, arena_state, message_history) -> str:
        """Generate strategic shift description"""
        shifts = [
            "Cooperation now yields greater rewards than pure competition",
            "Individual excellence is being weighted more heavily",
            "Risk-taking is now more advantageous than safe strategies",
            "Collaborative solutions are needed for the next phase",
            "Innovation and creativity are becoming more valuable"
        ]
        import random
        return random.choice(shifts)
    
    def _detect_alliance_threat(self, arena_state, message_history) -> str:
        """Detect and describe alliance threats"""
        return "certain partnerships may be gaining unfair advantages"
    
    def _describe_power_shift(self, arena_state, message_history) -> str:
        """Describe power dynamics changes"""
        return "the competitive balance is becoming asymmetric"
    
    def _generate_disruptive_info(self, arena_state, message_history) -> str:
        """Generate alliance-disrupting information"""
        disruptions = [
            "Individual achievement metrics are now weighted more heavily",
            "Hidden agendas may not align with stated cooperation",
            "Resource scarcity requires more independent strategies",
            "Trust relationships may not be as stable as they appear"
        ]
        import random
        return random.choice(disruptions)
    
    def _get_intervention_targets(
        self,
        intervention_type: InterventionType,
        arena_state: ArenaState,
        message_history: List[Message]
    ) -> List[str]:
        """Determine which agents are targeted by the intervention"""
        
        active_agents = arena_state.get_active_agent_ids()
        
        if intervention_type == InterventionType.ALLIANCE_BREAK:
            # Target agents with highest scores (likely alliance members)
            scores = arena_state.get_agent_scores()
            if scores:
                sorted_agents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                return [agent_id for agent_id, _ in sorted_agents[:2]]
        
        elif intervention_type == InterventionType.ELIMINATION_WARNING:
            # Target agents with lowest scores
            scores = arena_state.get_agent_scores()
            if scores:
                sorted_agents = sorted(scores.items(), key=lambda x: x[1])
                return [agent_id for agent_id, _ in sorted_agents[:2]]
        
        # Default: target all active agents
        return active_agents
    
    def record_intervention(self, intervention: Intervention) -> None:
        """Record an intervention in the history"""
        self.interventions_history.append(intervention)
        
        # Keep history manageable
        if len(self.interventions_history) > 50:
            self.interventions_history = self.interventions_history[-25:]
    
    def get_intervention_summary(self) -> Dict[str, Any]:
        """Get summary of all interventions"""
        return {
            "total_interventions": len(self.interventions_history),
            "by_type": {
                t.value: len([i for i in self.interventions_history if i.intervention_type == t])
                for t in InterventionType
            },
            "recent_interventions": [
                {
                    "type": i.intervention_type.value,
                    "turn": i.turn,
                    "content": i.content[:100] + "..." if len(i.content) > 100 else i.content
                }
                for i in self.interventions_history[-5:]
            ]
        }