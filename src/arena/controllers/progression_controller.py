"""
Progression Controller for Arena Conversations

Adapted from the talks project to prevent agents from cycling through
the same topics without meaningful progression in arena discussions.
"""

import asyncio
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging

from ..utils.topic_extractor import TopicExtractor
from ..utils.entailment_detector import EntailmentDetector, EntailmentType
from ..utils.redundancy_checker import RedundancyChecker, ConversationHistoryTracker

logger = logging.getLogger(__name__)


@dataclass
class TopicTensionState:
    """State tracking for topic tensions in arena discussions"""
    tension_pair: Tuple[str, str]
    cycles: int = 0
    last_new_entailment_turn: int = -1
    max_cycles: int = 2
    consequence_tests_attempted: int = 0
    max_consequence_tests: int = 2
    needs_pivot: bool = False
    
    def increment_cycle(self):
        """Increment the cycle count for this tension"""
        self.cycles += 1
        logger.debug(f"Tension {self.tension_pair} cycle incremented to {self.cycles}")
    
    def record_entailment(self, turn: int):
        """Record that a new entailment was detected"""
        self.last_new_entailment_turn = turn
        self.cycles = 0  # Reset cycles when new entailment found
        logger.debug(f"Tension {self.tension_pair} cycles reset due to new entailment at turn {turn}")
    
    def should_inject_test(self) -> bool:
        """Check if we should inject a consequence test"""
        return (self.cycles >= self.max_cycles and 
                self.consequence_tests_attempted < self.max_consequence_tests)
    
    def should_pivot(self) -> bool:
        """Check if we should force a topic pivot"""
        return (self.cycles >= self.max_cycles and 
                self.consequence_tests_attempted >= self.max_consequence_tests)
    
    def is_saturated(self) -> bool:
        """Check if this tension is saturated (needs intervention)"""
        return self.cycles >= self.max_cycles


@dataclass
class ProgressionConfig:
    """Configuration for arena progression control"""
    cycles_threshold: int = 2
    max_consequence_tests: int = 2
    synthesis_interval: int = 8  # Shorter for faster arena games
    entailment_required: bool = True
    enable_progression: bool = True
    topic_window: int = 3
    test_timeout_turns: int = 2
    redundancy_threshold: float = 0.85
    max_conversation_history: int = 10
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ProgressionConfig':
        """Create config from dictionary"""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})


@dataclass 
class ProgressionState:
    """Current state of progression control for arena"""
    topic_tensions: Dict[Tuple[str, str], TopicTensionState] = field(default_factory=dict)
    turn_index: int = 0
    last_pivot_turn: int = -1
    recent_topics: List[Set[str]] = field(default_factory=list)
    current_tensions: List[Tuple[str, str]] = field(default_factory=list)
    pending_tests: List[Dict[str, Any]] = field(default_factory=list)
    speaker_turn_counts: Dict[str, int] = field(default_factory=dict)
    
    def get_or_create_tension_state(self, tension: Tuple[str, str], config: ProgressionConfig) -> TopicTensionState:
        """Get or create a tension state"""
        if tension not in self.topic_tensions:
            self.topic_tensions[tension] = TopicTensionState(
                tension_pair=tension,
                max_cycles=config.cycles_threshold,
                max_consequence_tests=config.max_consequence_tests
            )
        return self.topic_tensions[tension]


class ProgressionController:
    """Main controller for arena discussion progression and anti-repetition"""
    
    def __init__(self, config: ProgressionConfig):
        """Initialize progression controller"""
        self.config = config
        
        # Core components
        self.topic_extractor = TopicExtractor()
        self.entailment_detector = EntailmentDetector() 
        self.redundancy_checker = RedundancyChecker(similarity_threshold=config.redundancy_threshold)
        self.conversation_tracker = ConversationHistoryTracker(max_history=config.max_conversation_history)
        
        # State
        self.state = ProgressionState()
        
        # Metrics for monitoring
        self.metrics = {
            "total_turns": 0,
            "repetition_cycles_detected": 0,
            "consequence_tests_injected": 0,
            "forced_pivots": 0,
            "entailments_detected": 0,
            "redundant_responses_blocked": 0,
            "tensions_tracked": 0
        }
    
    async def process_agent_response(self, agent_name: str, response: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process an agent response and determine any interventions needed
        
        Args:
            agent_name: Name of the responding agent
            response: The agent's response text
            context: Additional context information
            
        Returns:
            Dictionary with intervention instructions and state updates
        """
        if not self.config.enable_progression:
            return {"interventions": [], "allow_response": True, "state_update": {}}
        
        self.state.turn_index += 1
        self.metrics["total_turns"] += 1
        
        # Track speaker participation
        self.state.speaker_turn_counts[agent_name] = self.state.speaker_turn_counts.get(agent_name, 0) + 1
        
        # Check for redundancy first (semantic similarity)
        recent_messages = self.conversation_tracker.get_recent_messages(exclude_speaker=agent_name)
        is_redundant = self.redundancy_checker.is_redundant(response, recent_messages)
        
        # Check for meaningful entailments (required for substantive responses) 
        entailments = self.entailment_detector.detect(response)
        has_entailment = len(entailments) > 0
        
        # Enhanced validation: Reject responses that are redundant OR lack entailments
        should_reject = False
        rejection_reason = ""
        feedback_message = ""
        
        if is_redundant and not has_entailment:
            # Both redundant AND no entailments - strong rejection
            should_reject = True
            rejection_reason = "redundant_without_entailment"
            feedback_message = f"@{agent_name}, your response paraphrases previous ideas without adding meaningful implications. Please either: 1) Add specific predictions, consequences, or action steps to existing ideas, OR 2) Introduce genuinely new business concepts."
        elif is_redundant:
            # Redundant but has some entailments - still reject but with different feedback
            should_reject = True
            rejection_reason = "semantic_repetition"
            feedback_message = f"@{agent_name}, your response is too similar to recent contributions despite having some implications. Please focus on significantly different aspects or entirely new approaches."
        elif not has_entailment:
            # Not redundant but lacks substance - require entailments for substantive discussion
            should_reject = True
            rejection_reason = "lacks_entailment"
            feedback_message = f"@{agent_name}, your response needs specific implications or consequences. Add: concrete predictions (what will happen by when?), action steps (what should we do?), or explicit implications (if X then Y)."
        
        if should_reject:
            self.metrics["redundant_responses_blocked"] += 1
            return {
                "interventions": [{
                    "type": "redundancy_block",
                    "message": feedback_message,
                    "agent": agent_name
                }],
                "allow_response": False,
                "reason": rejection_reason,
                "feedback": feedback_message,
                "details": {
                    "is_redundant": is_redundant,
                    "has_entailment": has_entailment,
                    "entailment_count": len(entailments),
                    "similarity_details": self.redundancy_checker.get_similarity_details(response, recent_messages) if hasattr(self.redundancy_checker, 'get_similarity_details') else {}
                }
            }
        
        # Extract topics and detect tensions
        current_topics = self.topic_extractor.extract_topics(response)
        self.state.recent_topics.append(current_topics)
        
        # Keep only recent topic history
        if len(self.state.recent_topics) > self.config.topic_window + 1:
            self.state.recent_topics.pop(0)
        
        # Detect current tensions
        active_tensions = self.topic_extractor.detect_tensions(
            response, 
            self.state.recent_topics[:-1], 
            self.config.topic_window
        )
        self.state.current_tensions = active_tensions
        
        # Detect entailments
        entailments = self.entailment_detector.detect(response)
        has_entailment = len([e for e in entailments if e["confidence"] > 0.6]) > 0
        
        if has_entailment:
            self.metrics["entailments_detected"] += 1
        
        # Process tension states and determine interventions
        interventions = []
        
        for tension in active_tensions:
            self.metrics["tensions_tracked"] += 1
            tension_state = self.state.get_or_create_tension_state(tension, self.config)
            
            # Check if this is a cycle without new entailment
            if not has_entailment:
                tension_state.increment_cycle()
                self.metrics["repetition_cycles_detected"] += 1
                
                # Check if we need intervention
                if tension_state.should_inject_test():
                    test_intervention = self._create_consequence_test(tension_state, response, agent_name, context)
                    if test_intervention:
                        interventions.append(test_intervention)
                        tension_state.consequence_tests_attempted += 1
                        self.metrics["consequence_tests_injected"] += 1
                
                elif tension_state.should_pivot():
                    pivot_intervention = self._create_pivot_intervention(tension_state, agent_name, context)
                    if pivot_intervention:
                        interventions.append(pivot_intervention)
                        self.metrics["forced_pivots"] += 1
                        self.state.last_pivot_turn = self.state.turn_index
                        tension_state.needs_pivot = True
            else:
                # Record entailment and reset cycles
                tension_state.record_entailment(self.state.turn_index)
        
        # Add to conversation history if response is allowed
        self.conversation_tracker.add_message(agent_name, response, self.state.turn_index)
        
        # Check for global orbiting (multiple tensions saturated)
        saturated_tensions = [ts for ts in self.state.topic_tensions.values() if ts.is_saturated()]
        if len(saturated_tensions) > 1:
            global_pivot = self._create_global_pivot_intervention(context)
            if global_pivot:
                interventions.append(global_pivot)
        
        # Periodic synthesis check for arena discussions
        if (self.state.turn_index % self.config.synthesis_interval == 0 and 
            self.state.turn_index > self.config.synthesis_interval):
            synthesis_intervention = self._create_synthesis_intervention(context)
            if synthesis_intervention:
                interventions.append(synthesis_intervention)
        
        return {
            "interventions": interventions,
            "allow_response": True,
            "state_update": {
                "turn": self.state.turn_index,
                "topics": list(current_topics),
                "tensions": [list(t) for t in active_tensions],
                "entailments_count": len(entailments),
                "has_entailment": has_entailment
            },
            "metrics": self.metrics.copy()
        }
    
    def _create_consequence_test(self, tension_state: TopicTensionState, response: str, agent_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a consequence test intervention"""
        tension = tension_state.tension_pair
        
        # Arena-specific consequence test templates
        test_templates = {
            ('ai_technology', 'energy'): [
                f"@{agent_name}, you've mentioned AI and energy integration. Specifically, if your approach succeeds, what will be the exact energy consumption per AI operation by 2026?",
                f"@{agent_name}, your AI-energy concept - what specific infrastructure investment (in billions) would be needed to achieve trillion-dollar scale?",
            ],
            ('innovation', 'execution'): [
                f"@{agent_name}, you've discussed innovation and execution. What concrete milestone would prove your concept is executable within 18 months?",
                f"@{agent_name}, regarding innovation vs execution - what specific team size and skills would you need to build this trillion-dollar company?",
            ],
            ('market', 'technology_infrastructure'): [
                f"@{agent_name}, you've covered market and tech infrastructure. What specific market penetration percentage would make this trillion-dollar viable?",
                f"@{agent_name}, your market-tech approach - what exact customer acquisition cost would make the unit economics work?",
            ]
        }
        
        # Get templates for this tension or use generic ones
        templates = test_templates.get(tension, [
            f"@{agent_name}, you've discussed {tension[0]} and {tension[1]}. What specific, measurable outcome would prove your approach within 2 years?",
            f"@{agent_name}, regarding {tension[0]} vs {tension[1]} - what concrete prediction can you make that we could verify?"
        ])
        
        # Select template based on attempt number
        template_index = tension_state.consequence_tests_attempted % len(templates)
        test_message = templates[template_index]
        
        return {
            "type": "consequence_test",
            "message": test_message,
            "agent": agent_name,
            "tension": tension,
            "test_number": tension_state.consequence_tests_attempted + 1
        }
    
    def _create_pivot_intervention(self, tension_state: TopicTensionState, agent_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a topic pivot intervention"""
        tension = tension_state.tension_pair
        
        # Suggest new directions for arena discussions
        pivot_suggestions = [
            "Let's pivot to customer acquisition strategies and go-to-market approach.",
            "Time to focus on competitive differentiation and unique value propositions.", 
            "Let's examine the regulatory landscape and compliance requirements.",
            "How about discussing talent acquisition and organizational structure?",
            "Let's explore partnership opportunities and strategic alliances.",
            "Time to analyze potential risks and mitigation strategies.",
            "Let's focus on financial projections and funding requirements.",
            "How about examining scalability challenges and solutions?"
        ]
        
        # Select suggestion (could be more intelligent based on context)
        suggestion_index = len(self.state.topic_tensions) % len(pivot_suggestions)
        suggestion = pivot_suggestions[suggestion_index]
        
        return {
            "type": "forced_pivot",
            "message": f"We've been cycling through {tension[0]} and {tension[1]} without new insights. {suggestion}",
            "agent": agent_name,
            "previous_tension": tension,
            "suggested_direction": suggestion
        }
    
    def _create_global_pivot_intervention(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create intervention when multiple tensions are saturated"""
        return {
            "type": "global_pivot", 
            "message": "The discussion is orbiting multiple topics without progression. Let's focus on ONE concrete business model and develop it with specific details.",
            "saturated_tensions": len([ts for ts in self.state.topic_tensions.values() if ts.is_saturated()])
        }
    
    def _create_synthesis_intervention(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create periodic synthesis intervention for arena"""
        game_context = context.get('game_context', {}) if context else {}
        current_turn = self.state.turn_index
        
        return {
            "type": "synthesis",
            "message": f"[Turn {current_turn}] Let's synthesize our discussion so far and identify the most promising trillion-dollar opportunity that combines our insights.",
            "turn": current_turn,
            "topics_covered": len(set().union(*self.state.recent_topics)) if self.state.recent_topics else 0,
            "tensions_explored": len(self.state.topic_tensions)
        }
    
    def get_progression_status(self) -> Dict[str, Any]:
        """Get current progression status for monitoring"""
        saturated_count = len([ts for ts in self.state.topic_tensions.values() if ts.is_saturated()])
        
        return {
            "turn": self.state.turn_index,
            "active_tensions": len(self.state.current_tensions),
            "saturated_tensions": saturated_count,
            "total_topics_discussed": len(set().union(*self.state.recent_topics)) if self.state.recent_topics else 0,
            "speaker_distribution": self.state.speaker_turn_counts.copy(),
            "metrics": self.metrics.copy(),
            "needs_intervention": saturated_count > 0
        }
    
    def reset_state(self):
        """Reset progression state for new game"""
        self.state = ProgressionState()
        self.conversation_tracker.clear()
        
        # Reset metrics except cumulative ones
        self.metrics = {
            "total_turns": 0,
            "repetition_cycles_detected": 0,
            "consequence_tests_injected": 0,
            "forced_pivots": 0,
            "entailments_detected": 0,
            "redundant_responses_blocked": 0,
            "tensions_tracked": 0
        }
        
        logger.info("Progression controller state reset for new game")


class ArenaProgressionOrchestrator:
    """Orchestrates progression control across the entire arena game"""
    
    def __init__(self, config: ProgressionConfig):
        """Initialize arena progression orchestrator"""
        self.config = config
        self.controller = ProgressionController(config)
        self.intervention_queue = []
        
    async def process_turn(self, agent_name: str, response: str, game_context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a complete turn including interventions"""
        
        # Process the response
        result = await self.controller.process_agent_response(
            agent_name, response, {"game_context": game_context}
        )
        
        # Handle any interventions
        if result["interventions"]:
            for intervention in result["interventions"]:
                await self._execute_intervention(intervention, game_context)
        
        return result
    
    async def _execute_intervention(self, intervention: Dict[str, Any], game_context: Dict[str, Any]):
        """Execute a progression intervention"""
        intervention_type = intervention["type"]
        
        logger.info(f"Executing {intervention_type} intervention: {intervention.get('message', 'No message')}")
        
        if intervention_type == "consequence_test":
            # Queue the test for the game orchestrator to inject
            self.intervention_queue.append({
                "type": "inject_message",
                "speaker": "Judge",
                "content": intervention["message"],
                "metadata": {"intervention": intervention_type}
            })
        
        elif intervention_type in ["forced_pivot", "global_pivot"]:
            # Queue pivot message
            self.intervention_queue.append({
                "type": "inject_message", 
                "speaker": "Narrator",
                "content": intervention["message"],
                "metadata": {"intervention": intervention_type}
            })
        
        elif intervention_type == "synthesis":
            # Queue synthesis request
            self.intervention_queue.append({
                "type": "inject_message",
                "speaker": "Narrator", 
                "content": intervention["message"],
                "metadata": {"intervention": intervention_type}
            })
        
        elif intervention_type == "redundancy_block":
            # This blocks the response, handled by caller
            pass
    
    def get_pending_interventions(self) -> List[Dict[str, Any]]:
        """Get and clear pending interventions"""
        interventions = self.intervention_queue.copy()
        self.intervention_queue.clear()
        return interventions
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status"""
        return {
            "progression_status": self.controller.get_progression_status(),
            "pending_interventions": len(self.intervention_queue),
            "config": self.config.__dict__
        }