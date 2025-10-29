"""
Turn Selector Agent for Arena

This module implements the Turn Selector agent that uses game theory
to decide which agent should speak next. Adapted from AI-Talks turn
selection patterns.

Features:
- Game theory-based selection
- Fairness and diversity promotion
- Strategic turn allocation
- Performance-based weighting
- Elimination consideration

Author: Homunculus Team
"""

import logging
import random
import math
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

from .base_agent import BaseAgent, AgentConfig, AgentRole
from ..models import Message, MessageType, AgentState, ArenaState


logger = logging.getLogger(__name__)


class TurnSelectorAgent(BaseAgent):
    """
    Turn Selector agent that decides speaking order using game theory.
    
    The Turn Selector uses various strategies to determine who speaks next:
    - Fairness: Ensure all agents get opportunities
    - Merit: High-performing agents get more chances
    - Diversity: Encourage different perspectives
    - Strategy: Create interesting dynamics
    - Tension: Build dramatic moments
    """
    
    def __init__(self, config: AgentConfig):
        """
        Initialize the Turn Selector agent.
        
        Args:
            config: Agent configuration
        """
        # Ensure correct role
        config.role = AgentRole.TURN_SELECTOR
        super().__init__(config)
        
        # Selection strategy weights
        self.strategy_weights = config.metadata.get("strategy_weights", {
            "fairness": 0.25,      # Equal opportunity
            "merit": 0.25,         # Performance-based
            "diversity": 0.20,     # Different speakers
            "momentum": 0.15,      # Building on ideas
            "tension": 0.15        # Creating drama
        })
        
        # Tracking for selection
        self.turn_history: List[str] = []
        self.agent_speak_counts: Dict[str, int] = defaultdict(int)
        self.agent_last_spoke: Dict[str, int] = {}
        self.agent_performance: Dict[str, float] = {}
        self.recent_speakers: List[str] = []
        
        # Game theory parameters
        self.epsilon = 0.1  # Exploration vs exploitation
        self.temperature = 1.0  # Randomness in selection
        self.fairness_threshold = 3  # Max turns before forcing fairness
        
        # Pattern tracking
        self.interaction_matrix: Dict[Tuple[str, str], float] = defaultdict(float)
        self.contribution_quality: Dict[str, List[float]] = defaultdict(list)
    
    async def initialize(self) -> None:
        """Initialize the Turn Selector agent."""
        logger.info(f"Turn Selector {self.agent_name} initializing")
        
        # Subscribe to relevant topics
        self.config.kafka_topics = [
            "arena.game.turns",
            "arena.scoring.metrics",
            "arena.agent.lifecycle"
        ]
    
    async def process_message(self, message: Message) -> Optional[Message]:
        """
        Process messages and make turn selection decisions.
        
        Args:
            message: Incoming message
            
        Returns:
            Turn selection message if applicable
        """
        if message.message_type == "turn_request":
            return await self._select_next_speaker(message)
            
        elif message.message_type == "scoring":
            # Update performance metrics
            self._update_performance(message)
            
        elif message.message_type == "contribution":
            # Track speaking patterns
            self._track_contribution(message)
            
        elif message.message_type == "elimination":
            # Remove eliminated agent from consideration
            self._handle_elimination(message)
        
        return None
    
    async def generate_action(self, context: Dict[str, Any]) -> Optional[Message]:
        """
        Generate turn selection based on context.
        
        Args:
            context: Current game context
            
        Returns:
            Turn selection message
        """
        arena_state: ArenaState = context.get("arena_state")
        if not arena_state or not arena_state.is_active:
            return None
        
        # Check if selection needed
        if context.get("need_turn_selection", False):
            return await self._make_selection(arena_state)
        
        return None
    
    async def update_state(self, state: Dict[str, Any]) -> None:
        """
        Update turn selector state.
        
        Args:
            state: New state information
        """
        # Update strategy weights if provided
        if "strategy_weights" in state:
            self.strategy_weights.update(state["strategy_weights"])
        
        # Update game theory parameters
        if "temperature" in state:
            self.temperature = state["temperature"]
        if "epsilon" in state:
            self.epsilon = state["epsilon"]
    
    async def _select_next_speaker(self, request: Message) -> Message:
        """
        Select the next speaker using game theory.
        
        Args:
            request: Turn selection request
            
        Returns:
            Turn selection message
        """
        # Get available agents
        available_agents = request.metadata.get("available_agents", [])
        if not available_agents:
            logger.warning("No available agents for turn selection")
            return None
        
        current_turn = request.metadata.get("current_turn", 0)
        
        # Calculate selection probabilities
        probabilities = self._calculate_selection_probabilities(
            available_agents,
            current_turn
        )
        
        # Select agent
        selected_agent = self._weighted_selection(available_agents, probabilities)
        
        # Update tracking
        self._update_turn_tracking(selected_agent, current_turn)
        
        # Generate explanation
        explanation = self._generate_selection_explanation(
            selected_agent,
            probabilities,
            available_agents
        )
        
        return Message(
            sender_id=self.agent_id,
            sender_name=self.agent_name,
            sender_type="turn_selector",
            message_type="turn_selection",
            content=explanation,
            metadata={
                "selected_agent": selected_agent,
                "turn_number": current_turn,
                "selection_probabilities": dict(zip(available_agents, probabilities)),
                "selection_strategy": self._get_dominant_strategy(probabilities)
            }
        )
    
    def _calculate_selection_probabilities(
        self,
        agents: List[str],
        current_turn: int
    ) -> List[float]:
        """
        Calculate selection probabilities for each agent.
        
        Args:
            agents: List of available agent IDs
            current_turn: Current turn number
            
        Returns:
            List of probabilities (same order as agents)
        """
        scores = []
        
        for agent_id in agents:
            # Calculate component scores
            fairness_score = self._calculate_fairness_score(agent_id, current_turn)
            merit_score = self._calculate_merit_score(agent_id)
            diversity_score = self._calculate_diversity_score(agent_id)
            momentum_score = self._calculate_momentum_score(agent_id)
            tension_score = self._calculate_tension_score(agent_id, current_turn)
            
            # Weighted combination
            total_score = (
                self.strategy_weights["fairness"] * fairness_score +
                self.strategy_weights["merit"] * merit_score +
                self.strategy_weights["diversity"] * diversity_score +
                self.strategy_weights["momentum"] * momentum_score +
                self.strategy_weights["tension"] * tension_score
            )
            
            scores.append(total_score)
        
        # Convert to probabilities using softmax with temperature
        probabilities = self._softmax(scores, self.temperature)
        
        # Apply epsilon-greedy for exploration
        if random.random() < self.epsilon:
            # Random selection with epsilon probability
            uniform_prob = 1.0 / len(agents)
            probabilities = [uniform_prob] * len(agents)
        
        return probabilities
    
    def _calculate_fairness_score(self, agent_id: str, current_turn: int) -> float:
        """
        Calculate fairness score (inverse of recent speaking).
        
        Args:
            agent_id: Agent to score
            current_turn: Current turn number
            
        Returns:
            Fairness score (0.0 to 1.0)
        """
        # How many turns since last spoke
        last_spoke = self.agent_last_spoke.get(agent_id, -1)
        if last_spoke == -1:
            return 1.0  # Never spoke, highest fairness
        
        turns_since = current_turn - last_spoke
        
        # Enforce fairness threshold
        if turns_since >= self.fairness_threshold:
            return 1.0
        
        # Linear decay
        return min(1.0, turns_since / self.fairness_threshold)
    
    def _calculate_merit_score(self, agent_id: str) -> float:
        """
        Calculate merit score based on performance.
        
        Args:
            agent_id: Agent to score
            
        Returns:
            Merit score (0.0 to 1.0)
        """
        performance = self.agent_performance.get(agent_id, 0.5)
        
        # Sigmoid transformation to keep in [0, 1]
        return 1 / (1 + math.exp(-5 * (performance - 0.5)))
    
    def _calculate_diversity_score(self, agent_id: str) -> float:
        """
        Calculate diversity score (avoid repetition).
        
        Args:
            agent_id: Agent to score
            
        Returns:
            Diversity score (0.0 to 1.0)
        """
        if agent_id in self.recent_speakers[-3:]:
            return 0.1  # Recently spoke, low diversity
        
        if agent_id in self.recent_speakers[-5:]:
            return 0.5  # Somewhat recent
        
        return 1.0  # Hasn't spoken recently, high diversity
    
    def _calculate_momentum_score(self, agent_id: str) -> float:
        """
        Calculate momentum score (building on recent ideas).
        
        Args:
            agent_id: Agent to score
            
        Returns:
            Momentum score (0.0 to 1.0)
        """
        if not self.recent_speakers:
            return 0.5  # Neutral
        
        last_speaker = self.recent_speakers[-1] if self.recent_speakers else None
        if last_speaker:
            # Check interaction history
            interaction_key = (last_speaker, agent_id)
            interaction_strength = self.interaction_matrix.get(interaction_key, 0)
            
            # Normalize to [0, 1]
            return min(1.0, interaction_strength)
        
        return 0.5
    
    def _calculate_tension_score(self, agent_id: str, current_turn: int) -> float:
        """
        Calculate tension score (create dramatic moments).
        
        Args:
            agent_id: Agent to score
            current_turn: Current turn number
            
        Returns:
            Tension score (0.0 to 1.0)
        """
        # Create tension by selecting high-performers at key moments
        is_key_moment = current_turn % 10 == 0  # Every 10th turn
        
        if is_key_moment:
            # Favor high performers for dramatic moments
            performance = self.agent_performance.get(agent_id, 0.5)
            return performance
        
        # Check for conflict potential
        if self.recent_speakers:
            # Has this agent disagreed with recent speakers?
            conflict_score = 0.0
            for recent in self.recent_speakers[-3:]:
                interaction = self.interaction_matrix.get((agent_id, recent), 0)
                if interaction < 0:  # Negative interaction
                    conflict_score += abs(interaction)
            
            return min(1.0, conflict_score / 3)
        
        return 0.5
    
    def _softmax(self, scores: List[float], temperature: float) -> List[float]:
        """
        Apply softmax with temperature to convert scores to probabilities.
        
        Args:
            scores: List of scores
            temperature: Temperature parameter (higher = more random)
            
        Returns:
            List of probabilities
        """
        if not scores:
            return []
        
        # Apply temperature
        scaled_scores = [s / temperature for s in scores]
        
        # Compute softmax
        max_score = max(scaled_scores)
        exp_scores = [math.exp(s - max_score) for s in scaled_scores]
        sum_exp = sum(exp_scores)
        
        if sum_exp == 0:
            # Uniform if all zeros
            return [1.0 / len(scores)] * len(scores)
        
        return [e / sum_exp for e in exp_scores]
    
    def _weighted_selection(self, items: List[str], weights: List[float]) -> str:
        """
        Select an item based on weights.
        
        Args:
            items: List of items to select from
            weights: Weights for each item
            
        Returns:
            Selected item
        """
        if not items or not weights:
            return None
        
        # Normalize weights
        total = sum(weights)
        if total == 0:
            # Uniform selection if all weights are 0
            return random.choice(items)
        
        normalized = [w / total for w in weights]
        
        # Random selection based on weights
        r = random.random()
        cumulative = 0.0
        
        for item, weight in zip(items, normalized):
            cumulative += weight
            if r <= cumulative:
                return item
        
        return items[-1]  # Fallback
    
    def _update_turn_tracking(self, selected_agent: str, turn: int) -> None:
        """
        Update tracking after selection.
        
        Args:
            selected_agent: Selected agent ID
            turn: Current turn number
        """
        self.turn_history.append(selected_agent)
        self.agent_speak_counts[selected_agent] += 1
        self.agent_last_spoke[selected_agent] = turn
        
        self.recent_speakers.append(selected_agent)
        if len(self.recent_speakers) > 10:
            self.recent_speakers.pop(0)
    
    def _track_contribution(self, message: Message) -> None:
        """
        Track contribution patterns.
        
        Args:
            message: Contribution message
        """
        agent_id = message.sender_id
        
        # Track quality if score available
        if "score" in message.metadata:
            score = message.metadata["score"]
            self.contribution_quality[agent_id].append(score)
            
            # Update performance
            if len(self.contribution_quality[agent_id]) > 0:
                self.agent_performance[agent_id] = sum(
                    self.contribution_quality[agent_id]
                ) / len(self.contribution_quality[agent_id])
    
    def _update_performance(self, scoring_message: Message) -> None:
        """
        Update agent performance based on scoring.
        
        Args:
            scoring_message: Scoring message from Judge
        """
        agent_id = scoring_message.metadata.get("agent_id")
        score = scoring_message.metadata.get("weighted_score", 0)
        
        if agent_id:
            # Exponential moving average
            alpha = 0.3
            current = self.agent_performance.get(agent_id, 0.5)
            self.agent_performance[agent_id] = alpha * score + (1 - alpha) * current
    
    def _handle_elimination(self, message: Message) -> None:
        """
        Handle agent elimination.
        
        Args:
            message: Elimination message
        """
        eliminated_agent = message.metadata.get("eliminated_agent_id")
        if eliminated_agent:
            # Remove from performance tracking
            self.agent_performance.pop(eliminated_agent, None)
            self.contribution_quality.pop(eliminated_agent, None)
    
    def _generate_selection_explanation(
        self,
        selected: str,
        probabilities: List[float],
        agents: List[str]
    ) -> str:
        """
        Generate explanation for selection.
        
        Args:
            selected: Selected agent
            probabilities: Selection probabilities
            agents: Available agents
            
        Returns:
            Explanation text
        """
        # Find selected agent's probability
        selected_idx = agents.index(selected) if selected in agents else -1
        selected_prob = probabilities[selected_idx] if selected_idx >= 0 else 0
        
        # Identify strategy used
        strategy = self._get_dominant_strategy(probabilities)
        
        explanation = f"Selected {selected} (probability: {selected_prob:.2f}). "
        explanation += f"Primary strategy: {strategy}. "
        
        # Add context
        speak_count = self.agent_speak_counts.get(selected, 0)
        explanation += f"This agent has spoken {speak_count} times."
        
        return explanation
    
    def _get_dominant_strategy(self, probabilities: List[float]) -> str:
        """
        Identify the dominant selection strategy.
        
        Args:
            probabilities: Selection probabilities
            
        Returns:
            Name of dominant strategy
        """
        # Simple heuristic - could be more sophisticated
        if max(probabilities) > 0.5:
            return "merit"  # Strong preference
        elif min(probabilities) > 0.1:
            return "fairness"  # Even distribution
        else:
            return "diversity"  # Mixed
    
    async def _make_selection(self, arena_state: ArenaState) -> Message:
        """
        Make turn selection for current state.
        
        Args:
            arena_state: Current arena state
            
        Returns:
            Turn selection message
        """
        # Get active agents
        available_agents = arena_state.get_active_agent_ids()
        
        # Create request
        request = Message(
            sender_id="system",
            content="Select next speaker",
            metadata={
                "available_agents": available_agents,
                "current_turn": arena_state.current_turn
            }
        )
        
        return await self._select_next_speaker(request)
    
    def get_selection_statistics(self) -> Dict[str, Any]:
        """
        Get turn selection statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "total_selections": len(self.turn_history),
            "agent_speak_counts": dict(self.agent_speak_counts),
            "agent_performance": dict(self.agent_performance),
            "recent_speakers": self.recent_speakers[-5:],
            "strategy_weights": self.strategy_weights,
            "temperature": self.temperature,
            "epsilon": self.epsilon
        }