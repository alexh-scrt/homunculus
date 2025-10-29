"""
Game Strategies for Arena

This module implements various game theory strategies that agents
can employ in the competitive environment.

Features:
- Classic game theory strategies
- Adaptive strategies
- Meta-game strategies
- Strategy evaluation
- Equilibrium analysis

Author: Homunculus Team
"""

import logging
import random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class Action(Enum):
    """Possible actions in the game."""
    COOPERATE = "cooperate"
    DEFECT = "defect"
    CONTRIBUTE = "contribute"
    ACCUSE = "accuse"
    SUPPORT = "support"
    ABSTAIN = "abstain"


@dataclass
class GameState:
    """Current state of the game."""
    turn: int
    active_agents: List[str]
    eliminated_agents: List[str]
    scores: Dict[str, float]
    recent_actions: List[Tuple[str, Action]]
    phase: str  # early, mid, late, final
    
    @property
    def agents_remaining(self) -> int:
        """Number of active agents."""
        return len(self.active_agents)
    
    @property
    def elimination_rate(self) -> float:
        """Rate of elimination."""
        total = len(self.active_agents) + len(self.eliminated_agents)
        if total == 0:
            return 0.0
        return len(self.eliminated_agents) / total


@dataclass
class StrategyMemory:
    """Memory for strategy decisions."""
    my_actions: List[Action]
    opponent_actions: Dict[str, List[Action]]
    payoffs: List[float]
    cooperation_history: Dict[str, bool]
    
    def remember_action(self, agent: str, action: Action) -> None:
        """Remember an agent's action."""
        if agent not in self.opponent_actions:
            self.opponent_actions[agent] = []
        self.opponent_actions[agent].append(action)
    
    def get_last_action(self, agent: str) -> Optional[Action]:
        """Get agent's last action."""
        if agent in self.opponent_actions and self.opponent_actions[agent]:
            return self.opponent_actions[agent][-1]
        return None


class GameStrategy(ABC):
    """Abstract base class for game strategies."""
    
    def __init__(self, agent_id: str):
        """
        Initialize strategy.
        
        Args:
            agent_id: ID of agent using strategy
        """
        self.agent_id = agent_id
        self.memory = StrategyMemory(
            my_actions=[],
            opponent_actions={},
            payoffs=[],
            cooperation_history={}
        )
    
    @abstractmethod
    def decide_action(
        self,
        state: GameState,
        opponent: Optional[str] = None
    ) -> Action:
        """
        Decide next action.
        
        Args:
            state: Current game state
            opponent: Specific opponent if applicable
            
        Returns:
            Chosen action
        """
        pass
    
    @abstractmethod
    def update(
        self,
        action: Action,
        payoff: float,
        opponent_action: Optional[Action] = None
    ) -> None:
        """
        Update strategy based on outcome.
        
        Args:
            action: Action taken
            payoff: Resulting payoff
            opponent_action: Opponent's action if known
        """
        pass
    
    def reset(self) -> None:
        """Reset strategy memory."""
        self.memory = StrategyMemory(
            my_actions=[],
            opponent_actions={},
            payoffs=[],
            cooperation_history={}
        )


class TitForTat(GameStrategy):
    """
    Tit-for-Tat strategy: Cooperate first, then copy opponent's last action.
    
    Classic strategy that's simple but effective.
    """
    
    def __init__(self, agent_id: str, forgiveness: float = 0.0):
        """
        Initialize Tit-for-Tat.
        
        Args:
            agent_id: Agent ID
            forgiveness: Probability of forgiving defection
        """
        super().__init__(agent_id)
        self.forgiveness = forgiveness
    
    def decide_action(
        self,
        state: GameState,
        opponent: Optional[str] = None
    ) -> Action:
        """Cooperate first, then copy opponent."""
        if not opponent:
            # No specific opponent, default to cooperate
            return Action.COOPERATE
        
        last_action = self.memory.get_last_action(opponent)
        
        if last_action is None:
            # First interaction, cooperate
            return Action.COOPERATE
        
        if last_action == Action.DEFECT:
            # Opponent defected, maybe forgive
            if random.random() < self.forgiveness:
                return Action.COOPERATE
            return Action.DEFECT
        
        # Copy opponent's cooperation
        return Action.COOPERATE
    
    def update(
        self,
        action: Action,
        payoff: float,
        opponent_action: Optional[Action] = None
    ) -> None:
        """Update memory."""
        self.memory.my_actions.append(action)
        self.memory.payoffs.append(payoff)


class AlwaysCooperate(GameStrategy):
    """Always cooperate strategy - maximally trusting."""
    
    def decide_action(
        self,
        state: GameState,
        opponent: Optional[str] = None
    ) -> Action:
        """Always cooperate."""
        return Action.COOPERATE
    
    def update(
        self,
        action: Action,
        payoff: float,
        opponent_action: Optional[Action] = None
    ) -> None:
        """Update memory."""
        self.memory.my_actions.append(action)
        self.memory.payoffs.append(payoff)


class AlwaysDefect(GameStrategy):
    """Always defect strategy - maximally selfish."""
    
    def decide_action(
        self,
        state: GameState,
        opponent: Optional[str] = None
    ) -> Action:
        """Always defect."""
        return Action.DEFECT
    
    def update(
        self,
        action: Action,
        payoff: float,
        opponent_action: Optional[Action] = None
    ) -> None:
        """Update memory."""
        self.memory.my_actions.append(action)
        self.memory.payoffs.append(payoff)


class AdaptiveStrategy(GameStrategy):
    """
    Adaptive strategy that learns from game history.
    
    Uses reinforcement learning principles to adapt behavior.
    """
    
    def __init__(
        self,
        agent_id: str,
        learning_rate: float = 0.1,
        exploration_rate: float = 0.1
    ):
        """
        Initialize adaptive strategy.
        
        Args:
            agent_id: Agent ID
            learning_rate: Learning rate for updates
            exploration_rate: Probability of exploration
        """
        super().__init__(agent_id)
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        
        # Q-values for state-action pairs
        self.q_values: Dict[Tuple[str, Action], float] = defaultdict(float)
        
        # State features
        self.state_encoder = StateEncoder()
    
    def decide_action(
        self,
        state: GameState,
        opponent: Optional[str] = None
    ) -> Action:
        """Choose action based on learned Q-values."""
        # Encode state
        state_key = self.state_encoder.encode(state, opponent)
        
        # Exploration vs exploitation
        if random.random() < self.exploration_rate:
            # Explore: random action
            return random.choice(list(Action))
        
        # Exploit: choose best action
        best_action = None
        best_value = float('-inf')
        
        for action in Action:
            q_value = self.q_values.get((state_key, action), 0.0)
            if q_value > best_value:
                best_value = q_value
                best_action = action
        
        return best_action or Action.COOPERATE
    
    def update(
        self,
        action: Action,
        payoff: float,
        opponent_action: Optional[Action] = None
    ) -> None:
        """Update Q-values based on outcome."""
        self.memory.my_actions.append(action)
        self.memory.payoffs.append(payoff)
        
        if len(self.memory.payoffs) < 2:
            return
        
        # Calculate reward (change in payoff)
        reward = payoff - self.memory.payoffs[-2] if len(self.memory.payoffs) > 1 else payoff
        
        # Update Q-value
        state_key = self.state_encoder.get_last_state()
        if state_key:
            old_q = self.q_values.get((state_key, action), 0.0)
            self.q_values[(state_key, action)] = old_q + self.learning_rate * (reward - old_q)


class PavlovStrategy(GameStrategy):
    """
    Pavlov strategy: Win-stay, lose-shift.
    
    If the last action resulted in good payoff, repeat it.
    Otherwise, switch.
    """
    
    def __init__(self, agent_id: str, threshold: float = 0.5):
        """
        Initialize Pavlov strategy.
        
        Args:
            agent_id: Agent ID
            threshold: Payoff threshold for "winning"
        """
        super().__init__(agent_id)
        self.threshold = threshold
        self.last_action = Action.COOPERATE
    
    def decide_action(
        self,
        state: GameState,
        opponent: Optional[str] = None
    ) -> Action:
        """Win-stay, lose-shift."""
        if not self.memory.payoffs:
            # First action, cooperate
            self.last_action = Action.COOPERATE
            return Action.COOPERATE
        
        last_payoff = self.memory.payoffs[-1]
        
        if last_payoff >= self.threshold:
            # Won, stay with last action
            return self.last_action
        else:
            # Lost, switch action
            if self.last_action == Action.COOPERATE:
                self.last_action = Action.DEFECT
            else:
                self.last_action = Action.COOPERATE
            return self.last_action
    
    def update(
        self,
        action: Action,
        payoff: float,
        opponent_action: Optional[Action] = None
    ) -> None:
        """Update memory and last action."""
        self.memory.my_actions.append(action)
        self.memory.payoffs.append(payoff)
        self.last_action = action


class MetaStrategy(GameStrategy):
    """
    Meta-strategy that combines multiple strategies.
    
    Selects between strategies based on their performance.
    """
    
    def __init__(self, agent_id: str):
        """Initialize meta-strategy."""
        super().__init__(agent_id)
        
        # Portfolio of strategies
        self.strategies = {
            "tit_for_tat": TitForTat(agent_id, forgiveness=0.1),
            "adaptive": AdaptiveStrategy(agent_id),
            "pavlov": PavlovStrategy(agent_id)
        }
        
        # Performance tracking
        self.strategy_performance: Dict[str, List[float]] = defaultdict(list)
        self.current_strategy = "tit_for_tat"
        
        # Switch frequency
        self.switch_frequency = 10  # Evaluate every N turns
        self.turn_count = 0
    
    def decide_action(
        self,
        state: GameState,
        opponent: Optional[str] = None
    ) -> Action:
        """Delegate to current strategy."""
        # Check if should switch strategies
        if self.turn_count % self.switch_frequency == 0 and self.turn_count > 0:
            self._evaluate_and_switch()
        
        self.turn_count += 1
        
        # Use current strategy
        strategy = self.strategies[self.current_strategy]
        return strategy.decide_action(state, opponent)
    
    def update(
        self,
        action: Action,
        payoff: float,
        opponent_action: Optional[Action] = None
    ) -> None:
        """Update current strategy and track performance."""
        # Update current strategy
        strategy = self.strategies[self.current_strategy]
        strategy.update(action, payoff, opponent_action)
        
        # Track performance
        self.strategy_performance[self.current_strategy].append(payoff)
        
        # Update own memory
        self.memory.my_actions.append(action)
        self.memory.payoffs.append(payoff)
    
    def _evaluate_and_switch(self) -> None:
        """Evaluate strategies and switch if beneficial."""
        if not self.strategy_performance:
            return
        
        # Calculate average performance
        avg_performance = {}
        for name, payoffs in self.strategy_performance.items():
            if payoffs:
                avg_performance[name] = np.mean(payoffs[-10:])  # Recent performance
        
        if not avg_performance:
            return
        
        # Find best strategy
        best_strategy = max(avg_performance.items(), key=lambda x: x[1])[0]
        
        if best_strategy != self.current_strategy:
            logger.info(f"Switching strategy from {self.current_strategy} to {best_strategy}")
            self.current_strategy = best_strategy


class StateEncoder:
    """Encodes game state for strategy learning."""
    
    def __init__(self):
        """Initialize encoder."""
        self.last_state = None
    
    def encode(
        self,
        state: GameState,
        opponent: Optional[str] = None
    ) -> str:
        """
        Encode game state to string key.
        
        Args:
            state: Game state
            opponent: Specific opponent
            
        Returns:
            State key
        """
        # Simple encoding - could be more sophisticated
        features = [
            state.phase,
            self._discretize(state.agents_remaining, [3, 5, 10]),
            self._discretize(state.elimination_rate, [0.2, 0.5, 0.8])
        ]
        
        if opponent and opponent in state.scores:
            opponent_score = state.scores[opponent]
            features.append(self._discretize(opponent_score, [0.3, 0.5, 0.7]))
        
        state_key = "_".join(map(str, features))
        self.last_state = state_key
        return state_key
    
    def get_last_state(self) -> Optional[str]:
        """Get last encoded state."""
        return self.last_state
    
    def _discretize(self, value: float, bins: List[float]) -> str:
        """Discretize continuous value."""
        for i, threshold in enumerate(bins):
            if value < threshold:
                return f"bin{i}"
        return f"bin{len(bins)}"


class StrategyEvaluator:
    """
    Evaluates and compares strategies.
    """
    
    def __init__(self):
        """Initialize evaluator."""
        self.evaluation_results: List[Dict[str, Any]] = []
    
    def evaluate_strategies(
        self,
        strategies: List[GameStrategy],
        num_rounds: int = 100
    ) -> Dict[str, float]:
        """
        Evaluate strategies in round-robin tournament.
        
        Args:
            strategies: List of strategies to evaluate
            num_rounds: Number of rounds per matchup
            
        Returns:
            Average scores for each strategy
        """
        scores = defaultdict(list)
        
        # Round-robin tournament
        for i, strategy1 in enumerate(strategies):
            for strategy2 in strategies[i+1:]:
                payoffs1, payoffs2 = self._play_match(
                    strategy1, strategy2, num_rounds
                )
                
                scores[strategy1.agent_id].extend(payoffs1)
                scores[strategy2.agent_id].extend(payoffs2)
        
        # Calculate averages
        avg_scores = {}
        for agent_id, payoffs in scores.items():
            avg_scores[agent_id] = np.mean(payoffs) if payoffs else 0.0
        
        return avg_scores
    
    def _play_match(
        self,
        strategy1: GameStrategy,
        strategy2: GameStrategy,
        num_rounds: int
    ) -> Tuple[List[float], List[float]]:
        """
        Play a match between two strategies.
        
        Args:
            strategy1: First strategy
            strategy2: Second strategy
            num_rounds: Number of rounds
            
        Returns:
            Payoffs for both strategies
        """
        payoffs1 = []
        payoffs2 = []
        
        # Simple prisoner's dilemma payoff matrix
        payoff_matrix = {
            (Action.COOPERATE, Action.COOPERATE): (3, 3),
            (Action.COOPERATE, Action.DEFECT): (0, 5),
            (Action.DEFECT, Action.COOPERATE): (5, 0),
            (Action.DEFECT, Action.DEFECT): (1, 1)
        }
        
        # Create dummy game state
        state = GameState(
            turn=0,
            active_agents=[strategy1.agent_id, strategy2.agent_id],
            eliminated_agents=[],
            scores={strategy1.agent_id: 0, strategy2.agent_id: 0},
            recent_actions=[],
            phase="mid"
        )
        
        for round_num in range(num_rounds):
            state.turn = round_num
            
            # Get actions
            action1 = strategy1.decide_action(state, strategy2.agent_id)
            action2 = strategy2.decide_action(state, strategy1.agent_id)
            
            # Map to cooperate/defect if needed
            if action1 not in [Action.COOPERATE, Action.DEFECT]:
                action1 = Action.COOPERATE
            if action2 not in [Action.COOPERATE, Action.DEFECT]:
                action2 = Action.COOPERATE
            
            # Get payoffs
            pay1, pay2 = payoff_matrix.get(
                (action1, action2),
                (0, 0)
            )
            
            # Update strategies
            strategy1.update(action1, pay1, action2)
            strategy2.update(action2, pay2, action1)
            
            # Update memory
            strategy1.memory.remember_action(strategy2.agent_id, action2)
            strategy2.memory.remember_action(strategy1.agent_id, action1)
            
            # Record payoffs
            payoffs1.append(pay1)
            payoffs2.append(pay2)
            
            # Update state
            state.scores[strategy1.agent_id] += pay1
            state.scores[strategy2.agent_id] += pay2
            state.recent_actions = [(strategy1.agent_id, action1), 
                                   (strategy2.agent_id, action2)]
        
        return payoffs1, payoffs2
    
    def find_equilibrium(
        self,
        strategies: List[GameStrategy],
        tolerance: float = 0.01,
        max_iterations: int = 100
    ) -> Optional[Dict[str, float]]:
        """
        Find Nash equilibrium if it exists.
        
        Args:
            strategies: List of strategies
            tolerance: Convergence tolerance
            max_iterations: Maximum iterations
            
        Returns:
            Equilibrium strategy weights if found
        """
        # Simplified equilibrium finding
        # In practice, would use more sophisticated methods
        
        weights = {s.agent_id: 1.0/len(strategies) for s in strategies}
        
        for iteration in range(max_iterations):
            # Evaluate current weights
            scores = self.evaluate_strategies(strategies, num_rounds=50)
            
            # Update weights based on scores
            new_weights = {}
            total_score = sum(scores.values())
            
            if total_score > 0:
                for agent_id in weights:
                    new_weights[agent_id] = scores.get(agent_id, 0) / total_score
            else:
                new_weights = weights.copy()
            
            # Check convergence
            diff = sum(abs(new_weights[a] - weights[a]) for a in weights)
            
            if diff < tolerance:
                logger.info(f"Equilibrium found at iteration {iteration}")
                return new_weights
            
            weights = new_weights
        
        logger.warning("No equilibrium found within max iterations")
        return None