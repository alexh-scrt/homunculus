"""
Game Orchestrator for Arena

This module implements the main game orchestrator using LangGraph
to manage the complete game flow with state machines, checkpointing,
and recovery.

Features:
- LangGraph state machine for game flow
- Node-based architecture for modularity
- Edge conditions for transitions
- Checkpoint support for recovery
- Parallel agent execution
- Message routing through graph

Author: Homunculus Team
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Callable, TypedDict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint import MemorySaver
    from langgraph.prebuilt import ToolExecutor
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = None
    MemorySaver = None
    ToolExecutor = None

from ..models import Message, AgentState, ArenaState
from ..agents import BaseAgent
from ..game_theory import ScoringEngine, EliminationEngine
from .game_state import GameStateManager, GamePhase

logger = logging.getLogger(__name__)


class GameNode(Enum):
    """Nodes in the game orchestration graph."""
    START = "start"
    SETUP = "setup"
    TURN_START = "turn_start"
    AGENT_SELECT = "agent_select"
    AGENT_ACTION = "agent_action"
    MESSAGE_PROCESS = "message_process"
    SCORING = "scoring"
    ELIMINATION_CHECK = "elimination_check"
    ELIMINATION = "elimination"
    PHASE_CHECK = "phase_check"
    TURN_END = "turn_end"
    GAME_END = "game_end"
    ERROR = "error"


class GameEdge(Enum):
    """Edges in the game orchestration graph."""
    CONTINUE = "continue"
    ELIMINATE = "eliminate"
    NO_ELIMINATE = "no_eliminate"
    PHASE_CHANGE = "phase_change"
    NEXT_TURN = "next_turn"
    END_GAME = "end_game"
    ERROR = "error"
    RETRY = "retry"


class GameState(TypedDict):
    """State for the game orchestration graph."""
    game_id: str
    turn: int
    phase: str
    active_agents: List[str]
    eliminated_agents: List[str]
    scores: Dict[str, float]
    messages: List[Dict[str, Any]]
    current_speaker: Optional[str]
    elimination_pending: bool
    game_over: bool
    error: Optional[str]
    metadata: Dict[str, Any]


@dataclass
class OrchestratorConfig:
    """Configuration for the game orchestrator."""
    game_id: str
    max_turns: int = 100
    min_agents: int = 3
    checkpoint_frequency: int = 5
    enable_recovery: bool = True
    parallel_execution: bool = True
    timeout_seconds: int = 300
    retry_attempts: int = 3


class GameOrchestrator:
    """
    Main game orchestrator using LangGraph.
    
    Manages the complete game flow through a state machine
    with nodes for each game phase and edges for transitions.
    """
    
    def __init__(
        self,
        config: OrchestratorConfig,
        agents: List[BaseAgent],
        scoring_engine: ScoringEngine,
        elimination_engine: EliminationEngine
    ):
        """
        Initialize the game orchestrator.
        
        Args:
            config: Orchestrator configuration
            agents: List of participating agents
            scoring_engine: Scoring engine instance
            elimination_engine: Elimination engine instance
        """
        if not LANGGRAPH_AVAILABLE:
            raise ImportError("LangGraph is required for orchestration. Install with: pip install langgraph")
        
        self.config = config
        self.agents = {agent.agent_id: agent for agent in agents}
        self.scoring_engine = scoring_engine
        self.elimination_engine = elimination_engine
        
        # State management
        self.state_manager = GameStateManager(config.game_id)
        
        # Build the graph
        self.graph = self._build_graph()
        
        # Checkpoint saver for recovery
        self.checkpointer = MemorySaver() if config.enable_recovery else None
        
        # Compile the graph
        self.app = self.graph.compile(checkpointer=self.checkpointer)
        
        # Track execution
        self.execution_history: List[Dict[str, Any]] = []
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph state machine.
        
        Returns:
            Configured StateGraph
        """
        # Create graph with state type
        graph = StateGraph(GameState)
        
        # Add nodes
        graph.add_node(GameNode.START.value, self._start_node)
        graph.add_node(GameNode.SETUP.value, self._setup_node)
        graph.add_node(GameNode.TURN_START.value, self._turn_start_node)
        graph.add_node(GameNode.AGENT_SELECT.value, self._agent_select_node)
        graph.add_node(GameNode.AGENT_ACTION.value, self._agent_action_node)
        graph.add_node(GameNode.MESSAGE_PROCESS.value, self._message_process_node)
        graph.add_node(GameNode.SCORING.value, self._scoring_node)
        graph.add_node(GameNode.ELIMINATION_CHECK.value, self._elimination_check_node)
        graph.add_node(GameNode.ELIMINATION.value, self._elimination_node)
        graph.add_node(GameNode.PHASE_CHECK.value, self._phase_check_node)
        graph.add_node(GameNode.TURN_END.value, self._turn_end_node)
        graph.add_node(GameNode.GAME_END.value, self._game_end_node)
        graph.add_node(GameNode.ERROR.value, self._error_node)
        
        # Set entry point
        graph.set_entry_point(GameNode.START.value)
        
        # Add edges with conditions
        graph.add_edge(GameNode.START.value, GameNode.SETUP.value)
        graph.add_edge(GameNode.SETUP.value, GameNode.TURN_START.value)
        graph.add_edge(GameNode.TURN_START.value, GameNode.AGENT_SELECT.value)
        graph.add_edge(GameNode.AGENT_SELECT.value, GameNode.AGENT_ACTION.value)
        graph.add_edge(GameNode.AGENT_ACTION.value, GameNode.MESSAGE_PROCESS.value)
        graph.add_edge(GameNode.MESSAGE_PROCESS.value, GameNode.SCORING.value)
        graph.add_edge(GameNode.SCORING.value, GameNode.ELIMINATION_CHECK.value)
        
        # Conditional edges
        graph.add_conditional_edges(
            GameNode.ELIMINATION_CHECK.value,
            self._should_eliminate,
            {
                True: GameNode.ELIMINATION.value,
                False: GameNode.PHASE_CHECK.value
            }
        )
        
        graph.add_edge(GameNode.ELIMINATION.value, GameNode.PHASE_CHECK.value)
        
        graph.add_conditional_edges(
            GameNode.PHASE_CHECK.value,
            self._check_game_over,
            {
                True: GameNode.GAME_END.value,
                False: GameNode.TURN_END.value
            }
        )
        
        graph.add_conditional_edges(
            GameNode.TURN_END.value,
            self._should_continue,
            {
                True: GameNode.TURN_START.value,
                False: GameNode.GAME_END.value
            }
        )
        
        # Error handling
        for node in GameNode:
            if node not in [GameNode.ERROR, GameNode.GAME_END]:
                graph.add_conditional_edges(
                    node.value,
                    self._check_error,
                    {
                        True: GameNode.ERROR.value,
                        False: None  # Continue normal flow
                    }
                )
        
        # End states
        graph.add_edge(GameNode.GAME_END.value, END)
        graph.add_edge(GameNode.ERROR.value, END)
        
        return graph
    
    async def _start_node(self, state: GameState) -> GameState:
        """Initialize the game."""
        logger.info(f"Starting game {self.config.game_id}")
        
        state["game_id"] = self.config.game_id
        state["turn"] = 0
        state["phase"] = GamePhase.EARLY.value
        state["active_agents"] = list(self.agents.keys())
        state["eliminated_agents"] = []
        state["scores"] = {agent_id: 0.0 for agent_id in self.agents}
        state["messages"] = []
        state["current_speaker"] = None
        state["elimination_pending"] = False
        state["game_over"] = False
        state["error"] = None
        state["metadata"] = {
            "start_time": datetime.utcnow().isoformat(),
            "total_agents": len(self.agents)
        }
        
        return state
    
    async def _setup_node(self, state: GameState) -> GameState:
        """Set up game components."""
        logger.info("Setting up game components")
        
        # Initialize agents
        for agent_id, agent in self.agents.items():
            await agent.initialize()
        
        # Initialize state manager
        self.state_manager.initialize_state(
            active_agents=state["active_agents"],
            phase=GamePhase[state["phase"]]
        )
        
        return state
    
    async def _turn_start_node(self, state: GameState) -> GameState:
        """Start a new turn."""
        state["turn"] += 1
        logger.info(f"Starting turn {state['turn']}")
        
        # Clear turn messages
        state["messages"] = []
        
        # Check for checkpoint
        if state["turn"] % self.config.checkpoint_frequency == 0:
            self._create_checkpoint(state)
        
        return state
    
    async def _agent_select_node(self, state: GameState) -> GameState:
        """Select the next agent to speak."""
        # Use turn selector agent if available
        # For now, simple round-robin
        active = state["active_agents"]
        if not active:
            state["game_over"] = True
            return state
        
        current_idx = 0
        if state["current_speaker"] in active:
            current_idx = active.index(state["current_speaker"])
            current_idx = (current_idx + 1) % len(active)
        
        state["current_speaker"] = active[current_idx]
        logger.info(f"Selected speaker: {state['current_speaker']}")
        
        return state
    
    async def _agent_action_node(self, state: GameState) -> GameState:
        """Execute agent action."""
        agent_id = state["current_speaker"]
        if not agent_id or agent_id not in self.agents:
            return state
        
        agent = self.agents[agent_id]
        
        try:
            # Generate agent action
            context = {
                "turn": state["turn"],
                "phase": state["phase"],
                "scores": state["scores"],
                "active_agents": state["active_agents"]
            }
            
            message = await agent.generate_action(context)
            
            if message:
                state["messages"].append(message.to_dict())
                logger.info(f"Agent {agent_id} generated action: {message.message_type}")
        
        except Exception as e:
            logger.error(f"Error in agent action: {e}")
            state["error"] = str(e)
        
        return state
    
    async def _message_process_node(self, state: GameState) -> GameState:
        """Process messages through the system."""
        for msg_dict in state["messages"]:
            message = Message.from_dict(msg_dict)
            
            # Route to all agents
            if self.config.parallel_execution:
                tasks = []
                for agent_id, agent in self.agents.items():
                    if agent_id != message.sender_id:
                        tasks.append(agent.process_message(message))
                
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
            else:
                for agent_id, agent in self.agents.items():
                    if agent_id != message.sender_id:
                        await agent.process_message(message)
        
        return state
    
    async def _scoring_node(self, state: GameState) -> GameState:
        """Score contributions."""
        from ..game_theory.scoring_engine import ScoringContext
        
        context = ScoringContext(
            game_phase=state["phase"],
            turn_number=state["turn"],
            total_agents=len(self.agents),
            eliminated_agents=len(state["eliminated_agents"]),
            recent_scores=list(state["scores"].values())[-5:],
            problem_complexity=0.6
        )
        
        for msg_dict in state["messages"]:
            message = Message.from_dict(msg_dict)
            
            if message.message_type == "contribution":
                metrics = self.scoring_engine.score_contribution(message, context)
                
                # Update scores
                if message.sender_id in state["scores"]:
                    state["scores"][message.sender_id] += metrics.weighted_score
                
                logger.info(f"Scored {message.sender_id}: {metrics.weighted_score:.3f}")
        
        return state
    
    async def _elimination_check_node(self, state: GameState) -> GameState:
        """Check if elimination should occur."""
        # Check elimination conditions
        should_eliminate = (
            state["turn"] > 20 and
            state["turn"] % 10 == 0 and
            len(state["active_agents"]) > self.config.min_agents
        )
        
        state["elimination_pending"] = should_eliminate
        
        return state
    
    async def _elimination_node(self, state: GameState) -> GameState:
        """Process eliminations."""
        from ..game_theory.elimination_mechanics import EliminationContext
        
        context = EliminationContext(
            turn_number=state["turn"],
            total_agents=len(self.agents),
            active_agents=len(state["active_agents"]),
            elimination_round=state["turn"] // 10,
            scores=state["scores"],
            accusations={},
            protections={}
        )
        
        # Create agent states
        agent_states = {
            agent_id: AgentState(
                agent_id=agent_id,
                is_active=agent_id in state["active_agents"],
                score=state["scores"].get(agent_id, 0)
            )
            for agent_id in self.agents
        }
        
        # Process elimination
        eliminated = self.elimination_engine.process_elimination_round(
            context, agent_states
        )
        
        # Update state
        for agent_id in eliminated:
            if agent_id in state["active_agents"]:
                state["active_agents"].remove(agent_id)
                state["eliminated_agents"].append(agent_id)
                logger.info(f"Eliminated agent: {agent_id}")
        
        state["elimination_pending"] = False
        
        return state
    
    async def _phase_check_node(self, state: GameState) -> GameState:
        """Check and update game phase."""
        turn = state["turn"]
        total_agents = len(self.agents)
        eliminated = len(state["eliminated_agents"])
        
        # Determine phase
        if turn < 20:
            new_phase = GamePhase.EARLY
        elif turn < 50:
            new_phase = GamePhase.MID
        elif turn < 80:
            new_phase = GamePhase.LATE
        else:
            new_phase = GamePhase.FINAL
        
        # Also check elimination rate
        if eliminated / total_agents > 0.5:
            new_phase = GamePhase.LATE
        if eliminated / total_agents > 0.75:
            new_phase = GamePhase.FINAL
        
        if new_phase.value != state["phase"]:
            logger.info(f"Phase transition: {state['phase']} -> {new_phase.value}")
            state["phase"] = new_phase.value
        
        # Check game over conditions
        if len(state["active_agents"]) <= 1:
            state["game_over"] = True
        elif state["turn"] >= self.config.max_turns:
            state["game_over"] = True
        
        return state
    
    async def _turn_end_node(self, state: GameState) -> GameState:
        """End the current turn."""
        logger.info(f"Turn {state['turn']} complete")
        
        # Update state manager
        self.state_manager.update_turn(
            turn_number=state["turn"],
            active_agents=state["active_agents"],
            scores=state["scores"]
        )
        
        return state
    
    async def _game_end_node(self, state: GameState) -> GameState:
        """End the game."""
        logger.info(f"Game {self.config.game_id} ending")
        
        # Determine winner
        if state["active_agents"]:
            winner = max(
                state["active_agents"],
                key=lambda x: state["scores"].get(x, 0)
            )
            state["metadata"]["winner"] = winner
            logger.info(f"Winner: {winner}")
        else:
            state["metadata"]["winner"] = None
        
        # Final statistics
        state["metadata"]["end_time"] = datetime.utcnow().isoformat()
        state["metadata"]["total_turns"] = state["turn"]
        state["metadata"]["final_scores"] = state["scores"]
        
        return state
    
    async def _error_node(self, state: GameState) -> GameState:
        """Handle errors."""
        logger.error(f"Error in game: {state.get('error', 'Unknown')}")
        
        # Attempt recovery if enabled
        if self.config.enable_recovery and self.checkpointer:
            # Try to restore from last checkpoint
            try:
                restored = await self._restore_from_checkpoint()
                if restored:
                    return restored
            except Exception as e:
                logger.error(f"Recovery failed: {e}")
        
        state["game_over"] = True
        return state
    
    def _should_eliminate(self, state: GameState) -> bool:
        """Check if elimination should occur."""
        return state.get("elimination_pending", False)
    
    def _check_game_over(self, state: GameState) -> bool:
        """Check if game is over."""
        return state.get("game_over", False)
    
    def _should_continue(self, state: GameState) -> bool:
        """Check if game should continue."""
        return not state.get("game_over", False)
    
    def _check_error(self, state: GameState) -> bool:
        """Check for errors."""
        return state.get("error") is not None
    
    def _create_checkpoint(self, state: GameState) -> None:
        """Create a checkpoint for recovery."""
        if self.checkpointer:
            checkpoint = {
                "state": state.copy(),
                "timestamp": datetime.utcnow().isoformat(),
                "turn": state["turn"]
            }
            logger.info(f"Created checkpoint at turn {state['turn']}")
            # In production, save to persistent storage
    
    async def _restore_from_checkpoint(self) -> Optional[GameState]:
        """Restore from last checkpoint."""
        # In production, load from persistent storage
        logger.info("Attempting to restore from checkpoint")
        return None
    
    async def run_game(self) -> Dict[str, Any]:
        """
        Run a complete game.
        
        Returns:
            Game results
        """
        logger.info(f"Starting game orchestration for {self.config.game_id}")
        
        # Initial state
        initial_state: GameState = {
            "game_id": self.config.game_id,
            "turn": 0,
            "phase": GamePhase.EARLY.value,
            "active_agents": [],
            "eliminated_agents": [],
            "scores": {},
            "messages": [],
            "current_speaker": None,
            "elimination_pending": False,
            "game_over": False,
            "error": None,
            "metadata": {}
        }
        
        try:
            # Run the graph
            config = {"configurable": {"thread_id": self.config.game_id}}
            
            async for event in self.app.astream(initial_state, config):
                # Log progress
                for node, state in event.items():
                    logger.debug(f"Node {node} completed")
                    self.execution_history.append({
                        "node": node,
                        "turn": state.get("turn"),
                        "timestamp": datetime.utcnow()
                    })
                
                # Check timeout
                if len(self.execution_history) > self.config.max_turns * 10:
                    logger.warning("Execution limit reached")
                    break
            
            # Get final state
            final_state = await self.app.aget_state(config)
            
            return {
                "game_id": self.config.game_id,
                "winner": final_state.values.get("metadata", {}).get("winner"),
                "final_scores": final_state.values.get("scores", {}),
                "total_turns": final_state.values.get("turn", 0),
                "eliminated": final_state.values.get("eliminated_agents", []),
                "execution_history": self.execution_history
            }
        
        except Exception as e:
            logger.error(f"Game orchestration failed: {e}")
            return {
                "game_id": self.config.game_id,
                "error": str(e),
                "execution_history": self.execution_history
            }
    
    async def run_turn(self, state: GameState) -> GameState:
        """
        Run a single turn.
        
        Args:
            state: Current game state
            
        Returns:
            Updated game state
        """
        # Run one iteration through the graph
        config = {"configurable": {"thread_id": self.config.game_id}}
        
        result = await self.app.ainvoke(state, config)
        
        return result