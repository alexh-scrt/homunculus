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

from ..config.logging_config import setup_arena_logging, ArenaLogger

try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError as e:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = None
    MemorySaver = None

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
    NARRATOR_FINAL = "narrator_final"
    JUDGE_FINAL = "judge_final"
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
    recursion_limit: int = 250  # LangGraph recursion limit - configurable like talks project


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
        
        # Track conversation history for context
        self.conversation_history: List[Dict[str, Any]] = []
        
        # Set up Arena logging
        self.arena_logger = setup_arena_logging(config.game_id)
        
        # Set as current arena logger for agents to access
        from ..config.logging_config import set_arena_logger
        
        # Initialize seed question (will be set by start_game if provided)
        self.seed_question: Optional[str] = None
        set_arena_logger(self.arena_logger)
    
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
        graph.add_node(GameNode.NARRATOR_FINAL.value, self._narrator_final_node)
        graph.add_node(GameNode.JUDGE_FINAL.value, self._judge_final_node)
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
        
        # Error handling - errors will be handled within each node function
        # instead of using conditional edges to None (which is invalid)
        
        # End states - game flows through final summary and verdict before ending
        graph.add_edge(GameNode.GAME_END.value, GameNode.NARRATOR_FINAL.value)
        graph.add_edge(GameNode.NARRATOR_FINAL.value, GameNode.JUDGE_FINAL.value)
        graph.add_edge(GameNode.JUDGE_FINAL.value, END)
        graph.add_edge(GameNode.ERROR.value, END)
        
        return graph
    
    def _build_minimal_test_graph(self) -> StateGraph:
        """Build a minimal test graph to debug LangGraph issues."""
        from typing import TypedDict
        
        class MinimalState(TypedDict):
            test: str
        
        async def simple_node(state: MinimalState) -> MinimalState:
            return {"test": f"processed: {state['test']}"}
        
        graph = StateGraph(MinimalState)
        graph.add_node("simple", simple_node)
        graph.set_entry_point("simple")
        graph.add_edge("simple", END)
        
        return graph
    
    def _build_single_node_graph(self) -> StateGraph:
        """Build a graph with just our start node to test GameState compatibility."""
        graph = StateGraph(GameState)
        graph.add_node(GameNode.START.value, self._start_node)
        graph.set_entry_point(GameNode.START.value)
        graph.add_edge(GameNode.START.value, END)
        
        return graph
    
    async def _start_node(self, state: GameState) -> GameState:
        """Initialize the game."""
        try:
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
            
            logger.info(f"Start node completed - initialized {len(state['active_agents'])} agents")
            return state
            
        except Exception as e:
            logger.error(f"Error in start node: {e}")
            state["error"] = str(e)
            state["game_over"] = True
            return state
    
    async def _setup_node(self, state: GameState) -> GameState:
        """Set up game components."""
        logger.info("Setting up game components")
        
        # Initialize agents
        for agent_id, agent in self.agents.items():
            await agent.initialize()
        
        # Initialize state manager
        # Convert string phase back to enum
        phase_str = state["phase"]
        phase_enum = GamePhase(phase_str)  # This converts "early" -> GamePhase.EARLY
        
        self.state_manager.initialize_state(
            active_agents=state["active_agents"],
            phase=phase_enum
        )
        
        # Generate narrator's opening announcement
        await self._generate_opening_announcement(state)
        
        return state
    
    async def _generate_opening_announcement(self, state: GameState) -> None:
        """Generate narrator's opening announcement for the game."""
        # Find the narrator agent
        narrator_agent = None
        for agent_id, agent in self.agents.items():
            if hasattr(agent, 'config') and hasattr(agent.config, 'role') and agent.config.role.value == 'narrator':
                narrator_agent = agent
                break
        
        if not narrator_agent:
            logger.warning("No narrator agent found for opening announcement")
            return
        
        try:
            # Get gameplay agents (exclude narrator and judge)
            gameplay_agents = [
                agent_id for agent_id in state["active_agents"] 
                if not self._is_special_agent(agent_id)
            ]
            
            # Get agent names for announcement
            participant_names = []
            for agent_id in gameplay_agents:
                agent = self.agents.get(agent_id)
                if agent and hasattr(agent, 'agent_name'):
                    participant_names.append(agent.agent_name)
                else:
                    participant_names.append(agent_id)
            
            # Prepare context for narrator
            context = {
                "turn": 0,  # Opening announcement is turn 0
                "phase": state["phase"],
                "participants": participant_names,
                "max_turns": self.config.max_turns,
                "seed_question": getattr(self, 'seed_question', ''),
                "game_id": state["game_id"],
                "opening_announcement": True,
                "arena_state": state
            }
            
            # Generate opening announcement
            opening_message = await narrator_agent.generate_action(context)
            
            if opening_message:
                # Log the opening announcement with special formatting
                agent_name = opening_message.sender_name
                content = opening_message.content
                
                # Display with special formatting for opening
                # print("\n" + "🎭 " + "="*80)
                # print("ARENA BEGINS")
                # print("="*80)
                # print(f"[96m{agent_name}[0m: {content}")
                # print("="*80 + "\n")
                
        except Exception as e:
            logger.error(f"Error generating opening announcement: {e}")
    
    async def _turn_start_node(self, state: GameState) -> GameState:
        """Start a new turn."""
        state["turn"] += 1
        logger.info(f"Starting turn {state['turn']}")
        
        # Log turn start with clean format
        self.arena_logger.log_turn_start(
            turn=state["turn"], 
            current_speaker=state.get("current_speaker", "None"),
            active_agents=state["active_agents"]
        )
        
        # Clear turn messages
        state["messages"] = []
        
        # Check for checkpoint
        if state["turn"] % self.config.checkpoint_frequency == 0:
            self._create_checkpoint(state)
        
        return state
    
    async def _agent_select_node(self, state: GameState) -> GameState:
        """Select the next agent to speak."""
        try:
            logger.info(f"Agent selection - active_agents: {state.get('active_agents', [])}")
            # Filter out special agents (narrator/judge) from normal turn-based gameplay
            # They have dedicated final nodes and shouldn't participate in regular discussions
            gameplay_agents = [
                agent_id for agent_id in state["active_agents"] 
                if not self._is_special_agent(agent_id)
            ]
            
            if not gameplay_agents:
                logger.warning("No gameplay agents found, ending game")
                state["game_over"] = True
                return state
        
            current_idx = 0
            if state["current_speaker"] in gameplay_agents:
                current_idx = gameplay_agents.index(state["current_speaker"])
                current_idx = (current_idx + 1) % len(gameplay_agents)
            
            state["current_speaker"] = gameplay_agents[current_idx]
            logger.info(f"Selected speaker: {state['current_speaker']}")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in agent selection: {e}")
            state["error"] = str(e)
            state["game_over"] = True
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
                "active_agents": state["active_agents"],
                "my_turn": True,  # Tell the agent it's their turn to act
                "current_speaker": agent_id,
                "game_id": state["game_id"],
                "recent_messages": self._get_recent_messages(),  # Add conversation history
                "arena_state": state,  # Add full state for narrator agent
                "seed_question": self.seed_question  # Add seed question/topic for agents
            }
            
            message = await agent.generate_action(context)
            
            if message:
                state["messages"].append(message.to_dict())
                logger.info(f"Agent {agent_id} generated action: {message.message_type}")
                
                # Arena logging is now handled by the LLM client for both streaming and non-streaming modes
        
        except Exception as e:
            logger.error(f"Error in agent action: {e}")
            state["error"] = str(e)
        
        return state
    
    def _get_recent_messages(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent messages from conversation history.
        
        Args:
            limit: Maximum number of recent messages to return
            
        Returns:
            List of recent message dictionaries
        """
        return self.conversation_history[-limit:] if self.conversation_history else []
    
    async def _message_process_node(self, state: GameState) -> GameState:
        """Process messages through the system."""
        for msg_dict in state["messages"]:
            message = Message.from_dict(msg_dict)
            
            # Store in conversation history for context
            self.conversation_history.append({
                "sender_id": message.sender_id,
                "sender_name": message.sender_name,
                "content": message.content,
                "message_type": message.message_type,
                "turn": state["turn"],
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Keep conversation history manageable (last 50 messages)
            if len(self.conversation_history) > 50:
                self.conversation_history = self.conversation_history[-50:]
            
            # Route to all agents (narrator receives all messages, others skip their own)
            if self.config.parallel_execution:
                tasks = []
                for agent_id, agent in self.agents.items():
                    # Narrator agents receive ALL messages for observation
                    is_narrator = hasattr(agent, 'config') and hasattr(agent.config, 'role') and agent.config.role.value == 'narrator'
                    if is_narrator or agent_id != message.sender_id:
                        tasks.append(agent.process_message(message))
                
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
            else:
                for agent_id, agent in self.agents.items():
                    # Narrator agents receive ALL messages for observation
                    is_narrator = hasattr(agent, 'config') and hasattr(agent.config, 'role') and agent.config.role.value == 'narrator'
                    if is_narrator or agent_id != message.sender_id:
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
                
                # Log scoring event
                self.arena_logger.log_scoring(
                    message.sender_id, 
                    metrics.weighted_score, 
                    state["scores"][message.sender_id]
                )
        
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
                
                # Log elimination
                self.arena_logger.log_elimination(agent_id, "Low performance")
        
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
            logger.info(f"Game over: Only {len(state['active_agents'])} agents remaining")
        elif state["turn"] >= self.config.max_turns:
            state["game_over"] = True
            logger.info(f"Game over: Max turns ({self.config.max_turns}) reached")
        
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
        
        # Log game completion
        self.arena_logger.log_game_end(
            winner=state["metadata"].get("winner"),
            final_scores=state["scores"],
            total_turns=state["turn"]
        )
        
        return state
    
    async def _narrator_final_node(self, state: GameState) -> GameState:
        """Generate final narrator summary."""
        logger.info("Generating final narrator summary")
        
        # Find the narrator agent
        narrator_agent = None
        for agent in self.agents.values():
            if hasattr(agent, 'config') and hasattr(agent.config, 'role') and agent.config.role.value == 'narrator':
                narrator_agent = agent
                break
        
        if not narrator_agent:
            logger.warning("No narrator agent found for final summary")
            return state
        
        try:
            # Prepare game state context for final summary
            game_state_context = {
                "winner": state["metadata"].get("winner"),
                "scores": state["scores"],
                "turn": state["turn"],
                "seed_question": getattr(self, 'seed_question', ''),
                "game_id": state["game_id"],
                "active_agents": state["active_agents"],
                "eliminated_agents": state["eliminated_agents"],
                "phase": state["phase"]
            }
            
            # Generate final summary using narrator's generate_final_summary method if available
            final_summary_message = None
            if hasattr(narrator_agent, 'generate_final_summary'):
                final_summary_message = await narrator_agent.generate_final_summary(game_state_context)
            else:
                logger.warning("Narrator agent does not have generate_final_summary method")
            
            if final_summary_message:
                # Log the final summary with special formatting
                agent_name = final_summary_message.sender_name
                content = final_summary_message.content
                
                # Display with special formatting for final summary
                # print("\n" + "📚 " + "="*80)
                # print("FINAL NARRATOR SUMMARY")
                # print("="*80)
                # print(f"[94m{agent_name}[0m: {content}")
                # print("="*80 + "\n")
                
                # Store the final summary in metadata
                state["metadata"]["narrator_final_summary"] = content
                
        except Exception as e:
            logger.error(f"Error generating final narrator summary: {e}")
        
        return state
    
    async def _judge_final_node(self, state: GameState) -> GameState:
        """Generate final judge verdict."""
        logger.info("Generating final judge verdict")
        
        # Find the judge agent
        judge_agent = None
        for agent in self.agents.values():
            if hasattr(agent, 'config') and hasattr(agent.config, 'role') and agent.config.role.value == 'judge':
                judge_agent = agent
                break
        
        if not judge_agent:
            logger.warning("No judge agent found for final verdict")
            return state
        
        try:
            # Prepare game state context for final verdict
            game_state_context = {
                "winner": state["metadata"].get("winner"),
                "scores": state["scores"],
                "turn": state["turn"],
                "seed_question": getattr(self, 'seed_question', ''),
                "game_id": state["game_id"],
                "active_agents": state["active_agents"],
                "eliminated_agents": state["eliminated_agents"],
                "phase": state["phase"]
            }
            
            # Generate final verdict using judge's generate_final_verdict method if available
            final_verdict_message = None
            if hasattr(judge_agent, 'generate_final_verdict'):
                final_verdict_message = await judge_agent.generate_final_verdict(game_state_context, self.conversation_history)
            else:
                logger.warning("Judge agent does not have generate_final_verdict method")
            
            if final_verdict_message:
                # Log the final verdict with special formatting
                agent_name = final_verdict_message.sender_name
                content = final_verdict_message.content
                
                # Display with special formatting for final verdict
                # print("\n" + "⚖️ " + "="*80)
                # print("FINAL JUDGE VERDICT")
                # print("="*80)
                # print(f"[93m{agent_name}[0m: {content}")
                # print("="*80 + "\n")
                
                # Store the final verdict in metadata
                state["metadata"]["judge_final_verdict"] = content
                
        except Exception as e:
            logger.error(f"Error generating final judge verdict: {e}")
        
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
    
    
    def _is_special_agent(self, agent_id: str) -> bool:
        """Check if an agent is special (narrator/judge) that shouldn't participate in normal gameplay."""
        agent = self.agents.get(agent_id)
        if not agent or not hasattr(agent, 'config') or not hasattr(agent.config, 'role'):
            return False
        
        special_roles = {'narrator', 'judge'}
        return agent.config.role.value in special_roles
    
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
        logger.info(f"🎮 Starting game orchestration for {self.config.game_id}")
        logger.info(f"👥 Agents: {list(self.agents.keys())}")
        logger.info(f"🔄 Max turns: {self.config.max_turns}")
        logger.info(f"🔁 Recursion limit: {self.config.recursion_limit}")
        
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
            # Run the graph with proper configuration to handle recursion
            config = {
                "configurable": {"thread_id": self.config.game_id},
                "recursion_limit": self.config.recursion_limit,  # Use configurable recursion limit
            }
            logger.info(f"🚀 Starting LangGraph execution with {len(self.agents)} agents")
            logger.info(f"⚙️ Using recursion limit: {self.config.recursion_limit}")
            
            # Try the real LangGraph execution now that we fixed the termination logic
            logger.info("Attempting real LangGraph execution...")
            
            try:
                # Use astream for better monitoring
                final_state = None
                event_count = 0
                
                async for event in self.app.astream(initial_state, config):
                    event_count += 1
                    logger.info(f"Event {event_count}: {event}")
                    
                    # Get the latest state
                    if event:
                        final_state = list(event.values())[0] if isinstance(event, dict) else event
                    
                    # Safety check to prevent infinite loops
                    # Calculate dynamic limit based on max_turns (roughly 8 events per turn)
                    max_events = max(500, self.config.max_turns * 10)
                    if event_count > max_events:
                        logger.warning(f"Too many events ({event_count} > {max_events}), terminating for safety")
                        break
                        
                    # Break when game reaches final judge verdict (the last node before END)
                    if event and 'judge_final' in event:
                        logger.info("Game ended successfully")
                        break
                
                if final_state:
                    logger.info("✅ REAL LangGraph execution successful!")
                    return {
                        "game_id": self.config.game_id,
                        "winner": final_state.get("metadata", {}).get("winner"),
                        "final_scores": final_state.get("scores", {}),
                        "total_turns": final_state.get("turn", 0),
                        "eliminated": final_state.get("eliminated_agents", []),
                        "execution_history": [{"note": "Real LangGraph execution successful"}],
                        "real_mode": True,
                        "event_count": event_count
                    }
                else:
                    raise Exception("No final state received from LangGraph")
                    
            except Exception as e:
                logger.error(f"LangGraph execution failed: {e}")
                logger.info("Falling back to direct node execution...")
                
                # Fallback: Run nodes directly
                state = await self._start_node(initial_state.copy())
                state = await self._setup_node(state)
                
                # Run turns until game ends
                while not state.get("game_over", False) and state.get("turn", 0) < 10:
                    state = await self._turn_start_node(state)
                    state = await self._agent_select_node(state)
                    state = await self._agent_action_node(state)
                    state = await self._message_process_node(state)
                    state = await self._scoring_node(state)
                    state = await self._elimination_check_node(state)
                    
                    if state.get("elimination_pending", False):
                        state = await self._elimination_node(state)
                    
                    state = await self._phase_check_node(state)
                    
                    if state.get("game_over", False):
                        state = await self._game_end_node(state)
                        break
                    
                    state = await self._turn_end_node(state)
                
                logger.info("Fallback execution completed")
                
                return {
                    "game_id": self.config.game_id,
                    "winner": state.get("metadata", {}).get("winner"),
                    "final_scores": state.get("scores", {}),
                    "total_turns": state.get("turn", 0),
                    "eliminated": state.get("eliminated_agents", []),
                    "execution_history": [{"note": "Fallback direct node execution"}],
                    "fallback_mode": True
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