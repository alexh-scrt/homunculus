"""
Arena Orchestration Module

This module implements the LangGraph-based orchestration system
for managing Arena games with sophisticated state management,
checkpointing, and recovery.

Components:
- Game orchestrator with LangGraph
- State management and checkpointing
- Turn-based flow control
- Phase transitions
- Agent coordination
- Message routing
- Parallel execution

Author: Homunculus Team
"""

from .game_orchestrator import (
    GameOrchestrator,
    GameNode,
    GameEdge,
    OrchestratorConfig
)

from .game_state import (
    GameStateManager,
    GamePhase,
    TurnState,
    CheckpointManager,
    StateSnapshot
)

from .turn_manager import (
    TurnManager,
    TurnFlow,
    TurnContext,
    TurnResult
)

from .phase_controller import (
    PhaseController,
    PhaseTransition,
    TransitionCondition,
    PhaseMetrics
)

from .agent_coordinator import (
    AgentCoordinator,
    AgentScheduler,
    ExecutionPlan,
    CoordinationStrategy
)

# Message router and recovery manager to be implemented
# from .message_router import (
#     GraphMessageRouter,
#     RoutingRule,
#     MessageQueue,
#     DeliveryStatus
# )

# from .recovery_manager import (
#     RecoveryManager,
#     RecoveryStrategy,
#     FailureMode,
#     RecoveryPoint
# )

__all__ = [
    # Orchestrator
    "GameOrchestrator",
    "GameNode",
    "GameEdge",
    "OrchestratorConfig",
    
    # State Management
    "GameStateManager",
    "GamePhase",
    "TurnState",
    "CheckpointManager",
    "StateSnapshot",
    
    # Turn Management
    "TurnManager",
    "TurnFlow",
    "TurnContext",
    "TurnResult",
    
    # Phase Control
    "PhaseController",
    "PhaseTransition",
    "TransitionCondition",
    "PhaseMetrics",
    
    # Agent Coordination
    "AgentCoordinator",
    "AgentScheduler",
    "ExecutionPlan",
    "CoordinationStrategy",
    
    # Message Routing (to be implemented)
    # "GraphMessageRouter",
    # "RoutingRule",
    # "MessageQueue",
    # "DeliveryStatus",
    
    # Recovery (to be implemented)
    # "RecoveryManager",
    # "RecoveryStrategy",
    # "FailureMode",
    # "RecoveryPoint"
]