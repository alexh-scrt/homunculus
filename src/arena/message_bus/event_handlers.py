"""
Event Handlers for Arena Message Bus

This module defines handlers for different types of events in the Arena system.
Each handler processes specific message types and triggers appropriate actions.

Features:
- Handler interface definition
- Type-specific handlers
- Handler chaining
- Error handling and recovery
- Async and sync handler support

Author: Homunculus Team
"""

import logging
from typing import Optional, Any, Dict, List, Callable
from abc import ABC, abstractmethod
from datetime import datetime
import asyncio

from ..models import (
    Message, MessageType, AgentState, Accusation, 
    ArenaState, ScoringMetrics
)


logger = logging.getLogger(__name__)


class EventHandler(ABC):
    """
    Abstract base class for event handlers.
    
    All event handlers must implement the handle method and
    optionally implement pre/post processing hooks.
    """
    
    def __init__(self, handler_id: str):
        """
        Initialize the handler.
        
        Args:
            handler_id: Unique identifier for this handler
        """
        self.handler_id = handler_id
        self.processed_count = 0
        self.error_count = 0
        self.last_processed: Optional[datetime] = None
    
    @abstractmethod
    def can_handle(self, message: Message) -> bool:
        """
        Check if this handler can process a message.
        
        Args:
            message: Message to check
            
        Returns:
            True if handler can process the message
        """
        pass
    
    @abstractmethod
    def handle(self, message: Message, context: Dict[str, Any]) -> Any:
        """
        Process a message.
        
        Args:
            message: Message to process
            context: Processing context
            
        Returns:
            Processing result
        """
        pass
    
    def pre_process(self, message: Message, context: Dict[str, Any]) -> bool:
        """
        Pre-processing hook.
        
        Args:
            message: Message to process
            context: Processing context
            
        Returns:
            True to continue processing, False to skip
        """
        return True
    
    def post_process(
        self,
        message: Message,
        result: Any,
        context: Dict[str, Any]
    ) -> None:
        """
        Post-processing hook.
        
        Args:
            message: Processed message
            result: Processing result
            context: Processing context
        """
        self.processed_count += 1
        self.last_processed = datetime.utcnow()
    
    def handle_error(
        self,
        message: Message,
        error: Exception,
        context: Dict[str, Any]
    ) -> None:
        """
        Error handling hook.
        
        Args:
            message: Message that caused error
            error: Exception that occurred
            context: Processing context
        """
        self.error_count += 1
        logger.error(f"Handler {self.handler_id} error: {error}")


class ContributionHandler(EventHandler):
    """
    Handles agent contributions to problem-solving.
    
    This handler processes contribution messages, validates them,
    and triggers scoring evaluation.
    """
    
    def __init__(self):
        super().__init__("contribution_handler")
    
    def can_handle(self, message: Message) -> bool:
        """Check if this is a contribution message."""
        return message.message_type == "contribution"
    
    def handle(self, message: Message, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a contribution message.
        
        Args:
            message: Contribution message
            context: Must contain 'arena_state'
            
        Returns:
            Processing result with scoring request
        """
        arena_state: ArenaState = context.get("arena_state")
        if not arena_state:
            raise ValueError("arena_state required in context")
        
        # Validate agent is active
        agent = arena_state.get_agent(message.sender_id)
        if not agent or not agent.is_active:
            logger.warning(f"Contribution from inactive agent: {message.sender_id}")
            return {"status": "rejected", "reason": "agent_not_active"}
        
        # Update agent's contribution tracking
        agent.add_contribution(message.message_id)
        
        # Add to game history
        arena_state.add_message(message)
        
        # Request scoring
        return {
            "status": "accepted",
            "message_id": message.message_id,
            "agent_id": message.sender_id,
            "request_scoring": True,
            "turn_number": arena_state.current_turn
        }


class AccusationHandler(EventHandler):
    """
    Handles cheating accusations between agents.
    
    This handler processes accusation messages, validates evidence,
    and triggers judge evaluation.
    """
    
    def __init__(self):
        super().__init__("accusation_handler")
    
    def can_handle(self, message: Message) -> bool:
        """Check if this is an accusation message."""
        return message.message_type == "accusation"
    
    def handle(self, message: Message, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an accusation message.
        
        Args:
            message: Accusation message
            context: Must contain 'arena_state'
            
        Returns:
            Processing result with judge request
        """
        arena_state: ArenaState = context.get("arena_state")
        if not arena_state:
            raise ValueError("arena_state required in context")
        
        # Extract accusation from message metadata
        accusation_data = message.metadata.get("accusation")
        if not accusation_data:
            return {"status": "rejected", "reason": "no_accusation_data"}
        
        # Create accusation object
        accusation = Accusation(
            accuser_id=message.sender_id,
            accuser_name=message.sender_name,
            accused_id=message.target_agent_id or "",
            accused_name=message.metadata.get("accused_name", ""),
            accusation_type=message.metadata.get("accusation_type", "other"),
            claim=message.content,
            game_id=arena_state.game_id,
            turn_number=arena_state.current_turn
        )
        
        # Validate both agents exist and are active
        accuser = arena_state.get_agent(accusation.accuser_id)
        accused = arena_state.get_agent(accusation.accused_id)
        
        if not accuser or not accuser.is_active:
            return {"status": "rejected", "reason": "accuser_not_active"}
        
        if not accused:
            return {"status": "rejected", "reason": "accused_not_found"}
        
        # Add to game state
        arena_state.add_accusation(accusation)
        
        # Request judge evaluation
        return {
            "status": "accepted",
            "accusation_id": accusation.accusation_id,
            "request_judge": True,
            "priority": "high" if accused.is_active else "low"
        }


class EliminationHandler(EventHandler):
    """
    Handles agent elimination events.
    
    This handler processes elimination messages and updates
    game state accordingly.
    """
    
    def __init__(self):
        super().__init__("elimination_handler")
    
    def can_handle(self, message: Message) -> bool:
        """Check if this is an elimination message."""
        return message.message_type == "elimination"
    
    def handle(self, message: Message, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an elimination message.
        
        Args:
            message: Elimination message
            context: Must contain 'arena_state'
            
        Returns:
            Processing result
        """
        arena_state: ArenaState = context.get("arena_state")
        if not arena_state:
            raise ValueError("arena_state required in context")
        
        # Extract elimination details
        agent_id = message.target_agent_id or message.metadata.get("eliminated_agent_id")
        reason = message.content or "Unspecified reason"
        
        if not agent_id:
            return {"status": "rejected", "reason": "no_agent_id"}
        
        # Perform elimination
        success = arena_state.eliminate_agent(agent_id, reason)
        
        if success:
            # Check for game termination
            termination_reason = arena_state.check_termination()
            
            return {
                "status": "success",
                "eliminated_agent": agent_id,
                "remaining_agents": arena_state.agent_count,
                "check_termination": termination_reason is not None,
                "termination_reason": termination_reason
            }
        else:
            return {"status": "failed", "reason": "agent_not_found"}


class ScoringHandler(EventHandler):
    """
    Handles scoring events for agent contributions.
    
    This handler processes scoring results and updates agent scores.
    """
    
    def __init__(self):
        super().__init__("scoring_handler")
    
    def can_handle(self, message: Message) -> bool:
        """Check if this is a scoring message."""
        return message.message_type == "scoring" or message.message_type == "score_update"
    
    def handle(self, message: Message, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a scoring message.
        
        Args:
            message: Scoring message
            context: Must contain 'arena_state'
            
        Returns:
            Processing result
        """
        arena_state: ArenaState = context.get("arena_state")
        if not arena_state:
            raise ValueError("arena_state required in context")
        
        # Extract scoring metrics
        metrics_data = message.metadata.get("metrics")
        if not metrics_data:
            return {"status": "rejected", "reason": "no_metrics_data"}
        
        # Create ScoringMetrics object
        metrics = ScoringMetrics.from_dict(metrics_data)
        
        # Get agent ID
        agent_id = message.metadata.get("agent_id") or message.target_agent_id
        if not agent_id:
            return {"status": "rejected", "reason": "no_agent_id"}
        
        # Update score
        arena_state.update_score(agent_id, metrics)
        
        # Check for elimination threshold
        agent = arena_state.get_agent(agent_id)
        if agent and agent.score < -100:  # Elimination threshold
            return {
                "status": "success",
                "updated_score": agent.score,
                "request_elimination": True,
                "elimination_reason": "Score below threshold"
            }
        
        return {
            "status": "success",
            "updated_score": agent.score if agent else None,
            "leaderboard_update": True
        }


class TurnHandler(EventHandler):
    """
    Handles turn management events.
    
    This handler processes turn transitions and agent selections.
    """
    
    def __init__(self):
        super().__init__("turn_handler")
    
    def can_handle(self, message: Message) -> bool:
        """Check if this is a turn management message."""
        return message.message_type in ["turn_start", "turn_end", "turn_selection"]
    
    def handle(self, message: Message, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a turn management message.
        
        Args:
            message: Turn management message
            context: Must contain 'arena_state'
            
        Returns:
            Processing result
        """
        arena_state: ArenaState = context.get("arena_state")
        if not arena_state:
            raise ValueError("arena_state required in context")
        
        if message.message_type == "turn_start":
            arena_state.advance_turn()
            return {
                "status": "success",
                "current_turn": arena_state.current_turn,
                "max_turns": arena_state.max_turns
            }
        
        elif message.message_type == "turn_end":
            # Check for termination conditions
            termination_reason = arena_state.check_termination()
            
            return {
                "status": "success",
                "turn_completed": arena_state.current_turn,
                "check_termination": termination_reason is not None,
                "termination_reason": termination_reason
            }
        
        elif message.message_type == "turn_selection":
            # Handle agent selection for turn
            selected_agent = message.metadata.get("selected_agent")
            
            return {
                "status": "success",
                "selected_agent": selected_agent,
                "turn_number": arena_state.current_turn
            }
        
        return {"status": "unknown_turn_type"}


class SystemHandler(EventHandler):
    """
    Handles system-level events.
    
    This handler processes system messages like errors, status updates,
    and orchestration commands.
    """
    
    def __init__(self):
        super().__init__("system_handler")
    
    def can_handle(self, message: Message) -> bool:
        """Check if this is a system message."""
        return message.sender_type == "system" or message.message_type in [
            "error", "status", "command", "health_check"
        ]
    
    def handle(self, message: Message, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a system message.
        
        Args:
            message: System message
            context: Processing context
            
        Returns:
            Processing result
        """
        if message.message_type == "error":
            # Log error and potentially trigger recovery
            logger.error(f"System error: {message.content}")
            return {
                "status": "error_logged",
                "error_id": message.message_id,
                "severity": message.metadata.get("severity", "medium")
            }
        
        elif message.message_type == "health_check":
            # Respond to health check
            return {
                "status": "healthy",
                "handler_id": self.handler_id,
                "processed_count": self.processed_count,
                "error_count": self.error_count
            }
        
        elif message.message_type == "command":
            # Process system command
            command = message.metadata.get("command")
            
            if command == "reset_stats":
                self.processed_count = 0
                self.error_count = 0
                return {"status": "stats_reset"}
            
            return {"status": "command_received", "command": command}
        
        return {"status": "processed"}


class AsyncEventHandler(EventHandler):
    """
    Base class for asynchronous event handlers.
    
    Extends EventHandler with async/await support for handlers
    that need to perform async operations.
    """
    
    @abstractmethod
    async def handle_async(
        self,
        message: Message,
        context: Dict[str, Any]
    ) -> Any:
        """
        Asynchronously process a message.
        
        Args:
            message: Message to process
            context: Processing context
            
        Returns:
            Processing result
        """
        pass
    
    def handle(self, message: Message, context: Dict[str, Any]) -> Any:
        """
        Synchronous wrapper for async handler.
        
        Args:
            message: Message to process
            context: Processing context
            
        Returns:
            Processing result
        """
        # Run async handler in event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, schedule as task
            task = asyncio.create_task(self.handle_async(message, context))
            return {"status": "async_processing", "task": task}
        else:
            # Run synchronously
            return loop.run_until_complete(self.handle_async(message, context))


# Handler registry
HANDLER_REGISTRY = {
    "contribution": ContributionHandler,
    "accusation": AccusationHandler,
    "elimination": EliminationHandler,
    "scoring": ScoringHandler,
    "turn": TurnHandler,
    "system": SystemHandler
}


def get_handler(handler_type: str) -> Optional[EventHandler]:
    """
    Get a handler instance by type.
    
    Args:
        handler_type: Type of handler to get
        
    Returns:
        Handler instance or None if not found
    """
    handler_class = HANDLER_REGISTRY.get(handler_type)
    if handler_class:
        return handler_class()
    return None


def get_handlers_for_message(message: Message) -> List[EventHandler]:
    """
    Get all handlers that can process a message.
    
    Args:
        message: Message to check
        
    Returns:
        List of applicable handlers
    """
    handlers = []
    
    for handler_class in HANDLER_REGISTRY.values():
        handler = handler_class()
        if handler.can_handle(message):
            handlers.append(handler)
    
    return handlers