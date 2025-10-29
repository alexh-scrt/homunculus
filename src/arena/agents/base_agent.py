"""
Base Agent Interface for Arena

This module defines the base agent interface and abstract classes
that all Arena agents must implement.

Features:
- Abstract base agent with lifecycle methods
- Message handling interface
- State management
- Error handling and recovery
- Async operation support

Author: Homunculus Team
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
from enum import Enum

from ..models import Message, AgentState, ArenaState
# Kafka imports are optional - only import if needed
try:
    from ..message_bus import ArenaKafkaProducer, ArenaKafkaConsumer
except ImportError:
    ArenaKafkaProducer = None
    ArenaKafkaConsumer = None


logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Roles that agents can play in Arena."""
    CHARACTER = "character"      # Homunculus character agent
    NARRATOR = "narrator"        # Summarizes and contextualizes
    JUDGE = "judge"             # Evaluates and scores
    TURN_SELECTOR = "turn_selector"  # Decides who speaks next
    ORCHESTRATOR = "orchestrator"    # Manages game flow
    OBSERVER = "observer"           # Watches without participating


@dataclass
class AgentConfig:
    """
    Configuration for an Arena agent.
    
    Attributes:
        agent_id: Unique identifier for the agent
        agent_name: Display name for the agent
        role: Role the agent plays
        llm_config: LLM configuration if applicable
        kafka_topics: Topics to subscribe to
        max_retries: Maximum retry attempts for operations
        timeout_seconds: Timeout for operations
        metadata: Additional configuration data
    """
    
    agent_id: str
    agent_name: str
    role: AgentRole
    llm_config: Optional[Dict[str, Any]] = None
    kafka_topics: List[str] = field(default_factory=list)
    max_retries: int = 3
    timeout_seconds: float = 30.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Abstract base class for all Arena agents.
    
    This class defines the interface that all agents must implement
    and provides common functionality for state management, message
    handling, and lifecycle operations.
    """
    
    def __init__(self, config: AgentConfig):
        """
        Initialize the base agent.
        
        Args:
            config: Agent configuration
        """
        self.config = config
        self.agent_id = config.agent_id
        self.agent_name = config.agent_name
        self.role = config.role
        
        # State tracking
        self.is_active = False
        self.last_action_time: Optional[datetime] = None
        self.messages_processed = 0
        self.errors_encountered = 0
        
        # Message bus components
        self.producer: Optional[ArenaKafkaProducer] = None
        self.consumer: Optional[ArenaKafkaConsumer] = None
        
        # Callbacks
        self.message_callbacks: Dict[str, List[Callable]] = {}
        self.error_callbacks: List[Callable] = []
        
        logger.info(f"Initialized {self.role.value} agent: {self.agent_name} ({self.agent_id})")
    
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the agent's resources and connections.
        
        This method should set up any necessary connections,
        load models, and prepare the agent for operation.
        """
        pass
    
    @abstractmethod
    async def process_message(self, message: Message) -> Optional[Message]:
        """
        Process an incoming message.
        
        Args:
            message: Message to process
            
        Returns:
            Response message if applicable, None otherwise
        """
        pass
    
    @abstractmethod
    async def generate_action(self, context: Dict[str, Any]) -> Optional[Message]:
        """
        Generate an action based on current context.
        
        Args:
            context: Current game/conversation context
            
        Returns:
            Generated message if applicable, None otherwise
        """
        pass
    
    @abstractmethod
    async def update_state(self, state: Dict[str, Any]) -> None:
        """
        Update the agent's internal state.
        
        Args:
            state: New state information
        """
        pass
    
    async def start(self) -> None:
        """Start the agent and begin processing."""
        if self.is_active:
            logger.warning(f"Agent {self.agent_id} is already active")
            return
        
        logger.info(f"Starting agent {self.agent_id}")
        
        # Initialize resources
        await self.initialize()
        
        # Set up message bus if topics configured
        if self.config.kafka_topics:
            await self._setup_message_bus()
        
        self.is_active = True
        
        # Start message processing loop
        await self._message_loop()
    
    async def stop(self) -> None:
        """Stop the agent and clean up resources."""
        if not self.is_active:
            return
        
        logger.info(f"Stopping agent {self.agent_id}")
        
        self.is_active = False
        
        # Clean up message bus
        if self.producer:
            self.producer.close()
        if self.consumer:
            self.consumer.close()
        
        await self.cleanup()
    
    async def cleanup(self) -> None:
        """
        Clean up agent resources.
        
        Override this method to perform custom cleanup.
        """
        pass
    
    async def _setup_message_bus(self) -> None:
        """Set up Kafka producer and consumer."""
        # Create producer for sending messages
        self.producer = ArenaKafkaProducer()
        
        # Create consumer if topics configured
        if self.config.kafka_topics:
            self.consumer = ArenaKafkaConsumer(
                topics=self.config.kafka_topics,
                group_id=f"agent_{self.agent_id}",
                message_handler=self._handle_kafka_message
            )
    
    async def _message_loop(self) -> None:
        """Main message processing loop."""
        while self.is_active:
            try:
                if self.consumer:
                    # Process messages from Kafka
                    for message in self.consumer.consume_messages(
                        max_messages=10,
                        timeout_ms=1000
                    ):
                        if not self.is_active:
                            break
                        
                        await self._handle_message(message)
                else:
                    # No consumer, just sleep
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error in message loop for {self.agent_id}: {e}")
                self.errors_encountered += 1
                
                # Call error callbacks
                for callback in self.error_callbacks:
                    try:
                        callback(e)
                    except:
                        pass
                
                # Brief pause before retry
                await asyncio.sleep(1)
    
    def _handle_kafka_message(self, message: Message) -> None:
        """
        Handle message from Kafka (sync wrapper).
        
        Args:
            message: Message from Kafka
        """
        # Run async handler in event loop
        asyncio.create_task(self._handle_message(message))
    
    async def _handle_message(self, message: Message) -> None:
        """
        Handle an incoming message.
        
        Args:
            message: Message to handle
        """
        try:
            # Update statistics
            self.messages_processed += 1
            self.last_action_time = datetime.utcnow()
            
            # Call message callbacks
            for msg_type, callbacks in self.message_callbacks.items():
                if msg_type == "*" or message.message_type == msg_type:
                    for callback in callbacks:
                        try:
                            callback(message)
                        except:
                            pass
            
            # Process message
            response = await self.process_message(message)
            
            # Send response if generated
            if response and self.producer:
                self.producer.send_message(
                    topic=self._get_response_topic(message),
                    message=response
                )
                
        except Exception as e:
            logger.error(f"Error handling message in {self.agent_id}: {e}")
            self.errors_encountered += 1
    
    def _get_response_topic(self, original_message: Message) -> str:
        """
        Determine the appropriate response topic.
        
        Args:
            original_message: Original message being responded to
            
        Returns:
            Topic name for response
        """
        # Override in subclasses for custom routing
        return "arena.game.contributions"
    
    def register_message_callback(
        self,
        message_type: str,
        callback: Callable[[Message], None]
    ) -> None:
        """
        Register a callback for specific message types.
        
        Args:
            message_type: Type of message to listen for (* for all)
            callback: Function to call when message received
        """
        if message_type not in self.message_callbacks:
            self.message_callbacks[message_type] = []
        
        self.message_callbacks[message_type].append(callback)
    
    def register_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """
        Register a callback for errors.
        
        Args:
            callback: Function to call on error
        """
        self.error_callbacks.append(callback)
    
    async def send_message(
        self,
        content: str,
        message_type: str = "contribution",
        target_agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Send a message to the message bus.
        
        Args:
            content: Message content
            message_type: Type of message
            target_agent_id: Optional target agent
            metadata: Optional metadata
            
        Returns:
            True if message was sent successfully
        """
        if not self.producer:
            logger.warning(f"No producer configured for {self.agent_id}")
            return False
        
        message = Message(
            sender_id=self.agent_id,
            sender_name=self.agent_name,
            sender_type=self.role.value,
            message_type=message_type,
            content=content,
            target_agent_id=target_agent_id,
            metadata=metadata or {}
        )
        
        # Determine topic based on message type
        topic_map = {
            "contribution": "arena.game.contributions",
            "accusation": "arena.accusation.claims",
            "scoring": "arena.scoring.metrics",
            "turn_selection": "arena.game.turns",
            "system": "arena.system.orchestration"
        }
        
        topic = topic_map.get(message_type, "arena.game.contributions")
        
        return self.producer.send_message(topic, message)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get agent statistics.
        
        Returns:
            Dictionary with agent statistics
        """
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "role": self.role.value,
            "is_active": self.is_active,
            "messages_processed": self.messages_processed,
            "errors_encountered": self.errors_encountered,
            "last_action_time": self.last_action_time.isoformat() if self.last_action_time else None
        }


class LLMAgent(BaseAgent):
    """
    Base class for agents that use LLMs.
    
    Extends BaseAgent with LLM-specific functionality like
    prompt management, token tracking, and response generation.
    """
    
    def __init__(self, config: AgentConfig):
        """
        Initialize the LLM agent.
        
        Args:
            config: Agent configuration with LLM settings
        """
        super().__init__(config)
        
        # LLM configuration
        self.llm_config = config.llm_config or {}
        self.model_name = self.llm_config.get("model", "gpt-4")
        self.temperature = self.llm_config.get("temperature", 0.7)
        self.max_tokens = self.llm_config.get("max_tokens", 1000)
        
        # Token tracking
        self.total_tokens_used = 0
        self.prompt_tokens_used = 0
        self.completion_tokens_used = 0
        
        # Prompt templates
        self.system_prompt = ""
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history_length = 20
    
    @abstractmethod
    async def generate_prompt(self, context: Dict[str, Any]) -> str:
        """
        Generate a prompt for the LLM.
        
        Args:
            context: Current context
            
        Returns:
            Generated prompt
        """
        pass
    
    async def call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Call the LLM with a prompt.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt override
            
        Returns:
            LLM response
        """
        # This is a placeholder - actual implementation would call the LLM
        # In production, this would use langchain, openai, or anthropic SDK
        
        logger.info(f"LLM call for {self.agent_id}: {len(prompt)} chars")
        
        # Simulate token usage
        self.prompt_tokens_used += len(prompt) // 4
        self.completion_tokens_used += 100
        self.total_tokens_used = self.prompt_tokens_used + self.completion_tokens_used
        
        # Placeholder response
        return f"[LLM Response from {self.agent_name}]"
    
    def add_to_history(self, role: str, content: str) -> None:
        """
        Add an entry to conversation history.
        
        Args:
            role: Role (user/assistant/system)
            content: Message content
        """
        self.conversation_history.append({
            "role": role,
            "content": content
        })
        
        # Trim history if too long
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()
    
    def get_token_statistics(self) -> Dict[str, int]:
        """
        Get token usage statistics.
        
        Returns:
            Dictionary with token counts
        """
        return {
            "total_tokens": self.total_tokens_used,
            "prompt_tokens": self.prompt_tokens_used,
            "completion_tokens": self.completion_tokens_used,
            "estimated_cost": self._estimate_cost()
        }
    
    def _estimate_cost(self) -> float:
        """
        Estimate cost based on token usage.
        
        Returns:
            Estimated cost in dollars
        """
        # Rough estimates - adjust based on actual model pricing
        cost_per_1k_prompt = 0.03
        cost_per_1k_completion = 0.06
        
        prompt_cost = (self.prompt_tokens_used / 1000) * cost_per_1k_prompt
        completion_cost = (self.completion_tokens_used / 1000) * cost_per_1k_completion
        
        return prompt_cost + completion_cost