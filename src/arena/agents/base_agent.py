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

# LLM imports
try:
    from ..llm import ArenaLLMClient
    from ..config import arena_config
except ImportError:
    ArenaLLMClient = None
    arena_config = None


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
        
        # Initialize real LLM client
        self.llm_client = None
        self._initialize_llm_client()
    
    def _initialize_llm_client(self) -> None:
        """Initialize the LLM client based on configuration."""
        if not ArenaLLMClient:
            logger.warning(f"LLM client not available for {self.agent_id}, will use fallback responses")
            return
        
        try:
            # Get model and temperature from configuration
            model = self.llm_config.get("model", "llama3.3:70b")
            temperature = self.llm_config.get("temperature", 0.7)
            
            # Initialize simple Ollama client
            self.llm_client = ArenaLLMClient(model=model, temperature=temperature)
            logger.info(f"Initialized Ollama LLM client for {self.agent_id} with model {model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM client for {self.agent_id}: {e}")
            logger.warning("Will use fallback responses")
    
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
        logger.info(f"LLM call for {self.agent_id}: {len(prompt)} chars")
        
        # Use real LLM client if available
        if self.llm_client:
            try:
                # Extract context for character-aware generation
                context = {
                    "agent_id": self.agent_id,
                    "agent_name": self.agent_name,
                    "role": self.role.value,
                    "system_prompt": system_prompt
                }
                
                # Check if streaming is enabled
                if hasattr(self.llm_client, 'streaming_enabled') and self.llm_client.streaming_enabled:
                    # Use streaming response
                    response_stream = self.llm_client.generate_character_response_streaming(
                        character_id=self.agent_id,
                        character_name=self.agent_name,
                        prompt=prompt,
                        context=context
                    )
                    
                    # Get arena logger for streaming output
                    from ..config.logging_config import get_arena_logger
                    arena_logger = get_arena_logger()
                    
                    if arena_logger:
                        # Stream the response through arena logger
                        response = arena_logger.log_agent_response_streaming(
                            self.agent_name, 
                            response_stream, 
                            self.agent_id
                        )
                    else:
                        # Fallback: collect all tokens
                        response = ""
                        for token in response_stream:
                            response += token
                else:
                    # Use regular non-streaming response
                    response = await self.llm_client.generate_character_response(
                        character_id=self.agent_id,
                        character_name=self.agent_name,
                        prompt=prompt,
                        context=context
                    )
                
                # Update token tracking (approximate)
                self.prompt_tokens_used += len(prompt) // 4
                self.completion_tokens_used += len(response) // 4
                self.total_tokens_used = self.prompt_tokens_used + self.completion_tokens_used
                
                logger.info(f"Real LLM response for {self.agent_id}: {len(response)} chars")
                return response
                
            except Exception as e:
                logger.error(f"Real LLM call failed for {self.agent_id}: {e}")
                logger.warning("Falling back to character response generation")
        
        # Fallback to character-appropriate responses
        logger.info(f"Using fallback response generation for {self.agent_id}")
        
        # Simulate token usage for fallback
        self.prompt_tokens_used += len(prompt) // 4
        self.completion_tokens_used += 100
        self.total_tokens_used = self.prompt_tokens_used + self.completion_tokens_used
        
        # Generate character-appropriate response based on prompt
        response = await self._generate_character_response(prompt, system_prompt)
        return response
    
    async def _generate_character_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate character-appropriate response based on prompt context."""
        import random
        
        # Extract key context from prompt to determine response type
        prompt_lower = prompt.lower()
        
        # Different response types based on prompt content
        if "contribution" in prompt_lower or "discuss" in prompt_lower:
            return self._generate_contribution_response()
        elif "accusation" in prompt_lower or "accuse" in prompt_lower:
            return self._generate_accusation_response()
        elif "defense" in prompt_lower or "defend" in prompt_lower:
            return self._generate_defense_response()
        elif "summary" in prompt_lower or "summarize" in prompt_lower:
            return self._generate_summary_response()
        else:
            return self._generate_generic_response()
    
    def _generate_contribution_response(self) -> str:
        """Generate a contribution to the discussion."""
        import random
        
        # Character-specific contributions
        if "ada_lovelace" in self.agent_id.lower():
            ada_contributions = [
                "I believe we need to approach this systematically, analyzing each variable methodically.",
                "There's a logical pattern here that suggests a more algorithmic solution.",
                "My calculations indicate we should consider the mathematical relationships involved.",
                "From an analytical perspective, I see several optimization opportunities.",
                "The data suggests a different approach might yield better results.",
                "Let me propose a more structured framework for approaching this problem."
            ]
            return random.choice(ada_contributions)
        
        elif "captain_cosmos" in self.agent_id.lower():
            cosmos_contributions = [
                "Fellow explorers, I sense there are cosmic forces at play here we haven't considered.",
                "In my travels across the universe, I've encountered similar challenges - perhaps we need a broader perspective.",
                "The stellar patterns suggest we should think beyond conventional boundaries.",
                "My cosmic intuition tells me there's a more adventurous path forward.",
                "Like navigating through asteroid fields, sometimes the boldest route is the safest.",
                "We need to harness the energy of this moment and propel ourselves to new heights!"
            ]
            return random.choice(cosmos_contributions)
        
        else:
            # Generic contributions for other characters
            base_contributions = [
                "I think we need to consider the broader implications of this approach.",
                "There's an interesting angle we haven't explored yet - what if we tried a different strategy?",
                "I've been analyzing the situation, and I believe there's a more efficient solution.",
                "Let me propose an alternative that might address everyone's concerns.",
                "Building on what others have said, I'd like to add another perspective.",
                "I have some reservations about the current direction we're taking.",
                "There's a critical factor we need to discuss before moving forward."
            ]
            return random.choice(base_contributions)
    
    def _generate_accusation_response(self) -> str:
        """Generate an accusation."""
        import random
        
        accusations = [
            "I have concerns about someone's behavior in recent discussions.",
            "There seems to be inconsistency in how certain arguments are being presented.",
            "I question whether everyone is being entirely forthcoming about their motivations.",
            "Someone here isn't being completely honest with the group."
        ]
        
        return random.choice(accusations)
    
    def _generate_defense_response(self) -> str:
        """Generate a defense against accusations."""
        import random
        
        defenses = [
            "I stand by my contributions and believe they've been constructive throughout.",
            "These accusations seem unfounded - my track record speaks for itself.",
            "I've been transparent about my reasoning from the beginning.",
            "I think there's been a misunderstanding about my intentions."
        ]
        
        return random.choice(defenses)
    
    def _generate_summary_response(self) -> str:
        """Generate a summary of recent events."""
        import random
        
        summaries = [
            "The discussion has covered several key points, with varying perspectives on the best approach.",
            "We've seen some interesting developments in the conversation, including new strategic considerations.",
            "Recent exchanges have highlighted different viewpoints on how to proceed.",
            "The group has explored multiple angles, though consensus hasn't emerged yet."
        ]
        
        return random.choice(summaries)
    
    def _generate_generic_response(self) -> str:
        """Generate a generic response when type is unclear."""
        import random
        
        generic_responses = [
            f"As {self.agent_name}, I believe we should approach this thoughtfully.",
            f"From my perspective, there are several factors to consider here.",
            f"I'd like to share my thoughts on how we might move forward.",
            f"This is an interesting challenge that deserves careful consideration."
        ]
        
        return random.choice(generic_responses)
    
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