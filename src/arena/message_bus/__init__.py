"""Message Bus Module

Implements Kafka-based message exchange system for Arena.

Features:
- Full observability: All agents can see all messages
- Turn-restricted writing: Only selected agent can publish
- Message persistence: Complete history maintained
- Partitioning by game_id for scalability

Key Components:
- KafkaProducer: Send messages with retry logic
- KafkaConsumer: Consume messages with offset management
- MessageRouter: Route messages based on content and type
- EventDispatcher: Central event processing pipeline
- EventHandlers: Type-specific message processors
- Topics: Topic configuration and management
- Serialization: Message serialization utilities
"""

from .kafka_producer import ArenaKafkaProducer
from .kafka_consumer import ArenaKafkaConsumer, ConsumeMode
from .message_router import MessageRouter, Route, RouteType, Subscription
from .event_dispatcher import EventDispatcher, EventContext, PrioritizedEvent
from .event_handlers import (
    EventHandler,
    ContributionHandler,
    AccusationHandler,
    EliminationHandler,
    ScoringHandler,
    TurnHandler,
    SystemHandler,
    AsyncEventHandler,
    get_handler,
    get_handlers_for_message
)
from .topics import (
    ArenaTopics,
    TopicConfig,
    TopicCategory,
    get_contribution_topic,
    get_turn_topic,
    get_scoring_topic,
    get_accusation_topic,
    get_system_topic
)
from .serialization import (
    MessageSerializer,
    ModelSerializer,
    SerializationFormat,
    ArenaJSONEncoder,
    safe_deserialize
)

__all__ = [
    # Producer/Consumer
    "ArenaKafkaProducer",
    "ArenaKafkaConsumer",
    "ConsumeMode",
    
    # Routing
    "MessageRouter",
    "Route",
    "RouteType",
    "Subscription",
    
    # Event Processing
    "EventDispatcher",
    "EventContext",
    "PrioritizedEvent",
    
    # Handlers
    "EventHandler",
    "ContributionHandler",
    "AccusationHandler",
    "EliminationHandler",
    "ScoringHandler",
    "TurnHandler",
    "SystemHandler",
    "AsyncEventHandler",
    "get_handler",
    "get_handlers_for_message",
    
    # Topics
    "ArenaTopics",
    "TopicConfig",
    "TopicCategory",
    "get_contribution_topic",
    "get_turn_topic",
    "get_scoring_topic",
    "get_accusation_topic",
    "get_system_topic",
    
    # Serialization
    "MessageSerializer",
    "ModelSerializer",
    "SerializationFormat",
    "ArenaJSONEncoder",
    "safe_deserialize",
]