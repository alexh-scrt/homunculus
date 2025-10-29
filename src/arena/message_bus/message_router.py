"""
Message Router for Arena Message Bus

This module provides intelligent message routing based on message types,
game states, and agent targeting. It manages subscriptions and filters
to ensure messages reach the correct consumers.

Features:
- Content-based routing
- Topic-based routing
- Agent-specific routing
- Subscription management
- Message filtering
- Dead letter queue handling

Author: Homunculus Team
"""

import logging
from typing import Dict, List, Optional, Callable, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import re

from ..models import Message, MessageType
from .kafka_producer import ArenaKafkaProducer
from .kafka_consumer import ArenaKafkaConsumer
from .topics import ArenaTopics, get_contribution_topic, get_turn_topic


logger = logging.getLogger(__name__)


class RouteType(Enum):
    """Types of message routes."""
    BROADCAST = "broadcast"      # Send to all subscribers
    TARGETED = "targeted"        # Send to specific agent
    ROUND_ROBIN = "round_robin"  # Distribute among subscribers
    FILTERED = "filtered"        # Send based on filter criteria


@dataclass
class Route:
    """
    Represents a message route.
    
    Attributes:
        route_id: Unique route identifier
        source_pattern: Pattern to match source topics
        destination_topics: Target topics for messages
        route_type: Type of routing
        filters: Optional filters to apply
        transform: Optional transformation function
        active: Whether the route is active
    """
    
    route_id: str
    source_pattern: str
    destination_topics: List[str]
    route_type: RouteType = RouteType.BROADCAST
    filters: List[Callable[[Message], bool]] = field(default_factory=list)
    transform: Optional[Callable[[Message], Message]] = None
    active: bool = True
    
    def matches_source(self, topic: str) -> bool:
        """
        Check if a topic matches this route's source pattern.
        
        Args:
            topic: Topic name to check
            
        Returns:
            True if topic matches pattern
        """
        pattern = re.compile(self.source_pattern)
        return bool(pattern.match(topic))
    
    def should_route(self, message: Message) -> bool:
        """
        Check if a message should be routed.
        
        Args:
            message: Message to check
            
        Returns:
            True if message passes all filters
        """
        if not self.active:
            return False
        
        for filter_func in self.filters:
            if not filter_func(message):
                return False
        
        return True
    
    def process_message(self, message: Message) -> Message:
        """
        Apply transformation to message if configured.
        
        Args:
            message: Message to process
            
        Returns:
            Transformed message or original
        """
        if self.transform:
            return self.transform(message)
        return message


@dataclass
class Subscription:
    """
    Represents a message subscription.
    
    Attributes:
        subscription_id: Unique subscription identifier
        agent_id: Subscribing agent's ID
        topics: Topics to subscribe to
        message_types: Message types to receive
        handler: Message handler callback
        filters: Additional filters
        active: Whether subscription is active
    """
    
    subscription_id: str
    agent_id: str
    topics: Set[str] = field(default_factory=set)
    message_types: Set[MessageType] = field(default_factory=set)
    handler: Optional[Callable[[Message], None]] = None
    filters: List[Callable[[Message], bool]] = field(default_factory=list)
    active: bool = True
    
    def matches(self, message: Message, topic: str) -> bool:
        """
        Check if a message matches this subscription.
        
        Args:
            message: Message to check
            topic: Topic the message came from
            
        Returns:
            True if message matches subscription criteria
        """
        if not self.active:
            return False
        
        # Check topic match
        if self.topics and topic not in self.topics:
            return False
        
        # Check message type match
        if self.message_types and message.message_type not in self.message_types:
            return False
        
        # Check custom filters
        for filter_func in self.filters:
            if not filter_func(message):
                return False
        
        return True


class MessageRouter:
    """
    Central message router for Arena.
    
    This class manages message routing between topics, handles
    subscriptions, and ensures messages reach the correct destinations.
    """
    
    def __init__(
        self,
        producer: Optional[ArenaKafkaProducer] = None,
        topics_manager: Optional[ArenaTopics] = None
    ):
        """
        Initialize the message router.
        
        Args:
            producer: Kafka producer for sending messages
            topics_manager: Topic manager for topic configuration
        """
        self.producer = producer or ArenaKafkaProducer()
        self.topics_manager = topics_manager or ArenaTopics()
        
        # Routing tables
        self.routes: Dict[str, Route] = {}
        self.subscriptions: Dict[str, Subscription] = {}
        
        # Dead letter queue for failed messages
        self.dead_letter_queue: List[tuple[Message, str]] = []
        
        # Statistics
        self.routed_count = 0
        self.failed_count = 0
        
        # Initialize default routes
        self._initialize_default_routes()
        
        logger.info("Message router initialized")
    
    def _initialize_default_routes(self) -> None:
        """Initialize default routing rules."""
        
        # Route contributions to scoring
        self.add_route(Route(
            route_id="contributions_to_scoring",
            source_pattern=r"arena\.game\.contributions.*",
            destination_topics=["arena.scoring.metrics"],
            route_type=RouteType.BROADCAST
        ))
        
        # Route accusations to judge
        self.add_route(Route(
            route_id="accusations_to_judge",
            source_pattern=r"arena\.accusation\.claims.*",
            destination_topics=["arena.accusation.verdicts"],
            route_type=RouteType.BROADCAST
        ))
        
        # Route eliminations to all agents
        self.add_route(Route(
            route_id="eliminations_broadcast",
            source_pattern=r"arena\.agent\.lifecycle",
            destination_topics=["arena.game.state"],
            route_type=RouteType.BROADCAST,
            filters=[lambda m: m.message_type == "elimination"]
        ))
        
        # Route system errors to error topic
        self.add_route(Route(
            route_id="errors_to_error_topic",
            source_pattern=r"arena\..*",
            destination_topics=["arena.system.errors"],
            route_type=RouteType.BROADCAST,
            filters=[lambda m: m.message_type == "error"]
        ))
    
    def add_route(self, route: Route) -> None:
        """
        Add a routing rule.
        
        Args:
            route: Route to add
        """
        self.routes[route.route_id] = route
        logger.info(f"Added route: {route.route_id}")
    
    def remove_route(self, route_id: str) -> bool:
        """
        Remove a routing rule.
        
        Args:
            route_id: ID of route to remove
            
        Returns:
            True if route was removed
        """
        if route_id in self.routes:
            del self.routes[route_id]
            logger.info(f"Removed route: {route_id}")
            return True
        return False
    
    def add_subscription(self, subscription: Subscription) -> None:
        """
        Add a message subscription.
        
        Args:
            subscription: Subscription to add
        """
        self.subscriptions[subscription.subscription_id] = subscription
        logger.info(f"Added subscription: {subscription.subscription_id} for agent {subscription.agent_id}")
    
    def remove_subscription(self, subscription_id: str) -> bool:
        """
        Remove a subscription.
        
        Args:
            subscription_id: ID of subscription to remove
            
        Returns:
            True if subscription was removed
        """
        if subscription_id in self.subscriptions:
            del self.subscriptions[subscription_id]
            logger.info(f"Removed subscription: {subscription_id}")
            return True
        return False
    
    def route_message(
        self,
        message: Message,
        source_topic: str,
        override_routes: Optional[List[str]] = None
    ) -> Dict[str, bool]:
        """
        Route a message based on configured rules.
        
        Args:
            message: Message to route
            source_topic: Topic the message came from
            override_routes: Optional list of route IDs to use
            
        Returns:
            Dictionary mapping destination topics to send success
        """
        results = {}
        
        # Determine which routes to use
        if override_routes:
            routes_to_check = [
                self.routes[rid] for rid in override_routes 
                if rid in self.routes
            ]
        else:
            routes_to_check = [
                route for route in self.routes.values()
                if route.matches_source(source_topic)
            ]
        
        # Process each matching route
        for route in routes_to_check:
            if not route.should_route(message):
                continue
            
            # Apply transformation if configured
            routed_message = route.process_message(message)
            
            # Send to destination topics
            for dest_topic in route.destination_topics:
                try:
                    success = self.producer.send_message(dest_topic, routed_message)
                    results[dest_topic] = success
                    
                    if success:
                        self.routed_count += 1
                    else:
                        self.failed_count += 1
                        self._handle_failed_message(routed_message, dest_topic)
                        
                except Exception as e:
                    logger.error(f"Error routing to {dest_topic}: {e}")
                    results[dest_topic] = False
                    self.failed_count += 1
                    self._handle_failed_message(routed_message, dest_topic)
        
        return results
    
    def deliver_to_subscriptions(
        self,
        message: Message,
        source_topic: str
    ) -> Dict[str, bool]:
        """
        Deliver a message to matching subscriptions.
        
        Args:
            message: Message to deliver
            source_topic: Topic the message came from
            
        Returns:
            Dictionary mapping subscription IDs to delivery success
        """
        results = {}
        
        for sub_id, subscription in self.subscriptions.items():
            if not subscription.matches(message, source_topic):
                continue
            
            try:
                if subscription.handler:
                    subscription.handler(message)
                    results[sub_id] = True
                else:
                    logger.warning(f"Subscription {sub_id} has no handler")
                    results[sub_id] = False
                    
            except Exception as e:
                logger.error(f"Error delivering to subscription {sub_id}: {e}")
                results[sub_id] = False
        
        return results
    
    def broadcast(
        self,
        message: Message,
        topics: List[str]
    ) -> Dict[str, bool]:
        """
        Broadcast a message to multiple topics.
        
        Args:
            message: Message to broadcast
            topics: Target topics
            
        Returns:
            Dictionary mapping topics to send success
        """
        return self.producer.broadcast_to_topics(topics, message)
    
    def send_to_agent(
        self,
        message: Message,
        agent_id: str,
        topic: Optional[str] = None
    ) -> bool:
        """
        Send a message to a specific agent.
        
        Args:
            message: Message to send
            agent_id: Target agent ID
            topic: Optional specific topic (uses default if not provided)
            
        Returns:
            True if message was sent successfully
        """
        if topic is None:
            topic = "arena.agent.actions"
        
        # Set target agent
        message.target_agent_id = agent_id
        
        # Use agent ID as partition key for ordering
        return self.producer.send_message(topic, message, key=agent_id)
    
    def _handle_failed_message(
        self,
        message: Message,
        destination: str
    ) -> None:
        """
        Handle a message that failed to route.
        
        Args:
            message: Failed message
            destination: Intended destination
        """
        self.dead_letter_queue.append((message, destination))
        
        # Keep queue bounded
        if len(self.dead_letter_queue) > 1000:
            self.dead_letter_queue.pop(0)
        
        logger.warning(f"Added message {message.message_id} to dead letter queue")
    
    def retry_dead_letters(self) -> Dict[str, bool]:
        """
        Retry messages in the dead letter queue.
        
        Returns:
            Dictionary mapping message IDs to retry success
        """
        results = {}
        retry_queue = self.dead_letter_queue.copy()
        self.dead_letter_queue.clear()
        
        for message, destination in retry_queue:
            try:
                success = self.producer.send_message(destination, message)
                results[message.message_id] = success
                
                if not success:
                    self.dead_letter_queue.append((message, destination))
                    
            except Exception as e:
                logger.error(f"Error retrying dead letter {message.message_id}: {e}")
                results[message.message_id] = False
                self.dead_letter_queue.append((message, destination))
        
        return results
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """
        Get routing statistics.
        
        Returns:
            Dictionary with routing statistics
        """
        return {
            "routes_configured": len(self.routes),
            "active_routes": sum(1 for r in self.routes.values() if r.active),
            "subscriptions": len(self.subscriptions),
            "active_subscriptions": sum(1 for s in self.subscriptions.values() if s.active),
            "messages_routed": self.routed_count,
            "messages_failed": self.failed_count,
            "dead_letter_queue_size": len(self.dead_letter_queue)
        }
    
    def close(self) -> None:
        """Close the router and clean up resources."""
        if self.producer:
            self.producer.close()
        logger.info("Message router closed")