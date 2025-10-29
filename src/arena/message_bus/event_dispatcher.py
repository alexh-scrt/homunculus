"""
Event Dispatcher for Arena Message Bus

This module provides the central event dispatching mechanism that
coordinates between message consumers, handlers, and producers.

Features:
- Event loop management
- Handler orchestration
- Priority queue for events
- Concurrent handler execution
- Error recovery and retry logic

Author: Homunculus Team
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
from queue import PriorityQueue, Queue, Empty
from threading import Thread, Lock, Event as ThreadEvent
import time
from concurrent.futures import ThreadPoolExecutor, Future

from ..models import Message, ArenaState
from .kafka_consumer import ArenaKafkaConsumer, ConsumeMode
from .kafka_producer import ArenaKafkaProducer
from .message_router import MessageRouter
from .event_handlers import EventHandler, get_handlers_for_message
from .topics import ArenaTopics


logger = logging.getLogger(__name__)


@dataclass
class EventContext:
    """
    Context passed to event handlers.
    
    Attributes:
        arena_state: Current game state
        producer: Kafka producer for sending messages
        router: Message router for routing
        metadata: Additional context data
    """
    arena_state: Optional[ArenaState] = None
    producer: Optional[ArenaKafkaProducer] = None
    router: Optional[MessageRouter] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for handler use."""
        return {
            "arena_state": self.arena_state,
            "producer": self.producer,
            "router": self.router,
            **self.metadata
        }


@dataclass
class PrioritizedEvent:
    """
    Event with priority for queue ordering.
    
    Lower priority values are processed first.
    """
    priority: int
    message: Message
    source_topic: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    retry_count: int = 0
    
    def __lt__(self, other):
        """Compare by priority for queue ordering."""
        return self.priority < other.priority


class EventDispatcher:
    """
    Central event dispatcher for Arena.
    
    This class manages the event processing pipeline, coordinating
    between consumers, handlers, and producers.
    """
    
    def __init__(
        self,
        topics: List[str],
        group_id: str,
        arena_state: Optional[ArenaState] = None,
        max_workers: int = 10,
        queue_size: int = 1000
    ):
        """
        Initialize the event dispatcher.
        
        Args:
            topics: Topics to consume from
            group_id: Consumer group ID
            arena_state: Initial arena state
            max_workers: Maximum concurrent handler threads
            queue_size: Maximum event queue size
        """
        self.topics = topics
        self.group_id = group_id
        self.arena_state = arena_state or ArenaState()
        self.max_workers = max_workers
        self.queue_size = queue_size
        
        # Event queue with priority support
        self.event_queue: PriorityQueue = PriorityQueue(maxsize=queue_size)
        
        # Dead letter queue for failed events
        self.dead_letter_queue: Queue = Queue()
        
        # Kafka components
        self.consumer: Optional[ArenaKafkaConsumer] = None
        self.producer: Optional[ArenaKafkaProducer] = None
        self.router: Optional[MessageRouter] = None
        self.topics_manager: Optional[ArenaTopics] = None
        
        # Handler registry
        self.handlers: Dict[str, List[EventHandler]] = {}
        self.handler_executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Control flags
        self.running = False
        self.stop_event = ThreadEvent()
        self.consumer_thread: Optional[Thread] = None
        self.processor_thread: Optional[Thread] = None
        
        # Statistics
        self.stats_lock = Lock()
        self.events_received = 0
        self.events_processed = 0
        self.events_failed = 0
        self.handler_errors: Dict[str, int] = {}
        
        logger.info(f"Event dispatcher initialized for topics {topics}")
    
    def initialize_components(self) -> None:
        """Initialize Kafka components."""
        # Create producer
        self.producer = ArenaKafkaProducer(
            error_callback=self._handle_producer_error
        )
        
        # Create consumer
        self.consumer = ArenaKafkaConsumer(
            topics=self.topics,
            group_id=self.group_id,
            consume_mode=ConsumeMode.LATEST,
            error_callback=self._handle_consumer_error
        )
        
        # Create router
        self.router = MessageRouter(producer=self.producer)
        
        # Create topics manager
        self.topics_manager = ArenaTopics()
        
        logger.info("Dispatcher components initialized")
    
    def register_handler(
        self,
        message_type: str,
        handler: EventHandler
    ) -> None:
        """
        Register an event handler.
        
        Args:
            message_type: Message type to handle
            handler: Handler instance
        """
        if message_type not in self.handlers:
            self.handlers[message_type] = []
        
        self.handlers[message_type].append(handler)
        logger.info(f"Registered handler {handler.handler_id} for {message_type}")
    
    def start(self) -> None:
        """Start the event dispatcher."""
        if self.running:
            logger.warning("Dispatcher already running")
            return
        
        self.running = True
        self.stop_event.clear()
        
        # Initialize components if needed
        if not self.consumer:
            self.initialize_components()
        
        # Start consumer thread
        self.consumer_thread = Thread(
            target=self._consume_loop,
            name="EventConsumer"
        )
        self.consumer_thread.start()
        
        # Start processor thread
        self.processor_thread = Thread(
            target=self._process_loop,
            name="EventProcessor"
        )
        self.processor_thread.start()
        
        logger.info("Event dispatcher started")
    
    def stop(self, timeout: int = 30) -> None:
        """
        Stop the event dispatcher.
        
        Args:
            timeout: Maximum time to wait for graceful shutdown
        """
        if not self.running:
            return
        
        logger.info("Stopping event dispatcher...")
        self.running = False
        self.stop_event.set()
        
        # Wait for threads to stop
        if self.consumer_thread:
            self.consumer_thread.join(timeout=timeout/2)
        
        if self.processor_thread:
            self.processor_thread.join(timeout=timeout/2)
        
        # Shutdown executor
        self.handler_executor.shutdown(wait=True, timeout=timeout)
        
        # Close Kafka components
        if self.consumer:
            self.consumer.close()
        
        if self.producer:
            self.producer.close()
        
        if self.topics_manager:
            self.topics_manager.close()
        
        logger.info("Event dispatcher stopped")
    
    def _consume_loop(self) -> None:
        """Consumer loop that reads messages and queues them."""
        logger.info("Consumer loop started")
        
        while self.running:
            try:
                # Consume messages with timeout
                for message in self.consumer.consume_messages(
                    max_messages=10,
                    timeout_ms=1000
                ):
                    if not self.running:
                        break
                    
                    # Determine priority
                    priority = self._get_event_priority(message)
                    
                    # Get source topic from metadata
                    source_topic = message.metadata.get(
                        "kafka_topic",
                        self.topics[0]
                    )
                    
                    # Create prioritized event
                    event = PrioritizedEvent(
                        priority=priority,
                        message=message,
                        source_topic=source_topic
                    )
                    
                    # Queue event
                    try:
                        self.event_queue.put_nowait(event)
                        with self.stats_lock:
                            self.events_received += 1
                            
                    except:
                        logger.warning(f"Event queue full, dropping message {message.message_id}")
                
            except Exception as e:
                logger.error(f"Error in consumer loop: {e}")
                if self.running:
                    time.sleep(1)
        
        logger.info("Consumer loop stopped")
    
    def _process_loop(self) -> None:
        """Processor loop that handles queued events."""
        logger.info("Processor loop started")
        
        while self.running or not self.event_queue.empty():
            try:
                # Get event from queue with timeout
                event = self.event_queue.get(timeout=1)
                
                # Create context
                context = EventContext(
                    arena_state=self.arena_state,
                    producer=self.producer,
                    router=self.router,
                    metadata={
                        "source_topic": event.source_topic,
                        "retry_count": event.retry_count
                    }
                )
                
                # Process event
                future = self.handler_executor.submit(
                    self._process_event,
                    event,
                    context
                )
                
                # Add callback for statistics
                future.add_done_callback(
                    lambda f: self._handle_process_result(f, event)
                )
                
            except Empty:
                continue
                
            except Exception as e:
                logger.error(f"Error in processor loop: {e}")
        
        logger.info("Processor loop stopped")
    
    def _process_event(
        self,
        event: PrioritizedEvent,
        context: EventContext
    ) -> Dict[str, Any]:
        """
        Process a single event.
        
        Args:
            event: Event to process
            context: Processing context
            
        Returns:
            Processing results from handlers
        """
        message = event.message
        results = {}
        
        # Get applicable handlers
        handlers = get_handlers_for_message(message)
        
        # Also check registered handlers
        if message.message_type in self.handlers:
            handlers.extend(self.handlers[message.message_type])
        
        if not handlers:
            logger.debug(f"No handlers for message {message.message_id}")
            return {"status": "no_handlers"}
        
        # Process with each handler
        for handler in handlers:
            try:
                # Pre-process
                if not handler.pre_process(message, context.to_dict()):
                    continue
                
                # Handle
                result = handler.handle(message, context.to_dict())
                results[handler.handler_id] = result
                
                # Post-process
                handler.post_process(message, result, context.to_dict())
                
                # Process handler results
                self._process_handler_result(result, message, context)
                
            except Exception as e:
                handler.handle_error(message, e, context.to_dict())
                results[handler.handler_id] = {"status": "error", "error": str(e)}
                
                with self.stats_lock:
                    handler_id = handler.handler_id
                    self.handler_errors[handler_id] = self.handler_errors.get(handler_id, 0) + 1
        
        # Route message if needed
        if context.router and event.source_topic:
            context.router.route_message(message, event.source_topic)
        
        return results
    
    def _process_handler_result(
        self,
        result: Dict[str, Any],
        message: Message,
        context: EventContext
    ) -> None:
        """
        Process special handler results.
        
        Args:
            result: Handler result
            message: Processed message
            context: Processing context
        """
        # Check for scoring request
        if result.get("request_scoring"):
            self._trigger_scoring(message, result)
        
        # Check for judge request
        if result.get("request_judge"):
            self._trigger_judge(result.get("accusation_id"))
        
        # Check for elimination request
        if result.get("request_elimination"):
            self._trigger_elimination(
                result.get("agent_id", message.sender_id),
                result.get("elimination_reason", "Score threshold")
            )
        
        # Check for termination check
        if result.get("check_termination"):
            self._check_termination(result.get("termination_reason"))
    
    def _trigger_scoring(self, message: Message, result: Dict[str, Any]) -> None:
        """Trigger scoring evaluation for a message."""
        if not self.producer:
            return
        
        scoring_request = Message(
            sender_id="system",
            sender_name="Dispatcher",
            sender_type="system",
            message_type="scoring_request",
            content=f"Score contribution {message.message_id}",
            target_agent_id=message.sender_id,
            metadata={
                "message_id": message.message_id,
                "agent_id": message.sender_id,
                "turn_number": result.get("turn_number", 0)
            }
        )
        
        self.producer.send_message("arena.scoring.metrics", scoring_request)
    
    def _trigger_judge(self, accusation_id: str) -> None:
        """Trigger judge evaluation for an accusation."""
        if not self.producer:
            return
        
        judge_request = Message(
            sender_id="system",
            sender_name="Dispatcher",
            sender_type="system",
            message_type="judge_request",
            content=f"Evaluate accusation {accusation_id}",
            metadata={"accusation_id": accusation_id}
        )
        
        self.producer.send_message("arena.accusation.verdicts", judge_request)
    
    def _trigger_elimination(self, agent_id: str, reason: str) -> None:
        """Trigger agent elimination."""
        if not self.producer:
            return
        
        elimination_message = Message(
            sender_id="system",
            sender_name="Dispatcher",
            sender_type="system",
            message_type="elimination",
            content=reason,
            target_agent_id=agent_id
        )
        
        self.producer.send_message("arena.agent.lifecycle", elimination_message)
    
    def _check_termination(self, reason: Optional[str]) -> None:
        """Check and handle game termination."""
        if not reason or not self.arena_state:
            return
        
        self.arena_state.terminate_game(reason)
        
        if self.producer:
            termination_message = Message(
                sender_id="system",
                sender_name="Dispatcher",
                sender_type="system",
                message_type="game_terminated",
                content=f"Game terminated: {reason}",
                metadata={
                    "termination_reason": reason,
                    "winner_id": self.arena_state.winner_id,
                    "winner_name": self.arena_state.winner_name
                }
            )
            
            self.producer.send_message("arena.game.state", termination_message)
    
    def _get_event_priority(self, message: Message) -> int:
        """
        Determine priority for an event.
        
        Lower values = higher priority.
        
        Args:
            message: Message to prioritize
            
        Returns:
            Priority value
        """
        # Priority based on message type
        priority_map = {
            "elimination": 1,
            "accusation": 2,
            "judge_verdict": 2,
            "scoring": 3,
            "turn_selection": 4,
            "contribution": 5,
            "status": 8,
            "error": 9,
            "debug": 10
        }
        
        return priority_map.get(message.message_type, 7)
    
    def _handle_process_result(
        self,
        future: Future,
        event: PrioritizedEvent
    ) -> None:
        """Handle processing completion."""
        try:
            result = future.result()
            with self.stats_lock:
                self.events_processed += 1
                
        except Exception as e:
            logger.error(f"Processing failed for {event.message.message_id}: {e}")
            
            with self.stats_lock:
                self.events_failed += 1
            
            # Retry or send to dead letter queue
            if event.retry_count < 3:
                event.retry_count += 1
                event.priority += 1  # Lower priority for retries
                
                try:
                    self.event_queue.put_nowait(event)
                except:
                    self.dead_letter_queue.put(event)
            else:
                self.dead_letter_queue.put(event)
    
    def _handle_producer_error(self, message_id: str, error: Exception) -> None:
        """Handle producer errors."""
        logger.error(f"Producer error for {message_id}: {error}")
        with self.stats_lock:
            self.events_failed += 1
    
    def _handle_consumer_error(self, context: str, error: Exception) -> None:
        """Handle consumer errors."""
        logger.error(f"Consumer error in {context}: {error}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dispatcher statistics.
        
        Returns:
            Dictionary with statistics
        """
        with self.stats_lock:
            return {
                "running": self.running,
                "events_received": self.events_received,
                "events_processed": self.events_processed,
                "events_failed": self.events_failed,
                "queue_size": self.event_queue.qsize(),
                "dead_letter_size": self.dead_letter_queue.qsize(),
                "handler_errors": dict(self.handler_errors),
                "registered_handlers": {
                    k: len(v) for k, v in self.handlers.items()
                }
            }
    
    def process_dead_letters(self) -> int:
        """
        Process events in dead letter queue.
        
        Returns:
            Number of events reprocessed
        """
        count = 0
        
        while not self.dead_letter_queue.empty():
            try:
                event = self.dead_letter_queue.get_nowait()
                event.retry_count = 0  # Reset retry count
                event.priority = 0  # High priority for reprocess
                
                self.event_queue.put_nowait(event)
                count += 1
                
            except:
                break
        
        logger.info(f"Reprocessed {count} dead letter events")
        return count