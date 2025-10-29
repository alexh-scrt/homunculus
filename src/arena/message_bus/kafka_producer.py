"""
Kafka Producer for Arena Message Bus

This module provides a robust Kafka producer with retry logic,
error handling, and message serialization for Arena.

Features:
- Automatic retry on failures
- Message serialization to JSON
- Delivery confirmation tracking
- Batch sending support
- Error callbacks for monitoring

Author: Homunculus Team
"""

import json
import logging
from typing import Any, Dict, Optional, Callable, List
from datetime import datetime
import time

from kafka import KafkaProducer
from kafka.errors import KafkaError, KafkaTimeoutError

from ..models import Message, MessageBatch
from ..config.arena_settings import ArenaSettings


logger = logging.getLogger(__name__)


class ArenaKafkaProducer:
    """
    Kafka producer wrapper for Arena messages.
    
    This class provides a high-level interface for sending messages
    to Kafka topics with automatic serialization, retry logic, and
    error handling.
    
    Attributes:
        producer: Underlying Kafka producer
        settings: Arena configuration settings
        delivery_reports: Track message delivery confirmations
        error_callback: Optional callback for error handling
        success_callback: Optional callback for successful deliveries
    """
    
    def __init__(
        self,
        settings: Optional[ArenaSettings] = None,
        error_callback: Optional[Callable[[str, Exception], None]] = None,
        success_callback: Optional[Callable[[str, Dict], None]] = None
    ):
        """
        Initialize the Kafka producer.
        
        Args:
            settings: Arena settings (uses default if not provided)
            error_callback: Function called on send errors
            success_callback: Function called on successful sends
        """
        self.settings = settings or ArenaSettings()
        self.error_callback = error_callback
        self.success_callback = success_callback
        self.delivery_reports: List[Dict[str, Any]] = []
        
        # Configure producer with robust settings
        self.producer = KafkaProducer(
            bootstrap_servers=self.settings.kafka_bootstrap_servers,
            value_serializer=self._serialize_message,
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            acks='all',  # Wait for all replicas to acknowledge
            retries=3,
            max_in_flight_requests_per_connection=1,  # Ensure ordering
            compression_type='gzip',
            batch_size=16384,
            linger_ms=10,  # Small delay for batching
            request_timeout_ms=30000,
            retry_backoff_ms=100,
            api_version_auto_timeout_ms=10000
        )
        
        logger.info(f"Kafka producer initialized with servers: {self.settings.kafka_bootstrap_servers}")
    
    def _serialize_message(self, message: Any) -> bytes:
        """
        Serialize a message to JSON bytes.
        
        Args:
            message: Message to serialize (Message, dict, or JSON-serializable)
            
        Returns:
            JSON bytes representation
        """
        try:
            if isinstance(message, Message):
                data = message.to_dict()
            elif isinstance(message, dict):
                data = message
            else:
                data = {"content": str(message), "timestamp": datetime.utcnow().isoformat()}
            
            return json.dumps(data, default=str).encode('utf-8')
        except Exception as e:
            logger.error(f"Failed to serialize message: {e}")
            raise
    
    def send_message(
        self,
        topic: str,
        message: Message,
        key: Optional[str] = None,
        partition: Optional[int] = None,
        retry_count: int = 3
    ) -> bool:
        """
        Send a single message to a Kafka topic.
        
        Args:
            topic: Target Kafka topic
            message: Message to send
            key: Optional partition key (defaults to sender_id)
            partition: Optional specific partition
            retry_count: Number of retry attempts
            
        Returns:
            True if message was sent successfully
        """
        if key is None:
            key = message.sender_id
        
        for attempt in range(retry_count):
            try:
                # Send message and get future
                future = self.producer.send(
                    topic,
                    value=message,
                    key=key,
                    partition=partition
                )
                
                # Wait for confirmation (with timeout)
                record_metadata = future.get(timeout=10)
                
                # Record successful delivery
                delivery_report = {
                    "message_id": message.message_id,
                    "topic": record_metadata.topic,
                    "partition": record_metadata.partition,
                    "offset": record_metadata.offset,
                    "timestamp": record_metadata.timestamp,
                    "attempt": attempt + 1
                }
                self.delivery_reports.append(delivery_report)
                
                # Call success callback if provided
                if self.success_callback:
                    self.success_callback(message.message_id, delivery_report)
                
                logger.debug(
                    f"Message {message.message_id} sent to {topic}:{record_metadata.partition} "
                    f"at offset {record_metadata.offset}"
                )
                
                return True
                
            except KafkaTimeoutError as e:
                logger.warning(f"Timeout sending message {message.message_id} (attempt {attempt + 1}/{retry_count}): {e}")
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    
            except KafkaError as e:
                logger.error(f"Kafka error sending message {message.message_id}: {e}")
                if self.error_callback:
                    self.error_callback(message.message_id, e)
                    
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)
                    
            except Exception as e:
                logger.error(f"Unexpected error sending message {message.message_id}: {e}")
                if self.error_callback:
                    self.error_callback(message.message_id, e)
                break
        
        logger.error(f"Failed to send message {message.message_id} after {retry_count} attempts")
        return False
    
    def send_batch(
        self,
        topic: str,
        batch: MessageBatch,
        partition_strategy: str = "sender"
    ) -> Dict[str, bool]:
        """
        Send a batch of messages to a Kafka topic.
        
        Args:
            topic: Target Kafka topic
            batch: Batch of messages to send
            partition_strategy: How to partition messages ("sender", "turn", "round_robin")
            
        Returns:
            Dictionary mapping message IDs to send success status
        """
        results = {}
        
        for message in batch.messages:
            # Determine partition key based on strategy
            if partition_strategy == "sender":
                key = message.sender_id
            elif partition_strategy == "turn":
                key = str(message.turn_number)
            else:  # round_robin
                key = None
            
            success = self.send_message(topic, message, key=key)
            results[message.message_id] = success
        
        # Log batch results
        success_count = sum(1 for v in results.values() if v)
        logger.info(
            f"Batch send completed: {success_count}/{len(results)} messages sent successfully"
        )
        
        return results
    
    def send_system_message(
        self,
        topic: str,
        message_type: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Send a system message (from orchestrator/judge).
        
        Args:
            topic: Target Kafka topic
            message_type: Type of system message
            content: Message content
            metadata: Optional metadata
            
        Returns:
            True if message was sent successfully
        """
        system_message = Message(
            sender_id="system",
            sender_name="System",
            sender_type="system",
            message_type=message_type,
            content=content,
            metadata=metadata or {}
        )
        
        return self.send_message(topic, system_message, key="system")
    
    def broadcast_to_topics(
        self,
        topics: List[str],
        message: Message
    ) -> Dict[str, bool]:
        """
        Broadcast a message to multiple topics.
        
        Args:
            topics: List of target topics
            message: Message to broadcast
            
        Returns:
            Dictionary mapping topics to send success status
        """
        results = {}
        
        for topic in topics:
            success = self.send_message(topic, message)
            results[topic] = success
        
        return results
    
    def flush(self, timeout: Optional[float] = None) -> None:
        """
        Flush any pending messages.
        
        Args:
            timeout: Maximum time to wait for flush (None = infinite)
        """
        self.producer.flush(timeout=timeout)
        logger.info("Producer buffer flushed")
    
    def get_delivery_stats(self) -> Dict[str, Any]:
        """
        Get statistics about message deliveries.
        
        Returns:
            Dictionary with delivery statistics
        """
        if not self.delivery_reports:
            return {
                "total_sent": 0,
                "avg_attempts": 0,
                "topics": [],
                "partitions": []
            }
        
        total_attempts = sum(r["attempt"] for r in self.delivery_reports)
        unique_topics = set(r["topic"] for r in self.delivery_reports)
        unique_partitions = set((r["topic"], r["partition"]) for r in self.delivery_reports)
        
        return {
            "total_sent": len(self.delivery_reports),
            "avg_attempts": total_attempts / len(self.delivery_reports),
            "topics": list(unique_topics),
            "partitions": len(unique_partitions),
            "delivery_reports": self.delivery_reports[-10:]  # Last 10 reports
        }
    
    def close(self) -> None:
        """Close the producer and flush any pending messages."""
        try:
            self.flush(timeout=10)
            self.producer.close()
            logger.info("Kafka producer closed")
        except Exception as e:
            logger.error(f"Error closing producer: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.close()
        except:
            pass  # Suppress errors in destructor