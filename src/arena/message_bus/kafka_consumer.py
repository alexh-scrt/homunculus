"""
Kafka Consumer for Arena Message Bus

This module provides a robust Kafka consumer with offset management,
error handling, and message deserialization for Arena.

Features:
- Automatic offset management
- Message deserialization from JSON
- Error recovery and retry logic
- Batch consumption support
- Consumer group coordination

Author: Homunculus Team
"""

import json
import logging
from typing import Any, Dict, Optional, Callable, List, Generator
from datetime import datetime
import time
from enum import Enum

from kafka import KafkaConsumer, TopicPartition, OffsetAndMetadata
from kafka.errors import KafkaError, CommitFailedError
from kafka.structs import ConsumerRecord

from ..models import Message, MessageBatch
from ..config.arena_settings import ArenaSettings


logger = logging.getLogger(__name__)


class ConsumeMode(Enum):
    """Consumer operation modes."""
    LATEST = "latest"      # Start from latest messages
    EARLIEST = "earliest"  # Start from beginning
    COMMITTED = "committed"  # Start from last committed offset


class ArenaKafkaConsumer:
    """
    Kafka consumer wrapper for Arena messages.
    
    This class provides a high-level interface for consuming messages
    from Kafka topics with automatic deserialization, offset management,
    and error handling.
    
    Attributes:
        consumer: Underlying Kafka consumer
        settings: Arena configuration settings
        message_handler: Optional callback for processing messages
        error_callback: Optional callback for error handling
        consumed_count: Track number of messages consumed
        last_offset: Track last processed offset per partition
    """
    
    def __init__(
        self,
        topics: List[str],
        group_id: str,
        settings: Optional[ArenaSettings] = None,
        consume_mode: ConsumeMode = ConsumeMode.LATEST,
        message_handler: Optional[Callable[[Message], None]] = None,
        error_callback: Optional[Callable[[str, Exception], None]] = None
    ):
        """
        Initialize the Kafka consumer.
        
        Args:
            topics: List of topics to subscribe to
            group_id: Consumer group ID
            settings: Arena settings (uses default if not provided)
            consume_mode: Where to start consuming from
            message_handler: Optional callback for each message
            error_callback: Optional callback for errors
        """
        self.settings = settings or ArenaSettings()
        self.topics = topics
        self.group_id = group_id
        self.message_handler = message_handler
        self.error_callback = error_callback
        self.consumed_count = 0
        self.last_offset: Dict[TopicPartition, int] = {}
        
        # Configure consumer with robust settings
        self.consumer = KafkaConsumer(
            *topics,
            bootstrap_servers=self.settings.kafka_bootstrap_servers,
            group_id=group_id,
            value_deserializer=self._deserialize_message,
            key_deserializer=lambda k: k.decode('utf-8') if k else None,
            auto_offset_reset=consume_mode.value,
            enable_auto_commit=False,  # Manual commit for control
            max_poll_records=100,
            max_poll_interval_ms=300000,  # 5 minutes
            session_timeout_ms=30000,
            heartbeat_interval_ms=10000,
            consumer_timeout_ms=1000,  # Return from poll after 1 second
            api_version_auto_timeout_ms=10000
        )
        
        logger.info(
            f"Kafka consumer initialized for topics {topics} "
            f"with group_id={group_id}, mode={consume_mode.value}"
        )
    
    def _deserialize_message(self, data: bytes) -> Message:
        """
        Deserialize a message from JSON bytes.
        
        Args:
            data: JSON bytes to deserialize
            
        Returns:
            Deserialized Message object
        """
        try:
            json_data = json.loads(data.decode('utf-8'))
            
            # Handle both Message dicts and plain dicts
            if all(key in json_data for key in ['sender_id', 'content']):
                return Message.from_dict(json_data)
            else:
                # Create a basic message from plain data
                return Message(
                    sender_id=json_data.get('sender_id', 'unknown'),
                    sender_name=json_data.get('sender_name', 'Unknown'),
                    content=json_data.get('content', str(json_data)),
                    metadata=json_data
                )
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to deserialize message: {e}")
            # Return error message
            return Message(
                sender_id="error",
                sender_name="Error",
                sender_type="system",
                message_type="error",
                content=f"Deserialization error: {e}",
                metadata={"raw_data": data.decode('utf-8', errors='replace')}
            )
        except Exception as e:
            logger.error(f"Unexpected error deserializing message: {e}")
            raise
    
    def consume_messages(
        self,
        max_messages: Optional[int] = None,
        timeout_ms: int = 1000
    ) -> Generator[Message, None, None]:
        """
        Consume messages from subscribed topics.
        
        Args:
            max_messages: Maximum number of messages to consume (None = infinite)
            timeout_ms: Poll timeout in milliseconds
            
        Yields:
            Message objects as they are consumed
        """
        messages_consumed = 0
        
        while max_messages is None or messages_consumed < max_messages:
            try:
                # Poll for messages
                records = self.consumer.poll(timeout_ms=timeout_ms)
                
                if not records:
                    continue
                
                # Process records from all partitions
                for topic_partition, messages in records.items():
                    for record in messages:
                        try:
                            message = self._process_record(record)
                            
                            # Update tracking
                            self.consumed_count += 1
                            messages_consumed += 1
                            self.last_offset[topic_partition] = record.offset
                            
                            # Call handler if provided
                            if self.message_handler:
                                self.message_handler(message)
                            
                            yield message
                            
                            if max_messages and messages_consumed >= max_messages:
                                return
                                
                        except Exception as e:
                            logger.error(f"Error processing record: {e}")
                            if self.error_callback:
                                self.error_callback(f"record_{record.offset}", e)
                
                # Commit offsets after successful processing
                self.commit_offsets()
                
            except KafkaError as e:
                logger.error(f"Kafka error during consumption: {e}")
                if self.error_callback:
                    self.error_callback("consume", e)
                time.sleep(1)  # Brief pause before retry
                
            except Exception as e:
                logger.error(f"Unexpected error during consumption: {e}")
                if self.error_callback:
                    self.error_callback("consume", e)
                break
    
    def consume_batch(
        self,
        batch_size: int = 10,
        timeout_ms: int = 1000
    ) -> Optional[MessageBatch]:
        """
        Consume a batch of messages.
        
        Args:
            batch_size: Number of messages to consume
            timeout_ms: Poll timeout in milliseconds
            
        Returns:
            MessageBatch if messages were consumed, None otherwise
        """
        batch = MessageBatch(game_id="consumer_batch")
        
        for message in self.consume_messages(max_messages=batch_size, timeout_ms=timeout_ms):
            batch.add_message(message)
            
            if batch.size >= batch_size:
                break
        
        return batch if batch.size > 0 else None
    
    def _process_record(self, record: ConsumerRecord) -> Message:
        """
        Process a Kafka record into a Message.
        
        Args:
            record: Kafka consumer record
            
        Returns:
            Processed Message object
        """
        message = record.value
        
        # Add Kafka metadata to message
        message.metadata.update({
            "kafka_topic": record.topic,
            "kafka_partition": record.partition,
            "kafka_offset": record.offset,
            "kafka_timestamp": record.timestamp,
            "kafka_key": record.key
        })
        
        logger.debug(
            f"Processed message {message.message_id} from "
            f"{record.topic}:{record.partition}@{record.offset}"
        )
        
        return message
    
    def commit_offsets(self, async_commit: bool = False) -> bool:
        """
        Commit current offsets.
        
        Args:
            async_commit: Whether to commit asynchronously
            
        Returns:
            True if commit was successful
        """
        try:
            if async_commit:
                self.consumer.commit_async()
            else:
                self.consumer.commit()
            
            logger.debug(f"Committed offsets: {self.last_offset}")
            return True
            
        except CommitFailedError as e:
            logger.error(f"Failed to commit offsets: {e}")
            return False
            
        except Exception as e:
            logger.error(f"Unexpected error committing offsets: {e}")
            return False
    
    def seek_to_beginning(self, partitions: Optional[List[TopicPartition]] = None) -> None:
        """
        Seek to the beginning of partitions.
        
        Args:
            partitions: Specific partitions to seek (None = all)
        """
        if partitions is None:
            partitions = self.consumer.assignment()
        
        self.consumer.seek_to_beginning(*partitions)
        logger.info(f"Sought to beginning for {len(partitions)} partitions")
    
    def seek_to_end(self, partitions: Optional[List[TopicPartition]] = None) -> None:
        """
        Seek to the end of partitions.
        
        Args:
            partitions: Specific partitions to seek (None = all)
        """
        if partitions is None:
            partitions = self.consumer.assignment()
        
        self.consumer.seek_to_end(*partitions)
        logger.info(f"Sought to end for {len(partitions)} partitions")
    
    def seek_to_offset(self, topic: str, partition: int, offset: int) -> None:
        """
        Seek to a specific offset.
        
        Args:
            topic: Topic name
            partition: Partition number
            offset: Target offset
        """
        tp = TopicPartition(topic, partition)
        self.consumer.seek(tp, offset)
        logger.info(f"Sought to {topic}:{partition}@{offset}")
    
    def get_current_positions(self) -> Dict[str, Dict[int, int]]:
        """
        Get current position for all assigned partitions.
        
        Returns:
            Dictionary mapping topics to partition positions
        """
        positions = {}
        
        for tp in self.consumer.assignment():
            if tp.topic not in positions:
                positions[tp.topic] = {}
            
            positions[tp.topic][tp.partition] = self.consumer.position(tp)
        
        return positions
    
    def get_consumption_stats(self) -> Dict[str, Any]:
        """
        Get statistics about message consumption.
        
        Returns:
            Dictionary with consumption statistics
        """
        positions = self.get_current_positions()
        
        return {
            "consumed_count": self.consumed_count,
            "topics": self.topics,
            "group_id": self.group_id,
            "current_positions": positions,
            "last_offsets": {
                f"{tp.topic}:{tp.partition}": offset
                for tp, offset in self.last_offset.items()
            }
        }
    
    def pause_consumption(self, partitions: Optional[List[TopicPartition]] = None) -> None:
        """
        Pause consumption from partitions.
        
        Args:
            partitions: Specific partitions to pause (None = all)
        """
        if partitions is None:
            partitions = self.consumer.assignment()
        
        self.consumer.pause(*partitions)
        logger.info(f"Paused consumption from {len(partitions)} partitions")
    
    def resume_consumption(self, partitions: Optional[List[TopicPartition]] = None) -> None:
        """
        Resume consumption from partitions.
        
        Args:
            partitions: Specific partitions to resume (None = all)
        """
        if partitions is None:
            partitions = self.consumer.assignment()
        
        self.consumer.resume(*partitions)
        logger.info(f"Resumed consumption from {len(partitions)} partitions")
    
    def close(self) -> None:
        """Close the consumer and commit final offsets."""
        try:
            # Commit any pending offsets
            self.commit_offsets()
            
            # Close consumer
            self.consumer.close()
            logger.info("Kafka consumer closed")
            
        except Exception as e:
            logger.error(f"Error closing consumer: {e}")
    
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