"""
Kafka Topic Configuration and Management for Arena

This module defines and manages Kafka topics for the Arena message bus.
It provides topic creation, configuration, and validation utilities.

Features:
- Topic definitions and naming conventions
- Automatic topic creation
- Partition and replication configuration
- Topic validation and health checks

Author: Homunculus Team
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from kafka.admin import KafkaAdminClient, NewTopic, ConfigResource, ConfigResourceType
from kafka.errors import TopicAlreadyExistsError, KafkaError

from ..config.arena_settings import ArenaSettings


logger = logging.getLogger(__name__)


class TopicCategory(Enum):
    """Categories of Kafka topics in Arena."""
    GAME = "game"              # Game-wide messages
    AGENT = "agent"            # Agent-specific messages
    SYSTEM = "system"          # System and orchestration messages
    SCORING = "scoring"        # Scoring and evaluation messages
    ACCUSATION = "accusation"  # Accusation-related messages
    DEBUG = "debug"            # Debug and monitoring messages


@dataclass
class TopicConfig:
    """
    Configuration for a Kafka topic.
    
    Attributes:
        name: Topic name
        category: Topic category
        partitions: Number of partitions
        replication_factor: Replication factor
        retention_ms: Message retention time in milliseconds
        segment_ms: Segment roll time in milliseconds
        cleanup_policy: Cleanup policy (delete or compact)
        min_in_sync_replicas: Minimum in-sync replicas
        compression_type: Compression type
        max_message_bytes: Maximum message size
        description: Human-readable description
    """
    
    name: str
    category: TopicCategory
    partitions: int = 3
    replication_factor: int = 1
    retention_ms: int = 604800000  # 7 days
    segment_ms: int = 86400000  # 1 day
    cleanup_policy: str = "delete"
    min_in_sync_replicas: int = 1
    compression_type: str = "gzip"
    max_message_bytes: int = 1048576  # 1MB
    description: str = ""
    
    def to_kafka_config(self) -> Dict[str, str]:
        """
        Convert to Kafka configuration dictionary.
        
        Returns:
            Dictionary of Kafka topic configurations
        """
        return {
            "retention.ms": str(self.retention_ms),
            "segment.ms": str(self.segment_ms),
            "cleanup.policy": self.cleanup_policy,
            "min.insync.replicas": str(self.min_in_sync_replicas),
            "compression.type": self.compression_type,
            "max.message.bytes": str(self.max_message_bytes)
        }


class ArenaTopics:
    """
    Defines and manages all Kafka topics for Arena.
    
    This class centralizes topic definitions and provides methods
    for creating, configuring, and validating topics.
    """
    
    # Topic definitions
    TOPICS = {
        # Game-wide topics
        "arena.game.contributions": TopicConfig(
            name="arena.game.contributions",
            category=TopicCategory.GAME,
            partitions=6,  # Higher partition count for main game traffic
            description="Agent contributions to problem-solving"
        ),
        
        "arena.game.turns": TopicConfig(
            name="arena.game.turns",
            category=TopicCategory.GAME,
            partitions=1,  # Single partition to maintain order
            description="Turn management and selection"
        ),
        
        "arena.game.state": TopicConfig(
            name="arena.game.state",
            category=TopicCategory.GAME,
            partitions=1,
            cleanup_policy="compact",  # Keep latest state
            description="Game state updates"
        ),
        
        # Agent-specific topics
        "arena.agent.actions": TopicConfig(
            name="arena.agent.actions",
            category=TopicCategory.AGENT,
            partitions=3,
            description="Agent actions and decisions"
        ),
        
        "arena.agent.lifecycle": TopicConfig(
            name="arena.agent.lifecycle",
            category=TopicCategory.AGENT,
            partitions=3,
            description="Agent lifecycle events (join, elimination, etc.)"
        ),
        
        # System topics
        "arena.system.orchestration": TopicConfig(
            name="arena.system.orchestration",
            category=TopicCategory.SYSTEM,
            partitions=1,
            description="Orchestrator commands and state machine events"
        ),
        
        "arena.system.narrator": TopicConfig(
            name="arena.system.narrator",
            category=TopicCategory.SYSTEM,
            partitions=1,
            description="Narrator summaries and commentary"
        ),
        
        "arena.system.errors": TopicConfig(
            name="arena.system.errors",
            category=TopicCategory.SYSTEM,
            partitions=1,
            retention_ms=2592000000,  # 30 days for errors
            description="System errors and exceptions"
        ),
        
        # Scoring topics
        "arena.scoring.metrics": TopicConfig(
            name="arena.scoring.metrics",
            category=TopicCategory.SCORING,
            partitions=3,
            description="Scoring metrics for contributions"
        ),
        
        "arena.scoring.leaderboard": TopicConfig(
            name="arena.scoring.leaderboard",
            category=TopicCategory.SCORING,
            partitions=1,
            cleanup_policy="compact",
            description="Current leaderboard state"
        ),
        
        # Accusation topics
        "arena.accusation.claims": TopicConfig(
            name="arena.accusation.claims",
            category=TopicCategory.ACCUSATION,
            partitions=2,
            description="Cheating accusations"
        ),
        
        "arena.accusation.verdicts": TopicConfig(
            name="arena.accusation.verdicts",
            category=TopicCategory.ACCUSATION,
            partitions=1,
            description="Judge verdicts on accusations"
        ),
        
        # Debug topics
        "arena.debug.trace": TopicConfig(
            name="arena.debug.trace",
            category=TopicCategory.DEBUG,
            partitions=1,
            retention_ms=86400000,  # 1 day for debug
            description="Debug trace messages"
        )
    }
    
    def __init__(self, settings: Optional[ArenaSettings] = None):
        """
        Initialize the topic manager.
        
        Args:
            settings: Arena settings (uses default if not provided)
        """
        self.settings = settings or ArenaSettings()
        self.admin_client: Optional[KafkaAdminClient] = None
    
    def get_admin_client(self) -> KafkaAdminClient:
        """
        Get or create Kafka admin client.
        
        Returns:
            KafkaAdminClient instance
        """
        if self.admin_client is None:
            self.admin_client = KafkaAdminClient(
                bootstrap_servers=self.settings.kafka_bootstrap_servers,
                client_id="arena_topic_manager"
            )
        return self.admin_client
    
    def create_all_topics(self, dry_run: bool = False) -> Dict[str, bool]:
        """
        Create all defined topics.
        
        Args:
            dry_run: If True, only log what would be created
            
        Returns:
            Dictionary mapping topic names to creation success
        """
        results = {}
        
        for topic_name, config in self.TOPICS.items():
            if dry_run:
                logger.info(f"[DRY RUN] Would create topic: {topic_name}")
                results[topic_name] = True
            else:
                success = self.create_topic(config)
                results[topic_name] = success
        
        return results
    
    def create_topic(self, config: TopicConfig) -> bool:
        """
        Create a single topic.
        
        Args:
            config: Topic configuration
            
        Returns:
            True if topic was created or already exists
        """
        try:
            admin = self.get_admin_client()
            
            new_topic = NewTopic(
                name=config.name,
                num_partitions=config.partitions,
                replication_factor=config.replication_factor,
                topic_configs=config.to_kafka_config()
            )
            
            admin.create_topics([new_topic], validate_only=False)
            logger.info(f"Created topic: {config.name}")
            return True
            
        except TopicAlreadyExistsError:
            logger.info(f"Topic already exists: {config.name}")
            return True
            
        except KafkaError as e:
            logger.error(f"Failed to create topic {config.name}: {e}")
            return False
            
        except Exception as e:
            logger.error(f"Unexpected error creating topic {config.name}: {e}")
            return False
    
    def delete_topic(self, topic_name: str) -> bool:
        """
        Delete a topic.
        
        Args:
            topic_name: Name of topic to delete
            
        Returns:
            True if topic was deleted
        """
        try:
            admin = self.get_admin_client()
            admin.delete_topics([topic_name])
            logger.info(f"Deleted topic: {topic_name}")
            return True
            
        except KafkaError as e:
            logger.error(f"Failed to delete topic {topic_name}: {e}")
            return False
    
    def list_topics(self) -> List[str]:
        """
        List all existing topics.
        
        Returns:
            List of topic names
        """
        try:
            admin = self.get_admin_client()
            metadata = admin.list_topics()
            return list(metadata)
            
        except Exception as e:
            logger.error(f"Failed to list topics: {e}")
            return []
    
    def validate_topics(self) -> Dict[str, bool]:
        """
        Validate that all required topics exist.
        
        Returns:
            Dictionary mapping topic names to existence status
        """
        existing_topics = set(self.list_topics())
        
        results = {}
        for topic_name in self.TOPICS.keys():
            exists = topic_name in existing_topics
            results[topic_name] = exists
            
            if not exists:
                logger.warning(f"Required topic does not exist: {topic_name}")
        
        return results
    
    def get_topic_config(self, topic_name: str) -> Optional[TopicConfig]:
        """
        Get configuration for a topic.
        
        Args:
            topic_name: Topic name
            
        Returns:
            TopicConfig or None if not found
        """
        return self.TOPICS.get(topic_name)
    
    def get_topics_by_category(self, category: TopicCategory) -> List[str]:
        """
        Get all topics in a category.
        
        Args:
            category: Topic category
            
        Returns:
            List of topic names in the category
        """
        return [
            name for name, config in self.TOPICS.items()
            if config.category == category
        ]
    
    def get_game_topic(self, game_id: str, base_topic: str) -> str:
        """
        Get game-specific topic name.
        
        Args:
            game_id: Game ID
            base_topic: Base topic name
            
        Returns:
            Game-specific topic name
        """
        return f"{base_topic}.{game_id}"
    
    def close(self) -> None:
        """Close the admin client."""
        if self.admin_client:
            self.admin_client.close()
            self.admin_client = None
            logger.info("Topic manager closed")


# Convenience functions
def get_contribution_topic() -> str:
    """Get the main contribution topic name."""
    return "arena.game.contributions"


def get_turn_topic() -> str:
    """Get the turn management topic name."""
    return "arena.game.turns"


def get_scoring_topic() -> str:
    """Get the scoring metrics topic name."""
    return "arena.scoring.metrics"


def get_accusation_topic() -> str:
    """Get the accusation claims topic name."""
    return "arena.accusation.claims"


def get_system_topic() -> str:
    """Get the system orchestration topic name."""
    return "arena.system.orchestration"