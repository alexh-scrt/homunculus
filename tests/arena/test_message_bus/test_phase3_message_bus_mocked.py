"""
Comprehensive Unit Tests for Arena Phase 3 Message Bus (Mocked Version)

This test suite validates all message bus components with mocked Kafka dependencies.
This allows testing without requiring Kafka to be installed.

Author: Homunculus Team
"""

import pytest
import json
import sys
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch, call
from typing import Dict, Any

# Mock kafka module before imports
sys.modules['kafka'] = MagicMock()
sys.modules['kafka.admin'] = MagicMock()
sys.modules['kafka.errors'] = MagicMock()
sys.modules['kafka.structs'] = MagicMock()

# Define mock Kafka exceptions
class MockKafkaError(Exception):
    pass

class MockKafkaTimeoutError(MockKafkaError):
    pass

class MockTopicAlreadyExistsError(MockKafkaError):
    pass

class MockCommitFailedError(MockKafkaError):
    pass

# Set up mock exceptions
sys.modules['kafka.errors'].KafkaError = MockKafkaError
sys.modules['kafka.errors'].KafkaTimeoutError = MockKafkaTimeoutError
sys.modules['kafka.errors'].TopicAlreadyExistsError = MockTopicAlreadyExistsError
sys.modules['kafka.errors'].CommitFailedError = MockCommitFailedError

# Import models first (no Kafka dependencies)
from src.arena.models import Message, MessageBatch, AgentState, ArenaState, ScoringMetrics

# Import serialization (no direct Kafka dependencies)
from src.arena.message_bus.serialization import (
    MessageSerializer,
    SerializationFormat,
    ArenaJSONEncoder,
    ModelSerializer
)

# Import topics (will use mocked Kafka admin)
from src.arena.message_bus.topics import (
    ArenaTopics,
    TopicConfig,
    TopicCategory
)

# Import event handlers (no direct Kafka dependencies)
from src.arena.message_bus.event_handlers import (
    EventHandler,
    ContributionHandler,
    AccusationHandler,
    EliminationHandler,
    ScoringHandler,
    TurnHandler,
    SystemHandler
)


class TestMessageSerialization:
    """Test message serialization utilities."""
    
    def test_message_serialization(self):
        """Test serializing a message to JSON."""
        message = Message(
            sender_id="agent_001",
            sender_name="Ada",
            content="Test message",
            message_type="contribution"
        )
        
        # Serialize
        serialized = MessageSerializer.serialize(message)
        assert isinstance(serialized, bytes)
        
        # Deserialize
        deserialized = MessageSerializer.deserialize(
            serialized,
            Message
        )
        assert deserialized.sender_id == message.sender_id
        assert deserialized.content == message.content
        assert deserialized.message_id == message.message_id
    
    def test_batch_serialization(self):
        """Test batch message serialization."""
        batch = MessageBatch(game_id="test_game")
        
        for i in range(3):
            batch.add_message(Message(
                sender_id=f"agent_{i}",
                content=f"Message {i}",
                game_id="test_game"  # Must match batch game_id
            ))
        
        # Serialize
        serialized = MessageSerializer.serialize_batch(batch)
        assert isinstance(serialized, bytes)
        
        # Deserialize
        deserialized = MessageSerializer.deserialize_batch(serialized)
        assert deserialized.size == 3
        assert deserialized.messages[0].sender_id == "agent_0"
    
    def test_compressed_serialization(self):
        """Test compressed serialization."""
        message = Message(
            sender_id="agent_001",
            content="A" * 1000  # Large content
        )
        
        # Normal serialization
        normal = MessageSerializer.serialize(message)
        
        # Compressed serialization
        compressed = MessageSerializer.serialize(
            message,
            format=SerializationFormat.JSON_COMPRESSED
        )
        
        # Compressed should be smaller
        assert len(compressed) < len(normal)
        
        # Should deserialize correctly
        deserialized = MessageSerializer.deserialize(
            compressed,
            Message,
            format=SerializationFormat.JSON_COMPRESSED
        )
        assert deserialized.content == message.content
    
    def test_model_serializer(self):
        """Test generic model serialization."""
        agent = AgentState("agent_001", "Ada", {"test": "data"})
        
        # Serialize with type info
        serialized = ModelSerializer.serialize(agent, include_type=True)
        assert isinstance(serialized, bytes)
        
        # Deserialize with auto-detection
        deserialized = ModelSerializer.deserialize(serialized)
        assert deserialized.agent_id == agent.agent_id
        assert deserialized.character_name == agent.character_name
    
    def test_custom_json_encoder(self):
        """Test custom JSON encoder for special types."""
        data = {
            "datetime": datetime.utcnow(),
            "message": Message(sender_id="test", content="test"),
            "set": {1, 2, 3}
        }
        
        # Should encode without errors
        encoded = json.dumps(data, cls=ArenaJSONEncoder)
        assert isinstance(encoded, str)
        
        # Should contain ISO format datetime
        assert "T" in encoded  # ISO format indicator


class TestTopicConfiguration:
    """Test topic configuration and management."""
    
    def test_topic_definitions(self):
        """Test that all required topics are defined."""
        topics = ArenaTopics()
        
        # Check key topics exist
        assert "arena.game.contributions" in topics.TOPICS
        assert "arena.game.turns" in topics.TOPICS
        assert "arena.scoring.metrics" in topics.TOPICS
        assert "arena.accusation.claims" in topics.TOPICS
        
        # Check topic configurations
        contrib_topic = topics.TOPICS["arena.game.contributions"]
        assert contrib_topic.partitions == 6
        assert contrib_topic.category == TopicCategory.GAME
    
    def test_topic_config_to_kafka(self):
        """Test converting topic config to Kafka configuration."""
        config = TopicConfig(
            name="test_topic",
            category=TopicCategory.GAME,
            retention_ms=1000000,
            compression_type="snappy"
        )
        
        kafka_config = config.to_kafka_config()
        
        assert kafka_config["retention.ms"] == "1000000"
        assert kafka_config["compression.type"] == "snappy"
        assert "cleanup.policy" in kafka_config
    
    def test_get_topics_by_category(self):
        """Test filtering topics by category."""
        topics = ArenaTopics()
        
        game_topics = topics.get_topics_by_category(TopicCategory.GAME)
        assert "arena.game.contributions" in game_topics
        assert "arena.game.turns" in game_topics
        
        scoring_topics = topics.get_topics_by_category(TopicCategory.SCORING)
        assert "arena.scoring.metrics" in scoring_topics
    
    def test_game_specific_topics(self):
        """Test game-specific topic naming."""
        topics = ArenaTopics()
        
        game_topic = topics.get_game_topic("game_123", "arena.game.contributions")
        assert game_topic == "arena.game.contributions.game_123"


class TestEventHandlers:
    """Test event handler implementations."""
    
    def test_contribution_handler(self):
        """Test contribution message handler."""
        handler = ContributionHandler()
        
        # Create test message
        message = Message(
            sender_id="agent_001",
            sender_name="Ada",
            content="Solution proposal",
            message_type="contribution"
        )
        
        # Create context with arena state
        arena_state = ArenaState()
        agent = AgentState("agent_001", "Ada", {})
        arena_state.add_agent(agent)
        arena_state.start_game()
        
        context = {"arena_state": arena_state}
        
        # Test handler
        assert handler.can_handle(message)
        result = handler.handle(message, context)
        
        assert result["status"] == "accepted"
        assert result["request_scoring"] is True
        assert agent.turns_taken == 1
    
    def test_accusation_handler(self):
        """Test accusation message handler."""
        handler = AccusationHandler()
        
        # Create test message
        message = Message(
            sender_id="agent_001",
            sender_name="Ada",
            content="Agent 002 is cheating",
            message_type="accusation",
            target_agent_id="agent_002",
            metadata={
                "accusation": True,
                "accused_name": "Bob",
                "accusation_type": "manipulation"
            }
        )
        
        # Create context
        arena_state = ArenaState()
        accuser = AgentState("agent_001", "Ada", {})
        accused = AgentState("agent_002", "Bob", {})
        arena_state.add_agent(accuser)
        arena_state.add_agent(accused)
        arena_state.start_game()
        
        context = {"arena_state": arena_state}
        
        # Test handler
        assert handler.can_handle(message)
        result = handler.handle(message, context)
        
        assert result["status"] == "accepted"
        assert result["request_judge"] is True
        assert len(arena_state.accusation_history) == 1
    
    def test_elimination_handler(self):
        """Test elimination message handler."""
        handler = EliminationHandler()
        
        # Create test message
        message = Message(
            sender_id="system",
            sender_type="system",
            content="Poor performance",
            message_type="elimination",
            target_agent_id="agent_001"
        )
        
        # Create context
        arena_state = ArenaState()
        agent = AgentState("agent_001", "Ada", {})
        arena_state.add_agent(agent)
        arena_state.start_game()
        
        context = {"arena_state": arena_state}
        
        # Test handler
        assert handler.can_handle(message)
        result = handler.handle(message, context)
        
        assert result["status"] == "success"
        assert result["eliminated_agent"] == "agent_001"
        assert result["remaining_agents"] == 0
        assert agent.is_eliminated
    
    def test_scoring_handler(self):
        """Test scoring message handler."""
        handler = ScoringHandler()
        
        # Create test message
        metrics = ScoringMetrics(
            agent_id="agent_001",
            novelty=0.8,
            builds_on_others=0.6
        )
        
        message = Message(
            sender_id="judge",
            sender_type="judge",
            content="Score update",
            message_type="scoring",
            target_agent_id="agent_001",
            metadata={
                "metrics": metrics.to_dict(),
                "agent_id": "agent_001"
            }
        )
        
        # Create context
        arena_state = ArenaState()
        agent = AgentState("agent_001", "Ada", {})
        arena_state.add_agent(agent)
        
        context = {"arena_state": arena_state}
        
        # Test handler
        assert handler.can_handle(message)
        result = handler.handle(message, context)
        
        assert result["status"] == "success"
        assert agent.score > 0  # Score should be updated
    
    def test_turn_handler(self):
        """Test turn management handler."""
        handler = TurnHandler()
        
        # Test turn start
        start_msg = Message(
            sender_id="system",
            sender_type="system",
            message_type="turn_start",
            content="Starting turn"  # Content is required
        )
        
        arena_state = ArenaState()
        context = {"arena_state": arena_state}
        
        assert handler.can_handle(start_msg)
        result = handler.handle(start_msg, context)
        
        assert result["status"] == "success"
        assert result["current_turn"] == 1
        
        # Test turn selection
        select_msg = Message(
            sender_id="system",
            sender_type="system",
            message_type="turn_selection",
            metadata={"selected_agent": "agent_001"}
        )
        
        result = handler.handle(select_msg, context)
        assert result["selected_agent"] == "agent_001"
    
    def test_system_handler(self):
        """Test system message handler."""
        handler = SystemHandler()
        
        # Test error handling
        error_msg = Message(
            sender_id="system",
            sender_type="system",
            message_type="error",
            content="Test error",
            metadata={"severity": "high"}
        )
        
        assert handler.can_handle(error_msg)
        result = handler.handle(error_msg, {})
        
        assert result["status"] == "error_logged"
        assert result["severity"] == "high"
        
        # Test health check
        health_msg = Message(
            sender_id="system",
            sender_type="system",
            message_type="health_check",
            content="Health check"  # Content is required
        )
        
        result = handler.handle(health_msg, {})
        assert result["status"] == "healthy"
    
    def test_handler_lifecycle(self):
        """Test handler pre/post processing."""
        handler = ContributionHandler()
        
        message = Message(
            sender_id="test",
            content="test",
            message_type="contribution"
        )
        
        arena_state = ArenaState()
        agent = AgentState("test", "Test", {})
        arena_state.add_agent(agent)
        arena_state.start_game()
        
        context = {"arena_state": arena_state}
        
        # Pre-process should return True
        assert handler.pre_process(message, context)
        
        # Handle message
        result = handler.handle(message, context)
        
        # Post-process should update stats
        handler.post_process(message, result, context)
        assert handler.processed_count == 1
        assert handler.last_processed is not None
        
        # Error handling
        handler.handle_error(message, Exception("Test error"), context)
        assert handler.error_count == 1


class TestIntegration:
    """Integration tests for message bus components."""
    
    def test_serialization_roundtrip(self):
        """Test complete serialization roundtrip."""
        # Create complex message with metadata
        original = Message(
            sender_id="agent_001",
            sender_name="Ada",
            content="Complex message",
            message_type="contribution",
            turn_number=5,
            game_id="game_123",
            metadata={"score": 0.8, "tags": ["novel", "builds"]}
        )
        original.add_reference("ref_001")
        original.add_reference("ref_002")
        
        # Serialize
        serialized = MessageSerializer.serialize(original)
        
        # Deserialize
        restored = MessageSerializer.deserialize(serialized, Message)
        
        # Verify all fields preserved
        assert restored.sender_id == original.sender_id
        assert restored.sender_name == original.sender_name
        assert restored.content == original.content
        assert restored.message_type == original.message_type
        assert restored.turn_number == original.turn_number
        assert restored.game_id == original.game_id
        assert restored.metadata == original.metadata
        assert restored.references == original.references
        assert restored.message_id == original.message_id
    
    def test_handler_chain(self):
        """Test chaining multiple handlers."""
        # Create arena state
        arena_state = ArenaState()
        agent1 = AgentState("agent_001", "Ada", {})
        agent2 = AgentState("agent_002", "Bob", {})
        arena_state.add_agent(agent1)
        arena_state.add_agent(agent2)
        arena_state.start_game()
        
        # Process contribution
        contrib_handler = ContributionHandler()
        contrib_msg = Message(
            sender_id="agent_001",
            content="Solution",
            message_type="contribution"
        )
        
        context = {"arena_state": arena_state}
        result = contrib_handler.handle(contrib_msg, context)
        assert result["request_scoring"]
        
        # Process scoring
        scoring_handler = ScoringHandler()
        scoring_msg = Message(
            sender_id="judge",
            sender_type="judge",
            message_type="scoring",
            content="Scoring update",  # Content is required
            metadata={
                "agent_id": "agent_001",
                "metrics": ScoringMetrics(
                    agent_id="agent_001",
                    novelty=0.1  # Low score
                ).to_dict()
            }
        )
        
        result = scoring_handler.handle(scoring_msg, context)
        assert agent1.score > 0
        
        # Process accusation
        accuse_handler = AccusationHandler()
        accuse_msg = Message(
            sender_id="agent_002",
            message_type="accusation",
            target_agent_id="agent_001",
            content="Cheating detected",
            metadata={
                "accusation": True,
                "accused_name": "Ada",
                "accusation_type": "manipulation"
            }
        )
        
        result = accuse_handler.handle(accuse_msg, context)
        assert result["request_judge"]
        
        # Process elimination
        elim_handler = EliminationHandler()
        elim_msg = Message(
            sender_id="system",
            sender_type="system",
            message_type="elimination",
            target_agent_id="agent_001",
            content="Proven cheating"
        )
        
        result = elim_handler.handle(elim_msg, context)
        assert result["status"] == "success"
        assert agent1.is_eliminated
        assert arena_state.agent_count == 1
        
        # Should detect single survivor
        assert result["termination_reason"] == "single_survivor"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])