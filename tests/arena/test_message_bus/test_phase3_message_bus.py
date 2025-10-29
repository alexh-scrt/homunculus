"""
Comprehensive Unit Tests for Arena Phase 3 Message Bus

This test suite validates all message bus components created in Phase 3:
- Kafka producer with retry logic
- Kafka consumer with offset management
- Message serialization and deserialization
- Topic configuration and management
- Message routing
- Event handlers
- Event dispatcher

Author: Homunculus Team
"""

import pytest
import json
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch, call
from typing import Dict, Any

# Import message bus components
from src.arena.message_bus import (
    ArenaKafkaProducer,
    ArenaKafkaConsumer,
    ConsumeMode,
    MessageRouter,
    Route,
    RouteType,
    Subscription,
    EventDispatcher,
    EventContext,
    EventHandler,
    ContributionHandler,
    AccusationHandler,
    ArenaTopics,
    TopicConfig,
    TopicCategory,
    MessageSerializer,
    SerializationFormat
)

# Import models
from src.arena.models import Message, MessageBatch, AgentState, ArenaState, ScoringMetrics


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
                content=f"Message {i}"
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
    
    def test_custom_json_encoder(self):
        """Test custom JSON encoder for special types."""
        from src.arena.message_bus.serialization import ArenaJSONEncoder
        
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
    
    @patch('src.arena.message_bus.topics.KafkaAdminClient')
    def test_create_topics(self, mock_admin_class):
        """Test topic creation."""
        mock_admin = MagicMock()
        mock_admin_class.return_value = mock_admin
        
        topics = ArenaTopics()
        topics.create_topic(TopicConfig(
            name="test_topic",
            category=TopicCategory.DEBUG,
            partitions=2
        ))
        
        # Should create topic through admin client
        mock_admin.create_topics.assert_called_once()


class TestKafkaProducer:
    """Test Kafka producer functionality."""
    
    @patch('src.arena.message_bus.kafka_producer.KafkaProducer')
    def test_producer_initialization(self, mock_kafka_producer):
        """Test producer initialization."""
        producer = ArenaKafkaProducer()
        
        # Should initialize Kafka producer with correct config
        mock_kafka_producer.assert_called_once()
        call_kwargs = mock_kafka_producer.call_args[1]
        assert call_kwargs['acks'] == 'all'
        assert call_kwargs['retries'] == 3
        assert call_kwargs['compression_type'] == 'gzip'
    
    @patch('src.arena.message_bus.kafka_producer.KafkaProducer')
    def test_send_message(self, mock_kafka_producer):
        """Test sending a message."""
        mock_producer = MagicMock()
        mock_kafka_producer.return_value = mock_producer
        
        # Mock successful send
        mock_future = MagicMock()
        mock_future.get.return_value = MagicMock(
            topic="test_topic",
            partition=0,
            offset=100,
            timestamp=1234567890
        )
        mock_producer.send.return_value = mock_future
        
        producer = ArenaKafkaProducer()
        message = Message(sender_id="test", content="test message")
        
        success = producer.send_message("test_topic", message)
        
        assert success is True
        mock_producer.send.assert_called_once()
        assert len(producer.delivery_reports) == 1
    
    @patch('src.arena.message_bus.kafka_producer.KafkaProducer')
    def test_send_batch(self, mock_kafka_producer):
        """Test sending a batch of messages."""
        mock_producer = MagicMock()
        mock_kafka_producer.return_value = mock_producer
        
        # Mock successful sends
        mock_future = MagicMock()
        mock_future.get.return_value = MagicMock(
            topic="test_topic",
            partition=0,
            offset=100,
            timestamp=1234567890
        )
        mock_producer.send.return_value = mock_future
        
        producer = ArenaKafkaProducer()
        
        batch = MessageBatch(game_id="test_game")
        for i in range(3):
            batch.add_message(Message(
                sender_id=f"agent_{i}",
                content=f"Message {i}"
            ))
        
        results = producer.send_batch("test_topic", batch)
        
        assert len(results) == 3
        assert all(results.values())  # All should succeed
        assert mock_producer.send.call_count == 3
    
    @patch('src.arena.message_bus.kafka_producer.KafkaProducer')
    def test_retry_on_failure(self, mock_kafka_producer):
        """Test retry logic on send failure."""
        from kafka.errors import KafkaTimeoutError
        
        mock_producer = MagicMock()
        mock_kafka_producer.return_value = mock_producer
        
        # Mock failure then success
        mock_future = MagicMock()
        mock_future.get.side_effect = [
            KafkaTimeoutError("Timeout"),
            MagicMock(topic="test", partition=0, offset=100, timestamp=123)
        ]
        mock_producer.send.return_value = mock_future
        
        producer = ArenaKafkaProducer()
        message = Message(sender_id="test", content="test")
        
        success = producer.send_message("test_topic", message, retry_count=3)
        
        assert success is True
        assert mock_producer.send.call_count == 2  # Initial + 1 retry


class TestKafkaConsumer:
    """Test Kafka consumer functionality."""
    
    @patch('src.arena.message_bus.kafka_consumer.KafkaConsumer')
    def test_consumer_initialization(self, mock_kafka_consumer):
        """Test consumer initialization."""
        consumer = ArenaKafkaConsumer(
            topics=["test_topic"],
            group_id="test_group"
        )
        
        # Should initialize Kafka consumer with correct config
        mock_kafka_consumer.assert_called_once()
        call_args = mock_kafka_consumer.call_args
        assert "test_topic" in call_args[0]
        assert call_args[1]['group_id'] == "test_group"
        assert call_args[1]['enable_auto_commit'] is False
    
    @patch('src.arena.message_bus.kafka_consumer.KafkaConsumer')
    def test_consume_messages(self, mock_kafka_consumer):
        """Test consuming messages."""
        mock_consumer = MagicMock()
        mock_kafka_consumer.return_value = mock_consumer
        
        # Mock poll returning messages
        mock_record = MagicMock()
        mock_record.value = Message(sender_id="test", content="test message")
        mock_record.topic = "test_topic"
        mock_record.partition = 0
        mock_record.offset = 100
        mock_record.timestamp = 1234567890
        mock_record.key = "test_key"
        
        mock_consumer.poll.return_value = {
            MagicMock(): [mock_record]
        }
        
        consumer = ArenaKafkaConsumer(
            topics=["test_topic"],
            group_id="test_group"
        )
        
        messages = list(consumer.consume_messages(max_messages=1))
        
        assert len(messages) == 1
        assert messages[0].sender_id == "test"
        assert consumer.consumed_count == 1
    
    @patch('src.arena.message_bus.kafka_consumer.KafkaConsumer')
    def test_consume_batch(self, mock_kafka_consumer):
        """Test consuming a batch of messages."""
        mock_consumer = MagicMock()
        mock_kafka_consumer.return_value = mock_consumer
        
        # Mock multiple messages
        mock_records = []
        for i in range(3):
            record = MagicMock()
            record.value = Message(sender_id=f"agent_{i}", content=f"Message {i}")
            record.topic = "test_topic"
            record.partition = 0
            record.offset = 100 + i
            record.timestamp = 1234567890
            record.key = f"key_{i}"
            mock_records.append(record)
        
        mock_consumer.poll.return_value = {
            MagicMock(): mock_records
        }
        
        consumer = ArenaKafkaConsumer(
            topics=["test_topic"],
            group_id="test_group"
        )
        
        batch = consumer.consume_batch(batch_size=3)
        
        assert batch is not None
        assert batch.size == 3
        assert batch.messages[0].sender_id == "agent_0"


class TestMessageRouter:
    """Test message routing functionality."""
    
    def test_add_route(self):
        """Test adding a route."""
        router = MessageRouter()
        
        route = Route(
            route_id="test_route",
            source_pattern=r"arena\.game\..*",
            destination_topics=["arena.scoring.metrics"]
        )
        
        router.add_route(route)
        
        assert "test_route" in router.routes
        assert router.routes["test_route"] == route
    
    def test_route_matching(self):
        """Test route pattern matching."""
        route = Route(
            route_id="test_route",
            source_pattern=r"arena\.game\..*",
            destination_topics=["dest_topic"]
        )
        
        assert route.matches_source("arena.game.contributions")
        assert route.matches_source("arena.game.turns")
        assert not route.matches_source("arena.scoring.metrics")
    
    def test_route_filters(self):
        """Test route filtering."""
        route = Route(
            route_id="test_route",
            source_pattern=r".*",
            destination_topics=["dest_topic"],
            filters=[
                lambda m: m.message_type == "contribution"
            ]
        )
        
        contrib_msg = Message(sender_id="test", content="test", message_type="contribution")
        other_msg = Message(sender_id="test", content="test", message_type="status")
        
        assert route.should_route(contrib_msg)
        assert not route.should_route(other_msg)
    
    @patch('src.arena.message_bus.kafka_producer.ArenaKafkaProducer')
    def test_route_message(self, mock_producer_class):
        """Test routing a message."""
        mock_producer = MagicMock()
        mock_producer_class.return_value = mock_producer
        mock_producer.send_message.return_value = True
        
        router = MessageRouter()
        
        # Add a route
        route = Route(
            route_id="test_route",
            source_pattern=r"source_topic",
            destination_topics=["dest_topic1", "dest_topic2"]
        )
        router.add_route(route)
        
        # Route a message
        message = Message(sender_id="test", content="test")
        results = router.route_message(message, "source_topic")
        
        assert results == {"dest_topic1": True, "dest_topic2": True}
        assert mock_producer.send_message.call_count == 2


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


class TestEventDispatcher:
    """Test event dispatcher functionality."""
    
    @patch('src.arena.message_bus.event_dispatcher.ArenaKafkaConsumer')
    @patch('src.arena.message_bus.event_dispatcher.ArenaKafkaProducer')
    def test_dispatcher_initialization(self, mock_producer, mock_consumer):
        """Test dispatcher initialization."""
        dispatcher = EventDispatcher(
            topics=["test_topic"],
            group_id="test_group",
            max_workers=5
        )
        
        assert dispatcher.topics == ["test_topic"]
        assert dispatcher.group_id == "test_group"
        assert dispatcher.max_workers == 5
        assert not dispatcher.running
    
    def test_register_handler(self):
        """Test registering event handlers."""
        dispatcher = EventDispatcher(
            topics=["test_topic"],
            group_id="test_group"
        )
        
        handler = ContributionHandler()
        dispatcher.register_handler("contribution", handler)
        
        assert "contribution" in dispatcher.handlers
        assert handler in dispatcher.handlers["contribution"]
    
    def test_event_priority(self):
        """Test event prioritization."""
        dispatcher = EventDispatcher(
            topics=["test_topic"],
            group_id="test_group"
        )
        
        # Test priority assignment
        elim_msg = Message(sender_id="test", content="", message_type="elimination")
        contrib_msg = Message(sender_id="test", content="", message_type="contribution")
        debug_msg = Message(sender_id="test", content="", message_type="debug")
        
        assert dispatcher._get_event_priority(elim_msg) == 1  # Highest priority
        assert dispatcher._get_event_priority(contrib_msg) == 5
        assert dispatcher._get_event_priority(debug_msg) == 10  # Lowest priority
    
    def test_get_statistics(self):
        """Test statistics gathering."""
        dispatcher = EventDispatcher(
            topics=["test_topic"],
            group_id="test_group"
        )
        
        stats = dispatcher.get_statistics()
        
        assert "events_received" in stats
        assert "events_processed" in stats
        assert "events_failed" in stats
        assert "queue_size" in stats
        assert stats["running"] is False


class TestIntegration:
    """Integration tests for message bus components."""
    
    @patch('src.arena.message_bus.kafka_producer.KafkaProducer')
    @patch('src.arena.message_bus.kafka_consumer.KafkaConsumer')
    def test_producer_consumer_flow(self, mock_consumer_class, mock_producer_class):
        """Test end-to-end message flow."""
        # Setup mocks
        mock_producer = MagicMock()
        mock_consumer = MagicMock()
        mock_producer_class.return_value = mock_producer
        mock_consumer_class.return_value = mock_consumer
        
        # Mock successful send
        mock_future = MagicMock()
        mock_future.get.return_value = MagicMock(
            topic="test", partition=0, offset=100, timestamp=123
        )
        mock_producer.send.return_value = mock_future
        
        # Send a message
        producer = ArenaKafkaProducer()
        message = Message(sender_id="agent_001", content="Test message")
        success = producer.send_message("test_topic", message)
        assert success
        
        # Mock consuming the same message
        mock_record = MagicMock()
        mock_record.value = message
        mock_record.topic = "test_topic"
        mock_record.partition = 0
        mock_record.offset = 100
        mock_record.timestamp = 123
        mock_record.key = "agent_001"
        
        mock_consumer.poll.return_value = {MagicMock(): [mock_record]}
        
        # Consume the message
        consumer = ArenaKafkaConsumer(["test_topic"], "test_group")
        consumed = list(consumer.consume_messages(max_messages=1))
        
        assert len(consumed) == 1
        assert consumed[0].sender_id == message.sender_id
        assert consumed[0].content == message.content
    
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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])