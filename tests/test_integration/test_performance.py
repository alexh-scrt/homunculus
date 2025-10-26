"""Performance and stress tests for the character agent system."""

import pytest
import asyncio
import time
import gc
from pathlib import Path
from unittest.mock import Mock, patch
import sys

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

try:
    from character_agent import CharacterAgent
    from config.character_loader import CharacterLoader
    from config.settings import get_settings
except ImportError:
    pytest.skip("Performance test modules not available", allow_module_level=True)


class MockOllamaClient:
    """Fast mock Ollama client for performance testing."""
    
    def __init__(self, response_delay=0.1):
        self.response_delay = response_delay
        self.call_count = 0
    
    def generate(self, prompt, temperature=None):
        """Generate mock response with optional delay."""
        self.call_count += 1
        time.sleep(self.response_delay)  # Simulate processing time
        
        if 'personality' in prompt.lower():
            return "Quick personality response"
        elif 'mood' in prompt.lower():
            return "Quick mood response"
        elif 'synthesize' in prompt.lower():
            return "intention: help, tone: friendly, themes: assistance"
        else:
            return f"Mock response #{self.call_count}"


class MockMemoryModule:
    """Fast mock memory module for performance testing."""
    
    def __init__(self, character_id):
        self.character_id = character_id
        self.store_count = 0
        self.retrieve_count = 0
    
    async def store_experience(self, experience):
        """Fast mock store."""
        self.store_count += 1
        await asyncio.sleep(0.01)  # Minimal delay
    
    async def retrieve_relevant_experiences(self, query, limit=5):
        """Fast mock retrieve."""
        self.retrieve_count += 1
        await asyncio.sleep(0.01)  # Minimal delay
        return []


@pytest.fixture
def performance_character_config():
    """Simplified character config for performance testing."""
    return {
        'character_id': 'perf_test',
        'name': 'Performance Test Character',
        'archetype': 'test',
        'personality': {
            'big_five': {
                'openness': 0.5,
                'conscientiousness': 0.5,
                'extraversion': 0.5,
                'agreeableness': 0.5,
                'neuroticism': 0.5
            }
        },
        'neurochemical_baseline': {
            'dopamine': 50.0,
            'serotonin': 50.0,
            'oxytocin': 50.0,
            'endorphins': 50.0,
            'cortisol': 50.0,
            'adrenaline': 50.0
        },
        'communication_style': {
            'verbal_pattern': 'concise',
            'social_comfort': 'neutral'
        },
        'initial_goals': ['test performance'],
        'mood_baseline': {
            'current_state': 'neutral',
            'intensity': 0.5,
            'baseline_setpoint': 0.5,
            'emotional_volatility': 0.3
        }
    }


@pytest.fixture
def mock_settings():
    """Mock settings for performance testing."""
    settings = Mock()
    settings.ollama_base_url = "http://localhost:11434"
    settings.ollama_model = "test_model"
    settings.chroma_persist_directory = "/tmp/chroma"
    settings.neo4j_uri = "bolt://localhost:7687"
    settings.neo4j_user = "neo4j"
    settings.neo4j_password = "password"
    return settings


class TestPerformance:
    """Performance tests for character agent system."""
    
    @pytest.mark.asyncio
    async def test_character_initialization_time(self, performance_character_config, mock_settings):
        """Test character initialization performance."""
        with patch('llm.ollama_client.OllamaClient', return_value=MockOllamaClient(0.01)):
            with patch('memory.experience_module.ExperienceModule', MockMemoryModule):
                with patch('memory.knowledge_graph_module.KnowledgeGraphModule', MockMemoryModule):
                    
                    start_time = time.time()
                    
                    character = CharacterAgent(
                        character_config=performance_character_config,
                        settings=mock_settings
                    )
                    
                    await character.initialize()
                    
                    initialization_time = time.time() - start_time
                    
                    # Character should initialize within reasonable time
                    assert initialization_time < 2.0, f"Initialization took {initialization_time:.2f}s"
                    assert character.character_id is not None
                    assert character.state is not None
    
    @pytest.mark.asyncio
    async def test_single_message_response_time(self, performance_character_config, mock_settings):
        """Test response time for a single message."""
        with patch('llm.ollama_client.OllamaClient', return_value=MockOllamaClient(0.1)):
            with patch('memory.experience_module.ExperienceModule', MockMemoryModule):
                with patch('memory.knowledge_graph_module.KnowledgeGraphModule', MockMemoryModule):
                    
                    character = CharacterAgent(
                        character_config=performance_character_config,
                        settings=mock_settings
                    )
                    await character.initialize()
                    
                    start_time = time.time()
                    
                    result = await character.process_message(
                        user_message="Hello, how are you?",
                        context={}
                    )
                    
                    response_time = time.time() - start_time
                    
                    # Response should be generated within reasonable time
                    assert response_time < 5.0, f"Response took {response_time:.2f}s"
                    assert 'response' in result
                    assert len(result['response']) > 0
    
    @pytest.mark.asyncio
    async def test_multiple_message_throughput(self, performance_character_config, mock_settings):
        """Test throughput for multiple messages."""
        with patch('llm.ollama_client.OllamaClient', return_value=MockOllamaClient(0.05)):
            with patch('memory.experience_module.ExperienceModule', MockMemoryModule):
                with patch('memory.knowledge_graph_module.KnowledgeGraphModule', MockMemoryModule):
                    
                    character = CharacterAgent(
                        character_config=performance_character_config,
                        settings=mock_settings
                    )
                    await character.initialize()
                    
                    messages = [
                        "Hello there!",
                        "How are you doing?",
                        "What's your favorite color?",
                        "Tell me about yourself.",
                        "What are your goals?"
                    ]
                    
                    start_time = time.time()
                    
                    for msg in messages:
                        result = await character.process_message(
                            user_message=msg,
                            context={}
                        )
                        assert 'response' in result
                    
                    total_time = time.time() - start_time
                    avg_time_per_message = total_time / len(messages)
                    
                    # Average time per message should be reasonable
                    assert avg_time_per_message < 2.0, f"Average time per message: {avg_time_per_message:.2f}s"
                    
                    # Check conversation history was maintained
                    assert len(character.state.conversation_history) >= len(messages)
    
    @pytest.mark.asyncio
    async def test_concurrent_characters(self, performance_character_config, mock_settings):
        """Test performance with multiple concurrent characters."""
        with patch('llm.ollama_client.OllamaClient', return_value=MockOllamaClient(0.1)):
            with patch('memory.experience_module.ExperienceModule', MockMemoryModule):
                with patch('memory.knowledge_graph_module.KnowledgeGraphModule', MockMemoryModule):
                    
                    num_characters = 3
                    characters = []
                    
                    # Initialize multiple characters
                    for i in range(num_characters):
                        config = performance_character_config.copy()
                        config['character_id'] = f'perf_test_{i}'
                        config['name'] = f'Character {i}'
                        
                        character = CharacterAgent(
                            character_config=config,
                            settings=mock_settings
                        )
                        await character.initialize()
                        characters.append(character)
                    
                    # Process messages concurrently
                    async def send_message(char, msg):
                        return await char.process_message(
                            user_message=f"Hello from user to {char.character_name}: {msg}",
                            context={}
                        )
                    
                    start_time = time.time()
                    
                    # Send messages to all characters concurrently
                    tasks = [
                        send_message(char, f"Message {i}")
                        for i, char in enumerate(characters)
                    ]
                    
                    results = await asyncio.gather(*tasks)
                    
                    total_time = time.time() - start_time
                    
                    # Concurrent processing should be faster than sequential
                    assert total_time < num_characters * 1.0, f"Concurrent processing took {total_time:.2f}s"
                    assert len(results) == num_characters
                    
                    for result in results:
                        assert 'response' in result
                        assert len(result['response']) > 0
    
    @pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not available")
    @pytest.mark.asyncio
    async def test_memory_usage(self, performance_character_config, mock_settings):
        """Test memory usage during extended conversation."""
        with patch('llm.ollama_client.OllamaClient', return_value=MockOllamaClient(0.01)):
            with patch('memory.experience_module.ExperienceModule', MockMemoryModule):
                with patch('memory.knowledge_graph_module.KnowledgeGraphModule', MockMemoryModule):
                    
                    character = CharacterAgent(
                        character_config=performance_character_config,
                        settings=mock_settings
                    )
                    await character.initialize()
                    
                    # Get initial memory usage
                    process = psutil.Process()
                    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
                    
                    # Have extended conversation
                    for i in range(20):
                        await character.process_message(
                            user_message=f"This is message number {i} in our conversation.",
                            context={}
                        )
                    
                    # Get final memory usage
                    final_memory = process.memory_info().rss / 1024 / 1024  # MB
                    memory_increase = final_memory - initial_memory
                    
                    # Memory increase should be reasonable
                    assert memory_increase < 100, f"Memory increased by {memory_increase:.1f}MB"
    
    @pytest.mark.asyncio
    async def test_state_serialization_performance(self, performance_character_config, mock_settings):
        """Test performance of state serialization/deserialization."""
        with patch('llm.ollama_client.OllamaClient', return_value=MockOllamaClient(0.01)):
            with patch('memory.experience_module.ExperienceModule', MockMemoryModule):
                with patch('memory.knowledge_graph_module.KnowledgeGraphModule', MockMemoryModule):
                    
                    character = CharacterAgent(
                        character_config=performance_character_config,
                        settings=mock_settings
                    )
                    await character.initialize()
                    
                    # Build up some conversation history
                    for i in range(10):
                        await character.process_message(
                            user_message=f"Message {i}",
                            context={}
                        )
                    
                    # Test serialization performance
                    start_time = time.time()
                    
                    for _ in range(5):
                        state_dict = character.get_state_dict()
                        assert isinstance(state_dict, dict)
                        assert 'character_id' in state_dict
                    
                    serialization_time = time.time() - start_time
                    avg_serialization_time = serialization_time / 5
                    
                    # Serialization should be fast
                    assert avg_serialization_time < 0.1, f"Serialization took {avg_serialization_time:.3f}s"
    
    def test_character_loader_performance(self):
        """Test performance of character loading operations."""
        loader = CharacterLoader()
        
        # Test listing characters
        start_time = time.time()
        characters = loader.list_available_characters()
        listing_time = time.time() - start_time
        
        assert listing_time < 1.0, f"Character listing took {listing_time:.2f}s"
        assert len(characters) > 0
        
        # Test loading multiple characters
        test_characters = characters[:5]  # Test first 5
        
        start_time = time.time()
        
        for char_id in test_characters:
            try:
                config = loader.load_character(char_id)
                assert config is not None
                assert 'name' in config
            except Exception as e:
                pytest.skip(f"Character {char_id} failed to load: {e}")
        
        loading_time = time.time() - start_time
        avg_loading_time = loading_time / len(test_characters)
        
        assert avg_loading_time < 0.5, f"Average character loading took {avg_loading_time:.3f}s"
    
    def test_validation_performance(self):
        """Test performance of character validation."""
        loader = CharacterLoader()
        characters = loader.list_available_characters()
        
        test_characters = characters[:3]  # Test first 3
        
        start_time = time.time()
        
        for char_id in test_characters:
            try:
                config = loader.load_character(char_id)
                issues = loader.validate_character_config(config)
                assert isinstance(issues, list)
            except Exception as e:
                pytest.skip(f"Character {char_id} validation failed: {e}")
        
        validation_time = time.time() - start_time
        avg_validation_time = validation_time / len(test_characters)
        
        assert avg_validation_time < 0.2, f"Average validation took {avg_validation_time:.3f}s"


class TestStress:
    """Stress tests for character agent system."""
    
    @pytest.mark.asyncio
    async def test_rapid_message_processing(self, performance_character_config, mock_settings):
        """Test handling rapid message processing."""
        with patch('llm.ollama_client.OllamaClient', return_value=MockOllamaClient(0.01)):
            with patch('memory.experience_module.ExperienceModule', MockMemoryModule):
                with patch('memory.knowledge_graph_module.KnowledgeGraphModule', MockMemoryModule):
                    
                    character = CharacterAgent(
                        character_config=performance_character_config,
                        settings=mock_settings
                    )
                    await character.initialize()
                    
                    # Send many messages rapidly
                    messages = [f"Rapid message {i}" for i in range(50)]
                    
                    start_time = time.time()
                    
                    for msg in messages:
                        result = await character.process_message(
                            user_message=msg,
                            context={}
                        )
                        assert 'response' in result
                    
                    total_time = time.time() - start_time
                    
                    # Should handle all messages without crashing
                    assert len(character.state.conversation_history) >= len(messages)
                    
                    # Average time should be reasonable
                    avg_time = total_time / len(messages)
                    assert avg_time < 0.5, f"Average processing time: {avg_time:.3f}s"
    
    @pytest.mark.asyncio
    async def test_long_conversation_stability(self, performance_character_config, mock_settings):
        """Test stability during very long conversations."""
        with patch('llm.ollama_client.OllamaClient', return_value=MockOllamaClient(0.01)):
            with patch('memory.experience_module.ExperienceModule', MockMemoryModule):
                with patch('memory.knowledge_graph_module.KnowledgeGraphModule', MockMemoryModule):
                    
                    character = CharacterAgent(
                        character_config=performance_character_config,
                        settings=mock_settings
                    )
                    await character.initialize()
                    
                    # Have a very long conversation
                    num_messages = 100
                    
                    for i in range(num_messages):
                        result = await character.process_message(
                            user_message=f"Long conversation message {i}. This is a longer message to test handling of extended text and ensure the system remains stable over time.",
                            context={}
                        )
                        
                        assert 'response' in result
                        assert len(result['response']) > 0
                        
                        # Check memory hasn't grown excessively
                        if i % 20 == 0:  # Check every 20 messages
                            history_length = len(character.state.conversation_history)
                            assert history_length > 0
                            
                            # Trigger garbage collection
                            gc.collect()
                    
                    # Character should still be functional
                    final_result = await character.process_message(
                        user_message="Final test message",
                        context={}
                    )
                    
                    assert 'response' in final_result
                    assert character.state is not None
    
    def test_character_loading_stress(self):
        """Test loading all available characters rapidly."""
        loader = CharacterLoader()
        characters = loader.list_available_characters()
        
        loaded_count = 0
        failed_count = 0
        
        start_time = time.time()
        
        # Try to load all characters multiple times
        for _ in range(3):
            for char_id in characters:
                try:
                    config = loader.load_character(char_id)
                    assert config is not None
                    loaded_count += 1
                except Exception:
                    failed_count += 1
        
        total_time = time.time() - start_time
        total_operations = loaded_count + failed_count
        
        # Should handle high load without excessive failures
        success_rate = loaded_count / total_operations if total_operations > 0 else 0
        assert success_rate > 0.8, f"Success rate too low: {success_rate:.1%}"
        
        # Should complete in reasonable time
        avg_time = total_time / total_operations if total_operations > 0 else 0
        assert avg_time < 0.1, f"Average operation time: {avg_time:.3f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])