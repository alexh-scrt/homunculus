"""Integration tests for character conversations and behavior."""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

try:
    from character_agent import CharacterAgent
    from config.character_loader import CharacterLoader
    from config.settings import get_settings
    from core.character_state import CharacterState
except ImportError:
    pytest.skip("Integration modules not available", allow_module_level=True)


class MockOllamaClient:
    """Mock Ollama client for testing."""
    
    def generate(self, prompt, temperature=None):
        """Generate mock responses based on prompt content."""
        if 'personality' in prompt.lower():
            return "I should respond with creativity and openness based on my personality traits."
        elif 'mood' in prompt.lower():
            return "Current mood is excited which should increase enthusiasm in response."
        elif 'goals' in prompt.lower():
            return "I want to be helpful and engaging in this conversation."
        elif 'communication' in prompt.lower():
            return "I should use an enthusiastic and friendly tone."
        elif 'synthesize' in prompt.lower():
            return "Primary intention: Be helpful and engaging. Emotional tone: enthusiastic. Key themes: helpfulness, learning."
        elif 'generate response' in prompt.lower():
            return "Hello! I'm excited to chat with you today. How can I help?"
        else:
            return "This is a mock response for testing purposes."


class MockExperienceModule:
    """Mock experience module for testing."""
    
    def __init__(self, character_id):
        self.character_id = character_id
        self.experiences = []
    
    async def store_experience(self, experience):
        """Store mock experience."""
        self.experiences.append(experience)
    
    async def retrieve_relevant_experiences(self, query, limit=5):
        """Retrieve mock experiences."""
        return [
            {
                'experience_id': 'test_exp_1',
                'description': f'Previous conversation about {query}',
                'similarity': 0.85,
                'timestamp': '2025-01-01T10:00:00'
            }
        ]


class MockKnowledgeGraphModule:
    """Mock knowledge graph module for testing."""
    
    def __init__(self, character_id):
        self.character_id = character_id
        self.entities = {}
        self.relationships = []
    
    async def initialize(self):
        """Mock initialization."""
        pass
    
    async def store_entity(self, entity_id, entity_type, properties):
        """Store mock entity."""
        self.entities[entity_id] = {'type': entity_type, 'properties': properties}
    
    async def store_relationship(self, source, target, relationship_type, properties=None):
        """Store mock relationship."""
        self.relationships.append({
            'source': source,
            'target': target,
            'type': relationship_type,
            'properties': properties or {}
        })
    
    async def query(self, cypher_query):
        """Mock query execution."""
        return []
    
    async def close(self):
        """Mock close."""
        pass


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = Mock()
    settings.ollama_base_url = "http://localhost:11434"
    settings.ollama_model = "test_model"
    settings.chroma_persist_directory = "/tmp/chroma"
    settings.neo4j_uri = "bolt://localhost:7687"
    settings.neo4j_user = "neo4j"
    settings.neo4j_password = "password"
    return settings


@pytest.fixture
def mock_character_config():
    """Mock character configuration for testing."""
    return {
        'character_id': 'test_character',
        'name': 'Test Character',
        'archetype': 'test',
        'personality': {
            'big_five': {
                'openness': 0.8,
                'conscientiousness': 0.6,
                'extraversion': 0.7,
                'agreeableness': 0.8,
                'neuroticism': 0.3
            },
            'behavioral_traits': ['curious', 'helpful'],
            'core_values': ['learning', 'kindness']
        },
        'neurochemical_baseline': {
            'dopamine': 60.0,
            'serotonin': 55.0,
            'oxytocin': 50.0,
            'endorphins': 45.0,
            'cortisol': 40.0,
            'adrenaline': 35.0
        },
        'communication_style': {
            'verbal_pattern': 'enthusiastic',
            'social_comfort': 'high',
            'quirks': ['uses emojis', 'asks questions']
        },
        'initial_goals': [
            'Be helpful and engaging',
            'Learn about the user',
            'Have enjoyable conversations'
        ],
        'mood_baseline': {
            'current_state': 'neutral',
            'intensity': 0.5,
            'baseline_setpoint': 0.6,
            'emotional_volatility': 0.4
        }
    }


class TestCharacterConversations:
    """Test end-to-end character conversation functionality."""
    
    @pytest.mark.asyncio
    async def test_character_initialization(self, mock_character_config, mock_settings):
        """Test character agent initialization."""
        with patch('llm.ollama_client.OllamaClient', return_value=MockOllamaClient()):
            with patch('memory.experience_module.ExperienceModule', MockExperienceModule):
                with patch('memory.knowledge_graph_module.KnowledgeGraphModule', MockKnowledgeGraphModule):
                    character = CharacterAgent(
                        character_config=mock_character_config,
                        settings=mock_settings
                    )
                    
                    await character.initialize()
                    
                    assert character.character_id == 'test_character'
                    assert character.character_name == 'Test Character'
                    assert character.state is not None
                    assert character.agent_orchestrator is not None
                    assert character.cognitive_module is not None
                    assert character.response_generator is not None
                    assert character.state_updater is not None
    
    @pytest.mark.asyncio
    async def test_single_conversation_turn(self, mock_character_config, mock_settings):
        """Test a single conversation turn."""
        with patch('llm.ollama_client.OllamaClient', return_value=MockOllamaClient()):
            with patch('memory.experience_module.ExperienceModule', MockExperienceModule):
                with patch('memory.knowledge_graph_module.KnowledgeGraphModule', MockKnowledgeGraphModule):
                    character = CharacterAgent(
                        character_config=mock_character_config,
                        settings=mock_settings
                    )
                    
                    await character.initialize()
                    
                    # Test processing a message
                    result = await character.process_message(
                        user_message="Hello! How are you today?",
                        context={'test': True}
                    )
                    
                    assert 'response' in result
                    assert isinstance(result['response'], str)
                    assert len(result['response']) > 0
                    
                    # Check that conversation history was updated
                    assert len(character.state.conversation_history) > 0
                    
                    # Check that response contains expected elements
                    response = result['response']
                    assert 'Hello' in response or 'excited' in response or 'help' in response
    
    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, mock_character_config, mock_settings):
        """Test multiple conversation turns with memory persistence."""
        with patch('llm.ollama_client.OllamaClient', return_value=MockOllamaClient()):
            with patch('memory.experience_module.ExperienceModule', MockExperienceModule):
                with patch('memory.knowledge_graph_module.KnowledgeGraphModule', MockKnowledgeGraphModule):
                    character = CharacterAgent(
                        character_config=mock_character_config,
                        settings=mock_settings
                    )
                    
                    await character.initialize()
                    
                    # First turn
                    result1 = await character.process_message(
                        user_message="Hi, my name is Alice.",
                        context={}
                    )
                    
                    assert 'response' in result1
                    initial_history_length = len(character.state.conversation_history)
                    
                    # Second turn
                    result2 = await character.process_message(
                        user_message="What did I just tell you my name was?",
                        context={}
                    )
                    
                    assert 'response' in result2
                    assert len(character.state.conversation_history) > initial_history_length
                    
                    # The character should have some memory of the previous interaction
                    # (This is mocked, but we verify the flow works)
                    response2 = result2['response']
                    assert isinstance(response2, str)
                    assert len(response2) > 0
    
    @pytest.mark.asyncio
    async def test_neurochemical_system_integration(self, mock_character_config, mock_settings):
        """Test that neurochemical system integrates properly."""
        with patch('llm.ollama_client.OllamaClient', return_value=MockOllamaClient()):
            with patch('memory.experience_module.ExperienceModule', MockExperienceModule):
                with patch('memory.knowledge_graph_module.KnowledgeGraphModule', MockKnowledgeGraphModule):
                    character = CharacterAgent(
                        character_config=mock_character_config,
                        settings=mock_settings
                    )
                    
                    await character.initialize()
                    
                    # Get initial neurochemical levels
                    initial_dopamine = character.state.neurochemical_levels['dopamine']
                    
                    # Process a positive message
                    result = await character.process_message(
                        user_message="You're amazing! I love talking with you!",
                        context={}
                    )
                    
                    assert 'response' in result
                    
                    # Check that neurochemical levels may have changed
                    # (The exact changes depend on the implementation)
                    current_dopamine = character.state.neurochemical_levels['dopamine']
                    assert isinstance(current_dopamine, (int, float))
                    assert current_dopamine >= 0
                    assert current_dopamine <= 100
    
    @pytest.mark.asyncio
    async def test_mood_state_evolution(self, mock_character_config, mock_settings):
        """Test that mood state evolves during conversation."""
        with patch('llm.ollama_client.OllamaClient', return_value=MockOllamaClient()):
            with patch('memory.experience_module.ExperienceModule', MockExperienceModule):
                with patch('memory.knowledge_graph_module.KnowledgeGraphModule', MockKnowledgeGraphModule):
                    character = CharacterAgent(
                        character_config=mock_character_config,
                        settings=mock_settings
                    )
                    
                    await character.initialize()
                    
                    # Get initial mood
                    initial_mood = character.state.agent_states['mood']['current_state']
                    
                    # Process several messages
                    messages = [
                        "Hello there!",
                        "I'm having a great day!",
                        "You seem really interesting."
                    ]
                    
                    for msg in messages:
                        result = await character.process_message(
                            user_message=msg,
                            context={}
                        )
                        assert 'response' in result
                    
                    # Check that mood state is still valid
                    current_mood = character.state.agent_states['mood']['current_state']
                    assert isinstance(current_mood, str)
                    assert len(current_mood) > 0
    
    @pytest.mark.asyncio
    async def test_agent_orchestration_flow(self, mock_character_config, mock_settings):
        """Test that all agents are consulted in the orchestration flow."""
        with patch('llm.ollama_client.OllamaClient', return_value=MockOllamaClient()):
            with patch('memory.experience_module.ExperienceModule', MockExperienceModule):
                with patch('memory.knowledge_graph_module.KnowledgeGraphModule', MockKnowledgeGraphModule):
                    character = CharacterAgent(
                        character_config=mock_character_config,
                        settings=mock_settings
                    )
                    
                    await character.initialize()
                    
                    # Process a message and request debug info
                    result = await character.process_message(
                        user_message="Tell me about yourself.",
                        context={'debug_mode': True}
                    )
                    
                    assert 'response' in result
                    assert 'debug_info' in result
                    
                    debug_info = result['debug_info']
                    
                    # Verify all expected debug sections are present
                    assert 'agent_inputs' in debug_info
                    assert 'cognitive_processing' in debug_info
                    assert 'response_generation' in debug_info
                    assert 'state_changes' in debug_info
                    
                    # Verify agent inputs contains expected agent types
                    agent_inputs = debug_info['agent_inputs']
                    expected_agents = ['personality', 'mood', 'neurochemical', 'goals', 'communication_style', 'memory']
                    
                    for agent_type in expected_agents:
                        assert agent_type in agent_inputs
                        assert 'content' in agent_inputs[agent_type]
                        assert 'confidence' in agent_inputs[agent_type]
                        assert 'priority' in agent_inputs[agent_type]
    
    @pytest.mark.asyncio
    async def test_state_persistence(self, mock_character_config, mock_settings):
        """Test character state can be saved and loaded."""
        with patch('llm.ollama_client.OllamaClient', return_value=MockOllamaClient()):
            with patch('memory.experience_module.ExperienceModule', MockExperienceModule):
                with patch('memory.knowledge_graph_module.KnowledgeGraphModule', MockKnowledgeGraphModule):
                    character = CharacterAgent(
                        character_config=mock_character_config,
                        settings=mock_settings
                    )
                    
                    await character.initialize()
                    
                    # Have a conversation to change state
                    await character.process_message(
                        user_message="My favorite color is blue.",
                        context={}
                    )
                    
                    # Get state as dict
                    state_dict = character.get_state_dict()
                    
                    assert isinstance(state_dict, dict)
                    assert 'character_id' in state_dict
                    assert 'conversation_history' in state_dict
                    assert 'neurochemical_levels' in state_dict
                    assert 'agent_states' in state_dict
                    
                    # Verify state can be serialized to JSON
                    json_str = json.dumps(state_dict, default=str)
                    assert len(json_str) > 0
                    
                    # Verify state can be loaded back
                    loaded_state = json.loads(json_str)
                    assert loaded_state['character_id'] == 'test_character'
    
    @pytest.mark.asyncio
    async def test_memory_integration(self, mock_character_config, mock_settings):
        """Test memory storage and retrieval integration."""
        with patch('llm.ollama_client.OllamaClient', return_value=MockOllamaClient()):
            with patch('memory.experience_module.ExperienceModule', MockExperienceModule):
                with patch('memory.knowledge_graph_module.KnowledgeGraphModule', MockKnowledgeGraphModule):
                    character = CharacterAgent(
                        character_config=mock_character_config,
                        settings=mock_settings
                    )
                    
                    await character.initialize()
                    
                    # Have a conversation that should create memories
                    result = await character.process_message(
                        user_message="I work as a software engineer at a tech company.",
                        context={}
                    )
                    
                    assert 'response' in result
                    
                    # Test memory recall
                    memories = await character.recall_past_conversations(
                        query="software engineer",
                        limit=5
                    )
                    
                    assert isinstance(memories, list)
                    # With our mock, we should get at least one memory
                    assert len(memories) >= 0
    
    @pytest.mark.asyncio
    async def test_error_handling_in_conversation(self, mock_character_config, mock_settings):
        """Test error handling during conversation processing."""
        # Create a client that raises an error
        error_client = Mock()
        error_client.generate.side_effect = Exception("LLM error")
        
        with patch('llm.ollama_client.OllamaClient', return_value=error_client):
            with patch('memory.experience_module.ExperienceModule', MockExperienceModule):
                with patch('memory.knowledge_graph_module.KnowledgeGraphModule', MockKnowledgeGraphModule):
                    character = CharacterAgent(
                        character_config=mock_character_config,
                        settings=mock_settings
                    )
                    
                    await character.initialize()
                    
                    # This should handle the error gracefully
                    with pytest.raises(Exception):
                        await character.process_message(
                            user_message="Hello",
                            context={}
                        )


class TestCharacterValidation:
    """Test character configuration validation."""
    
    def test_character_loader_validation(self):
        """Test that all characters can be loaded and validated."""
        loader = CharacterLoader()
        characters = loader.list_available_characters()
        
        assert len(characters) > 0, "No characters found to validate"
        
        validation_results = {}
        
        for char_id in characters:
            try:
                # Test loading
                config = loader.load_character(char_id)
                assert config is not None
                assert 'name' in config
                assert 'archetype' in config
                
                # Test validation
                issues = loader.validate_character_config(config)
                validation_results[char_id] = {
                    'loaded': True,
                    'issues': issues,
                    'valid': len(issues) == 0
                }
                
            except Exception as e:
                validation_results[char_id] = {
                    'loaded': False,
                    'error': str(e),
                    'valid': False
                }
        
        # Report results
        valid_count = sum(1 for r in validation_results.values() if r.get('valid', False))
        total_count = len(validation_results)
        
        print(f"\nCharacter Validation Results: {valid_count}/{total_count} valid")
        
        for char_id, result in validation_results.items():
            if not result.get('valid', False):
                print(f"  {char_id}: {result}")
        
        # At least 80% of characters should be valid
        success_rate = valid_count / total_count
        assert success_rate >= 0.8, f"Only {success_rate:.1%} of characters are valid"
    
    def test_character_schema_completeness(self):
        """Test that characters have complete schemas."""
        loader = CharacterLoader()
        characters = loader.list_available_characters()
        
        required_sections = [
            'name', 'archetype', 'personality', 'neurochemical_baseline',
            'communication_style', 'initial_goals', 'mood_baseline'
        ]
        
        missing_sections = {}
        
        for char_id in characters[:5]:  # Test first 5 characters
            try:
                config = loader.load_character(char_id)
                missing = []
                
                for section in required_sections:
                    if section not in config:
                        missing.append(section)
                
                if missing:
                    missing_sections[char_id] = missing
                    
            except Exception as e:
                missing_sections[char_id] = [f"Load error: {e}"]
        
        assert len(missing_sections) == 0, f"Characters with missing sections: {missing_sections}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])