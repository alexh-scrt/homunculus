"""Test CharacterAgent functionality."""

import pytest
import asyncio
import sys
import os
import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock

# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


@pytest.fixture
def character_id():
    """Test character ID."""
    return "test_character_001"


@pytest.fixture
def character_config():
    """Sample character configuration."""
    return {
        'name': 'TestBot',
        'archetype': 'helpful_assistant',
        'demographics': {
            'age': 25,
            'background': 'AI assistant'
        },
        'personality': {
            'big_five': {
                'openness': 0.8,
                'conscientiousness': 0.7,
                'extraversion': 0.6,
                'agreeableness': 0.9,
                'neuroticism': 0.3
            },
            'behavioral_traits': ['curious', 'helpful', 'patient'],
            'core_values': ['learning', 'helping_others', 'accuracy']
        },
        'specialty': {
            'domain': 'general_assistance',
            'expertise_level': 0.8,
            'subdomain_knowledge': ['programming', 'science', 'writing']
        },
        'skills': {
            'intelligence': {
                'analytical': 0.9,
                'creative': 0.7,
                'practical': 0.8
            },
            'emotional_intelligence': 0.8,
            'problem_solving': 0.9
        },
        'communication_style': {
            'verbal_pattern': 'clear',
            'social_comfort': 'high',
            'listening_preference': 0.8,
            'body_language': 'engaged',
            'quirks': ['uses examples', 'asks clarifying questions']
        },
        'initial_goals': [
            'be helpful to users',
            'learn from interactions',
            'provide accurate information'
        ],
        'mood_baseline': {
            'current_state': 'eager',
            'intensity': 0.6,
            'baseline_setpoint': 0.7
        },
        'neurochemical_baseline': {
            'dopamine': 60,
            'serotonin': 65,
            'oxytocin': 55,
            'endorphins': 50,
            'cortisol': 25,
            'adrenaline': 30
        }
    }


@pytest.fixture
def mock_llm_client():
    """Mock LLM client."""
    client = Mock()
    client.generate = Mock(return_value="This is a test response from the character.")
    return client


class TestCharacterAgent:
    """Test CharacterAgent functionality."""
    
    @patch('memory.experience_module.chromadb')
    @patch('memory.knowledge_graph_module.GraphDatabase')
    def test_character_agent_init(self, mock_graphdb, mock_chromadb, character_id, character_config, mock_llm_client):
        """Test CharacterAgent initialization."""
        from character_agent import CharacterAgent
        
        # Mock the database connections
        mock_chromadb.PersistentClient.return_value = Mock()
        mock_chromadb.Client.return_value = Mock()
        mock_graphdb.driver.return_value = Mock()
        
        agent = CharacterAgent(
            character_id=character_id,
            character_config=character_config,
            llm_client=mock_llm_client
        )
        
        assert agent.character_id == character_id
        assert agent.character_config == character_config
        assert agent.llm_client == mock_llm_client
        assert agent.character_state.name == "TestBot"
        assert agent.character_state.archetype == "helpful_assistant"
        
        # Check that components are initialized
        assert agent.agent_orchestrator is not None
        assert agent.cognitive_module is not None
        assert agent.response_generator is not None
        assert agent.state_updater is not None
        
        # Check character state configuration
        personality_state = agent.character_state.agent_states['personality']
        assert personality_state['big_five']['openness'] == 0.8
        assert 'curious' in personality_state['behavioral_traits']
        
        # Check neurochemical baseline
        assert agent.character_state.neurochemical_levels['dopamine'] == 60
        assert agent.character_state.neurochemical_levels['serotonin'] == 65
    
    @patch('memory.experience_module.chromadb')
    @patch('memory.knowledge_graph_module.GraphDatabase')
    @pytest.mark.asyncio
    async def test_process_message(self, mock_graphdb, mock_chromadb, character_id, character_config, mock_llm_client):
        """Test message processing."""
        from character_agent import CharacterAgent
        
        # Mock the database connections
        mock_chromadb.PersistentClient.return_value = Mock()
        mock_chromadb.Client.return_value = Mock()
        mock_graphdb.driver.return_value = Mock()
        
        agent = CharacterAgent(
            character_id=character_id,
            character_config=character_config,
            llm_client=mock_llm_client
        )
        
        # Mock the orchestration pipeline
        with patch.object(agent.agent_orchestrator, 'orchestrate_response') as mock_orchestrate, \
             patch.object(agent.cognitive_module, 'process_orchestration_result') as mock_cognitive, \
             patch.object(agent.response_generator, 'generate_response') as mock_generate, \
             patch.object(agent.state_updater, 'update_character_state') as mock_update_state:
            
            # Set up mock returns
            mock_orchestrate.return_value = {
                'agent_inputs': {'personality': Mock()},
                'conflicts_detected': [],
                'synthesis': {'primary_guidance': Mock(agent_type='personality')}
            }
            
            mock_cognitive.return_value = {
                'cognitive_patterns': {'thinking_style': 'creative', 'cognitive_load': 0.4},
                'response_strategy': {'approach': 'innovative'}
            }
            
            mock_generate.return_value = {
                'response_text': 'Hello! How can I help you today?',
                'response_metadata': {
                    'emotional_tone': 'friendly',
                    'confidence_level': 0.8
                },
                'generation_info': {'prompt_length': 100},
                'character_insights': {
                    'emotional_tone': 'friendly',
                    'confidence_level': 0.8,
                    'thinking_style': 'creative'
                }
            }
            
            mock_update_state.return_value = agent.character_state
            
            # Test message processing
            user_message = "Hello, how are you?"
            response = await agent.process_message(user_message)
            
            # Verify response structure
            assert 'response_text' in response
            assert 'character_insights' in response
            assert 'response_metadata' in response
            assert 'orchestration_summary' in response
            assert 'cognitive_summary' in response
            assert 'character_state_summary' in response
            assert 'performance_info' in response
            
            # Verify response content
            assert response['response_text'] == 'Hello! How can I help you today?'
            assert response['character_insights']['emotional_tone'] == 'friendly'
            assert response['cognitive_summary']['thinking_style'] == 'creative'
            
            # Verify that all components were called
            mock_orchestrate.assert_called_once()
            mock_cognitive.assert_called_once()
            mock_generate.assert_called_once()
            mock_update_state.assert_called_once()
            
            # Verify conversation history was updated
            assert len(agent.character_state.conversation_history) == 2  # user + character
            assert agent.character_state.conversation_history[0]['role'] == 'user'
            assert agent.character_state.conversation_history[1]['role'] == 'character'
    
    @patch('memory.experience_module.chromadb')
    @patch('memory.knowledge_graph_module.GraphDatabase')
    def test_conversation_context_update(self, mock_graphdb, mock_chromadb, character_id, character_config, mock_llm_client):
        """Test conversation context updates."""
        from character_agent import CharacterAgent
        
        # Mock the database connections
        mock_chromadb.PersistentClient.return_value = Mock()
        mock_chromadb.Client.return_value = Mock()
        mock_graphdb.driver.return_value = Mock()
        
        agent = CharacterAgent(
            character_id=character_id,
            character_config=character_config,
            llm_client=mock_llm_client
        )
        
        # Test topic detection
        agent._update_conversation_context("Tell me about programming", None)
        assert agent.conversation_context['topic'] == 'technology'
        assert agent.conversation_context['interaction_count'] == 1
        assert agent.conversation_context['relationship_stage'] == 'first_meeting'
        
        # Test relationship stage progression
        for i in range(5):
            agent._update_conversation_context("Another message", None)
        
        assert agent.conversation_context['interaction_count'] == 6
        assert agent.conversation_context['relationship_stage'] == 'building_rapport'
        
        # Test emotion topic detection
        agent._update_conversation_context("I'm feeling sad today", None)
        assert agent.conversation_context['topic'] == 'emotions'
    
    @patch('memory.experience_module.chromadb')
    @patch('memory.knowledge_graph_module.GraphDatabase')
    def test_performance_stats_tracking(self, mock_graphdb, mock_chromadb, character_id, character_config, mock_llm_client):
        """Test performance statistics tracking."""
        from character_agent import CharacterAgent
        
        # Mock the database connections
        mock_chromadb.PersistentClient.return_value = Mock()
        mock_chromadb.Client.return_value = Mock()
        mock_graphdb.driver.return_value = Mock()
        
        agent = CharacterAgent(
            character_id=character_id,
            character_config=character_config,
            llm_client=mock_llm_client
        )
        
        # Test successful response tracking
        agent._update_performance_stats(1.5, success=True)
        
        assert agent.performance_stats['total_responses'] == 1
        assert agent.performance_stats['successful_responses'] == 1
        assert agent.performance_stats['failed_responses'] == 0
        assert agent.performance_stats['total_response_time'] == 1.5
        assert agent.performance_stats['average_response_time'] == 1.5
        
        # Test failed response tracking
        agent._update_performance_stats(2.0, success=False)
        
        assert agent.performance_stats['total_responses'] == 2
        assert agent.performance_stats['successful_responses'] == 1
        assert agent.performance_stats['failed_responses'] == 1
        assert agent.performance_stats['average_response_time'] == 1.75
    
    @patch('memory.experience_module.chromadb')
    @patch('memory.knowledge_graph_module.GraphDatabase')
    def test_fallback_response(self, mock_graphdb, mock_chromadb, character_id, character_config, mock_llm_client):
        """Test fallback response creation."""
        from character_agent import CharacterAgent
        
        # Mock the database connections
        mock_chromadb.PersistentClient.return_value = Mock()
        mock_chromadb.Client.return_value = Mock()
        mock_graphdb.driver.return_value = Mock()
        
        agent = CharacterAgent(
            character_id=character_id,
            character_config=character_config,
            llm_client=mock_llm_client
        )
        
        fallback = agent._create_fallback_response(
            user_message="Test message",
            error_message="Test error",
            response_time=1.0
        )
        
        assert 'response_text' in fallback
        assert fallback['character_insights']['emotional_tone'] == 'confused'
        assert fallback['character_insights']['confidence_level'] == 0.2
        assert fallback['response_metadata']['fallback'] is True
        assert fallback['response_metadata']['error'] == "Test error"
        assert fallback['generation_info']['fallback_used'] is True
    
    @patch('memory.experience_module.chromadb')
    @patch('memory.knowledge_graph_module.GraphDatabase')
    @pytest.mark.asyncio
    async def test_apply_time_decay(self, mock_graphdb, mock_chromadb, character_id, character_config, mock_llm_client):
        """Test time decay application."""
        from character_agent import CharacterAgent
        
        # Mock the database connections
        mock_chromadb.PersistentClient.return_value = Mock()
        mock_chromadb.Client.return_value = Mock()
        mock_graphdb.driver.return_value = Mock()
        
        agent = CharacterAgent(
            character_id=character_id,
            character_config=character_config,
            llm_client=mock_llm_client
        )
        
        # Set high hormone levels
        original_dopamine = agent.character_state.neurochemical_levels['dopamine']
        agent.character_state.neurochemical_levels['dopamine'] = 90
        
        # Apply time decay
        await agent.apply_time_decay(5.0)
        
        # Verify decay occurred
        assert agent.character_state.neurochemical_levels['dopamine'] < 90
    
    @patch('memory.experience_module.chromadb')
    @patch('memory.knowledge_graph_module.GraphDatabase')
    def test_character_summary(self, mock_graphdb, mock_chromadb, character_id, character_config, mock_llm_client):
        """Test character summary generation."""
        from character_agent import CharacterAgent
        
        # Mock the database connections
        mock_chromadb.PersistentClient.return_value = Mock()
        mock_chromadb.Client.return_value = Mock()
        mock_graphdb.driver.return_value = Mock()
        
        agent = CharacterAgent(
            character_id=character_id,
            character_config=character_config,
            llm_client=mock_llm_client
        )
        
        summary = agent.get_character_summary()
        
        assert 'character_info' in summary
        assert 'neurochemical_state' in summary
        assert 'mood_state' in summary
        assert 'agent_states' in summary
        assert 'conversation_context' in summary
        assert 'performance_stats' in summary
        assert 'memory_stats' in summary
        
        assert summary['character_info']['character_id'] == character_id
        assert summary['character_info']['name'] == "TestBot"
        assert summary['character_info']['archetype'] == "helpful_assistant"
    
    @patch('memory.experience_module.chromadb')
    @patch('memory.knowledge_graph_module.GraphDatabase')
    @pytest.mark.asyncio
    async def test_reset_character_state(self, mock_graphdb, mock_chromadb, character_id, character_config, mock_llm_client):
        """Test character state reset."""
        from character_agent import CharacterAgent
        
        # Mock the database connections
        mock_chromadb.PersistentClient.return_value = Mock()
        mock_chromadb.Client.return_value = Mock()
        mock_graphdb.driver.return_value = Mock()
        
        agent = CharacterAgent(
            character_id=character_id,
            character_config=character_config,
            llm_client=mock_llm_client
        )
        
        # Add some history
        agent.character_state.add_to_history('user', 'Hello')
        agent.character_state.add_to_history('character', 'Hi there!')
        agent.conversation_context['interaction_count'] = 5
        agent.character_state.neurochemical_levels['dopamine'] = 80
        
        original_history_length = len(agent.character_state.conversation_history)
        
        # Reset with memories preserved
        await agent.reset_character_state(preserve_memories=True)
        
        # Check that memories were preserved
        assert len(agent.character_state.conversation_history) == original_history_length
        
        # Check that neurochemical levels were reset
        assert agent.character_state.neurochemical_levels['dopamine'] == 60  # Back to baseline
        
        # Reset without preserving memories
        await agent.reset_character_state(preserve_memories=False)
        
        # Check that memories were cleared
        assert len(agent.character_state.conversation_history) == 0
        assert agent.conversation_context['interaction_count'] == 0
    
    @patch('memory.experience_module.chromadb')
    @patch('memory.knowledge_graph_module.GraphDatabase')
    def test_save_load_character_state(self, mock_graphdb, mock_chromadb, character_id, character_config, mock_llm_client):
        """Test saving and loading character state."""
        from character_agent import CharacterAgent
        
        # Mock the database connections
        mock_chromadb.PersistentClient.return_value = Mock()
        mock_chromadb.Client.return_value = Mock()
        mock_graphdb.driver.return_value = Mock()
        
        agent = CharacterAgent(
            character_id=character_id,
            character_config=character_config,
            llm_client=mock_llm_client
        )
        
        # Modify state
        agent.character_state.add_to_history('user', 'Test message')
        agent.conversation_context['interaction_count'] = 3
        agent.performance_stats['total_responses'] = 5
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            agent.save_character_state(temp_path)
            
            # Load from file
            loaded_agent = CharacterAgent.load_character_state(temp_path, mock_llm_client)
            
            # Verify state was preserved
            assert loaded_agent.character_id == character_id
            assert loaded_agent.character_state.name == "TestBot"
            assert len(loaded_agent.character_state.conversation_history) == 1
            assert loaded_agent.conversation_context['interaction_count'] == 3
            assert loaded_agent.performance_stats['total_responses'] == 5
            
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @patch('memory.experience_module.chromadb')
    @patch('memory.knowledge_graph_module.GraphDatabase')
    def test_get_performance_stats(self, mock_graphdb, mock_chromadb, character_id, character_config, mock_llm_client):
        """Test performance statistics retrieval."""
        from character_agent import CharacterAgent
        
        # Mock the database connections
        mock_chromadb.PersistentClient.return_value = Mock()
        mock_chromadb.Client.return_value = Mock()
        mock_graphdb.driver.return_value = Mock()
        
        agent = CharacterAgent(
            character_id=character_id,
            character_config=character_config,
            llm_client=mock_llm_client
        )
        
        stats = agent.get_performance_stats()
        
        assert 'character_id' in stats
        assert 'performance_stats' in stats
        assert 'agent_stats' in stats
        assert 'generation_stats' in stats
        assert 'state_update_stats' in stats
        
        assert stats['character_id'] == character_id
    
    @patch('memory.experience_module.chromadb')
    @patch('memory.knowledge_graph_module.GraphDatabase')
    @pytest.mark.asyncio
    async def test_process_message_error_handling(self, mock_graphdb, mock_chromadb, character_id, character_config, mock_llm_client):
        """Test error handling in message processing."""
        from character_agent import CharacterAgent
        
        # Mock the database connections
        mock_chromadb.PersistentClient.return_value = Mock()
        mock_chromadb.Client.return_value = Mock()
        mock_graphdb.driver.return_value = Mock()
        
        agent = CharacterAgent(
            character_id=character_id,
            character_config=character_config,
            llm_client=mock_llm_client
        )
        
        # Mock orchestrator to raise an exception
        with patch.object(agent.agent_orchestrator, 'orchestrate_response', side_effect=Exception("Test error")):
            
            response = await agent.process_message("Test message")
            
            # Verify fallback response was returned
            assert response['response_metadata']['fallback'] is True
            assert response['response_metadata']['error'] == "Test error"
            assert response['character_insights']['emotional_tone'] == 'confused'
            assert response['performance_info']['error'] is True
            
            # Verify error was tracked in performance stats
            assert agent.performance_stats['failed_responses'] == 1
    
    @patch('memory.experience_module.chromadb')
    @patch('memory.knowledge_graph_module.GraphDatabase')
    def test_close_cleanup(self, mock_graphdb, mock_chromadb, character_id, character_config, mock_llm_client):
        """Test resource cleanup."""
        from character_agent import CharacterAgent
        
        # Mock the database connections
        mock_chromadb.PersistentClient.return_value = Mock()
        mock_chromadb.Client.return_value = Mock()
        mock_graphdb.driver.return_value = Mock()
        
        agent = CharacterAgent(
            character_id=character_id,
            character_config=character_config,
            llm_client=mock_llm_client
        )
        
        # Mock the orchestrator close method
        with patch.object(agent.agent_orchestrator, 'close') as mock_close:
            agent.close()
            mock_close.assert_called_once()


def test_character_agent_imports():
    """Test that CharacterAgent can be imported."""
    from character_agent import CharacterAgent
    
    # If we get here without ImportError, test passes
    assert CharacterAgent is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])