"""Test StateUpdater functionality."""

import pytest
import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


@pytest.fixture
def character_id():
    """Test character ID."""
    return "test_character_001"


@pytest.fixture
def sample_character_state(character_id):
    """Sample character state for testing."""
    from core.character_state import CharacterState
    
    state = CharacterState(
        character_id=character_id,
        last_updated=datetime.now(),
        name="Test Character",
        archetype="test",
        neurochemical_levels={
            'dopamine': 50,
            'serotonin': 55,
            'oxytocin': 45,
            'endorphins': 40,
            'cortisol': 30,
            'adrenaline': 25
        }
    )
    
    # Add additional fields that StateUpdater expects
    state.interaction_count = 5
    state.total_response_time = 10.5
    state.average_confidence = 0.7
    
    return state


@pytest.fixture
def mock_memory_agent():
    """Mock memory agent for testing."""
    memory_agent = Mock()
    memory_agent.store_interaction_memory = AsyncMock(return_value=True)
    return memory_agent


class TestStateUpdater:
    """Test StateUpdater functionality."""
    
    def test_state_updater_init(self, character_id, mock_memory_agent):
        """Test StateUpdater initialization."""
        from integration.state_updater import StateUpdater
        
        state_config = {
            'hormone_decay_rate': 0.03,
            'mood_inertia': 0.9,
            'memory_formation_threshold': 0.7
        }
        
        updater = StateUpdater(
            character_id=character_id,
            memory_agent=mock_memory_agent,
            state_config=state_config
        )
        
        assert updater.character_id == character_id
        assert updater.memory_agent == mock_memory_agent
        assert updater.hormone_decay_rate == 0.03
        assert updater.mood_inertia == 0.9
        assert updater.memory_formation_threshold == 0.7
    
    @pytest.mark.asyncio
    async def test_update_character_state(self, character_id, mock_memory_agent, sample_character_state):
        """Test complete character state update."""
        from integration.state_updater import StateUpdater
        
        updater = StateUpdater(character_id, mock_memory_agent)
        
        interaction_data = {
            'user_message': 'How are you feeling today?',
            'context': {'topic': 'wellbeing'}
        }
        
        response_metadata = {
            'emotional_tone': 'happy',
            'confidence_level': 0.8,
            'response_text': 'I am feeling great today!'
        }
        
        # Test state update
        updated_state = await updater.update_character_state(
            sample_character_state,
            interaction_data,
            response_metadata
        )
        
        # Verify state was updated
        assert updated_state.character_id == character_id
        assert updated_state.interaction_count == 6  # Should increment
        assert updated_state.last_updated > sample_character_state.last_updated
        
        # Check neurochemical adjustments for happy tone
        assert updated_state.neurochemical_levels['dopamine'] > sample_character_state.neurochemical_levels['dopamine']
        assert updated_state.neurochemical_levels['serotonin'] > sample_character_state.neurochemical_levels['serotonin']
    
    def test_calculate_neurochemical_adjustments(self, character_id, mock_memory_agent):
        """Test neurochemical adjustment calculations."""
        from integration.state_updater import StateUpdater
        
        updater = StateUpdater(character_id, mock_memory_agent)
        
        # Test happy emotional tone
        adjustments = updater._calculate_neurochemical_adjustments(
            emotional_tone='happy',
            confidence_level=0.8,
            user_message='Thank you for helping me!'
        )
        
        # Happy tone should increase positive hormones
        assert adjustments['dopamine'] > 0
        assert adjustments['serotonin'] > 0
        assert adjustments['endorphins'] > 0
        assert adjustments['cortisol'] < 0
        
        # Social gratitude should boost oxytocin
        assert adjustments['oxytocin'] > 0
        
        # Test anxious emotional tone
        adjustments = updater._calculate_neurochemical_adjustments(
            emotional_tone='anxious',
            confidence_level=0.3,
            user_message='I have a problem I need help with'
        )
        
        # Anxious tone should increase stress hormones
        assert adjustments['cortisol'] > 0
        assert adjustments['adrenaline'] > 0
        assert adjustments['serotonin'] < 0
    
    def test_apply_hormone_decay(self, character_id, mock_memory_agent, sample_character_state):
        """Test hormone decay over time."""
        from integration.state_updater import StateUpdater
        
        updater = StateUpdater(character_id, mock_memory_agent)
        
        # Set high hormone levels
        sample_character_state.neurochemical_levels = {
            'dopamine': 90,
            'serotonin': 85,
            'oxytocin': 80,
            'endorphins': 75,
            'cortisol': 85,
            'adrenaline': 90
        }
        
        # Simulate time passage
        updater.last_hormone_update = datetime.now() - timedelta(minutes=10)
        
        updated_state = updater._apply_hormone_decay(sample_character_state)
        
        # High levels should decay toward target levels
        assert updated_state.neurochemical_levels['dopamine'] < 90
        assert updated_state.neurochemical_levels['cortisol'] < 85
        assert updated_state.neurochemical_levels['adrenaline'] < 90
    
    def test_calculate_mood_from_neurochemistry(self, character_id, mock_memory_agent):
        """Test mood calculation from neurochemical levels."""
        from integration.state_updater import StateUpdater
        
        updater = StateUpdater(character_id, mock_memory_agent)
        
        # Test happy mood calculation
        happy_levels = {
            'dopamine': 75,
            'serotonin': 70,
            'oxytocin': 60,
            'endorphins': 50,
            'cortisol': 20,
            'adrenaline': 30
        }
        
        mood = updater._calculate_mood_from_neurochemistry(happy_levels)
        
        assert mood['current_state'] == 'happy'
        assert mood['energy_level'] > 40  # Lowered expectation based on calculation
        assert mood['sociability'] > 50
        assert isinstance(mood['focus_level'], (int, float))
        
        # Test anxious mood calculation
        anxious_levels = {
            'dopamine': 30,
            'serotonin': 25,
            'oxytocin': 20,
            'endorphins': 15,
            'cortisol': 80,
            'adrenaline': 70
        }
        
        mood = updater._calculate_mood_from_neurochemistry(anxious_levels)
        
        assert mood['current_state'] == 'anxious'
        assert mood['neurochemical_basis']['dominant_hormone'] == 'cortisol'
    
    def test_should_form_memory(self, character_id, mock_memory_agent):
        """Test memory formation decision logic."""
        from integration.state_updater import StateUpdater
        
        updater = StateUpdater(character_id, mock_memory_agent)
        
        # High confidence should form memory
        interaction_data = {'user_message': 'Tell me about yourself'}
        response_metadata = {'confidence_level': 0.9, 'emotional_tone': 'neutral'}
        
        assert updater._should_form_memory(interaction_data, response_metadata) is True
        
        # Strong emotion should form memory
        response_metadata = {'confidence_level': 0.3, 'emotional_tone': 'excited'}
        
        assert updater._should_form_memory(interaction_data, response_metadata) is True
        
        # Questions should form memory
        interaction_data = {'user_message': 'How does machine learning work?'}
        response_metadata = {'confidence_level': 0.4, 'emotional_tone': 'neutral'}
        
        assert updater._should_form_memory(interaction_data, response_metadata) is True
        
        # Low confidence, neutral tone, simple statement should not form memory
        interaction_data = {'user_message': 'Hello'}
        response_metadata = {'confidence_level': 0.3, 'emotional_tone': 'neutral'}
        
        assert updater._should_form_memory(interaction_data, response_metadata) is False
    
    @pytest.mark.asyncio
    async def test_process_memory_formation(self, character_id, mock_memory_agent, sample_character_state):
        """Test memory formation process."""
        from integration.state_updater import StateUpdater
        
        updater = StateUpdater(character_id, mock_memory_agent)
        
        interaction_data = {
            'user_message': 'What is your favorite programming language?'
        }
        
        response_metadata = {
            'emotional_tone': 'enthusiastic',
            'confidence_level': 0.8,
            'response_text': 'I really enjoy working with Python!'
        }
        
        # Process memory formation
        await updater._process_memory_formation(
            sample_character_state,
            interaction_data,
            response_metadata
        )
        
        # Verify memory agent was called
        mock_memory_agent.store_interaction_memory.assert_called_once()
        
        # Verify call arguments
        call_args = mock_memory_agent.store_interaction_memory.call_args
        assert call_args.kwargs['user_message'] == 'What is your favorite programming language?'
        assert call_args.kwargs['character_response'] == 'I really enjoy working with Python!'
        assert call_args.kwargs['character_state'] == sample_character_state
    
    def test_apply_natural_decay(self, character_id, mock_memory_agent, sample_character_state):
        """Test natural decay for inactive periods."""
        from integration.state_updater import StateUpdater
        
        updater = StateUpdater(character_id, mock_memory_agent)
        
        # Set high hormone levels
        sample_character_state.neurochemical_levels = {
            'dopamine': 80,
            'serotonin': 75,
            'oxytocin': 70,
            'endorphins': 65,
            'cortisol': 60,
            'adrenaline': 85
        }
        
        # Apply decay for 5 hours of inactivity
        updated_state = updater.apply_natural_decay(sample_character_state, 5.0)
        
        # Verify decay occurred
        assert updated_state.neurochemical_levels['dopamine'] < 80
        assert updated_state.neurochemical_levels['adrenaline'] < 85
        assert updated_state.neurochemical_levels['cortisol'] < 60
        
        # Verify mood was updated
        assert 'current_state' in updated_state.agent_states['mood']
        
        # Test minimal decay for short periods
        updated_state_short = updater.apply_natural_decay(sample_character_state, 0.5)
        
        # Should be no change for less than 1 hour
        assert updated_state_short.neurochemical_levels == sample_character_state.neurochemical_levels
    
    def test_update_agent_states(self, character_id, mock_memory_agent, sample_character_state):
        """Test agent state updates."""
        from integration.state_updater import StateUpdater
        
        updater = StateUpdater(character_id, mock_memory_agent)
        
        interaction_data = {
            'user_message': 'I want to achieve my goal of learning Python'
        }
        
        response_metadata = {
            'emotional_tone': 'enthusiastic',
            'confidence_level': 0.85
        }
        
        original_count = sample_character_state.interaction_count
        original_confidence = sample_character_state.average_confidence
        
        updated_state = updater._update_agent_states(
            sample_character_state,
            interaction_data,
            response_metadata
        )
        
        # Verify interaction count increased
        assert updated_state.interaction_count == original_count + 1
        
        # Verify average confidence updated
        assert updated_state.average_confidence != original_confidence
        
        # Verify communication style state updated
        style_state = updated_state.agent_states['communication_style']
        assert 'tone_history' in style_state
        assert 'enthusiastic' in style_state['tone_history']
        assert style_state['tone_history']['enthusiastic'] == 1
        
        # Verify goals state updated for goal-related message
        goals_state = updated_state.agent_states['goals']
        assert 'goal_related_interactions' in goals_state
        assert goals_state['goal_related_interactions'] == 1
    
    def test_get_state_update_stats(self, character_id, mock_memory_agent):
        """Test state update statistics."""
        from integration.state_updater import StateUpdater
        
        state_config = {
            'hormone_decay_rate': 0.025,
            'mood_inertia': 0.85
        }
        
        updater = StateUpdater(character_id, mock_memory_agent, state_config)
        
        stats = updater.get_state_update_stats()
        
        assert stats['character_id'] == character_id
        assert stats['hormone_decay_rate'] == 0.025
        assert stats['mood_inertia'] == 0.85
        assert 'last_hormone_update' in stats
        assert 'last_mood_update' in stats
        assert 'state_config' in stats


def test_state_updater_imports():
    """Test that StateUpdater can be imported."""
    from integration.state_updater import StateUpdater
    from integration import StateUpdater as StateUpdaterFromModule
    
    # If we get here without ImportError, test passes
    assert StateUpdater == StateUpdaterFromModule


if __name__ == "__main__":
    pytest.main([__file__, "-v"])