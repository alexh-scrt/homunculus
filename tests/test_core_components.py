#!/usr/bin/env python3
"""Test script for core components before implementing memory systems."""

import sys
import os
import pytest
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from datetime import datetime
from typing import Dict, Any

# Test imports
def test_imports():
    """Test that all core modules can be imported."""
    from config.settings import get_settings
    from core.agent_input import AgentInput
    from core.character_state import CharacterState
    from core.experience import Experience
    from llm.ollama_client import OllamaClient
    from agents.base_agent import BaseAgent
    from agents.neurochemical_agent import NeurochemicalAgent
    from agents.personality_agent import PersonalityAgent
    from agents.mood_agent import MoodAgent
    from agents.communication_style_agent import CommunicationStyleAgent
    from agents.goals_agent import GoalsAgent
    
    # If we get here without ImportError, test passes
    assert True


def test_settings():
    """Test settings management."""
    from config.settings import get_settings
    settings = get_settings()
    assert settings is not None
    assert hasattr(settings, 'ollama_base_url')
    assert hasattr(settings, 'is_tavily_enabled')
    assert hasattr(settings, 'character_config_files')


def test_core_data_structures():
    """Test core data structures."""
    from core.agent_input import AgentInput
    from core.character_state import CharacterState
    from core.experience import Experience
    
    # Test AgentInput
    agent_input = AgentInput(
        agent_type="test",
        content="Test content",
        confidence=0.8,
        priority=0.7,
        emotional_tone="neutral",
        metadata={"test": True}
    )
    assert agent_input.agent_type == "test"
    assert agent_input.confidence == 0.8
    
    # Test CharacterState
    char_state = CharacterState(
        character_id="test_char",
        last_updated=datetime.now(),
        name="Test Character",
        archetype="test"
    )
    assert char_state.name == "Test Character"
    assert char_state.character_id == "test_char"
    assert "dopamine" in char_state.neurochemical_levels
    
    # Test Experience
    experience = Experience(
        experience_id="exp_test_001",
        character_id="test_char",
        timestamp=datetime.now(),
        experience_type="test",
        description="Test experience",
        participants=["human"],
        emotional_state="neutral"
    )
    assert experience.experience_type == "test"
    assert experience.character_id == "test_char"


def test_llm_client():
    """Test LLM client (without actual LLM connection)."""
    from config.settings import get_settings
    from llm.ollama_client import OllamaClient
    
    settings = get_settings()
    
    # Test client initialization
    llm_client = OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        tavily_api_key=settings.tavily_api_key
    )
    
    assert llm_client.base_url == settings.ollama_base_url
    assert hasattr(llm_client, 'web_search_enabled')
    
    # Get status (doesn't require actual connection)
    status = llm_client.get_status()
    assert status is not None


def test_neurochemical_agent():
    """Test neurochemical agent functionality."""
    from config.settings import get_settings
    from llm.ollama_client import OllamaClient
    from core.character_state import CharacterState
    from agents.neurochemical_agent import NeurochemicalAgent
    from datetime import datetime
    
    settings = get_settings()
    llm_client = OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        tavily_api_key=settings.tavily_api_key
    )
    
    char_state = CharacterState(
        character_id="test_char",
        last_updated=datetime.now(),
        name="Test Character",
        archetype="test"
    )
    
    neuro_config = {
        'baseline_levels': {
            'dopamine': 55.0,
            'serotonin': 50.0,
            'oxytocin': 60.0,
            'endorphins': 50.0,
            'cortisol': 45.0,
            'adrenaline': 50.0
        },
        'baseline_sensitivities': {
            'dopamine': 1.2,
            'serotonin': 1.0,
            'oxytocin': 1.3,
            'endorphins': 1.0,
            'cortisol': 0.8,
            'adrenaline': 1.0
        }
    }
    
    neuro_agent = NeurochemicalAgent(
        agent_id="test_neuro",
        character_id="test_char",
        llm_client=llm_client,
        neurochemical_config=neuro_config
    )
    
    assert neuro_agent.agent_id == "test_neuro"
    assert neuro_agent.character_id == "test_char"
    
    # Test hormone decay
    current_levels = char_state.neurochemical_levels.copy()
    current_levels['dopamine'] = 80.0  # Elevated level
    new_levels = neuro_agent.apply_decay(current_levels)
    assert new_levels['dopamine'] < current_levels['dopamine']  # Should decay
    
    # Test hormone change calculation
    changes = neuro_agent.calculate_hormone_change(
        stimulus_type="compliment",
        intensity=0.8,
        character_response="Thank you so much!"
    )
    assert isinstance(changes, dict)
    assert 'dopamine' in changes


def test_other_agents():
    """Test other agent initialization."""
    from config.settings import get_settings
    from llm.ollama_client import OllamaClient
    from agents.personality_agent import PersonalityAgent
    from agents.mood_agent import MoodAgent
    from agents.communication_style_agent import CommunicationStyleAgent
    from agents.goals_agent import GoalsAgent
    
    settings = get_settings()
    llm_client = OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        tavily_api_key=settings.tavily_api_key
    )
    
    personality_config = {
        'big_five': {
            'openness': 0.7,
            'conscientiousness': 0.6,
            'extraversion': 0.8,
            'agreeableness': 0.7,
            'neuroticism': 0.3
        },
        'behavioral_traits': [
            {'trait': 'curious', 'intensity': 0.8}
        ],
        'core_values': [
            {'value': 'learning', 'priority': 9}
        ]
    }
    
    personality_agent = PersonalityAgent(
        agent_id="test_personality",
        character_id="test_char",
        llm_client=llm_client,
        personality_config=personality_config
    )
    assert personality_agent.agent_type == "personality"
    
    mood_config = {
        'baseline_setpoint': 0.6,
        'emotional_volatility': 0.4,
        'default_state': 'content'
    }
    
    mood_agent = MoodAgent(
        agent_id="test_mood",
        character_id="test_char",
        llm_client=llm_client,
        mood_config=mood_config
    )
    assert mood_agent.agent_type == "mood"
    
    style_config = {
        'verbal_pattern': 'moderate',
        'social_comfort': 'assertive',
        'listening_preference': 0.4,
        'body_language': 'expressive',
        'quirks': ['Uses humor frequently', 'Makes analogies']
    }
    
    style_agent = CommunicationStyleAgent(
        agent_id="test_style",
        character_id="test_char",
        llm_client=llm_client,
        style_config=style_config
    )
    assert style_agent.agent_type == "communication_style"
    
    initial_goals = [
        {
            'goal_id': 'goal_1',
            'goal_type': 'short_term',
            'description': 'Have an engaging conversation',
            'priority': 8,
            'progress': 0.2
        }
    ]
    
    goals_agent = GoalsAgent(
        agent_id="test_goals",
        character_id="test_char",
        llm_client=llm_client,
        initial_goals=initial_goals
    )
    assert goals_agent.agent_type == "goals"


def test_character_state_updates():
    """Test character state update functionality."""
    from core.character_state import CharacterState
    from datetime import datetime
    
    char_state = CharacterState(
        character_id="test_char",
        last_updated=datetime.now(),
        name="Test Character",
        archetype="test"
    )
    
    # Test neurochemical updates
    old_dopamine = char_state.neurochemical_levels['dopamine']
    char_state.update_neurochemical_level('dopamine', 75.0)
    new_dopamine = char_state.neurochemical_levels['dopamine']
    assert new_dopamine == 75.0
    assert new_dopamine != old_dopamine
    
    # Test mood update from hormones
    char_state.update_mood_from_hormones()
    mood_state = char_state.agent_states['mood']
    assert 'current_state' in mood_state
    assert 'intensity' in mood_state
    
    # Test conversation history
    char_state.add_to_history('user', 'Hello, how are you?')
    char_state.add_to_history('character', 'I am doing well, thank you!')
    assert len(char_state.conversation_history) == 2
    
    # Test web search history
    char_state.add_web_search_record(
        query="test query",
        results=[{'title': 'Test Result', 'url': 'http://test.com'}],
        triggered_by="test"
    )
    assert len(char_state.web_search_history) == 1


def test_character_config_loading():
    """Test loading existing character configurations."""
    import yaml
    from core.character_state import CharacterState
    
    # Try to load one of the existing character configs
    char_files = ['m-playful.yaml', 'f-serious.yaml']
    schema_path = Path(__file__).parent.parent / 'schemas' / 'characters'
    
    loaded_configs = []
    for char_file in char_files:
        config_path = schema_path / char_file
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            assert 'character_id' in config
            assert 'name' in config
            assert 'archetype' in config
            
            # Test creating CharacterState from config
            char_state = CharacterState(
                character_id=config['character_id'],
                last_updated=datetime.now(),
                name=config['name'],
                archetype=config['archetype'],
                demographics=config.get('demographics', {})
            )
            
            # Update agent states from config
            char_state.agent_states['personality'] = config.get('personality', {})
            char_state.agent_states['mood'] = config.get('mood_baseline', {})
            char_state.agent_states['communication_style'] = config.get('communication_style', {})
            char_state.agent_states['goals'] = {'active_goals': config.get('initial_goals', [])}
            
            loaded_configs.append(config)
    
    # At least one config should be loadable
    assert len(loaded_configs) > 0


# Tests can be run with: pytest tests/test_core_components.py -v