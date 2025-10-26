"""Test debug view functionality."""

import pytest
import io
import sys
from unittest.mock import Mock, patch
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

try:
    from cli.debug_view import DebugView
    from rich.console import Console
except ImportError:
    pytest.skip("CLI modules not available", allow_module_level=True)


class TestDebugView:
    """Test DebugView functionality."""
    
    def setup_method(self):
        """Setup for each test."""
        # Use StringIO to capture console output
        self.output = io.StringIO()
        self.console = Console(file=self.output, width=80)
        self.debug_view = DebugView(self.console)
    
    def test_debug_view_initialization(self):
        """Test DebugView can be initialized."""
        # Test with custom console
        debug_view = DebugView(self.console)
        assert debug_view.console == self.console
        
        # Test with default console
        debug_view = DebugView()
        assert debug_view.console is not None
    
    def test_display_neurochemical_levels(self):
        """Test neurochemical levels display."""
        neuro_data = {
            'current_levels': {
                'dopamine': 75.0,
                'serotonin': 60.0,
                'oxytocin': 45.0,
                'endorphins': 55.0,
                'cortisol': 40.0,
                'adrenaline': 65.0
            },
            'baseline_levels': {
                'dopamine': 50.0,
                'serotonin': 55.0,
                'oxytocin': 50.0,
                'endorphins': 50.0,
                'cortisol': 45.0,
                'adrenaline': 50.0
            },
            'recent_changes': {
                'dopamine': 5.0,
                'serotonin': 2.0,
                'oxytocin': -3.0,
                'endorphins': 1.0,
                'cortisol': -2.0,
                'adrenaline': 8.0
            }
        }
        
        self.debug_view._display_neurochemical_levels(neuro_data)
        output = self.output.getvalue()
        
        # Check that all hormones are displayed
        assert 'Dopamine' in output
        assert 'Serotonin' in output
        assert 'Oxytocin' in output
        assert 'Endorphins' in output
        assert 'Cortisol' in output
        assert 'Adrenaline' in output
        
        # Check that values are displayed
        assert '75.0' in output  # dopamine current
        assert '50.0' in output  # baseline values
        assert '+5.0' in output  # positive change
        assert '-3.0' in output  # negative change
    
    def test_display_mood_state(self):
        """Test mood state display."""
        mood_data = {
            'current_state': 'excited',
            'intensity': 0.7,
            'emotional_volatility': 0.4,
            'energy_level': 0.8
        }
        
        self.debug_view._display_mood_state(mood_data)
        output = self.output.getvalue()
        
        assert 'Excited' in output
        assert '0.70' in output  # intensity
        assert '0.40' in output  # volatility
        assert '0.80' in output  # energy
        assert 'Mood State' in output
    
    def test_display_agent_inputs(self):
        """Test agent inputs display."""
        agent_data = {
            'personality': {
                'content': 'Character should respond with high openness and creativity',
                'confidence': 0.9,
                'priority': 8
            },
            'mood': {
                'content': 'Current excited state should increase response enthusiasm',
                'confidence': 0.8,
                'priority': 7
            },
            'goals': {
                'content': 'Long content that should be truncated because it is very long and exceeds the display limit for agent inputs which is set to 150 characters to keep the display clean and readable',
                'confidence': 0.6,
                'priority': 5
            }
        }
        
        self.debug_view._display_agent_inputs(agent_data)
        output = self.output.getvalue()
        
        assert 'PERSONALITY' in output
        assert 'MOOD' in output
        assert 'GOALS' in output
        assert 'openness and creativity' in output
        assert 'excited state' in output
        assert '0.90' in output  # confidence
        assert '...' in output  # truncation indicator
    
    def test_display_memory_retrieval(self):
        """Test memory retrieval display."""
        memory_data = {
            'query': 'test memories',
            'retrieved_memories': [
                {
                    'similarity': 0.95,
                    'description': 'First memory about testing',
                    'timestamp': '2025-01-01T10:00:00'
                },
                {
                    'similarity': 0.82,
                    'description': 'Second memory with lower similarity',
                    'timestamp': '2025-01-01T11:00:00'
                }
            ]
        }
        
        self.debug_view._display_memory_retrieval(memory_data)
        output = self.output.getvalue()
        
        assert 'test memories' in output
        assert 'First memory' in output
        assert 'Second memory' in output
        assert '0.950' in output  # high similarity
        assert '0.820' in output  # lower similarity
        assert '2025-01-01' in output  # timestamp
    
    def test_display_memory_retrieval_empty(self):
        """Test memory retrieval with no memories."""
        memory_data = {
            'query': 'no results',
            'retrieved_memories': []
        }
        
        self.debug_view._display_memory_retrieval(memory_data)
        output = self.output.getvalue()
        
        assert 'No relevant memories' in output
    
    def test_display_cognitive_processing(self):
        """Test cognitive processing display."""
        cognitive_data = {
            'synthesis': {
                'primary_intention': 'Respond helpfully',
                'emotional_tone': 'enthusiastic',
                'key_themes': ['learning', 'creativity', 'helpfulness']
            },
            'conflicts': [
                'Personality suggests creativity but goals suggest focus',
                'Mood is excited but context suggests calm response needed'
            ]
        }
        
        self.debug_view._display_cognitive_processing(cognitive_data)
        output = self.output.getvalue()
        
        assert 'Respond helpfully' in output
        assert 'enthusiastic' in output
        assert 'learning, creativity, helpfulness' in output
        assert 'Agent Conflicts' in output
        assert 'Personality suggests creativity' in output
    
    def test_display_response_generation(self):
        """Test response generation metadata display."""
        response_data = {
            'metadata': {
                'response_length': 150,
                'generation_time': 2.45,
                'style_applied': 'enthusiastic',
                'personality_influence': 0.8
            }
        }
        
        self.debug_view._display_response_generation(response_data)
        output = self.output.getvalue()
        
        assert '150 chars' in output
        assert '2.45s' in output
        assert 'enthusiastic' in output
        assert '0.80' in output
    
    def test_display_state_changes(self):
        """Test state changes display."""
        state_data = {
            'changes': {
                'neurochemical': [
                    'Dopamine increased by 5.0 due to positive interaction',
                    'Cortisol decreased by 2.0 due to reduced stress'
                ],
                'mood': [
                    'Mood shifted to excited with intensity 0.7'
                ],
                'memory': [
                    'Created new episodic memory about conversation'
                ]
            }
        }
        
        self.debug_view._display_state_changes(state_data)
        output = self.output.getvalue()
        
        assert 'State Changes' in output
        assert 'Neurochemical' in output
        assert 'Dopamine increased' in output
        assert 'Cortisol decreased' in output
        assert 'Mood shifted' in output
        assert 'Created new episodic' in output
    
    def test_display_state_changes_empty(self):
        """Test state changes with no changes."""
        state_data = {'changes': {}}
        
        self.debug_view._display_state_changes(state_data)
        output = self.output.getvalue()
        
        assert 'No significant state changes' in output
    
    def test_display_debug_info_comprehensive(self):
        """Test full debug info display with all sections."""
        debug_data = {
            'neurochemical_levels': {
                'current_levels': {'dopamine': 75.0, 'serotonin': 60.0},
                'baseline_levels': {'dopamine': 50.0, 'serotonin': 55.0},
                'recent_changes': {'dopamine': 5.0, 'serotonin': 2.0}
            },
            'mood': {
                'current_state': 'excited',
                'intensity': 0.7
            },
            'agent_inputs': {
                'personality': {'content': 'Test content', 'confidence': 0.9, 'priority': 8}
            },
            'memory_retrieval': {
                'query': 'test',
                'retrieved_memories': []
            },
            'cognitive_processing': {
                'synthesis': {'primary_intention': 'Test'}
            },
            'response_generation': {
                'metadata': {'response_length': 100}
            },
            'state_changes': {
                'changes': {'test': ['Test change']}
            }
        }
        
        self.debug_view.display_debug_info(debug_data)
        output = self.output.getvalue()
        
        assert '═══ DEBUG INFO ═══' in output
        assert '═══ END DEBUG ═══' in output
        assert 'Dopamine' in output
        assert 'Excited' in output
        assert 'Test content' in output
        assert 'No relevant memories' in output
    
    def test_display_character_selection(self):
        """Test character selection display."""
        characters = [
            {
                'id': 'ada_lovelace',
                'name': 'Ada Lovelace',
                'archetype': 'analytical_genius',
                'description': 'Brilliant mathematician and programmer'
            },
            {
                'id': 'zen_master',
                'name': 'Zen Master Kiku',
                'archetype': 'wise_contemplative',
                'description': 'Peaceful meditation teacher'
            }
        ]
        
        self.debug_view.display_character_selection(characters)
        output = self.output.getvalue()
        
        assert 'Available Characters' in output
        assert 'Ada Lovelace' in output
        assert 'Zen Master Kiku' in output
        assert 'analytical_genius' in output
        assert 'wise_contemplative' in output
        assert 'Brilliant mathematician' in output
    
    def test_display_message(self):
        """Test message display."""
        self.debug_view.display_message('User', 'Hello there!', style='blue')
        output = self.output.getvalue()
        
        assert 'User' in output
        assert 'Hello there!' in output
    
    def test_display_help(self):
        """Test help display."""
        self.debug_view.display_help()
        output = self.output.getvalue()
        
        assert 'Available Commands' in output
        assert '/exit' in output
        assert '/debug' in output
        assert '/save' in output
        assert '/load' in output
        assert '/memory' in output
        assert '/reset' in output
        assert '/help' in output
    
    def test_display_error(self):
        """Test error message display."""
        self.debug_view.display_error('Test error message')
        output = self.output.getvalue()
        
        assert 'Error' in output
        assert 'Test error message' in output
    
    def test_display_success(self):
        """Test success message display."""
        self.debug_view.display_success('Operation completed successfully')
        output = self.output.getvalue()
        
        assert 'Success' in output
        assert 'Operation completed successfully' in output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])