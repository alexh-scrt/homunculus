"""Test character configuration loader."""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, Mock
import sys
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


@pytest.fixture
def temp_characters_dir():
    """Create a temporary directory with test character files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        characters_dir = Path(temp_dir) / "characters"
        characters_dir.mkdir()
        
        # Create a valid test character
        test_character = {
            'name': 'Test Character',
            'archetype': 'test_archetype',
            'demographics': {'age': 25},
            'personality': {
                'big_five': {
                    'openness': 0.7,
                    'conscientiousness': 0.6,
                    'extraversion': 0.5,
                    'agreeableness': 0.8,
                    'neuroticism': 0.3
                }
            },
            'neurochemical_baseline': {
                'dopamine': 60.0,
                'serotonin': 55.0,
                'oxytocin': 50.0,
                'endorphins': 45.0,
                'cortisol': 30.0,
                'adrenaline': 35.0
            }
        }
        
        with open(characters_dir / "test_character.yaml", 'w') as f:
            yaml.dump(test_character, f)
        
        # Create an invalid character file
        invalid_character = {'name': 'Invalid Character'}  # Missing required 'archetype'
        
        with open(characters_dir / "invalid_character.yaml", 'w') as f:
            yaml.dump(invalid_character, f)
        
        # Create a character with .yml extension
        yml_character = {
            'name': 'YML Character',
            'archetype': 'yml_test'
        }
        
        with open(characters_dir / "yml_character.yml", 'w') as f:
            yaml.dump(yml_character, f)
        
        yield characters_dir


class TestCharacterLoader:
    """Test CharacterLoader functionality."""
    
    def test_character_loader_init(self, temp_characters_dir):
        """Test CharacterLoader initialization."""
        from config.character_loader import CharacterLoader
        
        loader = CharacterLoader(str(temp_characters_dir))
        
        assert loader.characters_dir == temp_characters_dir
        assert loader.characters_dir.exists()
    
    def test_list_available_characters(self, temp_characters_dir):
        """Test listing available character configurations."""
        from config.character_loader import CharacterLoader
        
        loader = CharacterLoader(str(temp_characters_dir))
        characters = loader.list_available_characters()
        
        assert 'test_character' in characters
        assert 'invalid_character' in characters
        assert 'yml_character' in characters
        assert len(characters) == 3
    
    def test_load_valid_character(self, temp_characters_dir):
        """Test loading a valid character configuration."""
        from config.character_loader import CharacterLoader
        
        loader = CharacterLoader(str(temp_characters_dir))
        config = loader.load_character('test_character')
        
        assert config['name'] == 'Test Character'
        assert config['archetype'] == 'test_archetype'
        assert config['character_id'] == 'test_character'
        assert 'personality' in config
        assert 'neurochemical_baseline' in config
        assert 'metadata' in config
        
        # Check that defaults were filled in
        assert 'specialty' in config
        assert 'skills' in config
        assert 'communication_style' in config
    
    def test_load_yml_extension(self, temp_characters_dir):
        """Test loading character with .yml extension."""
        from config.character_loader import CharacterLoader
        
        loader = CharacterLoader(str(temp_characters_dir))
        config = loader.load_character('yml_character')
        
        assert config['name'] == 'YML Character'
        assert config['archetype'] == 'yml_test'
    
    def test_load_nonexistent_character(self, temp_characters_dir):
        """Test loading a character that doesn't exist."""
        from config.character_loader import CharacterLoader, CharacterConfigurationError
        
        loader = CharacterLoader(str(temp_characters_dir))
        
        with pytest.raises(CharacterConfigurationError, match="Character configuration file not found"):
            loader.load_character('nonexistent_character')
    
    def test_load_invalid_character(self, temp_characters_dir):
        """Test loading an invalid character configuration."""
        from config.character_loader import CharacterLoader, CharacterConfigurationError
        
        loader = CharacterLoader(str(temp_characters_dir))
        
        with pytest.raises(CharacterConfigurationError, match="Missing required field 'archetype'"):
            loader.load_character('invalid_character')
    
    def test_validate_personality_config(self, temp_characters_dir):
        """Test personality configuration validation."""
        from config.character_loader import CharacterLoader, CharacterConfigurationError
        
        loader = CharacterLoader(str(temp_characters_dir))
        
        # Test invalid Big Five values
        invalid_personality = {
            'big_five': {
                'openness': 1.5,  # Invalid - should be 0.0-1.0
                'conscientiousness': 0.5,
                'extraversion': 0.5,
                'agreeableness': 0.5,
                'neuroticism': 0.5
            }
        }
        
        with pytest.raises(CharacterConfigurationError, match="Invalid Big Five trait"):
            loader._validate_personality_config(invalid_personality, 'test')
    
    def test_validate_neurochemical_baseline(self, temp_characters_dir):
        """Test neurochemical baseline validation."""
        from config.character_loader import CharacterLoader, CharacterConfigurationError
        
        loader = CharacterLoader(str(temp_characters_dir))
        
        # Test invalid neurochemical values
        invalid_neuro = {
            'dopamine': 150.0,  # Invalid - should be 0.0-100.0
            'serotonin': 50.0
        }
        
        with pytest.raises(CharacterConfigurationError, match="Invalid neurochemical level"):
            loader._validate_neurochemical_baseline(invalid_neuro, 'test')
    
    def test_load_all_characters(self, temp_characters_dir):
        """Test loading all available characters."""
        from config.character_loader import CharacterLoader
        
        loader = CharacterLoader(str(temp_characters_dir))
        all_characters = loader.load_all_characters()
        
        # Should load valid characters and skip invalid ones
        assert 'test_character' in all_characters
        assert 'yml_character' in all_characters
        assert 'invalid_character' not in all_characters  # Should be skipped due to validation error
        
        assert len(all_characters) == 2
    
    def test_validate_character_config(self, temp_characters_dir):
        """Test character configuration validation."""
        from config.character_loader import CharacterLoader
        
        loader = CharacterLoader(str(temp_characters_dir))
        
        # Test valid config
        valid_config = {
            'name': 'Test',
            'archetype': 'test',
            'personality': {
                'big_five': {
                    'openness': 0.5,
                    'conscientiousness': 0.5,
                    'extraversion': 0.5,
                    'agreeableness': 0.5,
                    'neuroticism': 0.5
                }
            }
        }
        
        issues = loader.validate_character_config(valid_config)
        assert len(issues) == 0
        
        # Test invalid config
        invalid_config = {'name': 'Test'}  # Missing archetype
        
        issues = loader.validate_character_config(invalid_config)
        assert len(issues) > 0
        assert any('archetype' in issue for issue in issues)
    
    def test_create_character_template(self, temp_characters_dir):
        """Test creating a character template."""
        from config.character_loader import CharacterLoader
        
        loader = CharacterLoader(str(temp_characters_dir))
        template = loader.create_character_template('new_character')
        
        assert template['name'] == 'Character New_Character'
        assert template['archetype'] == 'balanced'
        assert 'personality' in template
        assert 'neurochemical_baseline' in template
        assert 'initial_goals' in template
        
        # Validate the template
        issues = loader.validate_character_config(template)
        assert len(issues) == 0
    
    def test_save_character_config(self, temp_characters_dir):
        """Test saving a character configuration."""
        from config.character_loader import CharacterLoader
        
        loader = CharacterLoader(str(temp_characters_dir))
        
        config = {
            'name': 'Saved Character',
            'archetype': 'saved_test',
            'description': 'A test character for saving'
        }
        
        loader.save_character_config('saved_character', config)
        
        # Verify file was created
        saved_file = temp_characters_dir / 'saved_character.yaml'
        assert saved_file.exists()
        
        # Verify content
        with open(saved_file, 'r') as f:
            saved_config = yaml.safe_load(f)
        
        assert saved_config['name'] == 'Saved Character'
        assert saved_config['archetype'] == 'saved_test'
        assert 'character_id' not in saved_config  # Should be removed before saving
    
    def test_get_character_info(self, temp_characters_dir):
        """Test getting basic character information."""
        from config.character_loader import CharacterLoader
        
        loader = CharacterLoader(str(temp_characters_dir))
        
        # Test existing character
        info = loader.get_character_info('test_character')
        
        assert info['character_id'] == 'test_character'
        assert info['name'] == 'Test Character'
        assert info['archetype'] == 'test_archetype'
        
        # Test nonexistent character
        info = loader.get_character_info('nonexistent')
        
        assert info['character_id'] == 'nonexistent'
        assert info['name'] == 'Unknown'
        assert info['archetype'] == 'unknown'
    
    def test_process_config_sections(self, temp_characters_dir):
        """Test processing of different configuration sections."""
        from config.character_loader import CharacterLoader
        
        loader = CharacterLoader(str(temp_characters_dir))
        
        # Test personality processing
        personality = loader._process_personality_config({
            'big_five': {'openness': 0.8},
            'behavioral_traits': ['curious']
        })
        
        assert personality['big_five']['openness'] == 0.8
        assert personality['big_five']['conscientiousness'] == 0.5  # Default
        assert 'curious' in personality['behavioral_traits']
        
        # Test skills processing
        skills = loader._process_skills_config({
            'intelligence': {'analytical': 0.9},
            'emotional_intelligence': 0.7
        })
        
        assert skills['intelligence']['analytical'] == 0.9
        assert skills['intelligence']['creative'] == 0.5  # Default
        assert skills['emotional_intelligence'] == 0.7
        
        # Test communication style processing
        style = loader._process_communication_style_config({
            'verbal_pattern': 'elaborate',
            'quirks': ['uses metaphors']
        })
        
        assert style['verbal_pattern'] == 'elaborate'
        assert style['social_comfort'] == 'neutral'  # Default
        assert 'uses metaphors' in style['quirks']
    
    def test_nonexistent_directory(self):
        """Test behavior with nonexistent characters directory."""
        from config.character_loader import CharacterLoader
        
        loader = CharacterLoader('/nonexistent/path')
        
        # Should not crash, but return empty list
        characters = loader.list_available_characters()
        assert characters == []
    
    @patch('config.character_loader.get_settings')
    def test_default_directory_from_settings(self, mock_get_settings, temp_characters_dir):
        """Test using default directory from settings."""
        from config.character_loader import CharacterLoader
        
        # Mock settings
        mock_settings = Mock()
        mock_settings.character_schemas_dir = str(temp_characters_dir)
        mock_get_settings.return_value = mock_settings
        
        loader = CharacterLoader()  # No directory specified
        
        assert str(loader.characters_dir) == str(temp_characters_dir)


def test_character_loader_imports():
    """Test that CharacterLoader can be imported."""
    from config.character_loader import CharacterLoader, CharacterConfigurationError
    
    # If we get here without ImportError, test passes
    assert CharacterLoader is not None
    assert CharacterConfigurationError is not None


def test_real_character_files():
    """Test loading actual character files from the schemas directory."""
    from config.character_loader import CharacterLoader
    
    # Test with the actual schemas directory
    schemas_dir = Path(__file__).parent.parent / "schemas" / "characters"
    
    if schemas_dir.exists():
        loader = CharacterLoader(str(schemas_dir))
        characters = loader.list_available_characters()
        
        # Should find our created characters
        expected_characters = ['ada_lovelace', 'zen_master', 'captain_cosmos', 'grumpy_wizard', 'creative_artist']
        
        for char_id in expected_characters:
            if char_id in characters:
                # Test loading the character
                config = loader.load_character(char_id)
                assert config['name'] is not None
                assert config['archetype'] is not None
                
                # Validate the configuration
                issues = loader.validate_character_config(config)
                assert len(issues) == 0, f"Character {char_id} has validation issues: {issues}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])