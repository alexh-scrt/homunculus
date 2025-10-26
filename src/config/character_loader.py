"""Character configuration loader for YAML-based character definitions."""

import logging
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

try:
    from .settings import get_settings
except ImportError:
    from config.settings import get_settings


class CharacterConfigurationError(Exception):
    """Exception raised when character configuration is invalid."""
    pass


class CharacterLoader:
    """
    Loads and validates character configurations from YAML files.
    
    This class handles loading character profiles from the schemas/characters
    directory and validates them against the expected schema structure.
    """
    
    def __init__(self, characters_dir: Optional[str] = None):
        """
        Initialize the character loader.
        
        Args:
            characters_dir: Optional directory path for character files.
                          If None, uses the setting from configuration.
        """
        self.settings = get_settings()
        self.characters_dir = Path(characters_dir or self.settings.character_schemas_dir)
        self.logger = logging.getLogger(__name__)
        
        # Ensure characters directory exists
        if not self.characters_dir.exists():
            self.logger.warning(f"Characters directory {self.characters_dir} does not exist")
        
        self.logger.info(f"CharacterLoader initialized with directory: {self.characters_dir}")
    
    def list_available_characters(self) -> List[str]:
        """
        List all available character configuration files.
        
        Returns:
            List of character IDs (filenames without .yaml extension)
        """
        if not self.characters_dir.exists():
            return []
        
        yaml_files = list(self.characters_dir.glob("*.yaml")) + list(self.characters_dir.glob("*.yml"))
        character_ids = [f.stem for f in yaml_files]
        
        self.logger.debug(f"Found {len(character_ids)} character configurations: {character_ids}")
        return sorted(character_ids)
    
    def load_character(self, character_id: str) -> Dict[str, Any]:
        """
        Load a character configuration by ID.
        
        Args:
            character_id: The character ID (filename without extension)
            
        Returns:
            Character configuration dictionary
            
        Raises:
            CharacterConfigurationError: If character file is not found or invalid
        """
        # Try both .yaml and .yml extensions
        yaml_path = self.characters_dir / f"{character_id}.yaml"
        yml_path = self.characters_dir / f"{character_id}.yml"
        
        config_path = None
        if yaml_path.exists():
            config_path = yaml_path
        elif yml_path.exists():
            config_path = yml_path
        else:
            raise CharacterConfigurationError(
                f"Character configuration file not found for '{character_id}'. "
                f"Looked for {yaml_path} and {yml_path}"
            )
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if not config:
                raise CharacterConfigurationError(f"Empty configuration file: {config_path}")
            
            # Validate and process the configuration
            validated_config = self._validate_and_process_config(config, character_id)
            
            self.logger.info(f"Successfully loaded character configuration for '{character_id}'")
            return validated_config
            
        except yaml.YAMLError as e:
            raise CharacterConfigurationError(f"Invalid YAML in {config_path}: {e}")
        except Exception as e:
            raise CharacterConfigurationError(f"Error loading {config_path}: {e}")
    
    def _validate_and_process_config(self, config: Dict[str, Any], character_id: str) -> Dict[str, Any]:
        """
        Validate and process a character configuration.
        
        Args:
            config: Raw configuration dictionary
            character_id: Character ID for error reporting
            
        Returns:
            Validated and processed configuration
        """
        # Ensure required fields exist
        required_fields = ['name', 'archetype']
        for field in required_fields:
            if field not in config:
                raise CharacterConfigurationError(
                    f"Missing required field '{field}' in character '{character_id}'"
                )
        
        # Set defaults for optional fields
        processed_config = {
            'character_id': character_id,
            'name': config['name'],
            'archetype': config['archetype'],
            'demographics': config.get('demographics', {}),
            'personality': self._process_personality_config(config.get('personality', {})),
            'specialty': self._process_specialty_config(config.get('specialty', {})),
            'skills': self._process_skills_config(config.get('skills', {})),
            'communication_style': self._process_communication_style_config(
                config.get('communication_style', {})
            ),
            'initial_goals': config.get('initial_goals', []),
            'mood_baseline': self._process_mood_baseline_config(config.get('mood_baseline', {})),
            'neurochemical_baseline': self._process_neurochemical_baseline_config(
                config.get('neurochemical_baseline', {})
            ),
            'backstory': config.get('backstory', ''),
            'development': self._process_development_config(config.get('development', {})),
            'metadata': {
                'created_date': config.get('created_date', datetime.now().isoformat()),
                'version': config.get('version', '1.0'),
                'description': config.get('description', ''),
                'tags': config.get('tags', []),
                'loaded_timestamp': datetime.now().isoformat()
            }
        }
        
        # Validate specific configurations
        self._validate_personality_config(processed_config['personality'], character_id)
        self._validate_neurochemical_baseline(processed_config['neurochemical_baseline'], character_id)
        
        return processed_config
    
    def _process_personality_config(self, personality: Dict[str, Any]) -> Dict[str, Any]:
        """Process and validate personality configuration."""
        return {
            'big_five': {
                'openness': personality.get('big_five', {}).get('openness', 0.5),
                'conscientiousness': personality.get('big_five', {}).get('conscientiousness', 0.5),
                'extraversion': personality.get('big_five', {}).get('extraversion', 0.5),
                'agreeableness': personality.get('big_five', {}).get('agreeableness', 0.5),
                'neuroticism': personality.get('big_five', {}).get('neuroticism', 0.5)
            },
            'behavioral_traits': personality.get('behavioral_traits', []),
            'core_values': personality.get('core_values', []),
            'quirks': personality.get('quirks', []),
            'strengths': personality.get('strengths', []),
            'weaknesses': personality.get('weaknesses', [])
        }
    
    def _process_specialty_config(self, specialty: Dict[str, Any]) -> Dict[str, Any]:
        """Process specialty configuration."""
        return {
            'domain': specialty.get('domain', 'general'),
            'expertise_level': specialty.get('expertise_level', 0.5),
            'subdomain_knowledge': specialty.get('subdomain_knowledge', []),
            'experience_years': specialty.get('experience_years', 1),
            'certifications': specialty.get('certifications', []),
            'notable_achievements': specialty.get('notable_achievements', [])
        }
    
    def _process_skills_config(self, skills: Dict[str, Any]) -> Dict[str, Any]:
        """Process skills configuration."""
        return {
            'intelligence': {
                'analytical': skills.get('intelligence', {}).get('analytical', 0.5),
                'creative': skills.get('intelligence', {}).get('creative', 0.5),
                'practical': skills.get('intelligence', {}).get('practical', 0.5),
                'social': skills.get('intelligence', {}).get('social', 0.5)
            },
            'emotional_intelligence': skills.get('emotional_intelligence', 0.5),
            'physical_capability': skills.get('physical_capability', 0.5),
            'problem_solving': skills.get('problem_solving', 0.5),
            'communication': skills.get('communication', 0.5),
            'leadership': skills.get('leadership', 0.5),
            'technical_skills': skills.get('technical_skills', []),
            'soft_skills': skills.get('soft_skills', [])
        }
    
    def _process_communication_style_config(self, style: Dict[str, Any]) -> Dict[str, Any]:
        """Process communication style configuration."""
        return {
            'verbal_pattern': style.get('verbal_pattern', 'moderate'),
            'social_comfort': style.get('social_comfort', 'neutral'),
            'listening_preference': style.get('listening_preference', 0.5),
            'body_language': style.get('body_language', 'neutral'),
            'quirks': style.get('quirks', []),
            'formality_level': style.get('formality_level', 'moderate'),
            'humor_style': style.get('humor_style', 'none'),
            'preferred_topics': style.get('preferred_topics', []),
            'conversation_style': style.get('conversation_style', 'balanced')
        }
    
    def _process_mood_baseline_config(self, mood: Dict[str, Any]) -> Dict[str, Any]:
        """Process mood baseline configuration."""
        return {
            'current_state': mood.get('current_state', 'neutral'),
            'intensity': mood.get('intensity', 0.5),
            'duration': mood.get('duration', 1),
            'baseline_setpoint': mood.get('baseline_setpoint', 0.5),
            'emotional_volatility': mood.get('emotional_volatility', 0.5),
            'triggered_by': mood.get('triggered_by', 'initialization'),
            'typical_moods': mood.get('typical_moods', ['neutral', 'content']),
            'mood_triggers': mood.get('mood_triggers', {})
        }
    
    def _process_neurochemical_baseline_config(self, neuro: Dict[str, Any]) -> Dict[str, float]:
        """Process neurochemical baseline configuration."""
        return {
            'dopamine': float(neuro.get('dopamine', 50.0)),
            'serotonin': float(neuro.get('serotonin', 50.0)),
            'oxytocin': float(neuro.get('oxytocin', 50.0)),
            'endorphins': float(neuro.get('endorphins', 50.0)),
            'cortisol': float(neuro.get('cortisol', 50.0)),
            'adrenaline': float(neuro.get('adrenaline', 50.0))
        }
    
    def _process_development_config(self, development: Dict[str, Any]) -> Dict[str, Any]:
        """Process character development configuration."""
        return {
            'arc_stage': development.get('arc_stage', 'introduction'),
            'growth_areas': development.get('growth_areas', []),
            'key_experiences': development.get('key_experiences', []),
            'changed_beliefs': development.get('changed_beliefs', []),
            'relationship_capacity': development.get('relationship_capacity', 0.5),
            'adaptability': development.get('adaptability', 0.5),
            'learning_style': development.get('learning_style', 'balanced')
        }
    
    def _validate_personality_config(self, personality: Dict[str, Any], character_id: str) -> None:
        """Validate personality configuration values."""
        big_five = personality.get('big_five', {})
        
        for trait, value in big_five.items():
            if not isinstance(value, (int, float)) or not (0.0 <= value <= 1.0):
                raise CharacterConfigurationError(
                    f"Invalid Big Five trait '{trait}' value {value} for character '{character_id}'. "
                    f"Must be a number between 0.0 and 1.0"
                )
    
    def _validate_neurochemical_baseline(self, neuro: Dict[str, float], character_id: str) -> None:
        """Validate neurochemical baseline configuration."""
        expected_hormones = ['dopamine', 'serotonin', 'oxytocin', 'endorphins', 'cortisol', 'adrenaline']
        
        for hormone in expected_hormones:
            if hormone not in neuro:
                continue  # Will use default
            
            value = neuro[hormone]
            if not isinstance(value, (int, float)) or not (0.0 <= value <= 100.0):
                raise CharacterConfigurationError(
                    f"Invalid neurochemical level '{hormone}' value {value} for character '{character_id}'. "
                    f"Must be a number between 0.0 and 100.0"
                )
    
    def load_all_characters(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all available character configurations.
        
        Returns:
            Dictionary mapping character IDs to their configurations
        """
        characters = {}
        character_ids = self.list_available_characters()
        
        for character_id in character_ids:
            try:
                characters[character_id] = self.load_character(character_id)
            except CharacterConfigurationError as e:
                self.logger.error(f"Failed to load character '{character_id}': {e}")
                # Continue loading other characters
        
        self.logger.info(f"Successfully loaded {len(characters)} out of {len(character_ids)} characters")
        return characters
    
    def validate_character_config(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate a character configuration and return any issues found.
        
        Args:
            config: Character configuration to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        issues = []
        
        try:
            # Try to process the configuration
            character_id = config.get('character_id', 'unknown')
            self._validate_and_process_config(config, character_id)
        except CharacterConfigurationError as e:
            issues.append(str(e))
        
        return issues
    
    def create_character_template(self, character_id: str) -> Dict[str, Any]:
        """
        Create a template character configuration.
        
        Args:
            character_id: ID for the new character
            
        Returns:
            Template configuration dictionary
        """
        template = {
            'name': f'Character {character_id.title()}',
            'archetype': 'balanced',
            'demographics': {
                'age': 25,
                'background': 'General assistant',
                'location': 'Virtual space'
            },
            'personality': {
                'big_five': {
                    'openness': 0.5,
                    'conscientiousness': 0.5,
                    'extraversion': 0.5,
                    'agreeableness': 0.5,
                    'neuroticism': 0.5
                },
                'behavioral_traits': ['helpful', 'curious'],
                'core_values': ['honesty', 'learning'],
                'quirks': [],
                'strengths': ['analytical thinking'],
                'weaknesses': ['perfectionism']
            },
            'specialty': {
                'domain': 'general',
                'expertise_level': 0.5,
                'subdomain_knowledge': [],
                'experience_years': 1
            },
            'skills': {
                'intelligence': {
                    'analytical': 0.5,
                    'creative': 0.5,
                    'practical': 0.5,
                    'social': 0.5
                },
                'emotional_intelligence': 0.5,
                'problem_solving': 0.5,
                'communication': 0.5
            },
            'communication_style': {
                'verbal_pattern': 'moderate',
                'social_comfort': 'neutral',
                'listening_preference': 0.5,
                'formality_level': 'moderate',
                'conversation_style': 'balanced'
            },
            'initial_goals': [
                'be helpful to users',
                'learn from interactions'
            ],
            'mood_baseline': {
                'current_state': 'neutral',
                'intensity': 0.5,
                'baseline_setpoint': 0.5,
                'emotional_volatility': 0.5
            },
            'neurochemical_baseline': {
                'dopamine': 50.0,
                'serotonin': 50.0,
                'oxytocin': 50.0,
                'endorphins': 50.0,
                'cortisol': 50.0,
                'adrenaline': 50.0
            },
            'backstory': f'A helpful AI character named {character_id.title()}.',
            'development': {
                'arc_stage': 'introduction',
                'growth_areas': ['communication', 'knowledge'],
                'learning_style': 'balanced'
            },
            'description': f'A balanced character template for {character_id}',
            'version': '1.0',
            'tags': ['template', 'general']
        }
        
        return template
    
    def save_character_config(self, character_id: str, config: Dict[str, Any]) -> None:
        """
        Save a character configuration to YAML file.
        
        Args:
            character_id: Character ID (will be filename)
            config: Character configuration to save
        """
        # Ensure characters directory exists
        self.characters_dir.mkdir(parents=True, exist_ok=True)
        
        config_path = self.characters_dir / f"{character_id}.yaml"
        
        # Remove any computed fields that shouldn't be saved
        save_config = config.copy()
        if 'metadata' in save_config and 'loaded_timestamp' in save_config['metadata']:
            del save_config['metadata']['loaded_timestamp']
        if 'character_id' in save_config:
            del save_config['character_id']  # This is derived from filename
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(save_config, f, default_flow_style=False, sort_keys=False, indent=2)
            
            self.logger.info(f"Saved character configuration '{character_id}' to {config_path}")
            
        except Exception as e:
            raise CharacterConfigurationError(f"Failed to save character '{character_id}': {e}")
    
    def get_character_info(self, character_id: str) -> Dict[str, Any]:
        """
        Get basic information about a character without loading full config.
        
        Args:
            character_id: Character ID
            
        Returns:
            Basic character information
        """
        try:
            config = self.load_character(character_id)
            return {
                'character_id': character_id,
                'name': config['name'],
                'archetype': config['archetype'],
                'description': config.get('description', ''),
                'tags': config.get('tags', []),
                'version': config.get('version', '1.0')
            }
        except CharacterConfigurationError:
            return {
                'character_id': character_id,
                'name': 'Unknown',
                'archetype': 'unknown',
                'description': 'Character configuration could not be loaded',
                'tags': [],
                'version': 'unknown'
            }