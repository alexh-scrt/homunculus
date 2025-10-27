"""Main CharacterAgent class that orchestrates all character components."""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import json

try:
    from .core.character_state import CharacterState
except ImportError:
    from core.character_state import CharacterState

try:
    from .integration.agent_orchestrator import AgentOrchestrator
except ImportError:
    from integration.agent_orchestrator import AgentOrchestrator

try:
    from .integration.cognitive_module import CognitiveModule
except ImportError:
    from integration.cognitive_module import CognitiveModule

try:
    from .integration.response_generator import ResponseGenerator
except ImportError:
    from integration.response_generator import ResponseGenerator

try:
    from .integration.state_updater import StateUpdater
except ImportError:
    from integration.state_updater import StateUpdater

try:
    from .agents.memory_agent import MemoryAgent
except ImportError:
    from agents.memory_agent import MemoryAgent

try:
    from .llm.ollama_client import OllamaClient
except ImportError:
    from llm.ollama_client import OllamaClient

try:
    from .config.settings import get_settings
except ImportError:
    from config.settings import get_settings


class CharacterAgent:
    """
    Main character agent that orchestrates all character components.
    
    This is the primary interface for interacting with a character. It coordinates
    all the specialized agents, manages state, and generates responses while
    maintaining the character's personality, memory, and growth over time.
    """
    
    def __init__(
        self,
        character_id: str,
        character_config: Dict[str, Any],
        llm_client: Optional[OllamaClient] = None,
        agent_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the character agent.
        
        Args:
            character_id: Unique identifier for this character
            character_config: Character configuration (loaded from YAML)
            llm_client: Optional LLM client (will create default if not provided)
            agent_config: Optional configuration for agent behavior
        """
        self.character_id = character_id
        self.character_config = character_config
        self.agent_config = agent_config or {}
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)
        
        # Initialize LLM client
        if llm_client is None:
            self.llm_client = OllamaClient(
                base_url=self.settings.ollama_base_url,
                model=self.settings.ollama_model,
                tavily_api_key=self.settings.tavily_api_key,
                character_id=character_id
            )
        else:
            self.llm_client = llm_client
        
        # Initialize character state
        self.character_state = self._initialize_character_state()
        
        # Initialize core components
        self.agent_orchestrator = AgentOrchestrator(
            character_id=character_id,
            llm_client=self.llm_client,
            character_config=character_config
        )
        
        self.cognitive_module = CognitiveModule(
            character_id=character_id,
            llm_client=self.llm_client,
            cognitive_config=self.agent_config.get('cognitive', {})
        )
        
        self.response_generator = ResponseGenerator(
            character_id=character_id,
            llm_client=self.llm_client,
            generation_config=self.agent_config.get('generation', {})
        )
        
        # Get memory agent from orchestrator for state updater
        memory_agent = self.agent_orchestrator.agents.get('memory')
        
        self.state_updater = StateUpdater(
            character_id=character_id,
            memory_agent=memory_agent,
            state_config=self.agent_config.get('state_update', {})
        )
        
        # Conversation context
        self.conversation_context = {
            'topic': 'introduction',
            'relationship_stage': 'first_meeting',
            'conversation_start': datetime.now(),
            'interaction_count': 0
        }
        
        # Performance tracking
        self.performance_stats = {
            'total_responses': 0,
            'total_response_time': 0.0,
            'average_response_time': 0.0,
            'successful_responses': 0,
            'failed_responses': 0,
            'memory_formations': 0,
            'web_searches_triggered': 0
        }
        
        self.logger.info(f"CharacterAgent initialized for {character_id}")
    
    @property
    def character_name(self) -> str:
        """Get the character's name."""
        return self.character_state.name
    
    @property
    def state(self) -> CharacterState:
        """Get the current character state."""
        return self.character_state
    
    async def initialize(self) -> None:
        """Initialize the character agent (async operations)."""
        try:
            # Initialize any async components if needed
            # For now, most initialization is done in __init__
            self.logger.info(f"CharacterAgent async initialization completed for {self.character_id}")
        except Exception as e:
            self.logger.error(f"Error during async initialization: {e}")
            raise
    
    def _initialize_character_state(self) -> CharacterState:
        """Initialize character state from configuration."""
        
        # Extract basic info from config
        name = self.character_config.get('name', f'Character_{self.character_id}')
        archetype = self.character_config.get('archetype', 'balanced')
        demographics = self.character_config.get('demographics', {})
        
        # Create initial state
        state = CharacterState(
            character_id=self.character_id,
            last_updated=datetime.now(),
            name=name,
            archetype=archetype,
            demographics=demographics
        )
        
        # Initialize neurochemical levels from config if provided
        if 'neurochemical_baseline' in self.character_config:
            baseline = self.character_config['neurochemical_baseline']
            for hormone, level in baseline.items():
                if hormone in state.neurochemical_levels:
                    state.neurochemical_levels[hormone] = level
        
        # Add character-specific agent states
        self._configure_agent_states(state)
        
        return state
    
    def _configure_agent_states(self, state: CharacterState) -> None:
        """Configure agent states from character config."""
        
        # Update personality state
        if 'personality' in self.character_config:
            personality_config = self.character_config['personality']
            
            if 'big_five' in personality_config:
                state.agent_states['personality']['big_five'] = personality_config['big_five'].copy()
            
            if 'behavioral_traits' in personality_config:
                state.agent_states['personality']['behavioral_traits'] = personality_config['behavioral_traits'].copy()
            
            if 'core_values' in personality_config:
                state.agent_states['personality']['core_values'] = personality_config['core_values'].copy()
        
        # Update specialty state
        if 'specialty' in self.character_config:
            specialty_config = self.character_config['specialty']
            state.agent_states['specialty'].update(specialty_config)
        
        # Update skills state
        if 'skills' in self.character_config:
            skills_config = self.character_config['skills']
            state.agent_states['skills'].update(skills_config)
        
        # Update communication style
        if 'communication_style' in self.character_config:
            style_config = self.character_config['communication_style']
            state.agent_states['communication_style'].update(style_config)
        
        # Initialize goals from config
        if 'initial_goals' in self.character_config:
            initial_goals = self.character_config['initial_goals']
            state.agent_states['goals']['active_goals'] = initial_goals.copy()
        
        # Set mood baseline
        if 'mood_baseline' in self.character_config:
            mood_baseline = self.character_config['mood_baseline']
            state.agent_states['mood'].update(mood_baseline)
    
    async def process_message(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a user message and generate a character response.
        
        Args:
            user_message: The user's input message
            context: Optional additional context
            
        Returns:
            Complete response package with text, metadata, and insights
        """
        start_time = datetime.now()
        
        try:
            # Update conversation context
            self._update_conversation_context(user_message, context)
            
            # Add user message to history
            self.character_state.add_to_history('user', user_message)
            
            # Step 1: Agent Orchestration
            self.logger.debug("Starting agent orchestration")
            orchestration_result = await self.agent_orchestrator.orchestrate_response(
                context=self.conversation_context,
                character_state=self.character_state,
                user_message=user_message
            )
            
            # Step 2: Cognitive Processing
            self.logger.debug("Starting cognitive processing")
            cognitive_result = self.cognitive_module.process_orchestration_result(
                orchestration_result=orchestration_result,
                character_state=self.character_state,
                user_message=user_message,
                context=self.conversation_context
            )
            
            # Step 3: Response Generation
            self.logger.debug("Starting response generation")
            response_result = await self.response_generator.generate_response(
                orchestration_result=orchestration_result,
                cognitive_result=cognitive_result,
                character_state=self.character_state,
                user_message=user_message,
                context=self.conversation_context
            )
            
            # Step 4: State Update
            self.logger.debug("Starting state update")
            interaction_data = {
                'user_message': user_message,
                'context': self.conversation_context,
                'timestamp': start_time
            }
            
            self.character_state = await self.state_updater.update_character_state(
                character_state=self.character_state,
                interaction_data=interaction_data,
                response_metadata=response_result['response_metadata']
            )
            
            # Add character response to history
            character_response = response_result['response_text']
            self.character_state.add_to_history('character', character_response)
            
            # Update performance stats
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            self._update_performance_stats(response_time, success=True)
            
            # Track web search usage
            if any('web_search' in str(agent_input.metadata) for agent_input in orchestration_result.get('agent_inputs', {}).values()):
                self.performance_stats['web_searches_triggered'] += 1
            
            # Package complete response
            complete_response = {
                'response_text': character_response,
                'character_insights': response_result['character_insights'],
                'response_metadata': response_result['response_metadata'],
                'generation_info': response_result['generation_info'],
                'orchestration_summary': {
                    'agent_count': len(orchestration_result.get('agent_inputs', {})),
                    'conflicts_resolved': len(orchestration_result.get('conflicts_detected', [])),
                    'primary_agent': orchestration_result.get('synthesis', {}).get('primary_guidance').agent_type if orchestration_result.get('synthesis', {}).get('primary_guidance') else 'none'
                },
                'cognitive_summary': {
                    'thinking_style': cognitive_result.get('cognitive_patterns', {}).get('thinking_style', 'balanced'),
                    'cognitive_load': cognitive_result.get('cognitive_patterns', {}).get('cognitive_load', 0.5),
                    'response_strategy': cognitive_result.get('response_strategy', {}).get('approach', 'balanced')
                },
                'character_state_summary': {
                    'dominant_hormone': max(self.character_state.neurochemical_levels, key=self.character_state.neurochemical_levels.get),
                    'mood_state': self.character_state.agent_states.get('mood', {}).get('current_state', 'neutral'),
                    'interaction_count': getattr(self.character_state, 'interaction_count', 0),
                    'average_confidence': getattr(self.character_state, 'average_confidence', 0.5)
                },
                'performance_info': {
                    'response_time_seconds': response_time,
                    'total_responses': self.performance_stats['total_responses'],
                    'average_response_time': self.performance_stats['average_response_time']
                },
                'timestamp': end_time.isoformat()
            }
            
            self.logger.info(f"Successfully processed message for {self.character_id} in {response_time:.2f}s")
            return complete_response
            
        except Exception as e:
            # Handle failure
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            self._update_performance_stats(response_time, success=False)
            
            self.logger.error(f"Failed to process message for {self.character_id}: {e}")
            
            # Return fallback response
            return self._create_fallback_response(user_message, str(e), response_time)
    
    def _update_conversation_context(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]]
    ) -> None:
        """Update conversation context based on new message."""
        
        # Update interaction count
        self.conversation_context['interaction_count'] += 1
        
        # Update relationship stage based on interaction count
        interaction_count = self.conversation_context['interaction_count']
        if interaction_count == 1:
            self.conversation_context['relationship_stage'] = 'first_meeting'
        elif interaction_count <= 5:
            self.conversation_context['relationship_stage'] = 'getting_acquainted'
        elif interaction_count <= 15:
            self.conversation_context['relationship_stage'] = 'building_rapport'
        elif interaction_count <= 50:
            self.conversation_context['relationship_stage'] = 'established_relationship'
        else:
            self.conversation_context['relationship_stage'] = 'deep_connection'
        
        # Simple topic detection
        user_lower = user_message.lower()
        if any(word in user_lower for word in ['programming', 'code', 'software', 'development']):
            self.conversation_context['topic'] = 'technology'
        elif any(word in user_lower for word in ['feeling', 'emotion', 'mood', 'happy', 'sad', 'angry']):
            self.conversation_context['topic'] = 'emotions'
        elif any(word in user_lower for word in ['goal', 'plan', 'future', 'dream', 'ambition']):
            self.conversation_context['topic'] = 'goals_and_aspirations'
        elif any(word in user_lower for word in ['learn', 'teach', 'explain', 'understand']):
            self.conversation_context['topic'] = 'learning'
        elif any(word in user_lower for word in ['friend', 'relationship', 'family', 'love']):
            self.conversation_context['topic'] = 'relationships'
        else:
            self.conversation_context['topic'] = 'general_conversation'
        
        # Merge additional context
        if context:
            self.conversation_context.update(context)
    
    def _update_performance_stats(self, response_time: float, success: bool) -> None:
        """Update performance statistics."""
        self.performance_stats['total_responses'] += 1
        self.performance_stats['total_response_time'] += response_time
        
        if success:
            self.performance_stats['successful_responses'] += 1
        else:
            self.performance_stats['failed_responses'] += 1
        
        # Update average response time
        if self.performance_stats['total_responses'] > 0:
            self.performance_stats['average_response_time'] = (
                self.performance_stats['total_response_time'] / 
                self.performance_stats['total_responses']
            )
    
    def _create_fallback_response(
        self,
        user_message: str,
        error_message: str,
        response_time: float
    ) -> Dict[str, Any]:
        """Create a fallback response when processing fails."""
        
        # Simple fallback responses
        fallback_responses = [
            "I'm having some trouble organizing my thoughts right now. Could you give me a moment?",
            "That's interesting. I need to think about that a bit more.",
            "I'm feeling a bit scattered at the moment. What's your take on this?",
            "Let me process that for a second... Could you rephrase that?"
        ]
        
        # Choose response based on message length
        response_index = len(user_message) % len(fallback_responses)
        fallback_text = fallback_responses[response_index]
        
        return {
            'response_text': fallback_text,
            'character_insights': {
                'emotional_tone': 'confused',
                'confidence_level': 0.2,
                'thinking_style': 'uncertain',
                'cognitive_load': 1.0
            },
            'response_metadata': {
                'character_id': self.character_id,
                'timestamp': datetime.now().isoformat(),
                'fallback': True,
                'error': error_message,
                'emotional_tone': 'confused',
                'confidence_level': 0.2
            },
            'generation_info': {
                'fallback_used': True,
                'error_message': error_message,
                'response_time_seconds': response_time
            },
            'orchestration_summary': {
                'agent_count': 0,
                'conflicts_resolved': 0,
                'primary_agent': 'none'
            },
            'cognitive_summary': {
                'thinking_style': 'error_state',
                'cognitive_load': 1.0,
                'response_strategy': 'fallback'
            },
            'character_state_summary': {
                'dominant_hormone': 'cortisol',
                'mood_state': 'confused',
                'interaction_count': getattr(self.character_state, 'interaction_count', 0),
                'average_confidence': 0.2
            },
            'performance_info': {
                'response_time_seconds': response_time,
                'total_responses': self.performance_stats['total_responses'],
                'average_response_time': self.performance_stats['average_response_time'],
                'error': True
            },
            'timestamp': datetime.now().isoformat()
        }
    
    async def apply_time_decay(self, hours_passed: float) -> None:
        """Apply natural time decay when character has been inactive."""
        if hours_passed > 0:
            self.character_state = self.state_updater.apply_natural_decay(
                self.character_state, 
                hours_passed
            )
            self.logger.info(f"Applied {hours_passed:.1f} hours of natural decay to {self.character_id}")
    
    def get_character_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of the character's current state."""
        return {
            'character_info': {
                'character_id': self.character_id,
                'name': self.character_state.name,
                'archetype': self.character_state.archetype,
                'demographics': self.character_state.demographics
            },
            'neurochemical_state': self.character_state.neurochemical_levels.copy(),
            'mood_state': self.character_state.agent_states.get('mood', {}),
            'agent_states': {
                'personality': self.character_state.agent_states.get('personality', {}),
                'communication_style': self.character_state.agent_states.get('communication_style', {}),
                'goals': self.character_state.agent_states.get('goals', {}),
                'specialty': self.character_state.agent_states.get('specialty', {})
            },
            'conversation_context': self.conversation_context.copy(),
            'performance_stats': self.performance_stats.copy(),
            'memory_stats': {
                'conversation_history_length': len(self.character_state.conversation_history),
                'web_search_history_length': len(self.character_state.web_search_history),
                'knowledge_updates_length': len(self.character_state.knowledge_updates)
            },
            'last_updated': self.character_state.last_updated.isoformat(),
            'interaction_count': getattr(self.character_state, 'interaction_count', 0),
            'average_confidence': getattr(self.character_state, 'average_confidence', 0.5)
        }
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict[str, str]]:
        """Get recent conversation history."""
        return self.character_state.get_recent_context(limit)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'character_id': self.character_id,
            'performance_stats': self.performance_stats.copy(),
            'agent_stats': self.agent_orchestrator.get_agent_stats(),
            'generation_stats': self.response_generator.get_generation_stats(),
            'state_update_stats': self.state_updater.get_state_update_stats()
        }
    
    async def reset_character_state(self, preserve_memories: bool = True) -> None:
        """Reset character state while optionally preserving memories."""
        
        if preserve_memories:
            # Store current memories
            conversation_backup = self.character_state.conversation_history.copy()
            web_search_backup = self.character_state.web_search_history.copy()
            knowledge_backup = self.character_state.knowledge_updates.copy()
        
        # Re-initialize character state
        self.character_state = self._initialize_character_state()
        
        if preserve_memories:
            # Restore memories
            self.character_state.conversation_history = conversation_backup
            self.character_state.web_search_history = web_search_backup
            self.character_state.knowledge_updates = knowledge_backup
        
        # Reset conversation context
        self.conversation_context = {
            'topic': 'introduction',
            'relationship_stage': 'first_meeting' if not preserve_memories else self.conversation_context.get('relationship_stage', 'first_meeting'),
            'conversation_start': datetime.now(),
            'interaction_count': 0 if not preserve_memories else self.conversation_context.get('interaction_count', 0)
        }
        
        self.logger.info(f"Character state reset for {self.character_id} (memories preserved: {preserve_memories})")
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get character state as dictionary for saving.
        
        Returns:
            Dictionary containing complete character state
        """
        # Create a copy of conversation_context with serializable datetime
        conversation_context_copy = self.conversation_context.copy()
        if 'conversation_start' in conversation_context_copy:
            conversation_context_copy['conversation_start'] = conversation_context_copy['conversation_start'].isoformat()
        
        return {
            'character_state': self.character_state.to_dict(),
            'conversation_context': conversation_context_copy,
            'performance_stats': self.performance_stats,
            'character_config': self.character_config,
            'save_timestamp': datetime.now().isoformat()
        }
    
    async def load_state_dict(self, state_data: Dict[str, Any]) -> None:
        """Load character state from dictionary.
        
        Args:
            state_data: Dictionary containing character state data
        """
        try:
            # Extract data
            character_state_dict = state_data['character_state']
            conversation_context = state_data.get('conversation_context', {})
            performance_stats = state_data.get('performance_stats', {})
            
            # Convert conversation_start back to datetime if present
            if 'conversation_start' in conversation_context and isinstance(conversation_context['conversation_start'], str):
                conversation_context['conversation_start'] = datetime.fromisoformat(conversation_context['conversation_start'])
            
            # Restore state
            self.character_state = CharacterState.from_dict(character_state_dict)
            self.conversation_context = conversation_context
            self.performance_stats.update(performance_stats)
            
            self.logger.info(f"Character state loaded from dictionary for {self.character_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to load character state from dictionary: {e}")
            raise
    
    async def recall_past_conversations(self, query: str = "", limit: int = 5) -> List[Dict[str, Any]]:
        """Recall past conversations based on query.
        
        Args:
            query: Search query for memories
            limit: Maximum number of memories to return
            
        Returns:
            List of memory dictionaries with similarity scores
        """
        try:
            # Get the memory agent from orchestrator
            memory_agent = self.agent_orchestrator.agents.get('memory')
            if not memory_agent:
                return []
            
            # Use the memory agent to retrieve relevant experiences
            memories = await memory_agent.retrieve_relevant_experiences(
                query=query,
                limit=limit
            )
            
            return memories
            
        except Exception as e:
            self.logger.error(f"Error recalling past conversations: {e}")
            return []
    
    def save_character_state(self, filepath: str) -> None:
        """Save character state to file."""
        try:
            # Create a copy of conversation_context with serializable datetime
            conversation_context_copy = self.conversation_context.copy()
            if 'conversation_start' in conversation_context_copy:
                conversation_context_copy['conversation_start'] = conversation_context_copy['conversation_start'].isoformat()
            
            state_data = {
                'character_state': self.character_state.to_dict(),
                'conversation_context': conversation_context_copy,
                'performance_stats': self.performance_stats,
                'character_config': self.character_config,
                'save_timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            self.logger.info(f"Character state saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save character state: {e}")
            raise
    
    @classmethod
    def load_character_state(
        cls,
        filepath: str,
        llm_client: Optional[OllamaClient] = None
    ) -> 'CharacterAgent':
        """Load character state from file."""
        try:
            with open(filepath, 'r') as f:
                state_data = json.load(f)
            
            # Extract data
            character_state_dict = state_data['character_state']
            conversation_context = state_data['conversation_context']
            performance_stats = state_data['performance_stats']
            character_config = state_data['character_config']
            
            # Convert conversation_start back to datetime if present
            if 'conversation_start' in conversation_context and isinstance(conversation_context['conversation_start'], str):
                conversation_context['conversation_start'] = datetime.fromisoformat(conversation_context['conversation_start'])
            
            # Create character agent
            character_id = character_state_dict['character_id']
            agent = cls(
                character_id=character_id,
                character_config=character_config,
                llm_client=llm_client
            )
            
            # Restore state
            agent.character_state = CharacterState.from_dict(character_state_dict)
            agent.conversation_context = conversation_context
            agent.performance_stats = performance_stats
            
            logging.getLogger(__name__).info(f"Character state loaded from {filepath}")
            return agent
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to load character state: {e}")
            raise
    
    def close(self):
        """Clean up resources."""
        try:
            if hasattr(self, 'agent_orchestrator'):
                self.agent_orchestrator.close()
            if hasattr(self, 'logger'):
                self.logger.info(f"CharacterAgent closed for {self.character_id}")
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error closing CharacterAgent: {e}")
            else:
                print(f"Error closing CharacterAgent: {e}")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.close()
        except:
            pass