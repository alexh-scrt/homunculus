"""State updater for managing character state changes over time."""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import math

try:
    from ..core.character_state import CharacterState
except ImportError:
    from core.character_state import CharacterState

try:
    from ..core.experience import Experience
except ImportError:
    from core.experience import Experience

try:
    from ..agents.memory_agent import MemoryAgent
except ImportError:
    from agents.memory_agent import MemoryAgent


class StateUpdater:
    """
    Manages character state updates including hormone decay, mood changes,
    and memory formation from interactions.
    
    This module bridges the response generation with persistent state management,
    ensuring character evolution over time while maintaining consistency.
    """
    
    def __init__(
        self,
        character_id: str,
        memory_agent: MemoryAgent,
        state_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the state updater."""
        self.character_id = character_id
        self.memory_agent = memory_agent
        self.state_config = state_config or {}
        self.logger = logging.getLogger(__name__)
        
        # State update parameters
        self.hormone_decay_rate = self.state_config.get('hormone_decay_rate', 0.02)
        self.mood_inertia = self.state_config.get('mood_inertia', 0.8)
        self.memory_formation_threshold = self.state_config.get('memory_formation_threshold', 0.6)
        self.state_persistence_interval = self.state_config.get('persistence_interval', 300)  # 5 minutes
        
        # Track last update times
        self.last_hormone_update = datetime.now()
        self.last_mood_update = datetime.now()
        self.last_persistence = datetime.now()
        
        self.logger.info(f"StateUpdater initialized for character {character_id}")
    
    async def update_character_state(
        self,
        character_state: CharacterState,
        interaction_data: Dict[str, Any],
        response_metadata: Dict[str, Any]
    ) -> CharacterState:
        """
        Update character state based on interaction and response generation.
        
        Args:
            character_state: Current character state
            interaction_data: Data about the interaction (user message, context, etc.)
            response_metadata: Metadata from response generation
            
        Returns:
            Updated character state
        """
        try:
            # Create a copy to avoid modifying the original
            updated_state = self._copy_character_state(character_state)
            
            # Update neurochemical levels based on interaction
            updated_state = await self._update_neurochemical_state(
                updated_state, interaction_data, response_metadata
            )
            
            # Apply natural hormone decay
            updated_state = self._apply_hormone_decay(updated_state)
            
            # Update mood based on neurochemical changes
            updated_state = self._update_mood_state(
                updated_state, interaction_data, response_metadata
            )
            
            # Update agent states based on interaction
            updated_state = self._update_agent_states(
                updated_state, interaction_data, response_metadata
            )
            
            # Create memory if interaction is significant
            await self._process_memory_formation(
                updated_state, interaction_data, response_metadata
            )
            
            # Update timestamps
            updated_state.last_updated = datetime.now()
            
            # Persist state if enough time has passed
            if self._should_persist_state():
                await self._persist_character_state(updated_state)
            
            self.logger.debug(f"Character state updated for {self.character_id}")
            return updated_state
            
        except Exception as e:
            self.logger.error(f"State update failed: {e}")
            # Return original state if update fails
            return character_state
    
    def _copy_character_state(self, state: CharacterState) -> CharacterState:
        """Create a deep copy of character state."""
        new_state = CharacterState(
            character_id=state.character_id,
            last_updated=state.last_updated,
            name=state.name,
            archetype=state.archetype,
            demographics=state.demographics.copy() if hasattr(state, 'demographics') else {},
            neurochemical_levels=state.neurochemical_levels.copy(),
            agent_states={k: v.copy() if isinstance(v, dict) else v 
                         for k, v in state.agent_states.items()},
            conversation_history=state.conversation_history.copy() if hasattr(state, 'conversation_history') else [],
            relationship_state=state.relationship_state.copy() if hasattr(state, 'relationship_state') else {},
            web_search_history=state.web_search_history.copy() if hasattr(state, 'web_search_history') else [],
            knowledge_updates=state.knowledge_updates.copy() if hasattr(state, 'knowledge_updates') else []
        )
        
        # Copy additional dynamic fields if they exist
        if hasattr(state, 'interaction_count'):
            new_state.interaction_count = state.interaction_count
        if hasattr(state, 'total_response_time'):
            new_state.total_response_time = state.total_response_time
        if hasattr(state, 'average_confidence'):
            new_state.average_confidence = state.average_confidence
            
        return new_state
    
    async def _update_neurochemical_state(
        self,
        character_state: CharacterState,
        interaction_data: Dict[str, Any],
        response_metadata: Dict[str, Any]
    ) -> CharacterState:
        """Update neurochemical levels based on interaction."""
        
        # Extract relevant information
        emotional_tone = response_metadata.get('emotional_tone', 'neutral')
        confidence_level = response_metadata.get('confidence_level', 0.5)
        user_message = interaction_data.get('user_message', '')
        
        # Calculate neurochemical adjustments
        adjustments = self._calculate_neurochemical_adjustments(
            emotional_tone, confidence_level, user_message
        )
        
        # Apply adjustments to current levels
        current_levels = character_state.neurochemical_levels
        new_levels = {}
        
        for hormone, current_level in current_levels.items():
            adjustment = adjustments.get(hormone, 0)
            
            # Apply adjustment with bounds checking
            new_level = current_level + adjustment
            new_level = max(0, min(100, new_level))  # Clamp to 0-100
            
            new_levels[hormone] = new_level
        
        character_state.neurochemical_levels = new_levels
        
        self.logger.debug(f"Neurochemical levels updated: {adjustments}")
        return character_state
    
    def _calculate_neurochemical_adjustments(
        self,
        emotional_tone: str,
        confidence_level: float,
        user_message: str
    ) -> Dict[str, float]:
        """Calculate neurochemical level adjustments based on interaction."""
        adjustments = {
            'dopamine': 0,
            'serotonin': 0,
            'oxytocin': 0,
            'endorphins': 0,
            'cortisol': 0,
            'adrenaline': 0
        }
        
        # Base adjustments by emotional tone
        tone_adjustments = {
            'happy': {'dopamine': 8, 'serotonin': 6, 'endorphins': 4, 'cortisol': -3},
            'excited': {'dopamine': 10, 'adrenaline': 7, 'endorphins': 3, 'serotonin': 2},
            'confident': {'dopamine': 6, 'serotonin': 8, 'cortisol': -5, 'adrenaline': 2},
            'calm': {'serotonin': 8, 'cortisol': -6, 'adrenaline': -4, 'endorphins': 2},
            'curious': {'dopamine': 5, 'adrenaline': 3, 'serotonin': 2},
            'anxious': {'cortisol': 8, 'adrenaline': 6, 'serotonin': -4, 'dopamine': -3},
            'sad': {'serotonin': -6, 'dopamine': -4, 'cortisol': 4, 'endorphins': -2},
            'frustrated': {'cortisol': 6, 'adrenaline': 5, 'serotonin': -3, 'dopamine': -2},
            'tired': {'dopamine': -4, 'adrenaline': -6, 'serotonin': -2, 'cortisol': 2},
            'neutral': {}  # No adjustments for neutral
        }
        
        base_adj = tone_adjustments.get(emotional_tone, {})
        for hormone, change in base_adj.items():
            adjustments[hormone] += change
        
        # Confidence modulates the adjustments
        confidence_multiplier = 0.5 + (confidence_level * 0.5)  # 0.5 to 1.0
        for hormone in adjustments:
            adjustments[hormone] *= confidence_multiplier
        
        # Social interaction boosts oxytocin
        if any(word in user_message.lower() for word in ['thank', 'love', 'friend', 'help', 'appreciate']):
            adjustments['oxytocin'] += 4
        
        # Questions or learning boost dopamine
        if '?' in user_message or any(word in user_message.lower() for word in ['learn', 'how', 'what', 'why']):
            adjustments['dopamine'] += 3
        
        # Stress indicators increase cortisol
        if any(word in user_message.lower() for word in ['problem', 'issue', 'wrong', 'error', 'help']):
            adjustments['cortisol'] += 2
        
        return adjustments
    
    def _apply_hormone_decay(self, character_state: CharacterState) -> CharacterState:
        """Apply natural hormone decay over time."""
        
        current_time = datetime.now()
        time_delta = (current_time - self.last_hormone_update).total_seconds()
        
        # Only apply decay if enough time has passed (minimum 1 minute)
        if time_delta < 60:
            return character_state
        
        # Calculate decay amount based on time passed
        decay_minutes = time_delta / 60.0
        decay_amount = self.hormone_decay_rate * decay_minutes
        
        current_levels = character_state.neurochemical_levels
        new_levels = {}
        
        # Target levels for decay (hormones naturally return toward these values)
        target_levels = {
            'dopamine': 50,
            'serotonin': 55,
            'oxytocin': 45,
            'endorphins': 40,
            'cortisol': 30,
            'adrenaline': 25
        }
        
        for hormone, current_level in current_levels.items():
            target = target_levels.get(hormone, 50)
            
            # Decay toward target level
            if current_level > target:
                new_level = max(target, current_level - decay_amount * 100)
            elif current_level < target:
                new_level = min(target, current_level + decay_amount * 50)  # Slower recovery
            else:
                new_level = current_level
            
            new_levels[hormone] = new_level
        
        character_state.neurochemical_levels = new_levels
        self.last_hormone_update = current_time
        
        self.logger.debug(f"Applied hormone decay over {decay_minutes:.1f} minutes")
        return character_state
    
    def _update_mood_state(
        self,
        character_state: CharacterState,
        interaction_data: Dict[str, Any],
        response_metadata: Dict[str, Any]
    ) -> CharacterState:
        """Update mood state based on neurochemical changes and interaction."""
        
        current_mood = character_state.agent_states.get('mood', {})
        
        # Calculate new mood based on neurochemical state
        new_mood = self._calculate_mood_from_neurochemistry(character_state.neurochemical_levels)
        
        # Apply mood inertia (mood doesn't change instantly)
        blended_mood = {}
        for key in ['current_state', 'energy_level', 'sociability', 'focus_level']:
            if key in current_mood and key in new_mood:
                current_val = current_mood[key]
                new_val = new_mood[key]
                
                # Blend with inertia
                if isinstance(current_val, (int, float)) and isinstance(new_val, (int, float)):
                    blended_val = current_val * self.mood_inertia + new_val * (1 - self.mood_inertia)
                    blended_mood[key] = blended_val
                else:
                    # For string values, use threshold-based switching
                    blended_mood[key] = new_val if abs(hash(new_val) % 100) > 70 else current_val
            elif key in new_mood:
                blended_mood[key] = new_mood[key]
            elif key in current_mood:
                blended_mood[key] = current_mood[key]
        
        # Add interaction-specific mood modifiers
        emotional_tone = response_metadata.get('emotional_tone', 'neutral')
        if emotional_tone != 'neutral':
            blended_mood['recent_interaction_tone'] = emotional_tone
            blended_mood['interaction_impact'] = response_metadata.get('confidence_level', 0.5)
        
        # Preserve other mood fields from current state
        final_mood = current_mood.copy()
        final_mood.update(blended_mood)
        
        character_state.agent_states['mood'] = final_mood
        self.last_mood_update = datetime.now()
        
        self.logger.debug(f"Mood state updated based on neurochemical changes")
        return character_state
    
    def _calculate_mood_from_neurochemistry(self, levels: Dict[str, float]) -> Dict[str, Any]:
        """Calculate mood state from neurochemical levels."""
        
        # Determine current mood state
        current_state = "neutral"
        
        if levels['dopamine'] > 70 and levels['serotonin'] > 60:
            current_state = "happy"
        elif levels['dopamine'] > 75 and levels['adrenaline'] > 65:
            current_state = "excited"
        elif levels['cortisol'] > 70:
            current_state = "anxious"
        elif levels['serotonin'] < 30 and levels['dopamine'] < 40:
            current_state = "sad"
        elif levels['cortisol'] > 60 and levels['adrenaline'] > 60:
            current_state = "stressed"
        elif levels['serotonin'] > 70 and levels['cortisol'] < 40:
            current_state = "calm"
        elif levels['dopamine'] > 60 and levels['adrenaline'] < 40:
            current_state = "content"
        
        # Calculate energy level (0-100)
        energy_level = (
            levels['dopamine'] * 0.3 +
            levels['adrenaline'] * 0.4 +
            levels['endorphins'] * 0.2 -
            levels['cortisol'] * 0.1
        )
        energy_level = max(0, min(100, energy_level))
        
        # Calculate sociability (0-100)
        sociability = (
            levels['oxytocin'] * 0.4 +
            levels['serotonin'] * 0.3 +
            levels['dopamine'] * 0.2 -
            levels['cortisol'] * 0.1
        )
        sociability = max(0, min(100, sociability))
        
        # Calculate focus level (0-100)
        focus_level = (
            levels['dopamine'] * 0.3 +
            levels['adrenaline'] * 0.2 -
            levels['cortisol'] * 0.3 +
            levels['serotonin'] * 0.2
        )
        focus_level = max(0, min(100, focus_level))
        
        return {
            'current_state': current_state,
            'energy_level': energy_level,
            'sociability': sociability,
            'focus_level': focus_level,
            'neurochemical_basis': {
                'dominant_hormone': max(levels, key=levels.get),
                'hormone_balance': 'balanced' if max(levels.values()) - min(levels.values()) < 30 else 'imbalanced'
            }
        }
    
    def _update_agent_states(
        self,
        character_state: CharacterState,
        interaction_data: Dict[str, Any],
        response_metadata: Dict[str, Any]
    ) -> CharacterState:
        """Update agent-specific states based on interaction."""
        
        agent_states = character_state.agent_states.copy()
        
        # Update interaction statistics
        interaction_count = getattr(character_state, 'interaction_count', 0)
        character_state.interaction_count = interaction_count + 1
        
        # Update confidence tracking
        confidence = response_metadata.get('confidence_level', 0.5)
        current_avg_confidence = getattr(character_state, 'average_confidence', None)
        
        if current_avg_confidence is None:
            character_state.average_confidence = confidence
        else:
            # Running average
            total_interactions = character_state.interaction_count
            character_state.average_confidence = (
                (current_avg_confidence * (total_interactions - 1) + confidence) / 
                total_interactions
            )
        
        # Update communication style state
        if 'communication_style' not in agent_states:
            agent_states['communication_style'] = {}
        
        style_state = agent_states['communication_style']
        emotional_tone = response_metadata.get('emotional_tone', 'neutral')
        
        # Track tone usage
        if 'tone_history' not in style_state:
            style_state['tone_history'] = {}
        style_state['tone_history'][emotional_tone] = style_state['tone_history'].get(emotional_tone, 0) + 1
        
        # Update most common tone
        style_state['dominant_tone'] = max(style_state['tone_history'], key=style_state['tone_history'].get)
        
        # Update goals state
        if 'goals' not in agent_states:
            agent_states['goals'] = {}
        
        goals_state = agent_states['goals']
        
        # Track goal-related interactions
        user_message = interaction_data.get('user_message', '').lower()
        if any(word in user_message for word in ['goal', 'want', 'need', 'plan', 'achieve']):
            goals_state['goal_related_interactions'] = goals_state.get('goal_related_interactions', 0) + 1
        
        character_state.agent_states = agent_states
        return character_state
    
    async def _process_memory_formation(
        self,
        character_state: CharacterState,
        interaction_data: Dict[str, Any],
        response_metadata: Dict[str, Any]
    ) -> None:
        """Process memory formation if interaction is significant enough."""
        
        # Determine if interaction should form a memory
        should_form_memory = self._should_form_memory(interaction_data, response_metadata)
        
        if not should_form_memory:
            return
        
        try:
            # Extract memory information
            user_message = interaction_data.get('user_message', '')
            character_response = response_metadata.get('response_text', '')
            emotional_tone = response_metadata.get('emotional_tone', 'neutral')
            
            # Store the interaction as a memory
            await self.memory_agent.store_interaction_memory(
                user_message=user_message,
                character_response=character_response,
                character_state=character_state,
                emotional_context={
                    'emotional_tone': emotional_tone,
                    'confidence_level': response_metadata.get('confidence_level', 0.5),
                    'neurochemical_state': character_state.neurochemical_levels.copy(),
                    'mood_state': character_state.agent_states.get('mood', {}).copy()
                }
            )
            
            self.logger.debug("Memory formed from significant interaction")
            
        except Exception as e:
            self.logger.error(f"Memory formation failed: {e}")
    
    def _should_form_memory(
        self,
        interaction_data: Dict[str, Any],
        response_metadata: Dict[str, Any]
    ) -> bool:
        """Determine if an interaction should form a memory."""
        
        # High confidence interactions form memories
        confidence = response_metadata.get('confidence_level', 0.5)
        if confidence > self.memory_formation_threshold:
            return True
        
        # Strong emotional responses form memories
        emotional_tone = response_metadata.get('emotional_tone', 'neutral')
        if emotional_tone in ['excited', 'anxious', 'happy', 'frustrated', 'sad']:
            return True
        
        # Questions or learning interactions form memories
        user_message = interaction_data.get('user_message', '').lower()
        if any(word in user_message for word in ['?', 'learn', 'teach', 'explain', 'how', 'why', 'what']):
            return True
        
        # Social interactions form memories
        if any(word in user_message for word in ['thank', 'friend', 'love', 'hate', 'appreciate']):
            return True
        
        return False
    
    def _should_persist_state(self) -> bool:
        """Determine if character state should be persisted."""
        time_since_last = (datetime.now() - self.last_persistence).total_seconds()
        return time_since_last >= self.state_persistence_interval
    
    async def _persist_character_state(self, character_state: CharacterState) -> None:
        """Persist character state to storage."""
        try:
            # This would normally save to a database or file
            # For now, just update the timestamp
            self.last_persistence = datetime.now()
            
            self.logger.debug(f"Character state persisted for {self.character_id}")
            
        except Exception as e:
            self.logger.error(f"State persistence failed: {e}")
    
    def apply_natural_decay(self, character_state: CharacterState, time_passed_hours: float) -> CharacterState:
        """Apply natural decay when character is inactive for extended periods."""
        
        if time_passed_hours < 1.0:
            return character_state
        
        # Create updated state
        updated_state = self._copy_character_state(character_state)
        
        # Apply extended decay
        decay_factor = min(0.5, time_passed_hours * 0.1)  # Max 50% decay
        
        current_levels = updated_state.neurochemical_levels
        target_levels = {
            'dopamine': 45,
            'serotonin': 50,
            'oxytocin': 40,
            'endorphins': 35,
            'cortisol': 25,
            'adrenaline': 20
        }
        
        # Apply decay toward resting levels
        new_levels = {}
        for hormone, current_level in current_levels.items():
            target = target_levels.get(hormone, 45)
            new_level = current_level + (target - current_level) * decay_factor
            new_levels[hormone] = max(0, min(100, new_level))
        
        updated_state.neurochemical_levels = new_levels
        
        # Update mood based on new neurochemical state
        new_mood = self._calculate_mood_from_neurochemistry(new_levels)
        current_mood = updated_state.agent_states.get('mood', {})
        current_mood.update(new_mood)
        updated_state.agent_states['mood'] = current_mood
        
        # Reset agent states to neutral
        for agent_type in updated_state.agent_states:
            if 'energy_level' in updated_state.agent_states[agent_type]:
                current_energy = updated_state.agent_states[agent_type]['energy_level']
                updated_state.agent_states[agent_type]['energy_level'] = max(30, current_energy * (1 - decay_factor))
        
        self.logger.info(f"Applied natural decay after {time_passed_hours:.1f} hours of inactivity")
        return updated_state
    
    def get_state_update_stats(self) -> Dict[str, Any]:
        """Get statistics about state updates."""
        return {
            'character_id': self.character_id,
            'hormone_decay_rate': self.hormone_decay_rate,
            'mood_inertia': self.mood_inertia,
            'memory_formation_threshold': self.memory_formation_threshold,
            'last_hormone_update': self.last_hormone_update.isoformat(),
            'last_mood_update': self.last_mood_update.isoformat(),
            'last_persistence': self.last_persistence.isoformat(),
            'state_config': self.state_config
        }