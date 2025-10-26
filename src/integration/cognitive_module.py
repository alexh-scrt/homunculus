"""Cognitive module for processing and interpreting agent inputs."""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json

try:
    from ..core.character_state import CharacterState
except ImportError:
    from core.character_state import CharacterState

try:
    from ..core.agent_input import AgentInput
except ImportError:
    from core.agent_input import AgentInput


class CognitiveModule:
    """
    Processes agent inputs to understand the character's cognitive state.
    
    This module acts as the "mind" of the character, interpreting signals
    from various agents and forming a coherent understanding of how the
    character should think and respond.
    """
    
    def __init__(
        self,
        character_id: str,
        llm_client: Any,
        cognitive_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the cognitive module."""
        self.character_id = character_id
        self.llm_client = llm_client
        self.cognitive_config = cognitive_config or {}
        self.logger = logging.getLogger(__name__)
        
        # Cognitive processing parameters
        self.processing_depth = self.cognitive_config.get('processing_depth', 'moderate')
        self.reflection_tendency = self.cognitive_config.get('reflection_tendency', 0.5)
        self.emotional_awareness = self.cognitive_config.get('emotional_awareness', 0.7)
        self.analytical_tendency = self.cognitive_config.get('analytical_tendency', 0.6)
        
        self.logger.info(f"CognitiveModule initialized for character {character_id}")
    
    def process_orchestration_result(
        self,
        orchestration_result: Dict[str, Any],
        character_state: CharacterState,
        user_message: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process the orchestration result to understand cognitive state.
        
        Args:
            orchestration_result: Result from AgentOrchestrator
            character_state: Current character state
            user_message: User's message
            context: Conversation context
            
        Returns:
            Cognitive processing result with insights and recommendations
        """
        try:
            synthesis = orchestration_result.get('synthesis', {})
            agent_inputs = orchestration_result.get('agent_inputs', {})
            conflicts = orchestration_result.get('conflicts_detected', [])
            
            # Analyze cognitive patterns
            cognitive_patterns = self._analyze_cognitive_patterns(agent_inputs, character_state)
            
            # Process emotional intelligence
            emotional_intelligence = self._process_emotional_intelligence(
                synthesis, agent_inputs, character_state
            )
            
            # Generate cognitive insights
            cognitive_insights = self._generate_cognitive_insights(
                agent_inputs, cognitive_patterns, user_message
            )
            
            # Determine response strategy
            response_strategy = self._determine_response_strategy(
                synthesis, cognitive_patterns, emotional_intelligence, conflicts
            )
            
            # Create cognitive processing result
            cognitive_result = {
                'cognitive_patterns': cognitive_patterns,
                'emotional_intelligence': emotional_intelligence,
                'cognitive_insights': cognitive_insights,
                'response_strategy': response_strategy,
                'processing_metadata': {
                    'character_id': self.character_id,
                    'processing_depth': self.processing_depth,
                    'timestamp': datetime.now().isoformat(),
                    'conflict_count': len(conflicts)
                }
            }
            
            self.logger.debug(f"Cognitive processing completed for character {self.character_id}")
            return cognitive_result
            
        except Exception as e:
            self.logger.error(f"Cognitive processing failed: {e}")
            return self._create_fallback_cognitive_result(str(e))
    
    def _analyze_cognitive_patterns(
        self,
        agent_inputs: Dict[str, AgentInput],
        character_state: CharacterState
    ) -> Dict[str, Any]:
        """Analyze cognitive patterns from agent inputs."""
        patterns = {
            'thinking_style': self._determine_thinking_style(agent_inputs),
            'attention_focus': self._analyze_attention_focus(agent_inputs),
            'decision_making_mode': self._analyze_decision_making(agent_inputs),
            'cognitive_load': self._calculate_cognitive_load(agent_inputs),
            'information_processing': self._analyze_information_processing(agent_inputs)
        }
        
        return patterns
    
    def _determine_thinking_style(self, agent_inputs: Dict[str, AgentInput]) -> str:
        """Determine the character's current thinking style."""
        personality_input = agent_inputs.get('personality')
        mood_input = agent_inputs.get('mood')
        neurochemical_input = agent_inputs.get('neurochemical')
        
        # Default to balanced
        thinking_style = "balanced"
        
        if personality_input and personality_input.metadata:
            big_five = personality_input.metadata.get('big_five_scores', {})
            openness = big_five.get('openness', 0.5)
            conscientiousness = big_five.get('conscientiousness', 0.5)
            
            if openness > 0.7:
                thinking_style = "creative"
            elif conscientiousness > 0.7:
                thinking_style = "systematic"
            elif openness < 0.3 and conscientiousness < 0.3:
                thinking_style = "intuitive"
        
        # Mood can modify thinking style
        if mood_input and mood_input.metadata:
            mood_state = mood_input.metadata.get('mood_state', {})
            current_mood = mood_state.get('current_state', 'neutral')
            
            if current_mood in ['anxious', 'stressed']:
                thinking_style = "cautious"
            elif current_mood in ['excited', 'energetic']:
                thinking_style = "dynamic"
        
        return thinking_style
    
    def _analyze_attention_focus(self, agent_inputs: Dict[str, AgentInput]) -> Dict[str, Any]:
        """Analyze where the character's attention is focused."""
        focus_areas = {
            'internal_state': 0.0,
            'external_environment': 0.0,
            'relationships': 0.0,
            'goals': 0.0,
            'past_experiences': 0.0
        }
        
        # Neurochemical and mood inputs indicate internal focus
        if agent_inputs.get('neurochemical') and agent_inputs.get('mood'):
            internal_priority = (
                agent_inputs['neurochemical'].priority + 
                agent_inputs['mood'].priority
            ) / 2
            focus_areas['internal_state'] = internal_priority
        
        # Memory input indicates focus on past experiences
        if agent_inputs.get('memory'):
            memory_priority = agent_inputs['memory'].priority
            focus_areas['past_experiences'] = memory_priority
        
        # Goals input indicates future/strategic focus
        if agent_inputs.get('goals'):
            goals_priority = agent_inputs['goals'].priority
            focus_areas['goals'] = goals_priority
        
        # Communication style can indicate relationship focus
        if agent_inputs.get('communication_style'):
            style_input = agent_inputs['communication_style']
            if style_input.metadata and 'style_requirements' in style_input.metadata:
                style_req = style_input.metadata['style_requirements']
                if style_req.get('emotional_sensitivity_needed') == 'high':
                    focus_areas['relationships'] = 0.8
        
        # Normalize focus areas
        total_focus = sum(focus_areas.values())
        if total_focus > 0:
            focus_areas = {k: v/total_focus for k, v in focus_areas.items()}
        
        # Determine primary focus
        primary_focus = max(focus_areas, key=focus_areas.get)
        
        return {
            'focus_distribution': focus_areas,
            'primary_focus': primary_focus,
            'focus_intensity': max(focus_areas.values())
        }
    
    def _analyze_decision_making(self, agent_inputs: Dict[str, AgentInput]) -> str:
        """Analyze the character's decision-making mode."""
        # Check neurochemical state for decision-making indicators
        neurochemical_input = agent_inputs.get('neurochemical')
        if neurochemical_input and neurochemical_input.metadata:
            hormone_levels = neurochemical_input.metadata.get('hormone_levels', {})
            cortisol = hormone_levels.get('cortisol', 50)
            dopamine = hormone_levels.get('dopamine', 50)
            serotonin = hormone_levels.get('serotonin', 50)
            
            if cortisol > 70:
                return "stress_driven"
            elif dopamine > 70:
                return "reward_seeking"
            elif serotonin > 70:
                return "harmonious"
        
        # Check personality for decision-making style
        personality_input = agent_inputs.get('personality')
        if personality_input and personality_input.metadata:
            big_five = personality_input.metadata.get('big_five_scores', {})
            neuroticism = big_five.get('neuroticism', 0.5)
            conscientiousness = big_five.get('conscientiousness', 0.5)
            
            if neuroticism > 0.7:
                return "emotion_driven"
            elif conscientiousness > 0.7:
                return "analytical"
        
        return "balanced"
    
    def _calculate_cognitive_load(self, agent_inputs: Dict[str, AgentInput]) -> float:
        """Calculate the cognitive load based on agent inputs."""
        load_factors = []
        
        # High priority inputs increase cognitive load
        for agent_input in agent_inputs.values():
            if agent_input.priority > 0.7:
                load_factors.append(agent_input.priority)
        
        # Multiple high-confidence conflicting inputs increase load
        high_conf_count = sum(1 for inp in agent_inputs.values() if inp.confidence > 0.8)
        if high_conf_count > 3:
            load_factors.append(0.8)
        
        # Emotional intensity increases load
        mood_input = agent_inputs.get('mood')
        if mood_input and mood_input.metadata:
            mood_state = mood_input.metadata.get('mood_state', {})
            intensity = mood_state.get('intensity', 0.5)
            if intensity > 0.7:
                load_factors.append(intensity)
        
        # Calculate average load
        if load_factors:
            cognitive_load = sum(load_factors) / len(load_factors)
        else:
            cognitive_load = 0.3  # Low baseline load
        
        return min(1.0, cognitive_load)
    
    def _analyze_information_processing(self, agent_inputs: Dict[str, AgentInput]) -> Dict[str, str]:
        """Analyze how the character is processing information."""
        processing_style = {}
        
        # Check if memory is being heavily utilized
        memory_input = agent_inputs.get('memory')
        if memory_input and memory_input.priority > 0.8:
            processing_style['memory_usage'] = "high"
        else:
            processing_style['memory_usage'] = "moderate"
        
        # Check for analytical vs intuitive processing
        personality_input = agent_inputs.get('personality')
        if personality_input and personality_input.metadata:
            big_five = personality_input.metadata.get('big_five_scores', {})
            openness = big_five.get('openness', 0.5)
            
            if openness > 0.7:
                processing_style['approach'] = "intuitive"
            else:
                processing_style['approach'] = "analytical"
        else:
            processing_style['approach'] = "balanced"
        
        # Check processing speed based on mood and neurochemistry
        mood_input = agent_inputs.get('mood')
        if mood_input and mood_input.metadata:
            energy_level = mood_input.metadata.get('energy_level', 50)
            if energy_level > 70:
                processing_style['speed'] = "fast"
            elif energy_level < 30:
                processing_style['speed'] = "slow"
            else:
                processing_style['speed'] = "normal"
        else:
            processing_style['speed'] = "normal"
        
        return processing_style
    
    def _process_emotional_intelligence(
        self,
        synthesis: Dict[str, Any],
        agent_inputs: Dict[str, AgentInput],
        character_state: CharacterState
    ) -> Dict[str, Any]:
        """Process emotional intelligence aspects."""
        emotional_intelligence = {
            'self_awareness': self._assess_self_awareness(agent_inputs),
            'emotional_regulation': self._assess_emotional_regulation(agent_inputs),
            'empathy_level': self._assess_empathy(agent_inputs),
            'social_awareness': self._assess_social_awareness(agent_inputs),
            'emotional_impact_prediction': self._predict_emotional_impact(synthesis, agent_inputs)
        }
        
        return emotional_intelligence
    
    def _assess_self_awareness(self, agent_inputs: Dict[str, AgentInput]) -> float:
        """Assess the character's self-awareness level."""
        awareness_factors = []
        
        # High mood awareness indicates self-awareness
        mood_input = agent_inputs.get('mood')
        if mood_input:
            awareness_factors.append(mood_input.confidence)
        
        # Neurochemical awareness
        neurochemical_input = agent_inputs.get('neurochemical')
        if neurochemical_input:
            awareness_factors.append(neurochemical_input.confidence * 0.8)  # Slightly less weight
        
        # Memory reflection indicates self-awareness
        memory_input = agent_inputs.get('memory')
        if memory_input and memory_input.priority > 0.7:
            awareness_factors.append(0.8)
        
        # Calculate average
        if awareness_factors:
            return sum(awareness_factors) / len(awareness_factors)
        else:
            return 0.5
    
    def _assess_emotional_regulation(self, agent_inputs: Dict[str, AgentInput]) -> float:
        """Assess emotional regulation capability."""
        # Check if mood and neurochemical states are in balance
        mood_input = agent_inputs.get('mood')
        neurochemical_input = agent_inputs.get('neurochemical')
        
        if not mood_input or not neurochemical_input:
            return 0.5
        
        # Good regulation means balanced hormone levels
        if neurochemical_input.metadata:
            hormone_levels = neurochemical_input.metadata.get('hormone_levels', {})
            
            # Check for extreme hormone levels (indicates poor regulation)
            extreme_count = 0
            for level in hormone_levels.values():
                if level > 80 or level < 20:
                    extreme_count += 1
            
            regulation_score = max(0.2, 1.0 - (extreme_count * 0.2))
            return regulation_score
        
        return 0.5
    
    def _assess_empathy(self, agent_inputs: Dict[str, AgentInput]) -> float:
        """Assess empathy level based on agent inputs."""
        empathy_score = 0.5  # Default moderate empathy
        
        # Personality traits affect empathy
        personality_input = agent_inputs.get('personality')
        if personality_input and personality_input.metadata:
            big_five = personality_input.metadata.get('big_five_scores', {})
            agreeableness = big_five.get('agreeableness', 0.5)
            empathy_score = agreeableness
        
        # Communication style can indicate empathy
        style_input = agent_inputs.get('communication_style')
        if style_input and style_input.metadata:
            style_requirements = style_input.metadata.get('style_requirements', {})
            if style_requirements.get('emotional_sensitivity_needed') == 'high':
                empathy_score = min(1.0, empathy_score + 0.2)
        
        return empathy_score
    
    def _assess_social_awareness(self, agent_inputs: Dict[str, AgentInput]) -> float:
        """Assess social awareness level."""
        social_awareness = 0.5
        
        # Communication style indicates social awareness
        style_input = agent_inputs.get('communication_style')
        if style_input:
            social_awareness = style_input.confidence
        
        # Goals related to relationships indicate social awareness
        goals_input = agent_inputs.get('goals')
        if goals_input and goals_input.metadata:
            goal_analysis = goals_input.metadata.get('goal_analysis', {})
            if goal_analysis.get('relationship_building'):
                social_awareness = min(1.0, social_awareness + 0.2)
        
        return social_awareness
    
    def _predict_emotional_impact(
        self,
        synthesis: Dict[str, Any],
        agent_inputs: Dict[str, AgentInput]
    ) -> Dict[str, float]:
        """Predict emotional impact of potential responses."""
        impact_prediction = {
            'self_impact': 0.0,
            'other_impact': 0.0,
            'relationship_impact': 0.0
        }
        
        emotional_tone = synthesis.get('emotional_tone', 'neutral')
        confidence_level = synthesis.get('confidence_level', 0.5)
        
        # Positive emotional tones generally have positive impact
        if emotional_tone in ['happy', 'excited', 'confident']:
            impact_prediction['self_impact'] = 0.7
            impact_prediction['other_impact'] = 0.6
            impact_prediction['relationship_impact'] = 0.8
        elif emotional_tone in ['sad', 'anxious', 'frustrated']:
            impact_prediction['self_impact'] = 0.3
            impact_prediction['other_impact'] = 0.4
            impact_prediction['relationship_impact'] = 0.3
        else:
            impact_prediction['self_impact'] = 0.5
            impact_prediction['other_impact'] = 0.5
            impact_prediction['relationship_impact'] = 0.5
        
        # Adjust based on confidence level
        for key in impact_prediction:
            impact_prediction[key] *= confidence_level
        
        return impact_prediction
    
    def _generate_cognitive_insights(
        self,
        agent_inputs: Dict[str, AgentInput],
        cognitive_patterns: Dict[str, Any],
        user_message: str
    ) -> List[str]:
        """Generate cognitive insights based on processing."""
        insights = []
        
        thinking_style = cognitive_patterns.get('thinking_style', 'balanced')
        attention_focus = cognitive_patterns.get('attention_focus', {})
        cognitive_load = cognitive_patterns.get('cognitive_load', 0.5)
        
        # Insight about thinking style
        if thinking_style == 'creative':
            insights.append("Character is in a creative, open-minded thinking mode")
        elif thinking_style == 'systematic':
            insights.append("Character is thinking systematically and methodically")
        elif thinking_style == 'cautious':
            insights.append("Character is thinking cautiously due to current emotional state")
        
        # Insight about attention focus
        primary_focus = attention_focus.get('primary_focus', 'external_environment')
        if primary_focus == 'internal_state':
            insights.append("Character is highly self-focused right now")
        elif primary_focus == 'past_experiences':
            insights.append("Character is drawing heavily on past experiences")
        elif primary_focus == 'goals':
            insights.append("Character is strategically focused on their goals")
        
        # Insight about cognitive load
        if cognitive_load > 0.8:
            insights.append("Character is experiencing high cognitive load")
        elif cognitive_load < 0.3:
            insights.append("Character has mental bandwidth for complex thinking")
        
        # Limit to most relevant insights
        return insights[:4]
    
    def _determine_response_strategy(
        self,
        synthesis: Dict[str, Any],
        cognitive_patterns: Dict[str, Any],
        emotional_intelligence: Dict[str, Any],
        conflicts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Determine the optimal response strategy."""
        strategy = {
            'approach': 'balanced',
            'complexity_level': 'moderate',
            'emotional_engagement': 'moderate',
            'thinking_visibility': 'partial',
            'response_focus': 'comprehensive'
        }
        
        cognitive_load = cognitive_patterns.get('cognitive_load', 0.5)
        thinking_style = cognitive_patterns.get('thinking_style', 'balanced')
        self_awareness = emotional_intelligence.get('self_awareness', 0.5)
        
        # Adjust approach based on thinking style
        if thinking_style == 'creative':
            strategy['approach'] = 'innovative'
            strategy['thinking_visibility'] = 'high'
        elif thinking_style == 'systematic':
            strategy['approach'] = 'methodical'
            strategy['complexity_level'] = 'high'
        elif thinking_style == 'cautious':
            strategy['approach'] = 'careful'
            strategy['emotional_engagement'] = 'low'
        
        # Adjust complexity based on cognitive load
        if cognitive_load > 0.8:
            strategy['complexity_level'] = 'simple'
            strategy['response_focus'] = 'focused'
        elif cognitive_load < 0.3:
            strategy['complexity_level'] = 'high'
            strategy['response_focus'] = 'comprehensive'
        
        # Adjust emotional engagement based on self-awareness
        if self_awareness > 0.8:
            strategy['emotional_engagement'] = 'high'
            strategy['thinking_visibility'] = 'high'
        elif self_awareness < 0.3:
            strategy['emotional_engagement'] = 'low'
            strategy['thinking_visibility'] = 'minimal'
        
        # Handle conflicts
        if conflicts:
            strategy['approach'] = 'cautious'
            strategy['complexity_level'] = 'moderate'  # Don't overwhelm with complex responses when conflicted
        
        return strategy
    
    def _create_fallback_cognitive_result(self, error_message: str) -> Dict[str, Any]:
        """Create fallback cognitive result when processing fails."""
        return {
            'cognitive_patterns': {
                'thinking_style': 'balanced',
                'attention_focus': {'primary_focus': 'external_environment'},
                'decision_making_mode': 'balanced',
                'cognitive_load': 0.5,
                'information_processing': {'approach': 'balanced', 'speed': 'normal'}
            },
            'emotional_intelligence': {
                'self_awareness': 0.5,
                'emotional_regulation': 0.5,
                'empathy_level': 0.5,
                'social_awareness': 0.5,
                'emotional_impact_prediction': {'self_impact': 0.5, 'other_impact': 0.5}
            },
            'cognitive_insights': ['Cognitive processing system error'],
            'response_strategy': {
                'approach': 'balanced',
                'complexity_level': 'simple',
                'emotional_engagement': 'moderate'
            },
            'processing_metadata': {
                'character_id': self.character_id,
                'error': error_message,
                'timestamp': datetime.now().isoformat()
            }
        }