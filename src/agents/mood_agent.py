"""Mood agent for tracking and reflecting current emotional state."""

from typing import Dict, Any, Optional
try:
    from .base_agent import BaseAgent
except ImportError:
    from agents.base_agent import BaseAgent

try:
    from ..core.agent_input import AgentInput
except ImportError:
    from core.agent_input import AgentInput

try:
    from ..core.character_state import CharacterState
except ImportError:
    from core.character_state import CharacterState


class MoodAgent(BaseAgent):
    """
    Tracks and reflects current emotional state.
    
    Mood colors ALL responses and is updated dynamically based on 
    neurochemical levels and external events. This agent focuses
    on how the current emotional state should influence the response.
    """
    
    def __init__(
        self,
        agent_id: str,
        character_id: str,
        llm_client: Any,
        mood_config: Dict[str, Any]
    ):
        """Initialize mood agent."""
        super().__init__(agent_id, "mood", character_id, llm_client)
        
        self.mood_config = mood_config
        
        # Mood agents generally don't need web search
        self.web_search_enabled = False
        
        # Extract mood configuration
        self.baseline_setpoint = mood_config.get('baseline_setpoint', 0.5)
        self.emotional_volatility = mood_config.get('emotional_volatility', 0.5)
        self.default_state = mood_config.get('default_state', 'neutral')
    
    def consult(
        self,
        context: Dict[str, Any],
        character_state: CharacterState,
        user_message: str
    ) -> AgentInput:
        """
        Provide input on how current mood affects response approach.
        
        Analyzes emotional state and provides guidance on how it should
        color the response - energy level, emotional filters, patience, etc.
        """
        # Get current mood state
        current_mood = character_state.agent_states.get('mood', {})
        
        # Calculate energy level from neurochemical state
        energy_level = self._calculate_energy_from_neurochemicals(
            character_state.neurochemical_levels
        )
        
        # Analyze emotional impact of user message
        emotional_impact = self._analyze_emotional_impact(user_message, current_mood)
        
        # Build mood analysis prompt
        prompt = self._build_mood_prompt(
            current_mood,
            energy_level,
            emotional_impact,
            user_message,
            context
        )
        
        # Generate mood-based guidance
        response = self._call_llm(
            prompt=prompt,
            temperature=0.7,  # Moderate variability for emotional nuance
            max_tokens=200,
            use_web_search=False  # Mood is internal state
        )
        
        # Ensure we have valid content
        if not response or not response.strip():
            response = f"Current mood is {emotional_tone}. Energy level at {energy_level:.2f}. Maintaining emotional stability."
        
        # Mood is very important and reliable
        confidence = 0.85
        priority = 0.9
        
        # Emotional tone matches current mood
        emotional_tone = current_mood.get('current_state', 'neutral')
        
        # Package mood metadata for other agents
        metadata = {
            'mood_state': current_mood,
            'energy_level': energy_level,
            'emotional_impact': emotional_impact,
            'mood_stability': self._calculate_mood_stability(current_mood),
            'response_modifiers': self._get_response_modifiers(current_mood, energy_level)
        }
        
        return self._create_agent_input(
            content=response,
            confidence=confidence,
            priority=priority,
            emotional_tone=emotional_tone,
            metadata=metadata
        )
    
    def get_prompt_template(self) -> str:
        """Return mood analysis prompt template."""
        return """You are analyzing how a person's current emotional state affects their response approach.

CURRENT EMOTIONAL STATE:
Mood: {current_state} (intensity: {intensity}/1.0)
Duration: Has felt this way for {duration} conversation turns
Energy Level: {energy_level}/100
Emotional Volatility: {volatility}/1.0 (how quickly mood shifts)

EMOTIONAL IMPACT ANALYSIS:
Message Impact: {emotional_impact}
Mood Trend: {mood_trend}

USER MESSAGE: "{user_message}"

RECENT CONTEXT: {conversation_context}

Based on this emotional state, analyze how mood affects the response:

1. ENERGY & ENGAGEMENT:
   - Does this person have energy for a detailed response?
   - High energy = more elaborate, enthusiastic responses
   - Low energy = brief, minimal effort responses
   - Mood-energy interaction effects

2. EMOTIONAL FILTER:
   - How does current mood color perception of the message?
   - Positive mood = interpret charitably, see opportunities
   - Negative mood = more skeptical, see problems
   - Neutral mood = balanced interpretation

3. PATIENCE & TOLERANCE:
   - Is their current mood making them more/less patient?
   - Stressed/irritated = lower tolerance for complexity
   - Happy/content = more patience and flexibility
   - Tired = want simple, easy interactions

4. SOCIAL ENGAGEMENT DESIRE:
   - Does mood make them want to connect or withdraw?
   - Positive mood = more open, sharing, bonding
   - Negative mood = protective, distant, brief
   - Mixed mood = variable engagement

5. RESPONSE TIMING & STYLE:
   - Should response be immediate or thoughtful?
   - Energetic mood = quick, spontaneous responses
   - Contemplative mood = slower, more considered responses
   - Emotional mood = feelings-focused responses

Provide 2-3 sentences describing how this emotional state shapes the response style, engagement level, and emotional approach."""
    
    def _calculate_energy_from_neurochemicals(
        self,
        neurochemical_levels: Dict[str, float]
    ) -> float:
        """Calculate energy level from neurochemical state."""
        # Energy primarily from dopamine and serotonin
        # Reduced by cortisol (stress), temporarily boosted by adrenaline
        
        dopamine = neurochemical_levels.get('dopamine', 50)
        serotonin = neurochemical_levels.get('serotonin', 50)
        cortisol = neurochemical_levels.get('cortisol', 50)
        adrenaline = neurochemical_levels.get('adrenaline', 50)
        
        # Base energy from positive neurochemicals
        base_energy = (dopamine + serotonin) / 2
        
        # Stress penalty (cortisol reduces available energy)
        stress_penalty = max(0, (cortisol - 50) * 0.5)
        
        # Adrenaline can temporarily boost energy but also drain it
        adrenaline_effect = (adrenaline - 50) * 0.3
        
        energy = base_energy - stress_penalty + adrenaline_effect
        
        # Clamp to 0-100 range
        return max(0, min(100, energy))
    
    def _analyze_emotional_impact(
        self,
        user_message: str,
        current_mood: Dict[str, Any]
    ) -> str:
        """Analyze how the user message might impact current emotional state."""
        message_lower = user_message.lower()
        current_state = current_mood.get('current_state', 'neutral')
        intensity = current_mood.get('intensity', 0.5)
        
        # Detect message emotional content
        positive_indicators = [
            'great', 'awesome', 'wonderful', 'love', 'happy', 'excited',
            'thank', 'appreciate', 'amazing', 'perfect', 'excellent'
        ]
        
        negative_indicators = [
            'bad', 'terrible', 'hate', 'angry', 'sad', 'frustrated',
            'annoying', 'stupid', 'wrong', 'awful', 'horrible'
        ]
        
        neutral_indicators = [
            'what', 'how', 'when', 'where', 'tell me', 'explain', 'help'
        ]
        
        positive_score = sum(1 for word in positive_indicators if word in message_lower)
        negative_score = sum(1 for word in negative_indicators if word in message_lower)
        neutral_score = sum(1 for word in neutral_indicators if word in message_lower)
        
        # Determine impact
        if positive_score > negative_score and positive_score > 0:
            if current_state in ['happy', 'excited', 'content']:
                return "reinforcing_positive"
            elif current_state in ['sad', 'anxious', 'tired']:
                return "potentially_uplifting"
            else:
                return "positive_influence"
        
        elif negative_score > positive_score and negative_score > 0:
            if current_state in ['happy', 'excited', 'content']:
                return "potentially_dampening"
            elif current_state in ['sad', 'anxious', 'tired']:
                return "reinforcing_negative"
            else:
                return "negative_influence"
        
        elif neutral_score > 0:
            return "emotionally_neutral"
        
        else:
            return "ambiguous_impact"
    
    def _calculate_mood_stability(self, current_mood: Dict[str, Any]) -> float:
        """Calculate how stable the current mood is."""
        duration = current_mood.get('duration', 1)
        volatility = self.emotional_volatility
        intensity = current_mood.get('intensity', 0.5)
        
        # Longer duration + lower volatility + moderate intensity = more stable
        duration_factor = min(1.0, duration / 10)  # Normalize duration
        volatility_factor = 1.0 - volatility  # Lower volatility = more stable
        intensity_factor = 1.0 - abs(intensity - 0.5) * 2  # Extreme intensity less stable
        
        stability = (duration_factor + volatility_factor + intensity_factor) / 3
        return stability
    
    def _get_response_modifiers(
        self,
        current_mood: Dict[str, Any],
        energy_level: float
    ) -> Dict[str, str]:
        """Get specific response modifiers based on mood and energy."""
        modifiers = {}
        
        mood_state = current_mood.get('current_state', 'neutral')
        intensity = current_mood.get('intensity', 0.5)
        
        # Response length modifier
        if energy_level > 70:
            modifiers['length'] = 'verbose'
        elif energy_level < 40:
            modifiers['length'] = 'brief'
        else:
            modifiers['length'] = 'moderate'
        
        # Emotional expression modifier
        if mood_state in ['happy', 'excited'] and intensity > 0.6:
            modifiers['emotional_expression'] = 'enthusiastic'
        elif mood_state in ['sad', 'tired'] and intensity > 0.6:
            modifiers['emotional_expression'] = 'subdued'
        elif mood_state == 'anxious' and intensity > 0.6:
            modifiers['emotional_expression'] = 'cautious'
        else:
            modifiers['emotional_expression'] = 'balanced'
        
        # Social openness modifier
        if mood_state in ['happy', 'content', 'excited']:
            modifiers['social_openness'] = 'open'
        elif mood_state in ['anxious', 'sad', 'tired']:
            modifiers['social_openness'] = 'reserved'
        else:
            modifiers['social_openness'] = 'neutral'
        
        # Response speed modifier
        if energy_level > 80 or mood_state == 'excited':
            modifiers['response_speed'] = 'quick'
        elif mood_state in ['tired', 'sad'] or energy_level < 30:
            modifiers['response_speed'] = 'slow'
        else:
            modifiers['response_speed'] = 'normal'
        
        # Focus modifier
        if mood_state == 'anxious' or intensity > 0.8:
            modifiers['focus'] = 'scattered'
        elif mood_state == 'content' and energy_level > 50:
            modifiers['focus'] = 'clear'
        else:
            modifiers['focus'] = 'normal'
        
        return modifiers
    
    def _build_mood_prompt(
        self,
        current_mood: Dict[str, Any],
        energy_level: float,
        emotional_impact: str,
        user_message: str,
        context: Dict[str, Any]
    ) -> str:
        """Build mood analysis prompt with current state."""
        mood_state = current_mood.get('current_state', 'neutral')
        intensity = current_mood.get('intensity', 0.5)
        duration = current_mood.get('duration', 1)
        
        # Determine mood trend
        if duration == 1:
            mood_trend = "just_started"
        elif duration < 5:
            mood_trend = "developing"
        else:
            mood_trend = "persistent"
        
        # Get conversation context
        conversation_context = context.get('recent_messages', 'No recent context')
        
        return self.get_prompt_template().format(
            current_state=mood_state,
            intensity=intensity,
            duration=duration,
            energy_level=energy_level,
            volatility=self.emotional_volatility,
            emotional_impact=emotional_impact,
            mood_trend=mood_trend,
            user_message=user_message,
            conversation_context=conversation_context
        )
    
    def predict_mood_change(
        self,
        current_mood: Dict[str, Any],
        stimulus_type: str,
        intensity: float
    ) -> Dict[str, Any]:
        """Predict how mood will change based on a stimulus."""
        current_state = current_mood.get('current_state', 'neutral')
        current_intensity = current_mood.get('intensity', 0.5)
        volatility = self.emotional_volatility
        
        # Predict new mood based on stimulus
        mood_transitions = {
            'positive': {
                'sad': 'content',
                'anxious': 'relieved',
                'tired': 'energized',
                'neutral': 'happy',
                'happy': 'excited',
                'content': 'happy',
                'excited': 'excited'  # Already at peak positive
            },
            'negative': {
                'happy': 'neutral',
                'excited': 'content',
                'content': 'neutral',
                'neutral': 'sad',
                'sad': 'depressed',
                'anxious': 'anxious',  # Reinforces anxiety
                'tired': 'exhausted'
            },
            'stress': {
                'happy': 'anxious',
                'excited': 'anxious',
                'content': 'concerned',
                'neutral': 'anxious',
                'sad': 'overwhelmed',
                'anxious': 'panicked',
                'tired': 'stressed'
            }
        }
        
        # Get predicted new state
        transitions = mood_transitions.get(stimulus_type, {})
        new_state = transitions.get(current_state, current_state)
        
        # Calculate new intensity based on volatility and stimulus strength
        intensity_change = intensity * volatility
        
        if stimulus_type == 'positive':
            new_intensity = min(1.0, current_intensity + intensity_change)
        elif stimulus_type in ['negative', 'stress']:
            new_intensity = min(1.0, current_intensity + intensity_change)
        else:
            new_intensity = current_intensity
        
        return {
            'current_state': new_state,
            'intensity': new_intensity,
            'duration': 1,  # Reset duration for new mood
            'triggered_by': stimulus_type
        }