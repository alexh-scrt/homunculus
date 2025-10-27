"""Communication style agent for maintaining speech patterns and stylistic consistency."""

from typing import Dict, Any, Optional, List
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


class CommunicationStyleAgent(BaseAgent):
    """
    Determines HOW the character should express themselves.
    
    Handles verbal patterns, quirks, social comfort levels, and stylistic
    consistency. This agent ensures the character sounds like themselves
    across all interactions.
    """
    
    def __init__(
        self,
        agent_id: str,
        character_id: str,
        llm_client: Any,
        style_config: Dict[str, Any]
    ):
        """Initialize communication style agent."""
        super().__init__(agent_id, "communication_style", character_id, llm_client)
        
        self.style_config = style_config
        
        # Communication style agents don't typically need web search
        self.web_search_enabled = False
        
        # Extract style configuration
        self.verbal_pattern = style_config.get('verbal_pattern', 'moderate')
        self.social_comfort = style_config.get('social_comfort', 'neutral')
        self.listening_preference = style_config.get('listening_preference', 0.5)
        self.body_language = style_config.get('body_language', 'neutral')
        self.quirks = style_config.get('quirks', [])
    
    async def consult(
        self,
        context: Dict[str, Any],
        character_state: CharacterState,
        user_message: str
    ) -> AgentInput:
        """
        Provide guidance on communication style for the response.
        
        Focuses on HOW to express the response rather than WHAT to say,
        ensuring stylistic consistency with the character's communication patterns.
        """
        # Get current style state (may be modified by mood/energy)
        current_style = self._get_current_style_state(character_state)
        
        # Analyze style requirements for this specific interaction
        style_analysis = self._analyze_style_requirements(
            user_message,
            character_state,
            context
        )
        
        # Build style guidance prompt
        prompt = self._build_style_prompt(
            current_style,
            style_analysis,
            user_message,
            character_state
        )
        
        # Generate style guidance
        response = await self._call_llm(
            prompt=prompt,
            temperature=0.8,  # Higher creativity for natural style variation
            max_tokens=200,
            use_web_search=False  # Style is character-internal
        )
        
        # Style is very important for consistency
        confidence = 0.9
        priority = 0.85
        
        emotional_tone = "stylistic"
        
        # Package style metadata for response generation
        metadata = {
            'style_config': current_style,
            'style_requirements': style_analysis,
            'quirks_to_include': self._select_relevant_quirks(user_message),
            'response_modifiers': self._get_style_modifiers(character_state)
        }
        
        return self._create_agent_input(
            content=response,
            confidence=confidence,
            priority=priority,
            emotional_tone=emotional_tone,
            metadata=metadata
        )
    
    def get_prompt_template(self) -> str:
        """Return communication style analysis prompt template."""
        return """You are defining the communication style and expression approach for this character's response.

CHARACTER COMMUNICATION PROFILE:
Verbal Pattern: {verbal_pattern}
- "concise": Brief, to-the-point responses
- "moderate": Balanced length responses  
- "verbose": Detailed, elaborate responses
- "rambling": Stream-of-consciousness, tangential

Social Comfort Level: {social_comfort}
- "passive": Hesitant, deferential, avoids conflict
- "diplomatic": Tactful, considerate, balanced
- "assertive": Direct, confident, takes initiative
- "aggressive": Blunt, confrontational, dominant

Talking vs Listening Preference: {listening_preference}/1.0
- 0.0 = Dominates conversation, talks much more than listens
- 0.5 = Balanced between talking and listening
- 1.0 = Prefers to listen, asks questions, brief responses

Body Language Style: {body_language}
Physical Quirks & Mannerisms: {quirks}

CURRENT STATE MODIFIERS:
Mood: {current_mood} (affects tone and expressiveness)
Energy Level: {energy_level}/100 (affects response length and enthusiasm)
Style Adjustments: {style_modifiers}

USER MESSAGE: "{user_message}"
MESSAGE TYPE: {message_type}

Based on this communication profile, provide style guidance for the response:

1. RESPONSE LENGTH & STRUCTURE:
   - Should this be brief, moderate, or lengthy based on verbal pattern + current state?
   - How does energy level affect typical verbosity?
   - Any mood-based adjustments to usual pattern?

2. TONE & DELIVERY:
   - Formal, casual, playful, serious?
   - How does social comfort level manifest in this interaction?
   - Impact of current mood on tone?

3. LANGUAGE PATTERNS:
   - Technical, simple, complex, slang?
   - Vocabulary level and style choices
   - Regional or cultural speech patterns

4. INTERACTIVE STYLE:
   - Questions vs statements based on listening preference
   - Conversational flow and turn-taking
   - Engagement vs detachment

5. QUIRKS & MANNERISMS:
   - Which specific quirks fit naturally in this response?
   - How to include them without forcing?
   - Frequency and timing of quirk expression

Provide 2-3 sentences describing the optimal communication style approach for this specific response."""
    
    def _get_current_style_state(self, character_state: CharacterState) -> Dict[str, Any]:
        """Get current style state, potentially modified by mood/energy."""
        base_style = character_state.agent_states.get('communication_style', {})
        
        # Start with configured style
        current_style = {
            'verbal_pattern': base_style.get('verbal_pattern', self.verbal_pattern),
            'social_comfort': base_style.get('social_comfort', self.social_comfort),
            'listening_preference': base_style.get('listening_preference', self.listening_preference),
            'body_language': base_style.get('body_language', self.body_language),
            'quirks': base_style.get('quirks', self.quirks)
        }
        
        # Apply mood/energy modifiers
        mood_state = character_state.agent_states.get('mood', {})
        energy_level = self._calculate_energy_level(character_state.neurochemical_levels)
        
        # Adjust verbal pattern based on energy
        if energy_level < 30:
            # Low energy makes verbose people more concise
            if current_style['verbal_pattern'] == 'verbose':
                current_style['verbal_pattern'] = 'moderate'
            elif current_style['verbal_pattern'] == 'moderate':
                current_style['verbal_pattern'] = 'concise'
        elif energy_level > 80:
            # High energy can make people more verbose
            if current_style['verbal_pattern'] == 'concise':
                current_style['verbal_pattern'] = 'moderate'
        
        # Adjust social comfort based on mood
        current_mood = mood_state.get('current_state', 'neutral')
        if current_mood in ['anxious', 'sad']:
            # Negative moods make people more passive
            if current_style['social_comfort'] == 'assertive':
                current_style['social_comfort'] = 'diplomatic'
            elif current_style['social_comfort'] == 'aggressive':
                current_style['social_comfort'] = 'assertive'
        
        return current_style
    
    def _analyze_style_requirements(
        self,
        user_message: str,
        character_state: CharacterState,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze what style approach this specific interaction requires."""
        message_lower = user_message.lower()
        
        # Classify message type
        message_type = self._classify_message_type(user_message)
        
        # Determine appropriate response approach
        analysis = {
            'message_type': message_type,
            'formality_level': self._determine_formality_level(user_message),
            'emotional_sensitivity_needed': self._assess_emotional_sensitivity(user_message),
            'technical_complexity': self._assess_technical_complexity(user_message),
            'social_dynamics': self._analyze_social_dynamics(user_message, context)
        }
        
        return analysis
    
    def _classify_message_type(self, user_message: str) -> str:
        """Classify the type of user message for style adaptation."""
        message_lower = user_message.lower()
        
        question_words = ['what', 'how', 'when', 'where', 'why', 'who', 'which']
        greeting_words = ['hello', 'hi', 'hey', 'good morning', 'good afternoon']
        emotional_words = ['feel', 'emotion', 'sad', 'happy', 'angry', 'excited']
        help_words = ['help', 'advice', 'suggest', 'recommend']
        
        if any(word in message_lower for word in greeting_words):
            return 'greeting'
        elif any(word in message_lower for word in emotional_words):
            return 'emotional'
        elif any(word in message_lower for word in help_words):
            return 'request_for_help'
        elif any(message_lower.startswith(word) for word in question_words):
            return 'information_question'
        elif '?' in user_message:
            return 'general_question'
        elif '!' in user_message:
            return 'exclamation'
        else:
            return 'statement'
    
    def _determine_formality_level(self, user_message: str) -> str:
        """Determine appropriate formality level for response."""
        formal_indicators = ['please', 'thank you', 'sir', 'ma\'am', 'would you']
        casual_indicators = ['hey', 'yeah', 'ok', 'cool', 'awesome']
        
        message_lower = user_message.lower()
        
        formal_count = sum(1 for indicator in formal_indicators if indicator in message_lower)
        casual_count = sum(1 for indicator in casual_indicators if indicator in message_lower)
        
        if formal_count > casual_count:
            return 'formal'
        elif casual_count > formal_count:
            return 'casual'
        else:
            return 'neutral'
    
    def _assess_emotional_sensitivity(self, user_message: str) -> str:
        """Assess how much emotional sensitivity the response needs."""
        sensitive_topics = ['problem', 'difficult', 'struggle', 'help', 'sad', 'worried', 'stressed']
        message_lower = user_message.lower()
        
        sensitivity_score = sum(1 for topic in sensitive_topics if topic in message_lower)
        
        if sensitivity_score >= 2:
            return 'high'
        elif sensitivity_score == 1:
            return 'moderate'
        else:
            return 'low'
    
    def _assess_technical_complexity(self, user_message: str) -> str:
        """Assess the technical complexity of the topic."""
        technical_indicators = [
            'algorithm', 'system', 'process', 'method', 'technique', 'analysis',
            'implementation', 'configuration', 'optimization', 'architecture'
        ]
        
        message_lower = user_message.lower()
        tech_count = sum(1 for indicator in technical_indicators if indicator in message_lower)
        
        if tech_count >= 2:
            return 'high'
        elif tech_count == 1:
            return 'moderate'
        else:
            return 'low'
    
    def _analyze_social_dynamics(
        self,
        user_message: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze social dynamics affecting communication style."""
        return {
            'relationship_stage': self._assess_relationship_stage(context),
            'power_dynamics': 'equal',  # Simplified for now
            'social_context': 'casual_conversation'  # Simplified for now
        }
    
    def _assess_relationship_stage(self, context: Dict[str, Any]) -> str:
        """Assess the stage of relationship with user."""
        interaction_count = context.get('interaction_count', 0)
        
        if interaction_count == 0:
            return 'first_meeting'
        elif interaction_count < 5:
            return 'getting_acquainted'
        elif interaction_count < 20:
            return 'familiar'
        else:
            return 'established'
    
    def _select_relevant_quirks(self, user_message: str) -> List[str]:
        """Select which quirks are relevant for this response."""
        if not self.quirks:
            return []
        
        # Simple selection - in production could be more sophisticated
        relevant_quirks = []
        
        for quirk in self.quirks:
            quirk_lower = quirk.lower()
            message_lower = user_message.lower()
            
            # Match quirks to context
            if 'emoji' in quirk_lower and len(user_message) > 10:
                relevant_quirks.append(quirk)
            elif 'reference' in quirk_lower and 'culture' not in message_lower:
                relevant_quirks.append(quirk)
            elif 'humor' in quirk_lower:
                relevant_quirks.append(quirk)
        
        # Limit to 1-2 quirks per response to avoid overuse
        return relevant_quirks[:2]
    
    def _get_style_modifiers(self, character_state: CharacterState) -> Dict[str, str]:
        """Get style modifiers based on current character state."""
        mood_state = character_state.agent_states.get('mood', {})
        energy_level = self._calculate_energy_level(character_state.neurochemical_levels)
        
        modifiers = {}
        
        # Energy-based modifiers
        if energy_level > 80:
            modifiers['enthusiasm'] = 'high'
            modifiers['pace'] = 'quick'
        elif energy_level < 30:
            modifiers['enthusiasm'] = 'low'
            modifiers['pace'] = 'slow'
        else:
            modifiers['enthusiasm'] = 'moderate'
            modifiers['pace'] = 'normal'
        
        # Mood-based modifiers
        current_mood = mood_state.get('current_state', 'neutral')
        if current_mood in ['happy', 'excited']:
            modifiers['warmth'] = 'high'
            modifiers['playfulness'] = 'elevated'
        elif current_mood in ['sad', 'tired']:
            modifiers['warmth'] = 'subdued'
            modifiers['playfulness'] = 'minimal'
        elif current_mood == 'anxious':
            modifiers['caution'] = 'elevated'
            modifiers['formality'] = 'increased'
        
        return modifiers
    
    def _calculate_energy_level(self, neurochemical_levels: Dict[str, float]) -> float:
        """Calculate energy level from neurochemical state."""
        dopamine = neurochemical_levels.get('dopamine', 50)
        serotonin = neurochemical_levels.get('serotonin', 50)
        cortisol = neurochemical_levels.get('cortisol', 50)
        
        energy = (dopamine + serotonin) / 2 - (cortisol - 50) * 0.5
        return max(0, min(100, energy))
    
    def _build_style_prompt(
        self,
        current_style: Dict[str, Any],
        style_analysis: Dict[str, Any],
        user_message: str,
        character_state: CharacterState
    ) -> str:
        """Build style guidance prompt with current state."""
        mood_state = character_state.agent_states.get('mood', {})
        energy_level = self._calculate_energy_level(character_state.neurochemical_levels)
        
        # Format quirks for prompt
        quirks_text = "\n".join([f"- {quirk}" for quirk in self.quirks]) if self.quirks else "No specific quirks"
        
        # Get style modifiers
        style_modifiers = self._get_style_modifiers(character_state)
        modifiers_text = ", ".join([f"{k}: {v}" for k, v in style_modifiers.items()])
        
        return self.get_prompt_template().format(
            verbal_pattern=current_style['verbal_pattern'],
            social_comfort=current_style['social_comfort'],
            listening_preference=current_style['listening_preference'],
            body_language=current_style['body_language'],
            quirks=quirks_text,
            current_mood=mood_state.get('current_state', 'neutral'),
            energy_level=energy_level,
            style_modifiers=modifiers_text,
            user_message=user_message,
            message_type=style_analysis['message_type']
        )