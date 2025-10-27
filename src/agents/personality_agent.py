"""Personality agent for enforcing Big Five traits and behavioral patterns."""

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


class PersonalityAgent(BaseAgent):
    """
    Provides input based on Big Five traits and behavioral patterns.
    Ensures character stays true to personality while potentially using web search
    for personality-relevant information when appropriate.
    """
    
    def __init__(
        self,
        agent_id: str,
        character_id: str,
        llm_client: Any,
        personality_config: Dict[str, Any]
    ):
        """Initialize personality agent."""
        super().__init__(agent_id, "personality", character_id, llm_client)
        
        self.personality_config = personality_config
        
        # Personality agents may use web search for topics related to their interests
        self.web_search_enabled = True
        self.web_search_threshold = 0.3  # Moderately likely to search
        
        # Extract key personality dimensions
        self.big_five = personality_config.get('big_five', {})
        self.traits = personality_config.get('behavioral_traits', [])
        self.values = personality_config.get('core_values', [])
        
        # Determine search behavior based on openness
        openness = self.big_five.get('openness', 0.5)
        self.web_search_threshold = 0.2 + (openness * 0.6)  # Higher openness = more likely to search
    
    async def consult(
        self,
        context: Dict[str, Any],
        character_state: CharacterState,
        user_message: str
    ) -> AgentInput:
        """
        Analyze user message through personality lens and provide guidance.
        May use web search for topics that align with personality interests.
        """
        # Check if this topic aligns with personality interests and might benefit from web search
        should_search = self._should_search_for_personality_info(user_message)
        
        # Build base prompt with personality state
        prompt = self._build_personality_prompt(
            user_message,
            character_state,
            context
        )
        
        # Generate personality-aligned perspective
        response = await self._call_llm(
            prompt=prompt,
            temperature=0.6,  # Moderate creativity while staying consistent
            max_tokens=250,
            use_web_search=should_search
        )
        
        # Calculate confidence based on how well-defined the personality is
        confidence = self._calculate_confidence()
        
        # Personality is high priority for consistency
        priority = 0.8
        
        # Infer emotional tone from personality
        emotional_tone = self._infer_personality_tone()
        
        # Package personality metadata
        metadata = {
            'big_five': self.big_five,
            'dominant_traits': self._get_dominant_traits(),
            'core_values': [v['value'] for v in self.values],
            'personality_consistency_check': self._check_consistency(response)
        }
        
        return self._create_agent_input(
            content=response,
            confidence=confidence,
            priority=priority,
            emotional_tone=emotional_tone,
            metadata=metadata
        )
    
    def get_prompt_template(self) -> str:
        """Return personality analysis prompt template."""
        return """You are analyzing how a person with specific personality traits would naturally respond.

PERSONALITY PROFILE:
Big Five Traits:
- Openness: {openness}/1.0 (low=conventional/practical, high=creative/curious)
- Conscientiousness: {conscientiousness}/1.0 (low=spontaneous/flexible, high=organized/disciplined)
- Extraversion: {extraversion}/1.0 (low=introverted/reserved, high=extraverted/social)
- Agreeableness: {agreeableness}/1.0 (low=competitive/skeptical, high=cooperative/trusting)
- Neuroticism: {neuroticism}/1.0 (low=calm/stable, high=anxious/sensitive)

Behavioral Traits: {traits}
Core Values: {values}

CONVERSATION CONTEXT:
Recent History: {conversation_history}
Current Mood: {current_mood}
Energy Level: {energy_level}

USER MESSAGE: "{user_message}"

Based on this specific personality, analyze how they would naturally respond. Consider:

1. ENGAGEMENT STYLE: Would they engage deeply or keep it brief?
   - High openness + high extraversion = deep engagement with new topics
   - Low extraversion + high neuroticism = cautious, brief responses
   - High conscientiousness = thorough, organized responses

2. EMOTIONAL APPROACH: How would they handle the emotional aspects?
   - High agreeableness = warm, supportive tone
   - Low agreeableness + high neuroticism = potentially defensive
   - High emotional stability = calm, measured responses

3. CURIOSITY & EXPLORATION: How would their openness manifest?
   - High openness = ask follow-up questions, explore implications
   - Low openness = stick to practical, concrete responses

4. VALUES ALIGNMENT: Do any core values influence the response?
   - How do their priorities shape what they focus on?
   - What would they avoid saying that conflicts with their values?

5. SOCIAL COMFORT: How does their extraversion + agreeableness combo affect interaction?
   - Social energy level and preference for connection vs. autonomy

Provide 2-3 sentences describing the personality-driven approach to this response, focusing on HOW they would respond rather than WHAT they would say."""
    
    def _should_search_for_personality_info(self, user_message: str) -> bool:
        """Determine if web search would help based on personality traits."""
        if not self.should_search_web(user_message, {}):
            return False
        
        # High openness personalities are more likely to search for information
        openness = self.big_five.get('openness', 0.5)
        if openness < 0.4:
            return False  # Conservative personalities less likely to search
        
        # Check if message relates to personality interests
        message_lower = user_message.lower()
        
        # Topics that curious/open personalities might want to research
        research_topics = [
            'psychology', 'personality', 'behavior', 'human nature',
            'philosophy', 'culture', 'society', 'trends', 'innovation',
            'creative', 'art', 'science', 'technology', 'future'
        ]
        
        # Personal growth and learning topics
        growth_topics = [
            'learn', 'improve', 'develop', 'skill', 'habit', 'method',
            'technique', 'approach', 'strategy', 'best practice'
        ]
        
        personality_relevant = any(topic in message_lower for topic in research_topics)
        growth_relevant = any(topic in message_lower for topic in growth_topics)
        
        # High openness + relevant topic = search
        return personality_relevant or (openness > 0.7 and growth_relevant)
    
    def _build_personality_prompt(
        self,
        user_message: str,
        character_state: CharacterState,
        context: Dict[str, Any]
    ) -> str:
        """Build personality analysis prompt with current state."""
        # Get recent conversation history
        history = character_state.get_recent_context(3)
        history_text = self._format_conversation_history(history)
        
        # Get current mood and energy from neurochemical state
        current_mood = character_state.agent_states.get('mood', {}).get('current_state', 'neutral')
        
        # Calculate energy level from neurochemical levels
        neuro_levels = character_state.neurochemical_levels
        energy_level = (neuro_levels.get('dopamine', 50) + neuro_levels.get('serotonin', 50)) / 2
        
        # Format traits and values for prompt
        traits_text = self._format_traits(self.traits)
        values_text = self._format_values(self.values)
        
        return self.get_prompt_template().format(
            openness=self.big_five.get('openness', 0.5),
            conscientiousness=self.big_five.get('conscientiousness', 0.5),
            extraversion=self.big_five.get('extraversion', 0.5),
            agreeableness=self.big_five.get('agreeableness', 0.5),
            neuroticism=self.big_five.get('neuroticism', 0.5),
            traits=traits_text,
            values=values_text,
            conversation_history=history_text,
            current_mood=current_mood,
            energy_level=f"{energy_level:.0f}/100",
            user_message=user_message
        )
    
    def _format_traits(self, traits: list) -> str:
        """Format behavioral traits for prompt."""
        if not traits:
            return "No specific behavioral traits defined"
        
        return ", ".join([
            f"{trait['trait']} (intensity: {trait['intensity']}/1.0)"
            for trait in traits
        ])
    
    def _format_values(self, values: list) -> str:
        """Format core values for prompt."""
        if not values:
            return "No specific core values defined"
        
        return ", ".join([
            f"{value['value']} (priority: {value['priority']}/10)"
            for value in values
        ])
    
    def _calculate_confidence(self) -> float:
        """Calculate confidence based on personality definition completeness."""
        confidence = 0.7  # Base confidence
        
        # Higher confidence if Big Five traits are well-defined
        big_five_defined = len([v for v in self.big_five.values() if v != 0.5])
        confidence += (big_five_defined / len(self.big_five)) * 0.2
        
        # Higher confidence if behavioral traits are defined
        if self.traits:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _infer_personality_tone(self) -> str:
        """Infer emotional tone from personality traits."""
        extraversion = self.big_five.get('extraversion', 0.5)
        agreeableness = self.big_five.get('agreeableness', 0.5)
        neuroticism = self.big_five.get('neuroticism', 0.5)
        openness = self.big_five.get('openness', 0.5)
        
        if extraversion > 0.7 and agreeableness > 0.6:
            return "enthusiastic"
        elif neuroticism > 0.6:
            return "cautious"
        elif openness > 0.7:
            return "curious"
        elif agreeableness > 0.7:
            return "warm"
        elif extraversion < 0.4:
            return "reserved"
        else:
            return "balanced"
    
    def _get_dominant_traits(self) -> list:
        """Get the most prominent personality traits."""
        dominant = []
        
        # Check Big Five extremes
        for trait, value in self.big_five.items():
            if value > 0.75:
                dominant.append(f"high_{trait}")
            elif value < 0.25:
                dominant.append(f"low_{trait}")
        
        # Add behavioral traits with high intensity
        for trait_info in self.traits:
            if trait_info.get('intensity', 0) > 0.7:
                dominant.append(trait_info['trait'])
        
        return dominant
    
    def _check_consistency(self, response: str) -> Dict[str, Any]:
        """Check if the response aligns with personality traits."""
        response_lower = response.lower()
        
        consistency_check = {
            'extraversion_alignment': self._check_extraversion_alignment(response_lower),
            'agreeableness_alignment': self._check_agreeableness_alignment(response_lower),
            'openness_alignment': self._check_openness_alignment(response_lower),
            'overall_consistency': 'good'  # Simplified for now
        }
        
        return consistency_check
    
    def _check_extraversion_alignment(self, response: str) -> str:
        """Check if response length/engagement matches extraversion level."""
        extraversion = self.big_five.get('extraversion', 0.5)
        response_length = len(response.split())
        
        if extraversion > 0.7 and response_length < 15:
            return "potentially_too_brief"
        elif extraversion < 0.3 and response_length > 40:
            return "potentially_too_verbose"
        else:
            return "aligned"
    
    def _check_agreeableness_alignment(self, response: str) -> str:
        """Check if response tone matches agreeableness level."""
        agreeableness = self.big_five.get('agreeableness', 0.5)
        
        # Simple keyword analysis
        disagreeable_words = ['wrong', 'stupid', 'ridiculous', 'disagree', 'nonsense']
        agreeable_words = ['understand', 'appreciate', 'interesting', 'good point']
        
        disagreeable_count = sum(1 for word in disagreeable_words if word in response)
        agreeable_count = sum(1 for word in agreeable_words if word in response)
        
        if agreeableness > 0.7 and disagreeable_count > agreeable_count:
            return "potentially_too_disagreeable"
        elif agreeableness < 0.3 and agreeable_count > disagreeable_count + 1:
            return "potentially_too_agreeable"
        else:
            return "aligned"
    
    def _check_openness_alignment(self, response: str) -> str:
        """Check if response curiosity/exploration matches openness level."""
        openness = self.big_five.get('openness', 0.5)
        
        curious_indicators = ['?', 'interesting', 'wonder', 'explore', 'learn', 'think about']
        curiosity_score = sum(1 for indicator in curious_indicators if indicator in response)
        
        if openness > 0.7 and curiosity_score == 0:
            return "potentially_too_closed"
        elif openness < 0.3 and curiosity_score > 2:
            return "potentially_too_curious"
        else:
            return "aligned"
    
    def _should_consult_web_for_domain(self, topic: str, user_message: str) -> bool:
        """Override to check personality-specific research interests."""
        openness = self.big_five.get('openness', 0.5)
        
        # High openness personalities research more broadly
        if openness > 0.7:
            return True
        
        # Check against core values - research topics that align with values
        for value_info in self.values:
            if value_info['value'].lower() in user_message.lower():
                return True
        
        return False