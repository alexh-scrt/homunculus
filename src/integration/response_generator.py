"""Response generator for creating final character responses."""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

try:
    from ..core.character_state import CharacterState
except ImportError:
    from core.character_state import CharacterState


class ResponseGenerator:
    """
    Generates final character responses based on cognitive processing results.
    
    This module takes the synthesized agent inputs and cognitive analysis
    to create coherent, character-consistent responses.
    """
    
    def __init__(
        self,
        character_id: str,
        llm_client: Any,
        generation_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the response generator."""
        self.character_id = character_id
        self.llm_client = llm_client
        self.generation_config = generation_config or {}
        self.logger = logging.getLogger(__name__)
        
        # Response generation parameters
        self.default_temperature = self.generation_config.get('temperature', 0.8)
        self.default_max_tokens = self.generation_config.get('max_tokens', 400)
        self.response_style = self.generation_config.get('style', 'conversational')
        
        self.logger.info(f"ResponseGenerator initialized for character {character_id}")
    
    def generate_response(
        self,
        orchestration_result: Dict[str, Any],
        cognitive_result: Dict[str, Any],
        character_state: CharacterState,
        user_message: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate the final character response.
        
        Args:
            orchestration_result: Result from AgentOrchestrator
            cognitive_result: Result from CognitiveModule
            character_state: Current character state
            user_message: User's message
            context: Conversation context
            
        Returns:
            Generated response with metadata
        """
        try:
            # Extract key information
            synthesis = orchestration_result.get('synthesis', {})
            cognitive_patterns = cognitive_result.get('cognitive_patterns', {})
            response_strategy = cognitive_result.get('response_strategy', {})
            emotional_intelligence = cognitive_result.get('emotional_intelligence', {})
            
            # Build comprehensive prompt
            prompt = self._build_response_prompt(
                synthesis, cognitive_patterns, response_strategy,
                emotional_intelligence, character_state, user_message, context
            )
            
            # Determine generation parameters
            generation_params = self._determine_generation_parameters(
                response_strategy, cognitive_patterns, synthesis
            )
            
            # Generate response using LLM
            raw_response = self.llm_client.generate(
                prompt=prompt,
                temperature=generation_params['temperature'],
                max_tokens=generation_params['max_tokens']
            )
            
            # Post-process response
            processed_response = self._post_process_response(
                raw_response, response_strategy, character_state
            )
            
            # Create response metadata
            response_metadata = self._create_response_metadata(
                orchestration_result, cognitive_result, generation_params
            )
            
            # Package final response
            final_response = {
                'response_text': processed_response,
                'response_metadata': response_metadata,
                'generation_info': {
                    'prompt_length': len(prompt),
                    'raw_response_length': len(raw_response),
                    'processed_response_length': len(processed_response),
                    'generation_params': generation_params
                },
                'character_insights': {
                    'emotional_tone': synthesis.get('emotional_tone', 'neutral'),
                    'confidence_level': synthesis.get('confidence_level', 0.5),
                    'thinking_style': cognitive_patterns.get('thinking_style', 'balanced'),
                    'cognitive_load': cognitive_patterns.get('cognitive_load', 0.5)
                }
            }
            
            self.logger.debug(f"Response generated for character {self.character_id}")
            return final_response
            
        except Exception as e:
            self.logger.error(f"Response generation failed: {e}")
            return self._create_fallback_response(user_message, str(e))
    
    def _build_response_prompt(
        self,
        synthesis: Dict[str, Any],
        cognitive_patterns: Dict[str, Any],
        response_strategy: Dict[str, Any],
        emotional_intelligence: Dict[str, Any],
        character_state: CharacterState,
        user_message: str,
        context: Dict[str, Any]
    ) -> str:
        """Build the comprehensive response generation prompt."""
        
        prompt_template = """You are embodying a character with a sophisticated multi-agent cognitive system. Respond as this character would, based on their current mental state.

CHARACTER STATE:
Name: {character_name}
Archetype: {character_archetype}
Current Emotional State: {emotional_tone} (confidence: {confidence_level:.2f})
Thinking Style: {thinking_style}
Cognitive Load: {cognitive_load:.2f} (1.0 = overwhelmed, 0.0 = clear)

NEUROCHEMICAL STATE:
{neurochemical_summary}

COGNITIVE ANALYSIS:
- Primary Attention Focus: {attention_focus}
- Decision Making Mode: {decision_making_mode}
- Self-awareness Level: {self_awareness:.2f}
- Emotional Regulation: {emotional_regulation:.2f}

RESPONSE STRATEGY:
- Approach: {response_approach}
- Complexity Level: {complexity_level}
- Emotional Engagement: {emotional_engagement}
- Thinking Visibility: {thinking_visibility}

AGENT INSIGHTS:
{agent_insights}

MEMORY CONTEXT:
{memory_context}

CONVERSATION CONTEXT:
Previous Topic: {previous_topic}
Relationship Stage: {relationship_stage}

USER MESSAGE: "{user_message}"

Based on this comprehensive analysis, respond as this character. Your response should:

1. REFLECT THE COGNITIVE STATE:
   - Show the current thinking style and cognitive patterns
   - Demonstrate the level of self-awareness indicated
   - Match the emotional tone and regulation level

2. FOLLOW THE RESPONSE STRATEGY:
   - Use the specified approach and complexity level
   - Show appropriate emotional engagement
   - Reveal thinking process according to visibility setting

3. MAINTAIN CHARACTER CONSISTENCY:
   - Stay true to the character's archetype and personality
   - Reflect their current neurochemical and emotional state
   - Incorporate relevant memories and experiences

4. BE NATURAL AND ENGAGING:
   - Respond authentically as a real person would
   - Don't mention the agent system or cognitive analysis
   - Focus on genuine interaction with the user

CHARACTER RESPONSE:"""

        # Format the prompt with actual values
        return prompt_template.format(
            character_name=character_state.name,
            character_archetype=character_state.archetype,
            emotional_tone=synthesis.get('emotional_tone', 'neutral'),
            confidence_level=synthesis.get('confidence_level', 0.5),
            thinking_style=cognitive_patterns.get('thinking_style', 'balanced'),
            cognitive_load=cognitive_patterns.get('cognitive_load', 0.5),
            neurochemical_summary=self._format_neurochemical_summary(character_state),
            attention_focus=cognitive_patterns.get('attention_focus', {}).get('primary_focus', 'external'),
            decision_making_mode=cognitive_patterns.get('decision_making_mode', 'balanced'),
            self_awareness=emotional_intelligence.get('self_awareness', 0.5),
            emotional_regulation=emotional_intelligence.get('emotional_regulation', 0.5),
            response_approach=response_strategy.get('approach', 'balanced'),
            complexity_level=response_strategy.get('complexity_level', 'moderate'),
            emotional_engagement=response_strategy.get('emotional_engagement', 'moderate'),
            thinking_visibility=response_strategy.get('thinking_visibility', 'partial'),
            agent_insights=self._format_agent_insights(synthesis),
            memory_context=self._format_memory_context(synthesis),
            previous_topic=context.get('topic', 'general conversation'),
            relationship_stage=context.get('relationship_stage', 'getting_acquainted'),
            user_message=user_message
        )
    
    def _format_neurochemical_summary(self, character_state: CharacterState) -> str:
        """Format neurochemical levels for the prompt."""
        levels = character_state.neurochemical_levels
        
        # Identify notable levels
        notable = []
        for hormone, level in levels.items():
            if level > 70:
                notable.append(f"elevated {hormone}")
            elif level < 30:
                notable.append(f"low {hormone}")
        
        if notable:
            return f"Notable levels: {', '.join(notable)}"
        else:
            return "Hormone levels are balanced"
    
    def _format_agent_insights(self, synthesis: Dict[str, Any]) -> str:
        """Format agent insights for the prompt."""
        insights = synthesis.get('key_considerations', [])
        
        if not insights:
            return "No specific agent insights"
        
        # Format insights without revealing agent types
        formatted_insights = []
        for insight in insights[:3]:  # Limit to top 3
            # Remove agent type indicators
            clean_insight = insight.split('] ', 1)[-1] if '] ' in insight else insight
            formatted_insights.append(f"- {clean_insight}")
        
        return "\n".join(formatted_insights)
    
    def _format_memory_context(self, synthesis: Dict[str, Any]) -> str:
        """Format memory context for the prompt."""
        agent_metadata = synthesis.get('agent_metadata', {})
        memory_metadata = agent_metadata.get('memory', {})
        
        if not memory_metadata:
            return "No relevant memories retrieved"
        
        episodic = memory_metadata.get('episodic_memories', {})
        semantic = memory_metadata.get('semantic_knowledge', [])
        
        context_parts = []
        
        # Episodic memories
        similar_memories = episodic.get('similar', [])
        if similar_memories:
            context_parts.append(f"Relevant past experiences: {len(similar_memories)} similar situations")
        
        # Semantic knowledge
        if semantic:
            domains = list(set(fact.get('domain', 'general') for fact in semantic))
            context_parts.append(f"Related knowledge: {', '.join(domains)}")
        
        return "; ".join(context_parts) if context_parts else "No specific memory context"
    
    def _determine_generation_parameters(
        self,
        response_strategy: Dict[str, Any],
        cognitive_patterns: Dict[str, Any],
        synthesis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine LLM generation parameters based on character state."""
        params = {
            'temperature': self.default_temperature,
            'max_tokens': self.default_max_tokens
        }
        
        # Adjust temperature based on thinking style and cognitive load
        thinking_style = cognitive_patterns.get('thinking_style', 'balanced')
        cognitive_load = cognitive_patterns.get('cognitive_load', 0.5)
        
        if thinking_style == 'creative':
            params['temperature'] = min(0.95, self.default_temperature + 0.1)
        elif thinking_style == 'systematic':
            params['temperature'] = max(0.6, self.default_temperature - 0.1)
        elif thinking_style == 'cautious':
            params['temperature'] = max(0.5, self.default_temperature - 0.2)
        
        # Adjust max_tokens based on complexity level and cognitive load
        complexity = response_strategy.get('complexity_level', 'moderate')
        
        if complexity == 'high' and cognitive_load < 0.5:
            params['max_tokens'] = min(600, self.default_max_tokens + 100)
        elif complexity == 'simple' or cognitive_load > 0.8:
            params['max_tokens'] = max(200, self.default_max_tokens - 100)
        
        # Adjust based on emotional engagement
        engagement = response_strategy.get('emotional_engagement', 'moderate')
        if engagement == 'high':
            params['temperature'] = min(0.9, params['temperature'] + 0.05)
        elif engagement == 'low':
            params['temperature'] = max(0.6, params['temperature'] - 0.05)
        
        return params
    
    def _post_process_response(
        self,
        raw_response: str,
        response_strategy: Dict[str, Any],
        character_state: CharacterState
    ) -> str:
        """Post-process the generated response."""
        response = raw_response.strip()
        
        # Remove any system artifacts
        response = self._clean_system_artifacts(response)
        
        # Apply style adjustments based on strategy
        response = self._apply_style_adjustments(response, response_strategy, character_state)
        
        # Ensure appropriate length
        response = self._adjust_response_length(response, response_strategy)
        
        return response
    
    def _clean_system_artifacts(self, response: str) -> str:
        """Remove system artifacts from the response."""
        # Remove common AI artifacts
        artifacts = [
            "As an AI", "I'm an artificial", "I'm programmed", "My programming",
            "Based on my analysis", "According to my", "The system indicates",
            "Agent ", "[agent", "cognitive module", "neurochemical"
        ]
        
        for artifact in artifacts:
            if artifact.lower() in response.lower():
                # This is a simple cleanup - in production you might want more sophisticated processing
                response = response.replace(artifact, "")
        
        # Clean up any double spaces or artifacts
        response = " ".join(response.split())
        
        return response
    
    def _apply_style_adjustments(
        self,
        response: str,
        response_strategy: Dict[str, Any],
        character_state: CharacterState
    ) -> str:
        """Apply style adjustments based on response strategy."""
        
        # Get communication style from character state
        style_state = character_state.agent_states.get('communication_style', {})
        
        # Apply verbal pattern adjustments
        verbal_pattern = style_state.get('verbal_pattern', 'moderate')
        if verbal_pattern == 'concise' and len(response.split()) > 100:
            # Shorten verbose responses for concise speakers
            sentences = response.split('. ')
            response = '. '.join(sentences[:2]) + '.'
        elif verbal_pattern == 'verbose' and len(response.split()) < 50:
            # This is harder to adjust automatically - mainly keep as is
            pass
        
        return response
    
    def _adjust_response_length(
        self,
        response: str,
        response_strategy: Dict[str, Any]
    ) -> str:
        """Adjust response length based on strategy."""
        
        complexity_level = response_strategy.get('complexity_level', 'moderate')
        
        word_count = len(response.split())
        
        # Set target ranges based on complexity
        if complexity_level == 'simple':
            target_max = 80
        elif complexity_level == 'high':
            target_max = 300
        else:  # moderate
            target_max = 150
        
        # If too long, try to shorten by removing last sentence
        if word_count > target_max:
            sentences = response.split('. ')
            if len(sentences) > 1:
                response = '. '.join(sentences[:-1]) + '.'
        
        return response
    
    def _create_response_metadata(
        self,
        orchestration_result: Dict[str, Any],
        cognitive_result: Dict[str, Any],
        generation_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create metadata about the response generation process."""
        
        synthesis = orchestration_result.get('synthesis', {})
        conflicts = orchestration_result.get('conflicts_detected', [])
        
        return {
            'character_id': self.character_id,
            'timestamp': datetime.now().isoformat(),
            'emotional_tone': synthesis.get('emotional_tone', 'neutral'),
            'confidence_level': synthesis.get('confidence_level', 0.5),
            'agent_count': len(orchestration_result.get('agent_inputs', {})),
            'conflicts_resolved': len(conflicts),
            'cognitive_load': cognitive_result.get('cognitive_patterns', {}).get('cognitive_load', 0.5),
            'response_strategy': cognitive_result.get('response_strategy', {}),
            'generation_temperature': generation_params.get('temperature', 0.8),
            'max_tokens_used': generation_params.get('max_tokens', 400)
        }
    
    def _create_fallback_response(self, user_message: str, error_message: str) -> Dict[str, Any]:
        """Create a fallback response when generation fails."""
        
        # Simple fallback responses based on message type
        fallback_responses = [
            "I'm having trouble organizing my thoughts right now. Could you give me a moment?",
            "That's an interesting point. I need to think about that a bit more.",
            "I'm feeling a bit scattered at the moment. What do you think about this?",
            "Let me process that for a second..."
        ]
        
        # Choose based on message length (simple heuristic)
        response_index = len(user_message) % len(fallback_responses)
        fallback_text = fallback_responses[response_index]
        
        return {
            'response_text': fallback_text,
            'response_metadata': {
                'character_id': self.character_id,
                'timestamp': datetime.now().isoformat(),
                'fallback': True,
                'error': error_message,
                'emotional_tone': 'uncertain',
                'confidence_level': 0.3
            },
            'generation_info': {
                'fallback_used': True,
                'error_message': error_message
            },
            'character_insights': {
                'emotional_tone': 'uncertain',
                'confidence_level': 0.3,
                'thinking_style': 'confused',
                'cognitive_load': 1.0
            }
        }
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about response generation."""
        return {
            'character_id': self.character_id,
            'default_temperature': self.default_temperature,
            'default_max_tokens': self.default_max_tokens,
            'response_style': self.response_style,
            'generation_config': self.generation_config
        }