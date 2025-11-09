"""
Style Transfer Agent for Arena Characters

Based on the talks project's RAGStyleTransferAgent, this module transforms
agent responses to ensure personalized, character-appropriate voices while
eliminating formulaic language patterns.

Automatically detects when agents use research/web tools and rewrites responses
to sound like personal expertise rather than generic academic citations.

Author: Homunculus Team
"""

import re
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class PersonalityType(Enum):
    """Character personality types for voice customization"""
    VISIONARY = "visionary"          # Steve Jobs - innovation, user experience focus
    ANALYTICAL = "analytical"        # Bill Gates - logical, systematic, data-driven
    DISRUPTOR = "disruptor"         # Elon Musk - ambitious, unconventional, future-focused
    BUILDER = "builder"             # Jeff Bezos - operational, customer-obsessed, long-term


@dataclass
class VoiceGuidelines:
    """Voice guidelines for a specific personality type"""
    personal_pronouns: List[str]
    opening_patterns: List[str]
    forbidden_phrases: List[str]
    characteristic_expressions: List[str]
    expertise_claims: List[str]


class ArenaStyleTransferAgent:
    """
    Transforms agent responses to ensure personalized, character-appropriate voices.
    
    Based on the talks project's approach but adapted for arena characters.
    Eliminates formulaic beginnings and enforces personal perspective language.
    """
    
    def __init__(self):
        """Initialize style transfer agent with personality voice guidelines"""
        self.voice_guidelines = self._initialize_voice_guidelines()
        
        # Generic formulaic patterns to eliminate (from talks project)
        self.forbidden_generic_patterns = [
            r'^as we continue to explore',
            r'^building on the points? made by',
            r'^i\'d like to delve deeper into',
            r'^to move (the|this) discussion forward',
            r'^the research (suggests|shows|indicates)',
            r'^according to recent (research|studies)',
            r'^studies (show|indicate|suggest)',
            r'^it\'s (clear|widely accepted|commonly known)',
            r'^this (clearly shows|demonstrates)',
            r'^furthermore,?\s*we can see',
            r'^notably,?\s*(the research|studies)',
            r'^as we move forward',
            r'^moving forward',
            r'^furthermore,',
            r'^additionally,',
            r'^moreover,',
            r'^in addition to',
        ]
        
        # Impersonal to personal language transformations
        self.impersonal_to_personal = {
            r'\baccording to research\b': 'in my experience',
            r'\bstudies show\b': 'I\'ve found',
            r'\bresearch indicates\b': 'I believe',
            r'\bit\'s widely accepted\b': 'I\'m convinced',
            r'\bexperts agree\b': 'in my view',
            r'\bthe consensus is\b': 'I believe',
            r'\bit\'s commonly known\b': 'I know',
            r'\banalysts suggest\b': 'I think',
            r'\bdata shows\b': 'my analysis shows',
            r'\breports indicate\b': 'I\'ve observed',
        }
    
    def _initialize_voice_guidelines(self) -> Dict[PersonalityType, VoiceGuidelines]:
        """Initialize voice guidelines for each personality type"""
        return {
            PersonalityType.VISIONARY: VoiceGuidelines(
                personal_pronouns=['I believe', 'In my experience at Apple', 'I\'ve seen', 'I know'],
                opening_patterns=[
                    'Here\'s what I learned building Apple:',
                    'From my experience in consumer tech,',
                    'I\'ve always believed that',
                    'What I discovered at Pixar was',
                    'My approach has always been'
                ],
                forbidden_phrases=[
                    'according to research', 'studies show', 'experts agree', 
                    'it\'s widely accepted', 'the consensus is'
                ],
                characteristic_expressions=[
                    'revolutionary', 'game-changing', 'intuitive', 'elegant',
                    'user experience', 'design thinking', 'magical'
                ],
                expertise_claims=[
                    'From building Apple', 'In my consumer electronics experience',
                    'What I learned in product development', 'My design philosophy'
                ]
            ),
            
            PersonalityType.ANALYTICAL: VoiceGuidelines(
                personal_pronouns=['I analyze', 'In my research', 'I\'ve calculated', 'My data shows'],
                opening_patterns=[
                    'Based on my analysis,',
                    'I\'ve been researching this since my Microsoft days:',
                    'My foundation work shows',
                    'From my software engineering background,',
                    'I\'ve calculated that'
                ],
                forbidden_phrases=[
                    'recent studies', 'data suggests', 'research indicates',
                    'analysts believe', 'reports show'
                ],
                characteristic_expressions=[
                    'systematic approach', 'data-driven', 'scalable solution',
                    'evidence-based', 'measurable impact', 'optimization'
                ],
                expertise_claims=[
                    'From my Microsoft experience', 'In my software development work',
                    'My foundation research shows', 'Based on my technical background'
                ]
            ),
            
            PersonalityType.DISRUPTOR: VoiceGuidelines(
                personal_pronouns=['I\'m convinced', 'I see', 'I\'m building', 'My vision is'],
                opening_patterns=[
                    'Here\'s what I\'m doing at Tesla/SpaceX:',
                    'I\'m convinced the future requires',
                    'My approach at Neuralink shows',
                    'What I\'m building is',
                    'I see a future where'
                ],
                forbidden_phrases=[
                    'conventional wisdom', 'traditional approaches', 'industry standards',
                    'according to experts', 'established practices'
                ],
                characteristic_expressions=[
                    'first principles', 'exponential', 'breakthrough', 'disruptive',
                    'sustainable future', 'multiplanetary', 'neural interface'
                ],
                expertise_claims=[
                    'At Tesla, I\'ve learned', 'SpaceX has shown me',
                    'Building Neuralink taught me', 'My engineering experience'
                ]
            ),
            
            PersonalityType.BUILDER: VoiceGuidelines(
                personal_pronouns=['I\'ve built', 'At Amazon, I learned', 'I focus on', 'My approach'],
                opening_patterns=[
                    'Building Amazon taught me:',
                    'I\'ve learned that customer obsession means',
                    'My long-term approach is',
                    'What Amazon showed me is',
                    'I always start with the customer'
                ],
                forbidden_phrases=[
                    'market research', 'industry trends', 'business schools teach',
                    'consulting firms say', 'best practices suggest'
                ],
                characteristic_expressions=[
                    'customer obsession', 'long-term thinking', 'operational excellence',
                    'scale', 'efficiency', 'logistics', 'customer experience'
                ],
                expertise_claims=[
                    'Building Amazon from scratch', 'My e-commerce experience',
                    'What I learned scaling globally', 'My operations background'
                ]
            )
        }
    
    def detect_needs_style_transfer(self, response: str, used_tools: bool = False) -> bool:
        """
        Detect if response needs style transfer
        
        Args:
            response: Agent response text
            used_tools: Whether agent used web search or research tools
            
        Returns:
            True if response needs style transfer
        """
        response_lower = response.lower()
        
        # Always transfer if tools were used (like talks project)
        if used_tools:
            logger.debug("Style transfer needed - agent used research tools")
            return True
        
        # Check for forbidden generic patterns
        for pattern in self.forbidden_generic_patterns:
            if re.search(pattern, response_lower):
                logger.debug(f"Style transfer needed - found generic pattern: {pattern[:30]}...")
                return True
        
        # Check for impersonal language
        for impersonal_phrase in self.impersonal_to_personal.keys():
            if re.search(impersonal_phrase, response_lower):
                logger.debug(f"Style transfer needed - found impersonal language")
                return True
        
        return False
    
    def transfer_to_character_voice(
        self, 
        response: str, 
        personality_type: PersonalityType,
        character_name: str,
        used_tools: bool = False
    ) -> str:
        """
        Transfer response to character-appropriate voice
        
        Args:
            response: Original response
            personality_type: Character's personality type
            character_name: Character name for logging
            used_tools: Whether agent used research tools
            
        Returns:
            Response transformed to character voice
        """
        if not self.detect_needs_style_transfer(response, used_tools):
            return response
        
        logger.debug(f"Applying style transfer for {character_name} ({personality_type.value})")
        
        transformed = response
        guidelines = self.voice_guidelines[personality_type]
        
        # 1. Remove formulaic beginnings
        transformed = self._remove_formulaic_beginnings(transformed)
        
        # 2. Transform impersonal to personal language
        transformed = self._transform_impersonal_language(transformed)
        
        # 3. Add character-specific voice elements
        transformed = self._add_character_voice(transformed, guidelines, character_name)
        
        # 4. Ensure proper opening if needed
        transformed = self._ensure_character_opening(transformed, guidelines)
        
        if transformed != response:
            logger.info(f"Style transfer applied for {character_name}")
        
        return transformed
    
    def _remove_formulaic_beginnings(self, text: str) -> str:
        """Remove formulaic conversation beginnings"""
        text_trimmed = text.strip()
        
        for pattern in self.forbidden_generic_patterns:
            # Find and remove the pattern, then clean up
            match = re.search(pattern, text_trimmed, re.IGNORECASE)
            if match:
                # Get the start position of the match
                start_pos = match.start()
                end_pos = match.end()
                
                # If match is at the beginning, remove the entire formulaic phrase
                if start_pos == 0:
                    # Look for a comma or clause end after the formulaic beginning
                    clause_end = re.search(r'[,]\s+', text_trimmed[end_pos:])
                    if clause_end:
                        # Remove formulaic beginning up to the comma, then continue with rest
                        text_trimmed = text_trimmed[end_pos + clause_end.end():].strip()
                        # Capitalize first letter of remaining text
                        if text_trimmed:
                            text_trimmed = text_trimmed[0].upper() + text_trimmed[1:]
                    else:
                        # Remove entire formulaic sentence
                        sentence_end = re.search(r'[.!?]\s+', text_trimmed[end_pos:])
                        if sentence_end:
                            text_trimmed = text_trimmed[end_pos + sentence_end.end():].strip()
                        else:
                            # Remove the pattern and continue with remaining text
                            text_trimmed = text_trimmed[end_pos:].strip()
                            if text_trimmed:
                                text_trimmed = text_trimmed[0].upper() + text_trimmed[1:]
                break
        
        return text_trimmed
    
    def _transform_impersonal_language(self, text: str) -> str:
        """Transform impersonal language to personal voice"""
        transformed = text
        
        for impersonal, personal in self.impersonal_to_personal.items():
            transformed = re.sub(impersonal, personal, transformed, flags=re.IGNORECASE)
        
        return transformed
    
    def _add_character_voice(self, text: str, guidelines: VoiceGuidelines, character_name: str) -> str:
        """Add character-specific voice elements"""
        # This could be enhanced to inject characteristic expressions naturally
        # For now, just ensure we're not using forbidden phrases
        
        text_lower = text.lower()
        for forbidden in guidelines.forbidden_phrases:
            if forbidden in text_lower:
                # Replace with appropriate personal alternative
                if 'research' in forbidden:
                    text = re.sub(re.escape(forbidden), 'my experience', text, flags=re.IGNORECASE)
                elif 'studies' in forbidden:
                    text = re.sub(re.escape(forbidden), 'I\'ve found', text, flags=re.IGNORECASE)
                elif 'experts' in forbidden:
                    text = re.sub(re.escape(forbidden), 'I believe', text, flags=re.IGNORECASE)
        
        return text
    
    def _ensure_character_opening(self, text: str, guidelines: VoiceGuidelines) -> str:
        """Ensure response starts with character-appropriate voice"""
        text_trimmed = text.strip()
        
        if not text_trimmed:
            return text_trimmed
        
        # Check if text starts with a personal pronoun/voice
        if any(text_trimmed.lower().startswith(pronoun.lower()) for pronoun in guidelines.personal_pronouns):
            return text_trimmed
        
        # Check if it starts with a characteristic opening
        if any(text_trimmed.lower().startswith(opening.lower()) for opening in guidelines.opening_patterns):
            return text_trimmed
        
        # If text is too short or already has good personal voice, leave it
        if len(text_trimmed) < 20 or re.search(r'\bi\s+', text_trimmed.lower()[:30]):
            return text_trimmed
        
        # Add a characteristic opening if the text lacks personal voice
        import random
        characteristic_opening = random.choice(guidelines.opening_patterns)
        return f"{characteristic_opening} {text_trimmed[0].lower() + text_trimmed[1:]}"
    
    def get_character_personality(self, character_id: str) -> PersonalityType:
        """Map character ID to personality type"""
        character_mapping = {
            'jobs': PersonalityType.VISIONARY,
            'steve_jobs': PersonalityType.VISIONARY,
            'gates': PersonalityType.ANALYTICAL,
            'bill_gates': PersonalityType.ANALYTICAL,
            'musk': PersonalityType.DISRUPTOR,
            'elon_musk': PersonalityType.DISRUPTOR,
            'bezos': PersonalityType.BUILDER,
            'jeff_bezos': PersonalityType.BUILDER,
        }
        
        character_key = character_id.lower().replace(' ', '_')
        return character_mapping.get(character_key, PersonalityType.ANALYTICAL)  # Default fallback
    
    def get_anti_formulaic_instructions(self, personality_type: PersonalityType) -> str:
        """Get anti-formulaic instructions for character prompts"""
        guidelines = self.voice_guidelines[personality_type]
        
        instructions = f"""
CRITICAL VOICE INSTRUCTIONS:
- NEVER use formulaic beginnings like "As we continue to explore", "Building on the points made by", "I'd like to delve deeper"
- NEVER use impersonal language like "According to research", "Studies show", "Experts agree"
- ALWAYS speak from personal experience and expertise
- Start responses directly with your perspective using: {', '.join(guidelines.personal_pronouns[:3])}
- Use your characteristic expressions: {', '.join(guidelines.characteristic_expressions[:4])}
- Draw from your personal expertise: {', '.join(guidelines.expertise_claims[:2])}

FORBIDDEN PHRASES: {', '.join(guidelines.forbidden_phrases)}
PREFERRED OPENINGS: {', '.join(guidelines.opening_patterns[:3])}
"""
        return instructions