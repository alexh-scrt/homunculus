"""Memory agent for retrieving relevant past experiences and knowledge."""

from typing import Dict, Any, Optional, List
import logging
import asyncio

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

try:
    from ..memory.experience_module import ExperienceModule
except ImportError:
    from memory.experience_module import ExperienceModule

try:
    from ..memory.knowledge_graph_module import KnowledgeGraphModule, extract_concepts_from_text
except ImportError:
    from memory.knowledge_graph_module import KnowledgeGraphModule, extract_concepts_from_text


class MemoryAgent(BaseAgent):
    """
    Retrieves relevant memories and knowledge to inform character responses.
    
    Combines episodic memory (experiences) with semantic memory (facts and relationships)
    to provide context for the character's current interaction.
    """
    
    def __init__(
        self,
        agent_id: str,
        character_id: str,
        llm_client: Any
    ):
        """Initialize memory agent."""
        super().__init__(agent_id, "memory", character_id, llm_client)
        
        # Initialize memory modules
        self.experience_module = ExperienceModule(character_id)
        self.knowledge_graph = KnowledgeGraphModule(character_id)
        
        # Memory agents don't need web search - they retrieve stored knowledge
        self.web_search_enabled = False
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"MemoryAgent initialized for character {character_id}")
    
    def consult(
        self,
        context: Dict[str, Any],
        character_state: CharacterState,
        user_message: str
    ) -> AgentInput:
        """
        Retrieve and analyze relevant memories for the current interaction.
        
        Combines episodic memories (past experiences) with semantic knowledge
        to provide rich context for the character's response.
        """
        # Run memory retrieval asynchronously
        memory_data = asyncio.run(self._retrieve_all_relevant_memories(
            user_message, context, character_state
        ))
        
        # Build memory analysis prompt
        prompt = self._build_memory_prompt(
            memory_data,
            user_message,
            context
        )
        
        # Generate memory-informed guidance
        response = self._call_llm(
            prompt=prompt,
            temperature=0.6,  # Balanced between consistency and insight
            max_tokens=250,
            use_web_search=False  # Memory is internal knowledge
        )
        
        # Memory is highly reliable and important for consistency
        confidence = 0.95
        priority = 0.9
        
        emotional_tone = self._determine_memory_emotional_tone(memory_data)
        
        # Package memory metadata for other agents
        metadata = {
            'episodic_memories': memory_data['experiences'],
            'semantic_knowledge': memory_data['facts'],
            'goal_knowledge': memory_data['goal_info'],
            'memory_summary': memory_data['summary'],
            'retrieval_stats': memory_data['stats']
        }
        
        return self._create_agent_input(
            content=response,
            confidence=confidence,
            priority=priority,
            emotional_tone=emotional_tone,
            metadata=metadata
        )
    
    async def _retrieve_all_relevant_memories(
        self,
        user_message: str,
        context: Dict[str, Any],
        character_state: CharacterState
    ) -> Dict[str, Any]:
        """Retrieve all types of relevant memories."""
        
        # Extract concepts from the user message for semantic search
        concepts = await extract_concepts_from_text(user_message)
        
        # Get current goals for goal-related memory retrieval
        active_goals = character_state.agent_states.get('goals', {}).get('active_goals', [])
        goal_ids = [goal.get('goal_id') for goal in active_goals if goal.get('goal_id')]
        
        # Run all memory retrievals in parallel
        memory_tasks = [
            # Episodic memory - similar experiences
            self.experience_module.retrieve_similar_experiences(
                query_text=user_message,
                n_results=5,
                time_window_days=30
            ),
            
            # Recent experiences for context
            self.experience_module.get_recent_experiences(n_results=3, days_back=7),
            
            # Semantic knowledge - facts related to concepts
            self.knowledge_graph.retrieve_related_facts(
                concepts=concepts,
                limit=8,
                min_confidence=0.6
            ),
            
            # Goal-related knowledge
            self._get_goal_related_memories(goal_ids),
            
            # Memory statistics
            self._get_memory_statistics()
        ]
        
        try:
            results = await asyncio.gather(*memory_tasks, return_exceptions=True)
            
            similar_experiences = results[0] if not isinstance(results[0], Exception) else []
            recent_experiences = results[1] if not isinstance(results[1], Exception) else []
            related_facts = results[2] if not isinstance(results[2], Exception) else []
            goal_memories = results[3] if not isinstance(results[3], Exception) else {}
            memory_stats = results[4] if not isinstance(results[4], Exception) else {}
            
        except Exception as e:
            self.logger.error(f"Error retrieving memories: {e}")
            # Return empty results on error
            similar_experiences = []
            recent_experiences = []
            related_facts = []
            goal_memories = {}
            memory_stats = {}
        
        # Combine and summarize memories
        summary = self._create_memory_summary(
            similar_experiences, recent_experiences, related_facts, goal_memories
        )
        
        return {
            'experiences': {
                'similar': similar_experiences,
                'recent': recent_experiences
            },
            'facts': related_facts,
            'goal_info': goal_memories,
            'summary': summary,
            'stats': memory_stats,
            'concepts_searched': concepts
        }
    
    async def _get_goal_related_memories(self, goal_ids: List[str]) -> Dict[str, Any]:
        """Get memories related to active goals."""
        goal_memories = {}
        
        for goal_id in goal_ids:
            try:
                goal_knowledge = await self.knowledge_graph.get_goal_related_knowledge(goal_id)
                if goal_knowledge:
                    goal_memories[goal_id] = goal_knowledge
            except Exception as e:
                self.logger.warning(f"Failed to get memories for goal {goal_id}: {e}")
        
        return goal_memories
    
    async def _get_memory_statistics(self) -> Dict[str, Any]:
        """Get overall memory statistics."""
        try:
            exp_stats = await self.experience_module.get_memory_stats()
            kg_stats = await self.knowledge_graph.get_knowledge_stats()
            
            return {
                'episodic': exp_stats,
                'semantic': kg_stats
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get memory statistics: {e}")
            return {}
    
    def _create_memory_summary(
        self,
        similar_exp: List,
        recent_exp: List,
        facts: List[Dict],
        goal_memories: Dict
    ) -> str:
        """Create a concise summary of retrieved memories."""
        summary_parts = []
        
        # Episodic memory summary
        if similar_exp:
            summary_parts.append(f"Found {len(similar_exp)} similar past experiences")
        
        if recent_exp:
            summary_parts.append(f"{len(recent_exp)} recent experiences")
        
        # Semantic knowledge summary
        if facts:
            domains = list(set(fact.get('domain', 'general') for fact in facts))
            summary_parts.append(f"{len(facts)} relevant facts from domains: {', '.join(domains)}")
        
        # Goal-related memory summary
        if goal_memories:
            summary_parts.append(f"Knowledge for {len(goal_memories)} active goals")
        
        if not summary_parts:
            return "No specific relevant memories found"
        
        return "; ".join(summary_parts)
    
    def _determine_memory_emotional_tone(self, memory_data: Dict[str, Any]) -> str:
        """Determine emotional tone based on retrieved memories."""
        experiences = memory_data.get('experiences', {})
        similar_exp = experiences.get('similar', [])
        
        if not similar_exp:
            return "neutral"
        
        # Analyze emotional valence of similar experiences
        valences = []
        for exp in similar_exp:
            if hasattr(exp, 'emotional_valence'):
                valences.append(exp.emotional_valence)
        
        if not valences:
            return "neutral"
        
        avg_valence = sum(valences) / len(valences)
        
        if avg_valence > 0.3:
            return "positive_memory"
        elif avg_valence < -0.3:
            return "negative_memory"
        else:
            return "neutral_memory"
    
    def get_prompt_template(self) -> str:
        """Return memory analysis prompt template."""
        return """You are analyzing a character's relevant memories to inform their response approach.

MEMORY RETRIEVAL SUMMARY:
{memory_summary}

SIMILAR PAST EXPERIENCES:
{similar_experiences}

RECENT EXPERIENCES:
{recent_experiences}

RELEVANT KNOWLEDGE:
{relevant_facts}

GOAL-RELATED MEMORIES:
{goal_memories}

USER MESSAGE: "{user_message}"
CONVERSATION CONTEXT: {conversation_context}

Based on these memories, analyze how the character's past should influence their current response:

1. EXPERIENTIAL PATTERNS:
   - What patterns emerge from similar past experiences?
   - How did the character respond to similar situations before?
   - What worked well or poorly in the past?

2. EMOTIONAL CONTINUITY:
   - Do past experiences create emotional context for this interaction?
   - Are there unresolved feelings or associations to consider?
   - How might past emotional experiences influence current mood?

3. KNOWLEDGE INTEGRATION:
   - What relevant facts or knowledge should inform the response?
   - Are there connections between current topic and stored knowledge?
   - How can past learning enhance the current conversation?

4. GOAL AWARENESS:
   - How do past experiences relate to current goals?
   - What lessons or progress from memory apply to goal pursuit?
   - Are there opportunities to build on past goal-related actions?

5. CONSISTENCY MAINTENANCE:
   - How can the character maintain consistency with past behavior?
   - What aspects of character identity should be reinforced?
   - Are there contradictions that need to be addressed?

6. MEMORY-INFORMED OPPORTUNITIES:
   - Can past experiences provide unique insights for this conversation?
   - Are there interesting connections or callbacks to make?
   - How can memory enhance the depth of the response?

Provide 2-3 sentences on how the character's memories should shape their response approach, focusing on the most relevant insights from their past."""
    
    def _build_memory_prompt(
        self,
        memory_data: Dict[str, Any],
        user_message: str,
        context: Dict[str, Any]
    ) -> str:
        """Build memory analysis prompt with retrieved data."""
        
        # Format similar experiences
        similar_exp = memory_data['experiences']['similar']
        if similar_exp:
            exp_texts = []
            for exp in similar_exp[:3]:  # Limit to top 3
                similarity = getattr(exp, '_similarity_score', 0.0)
                exp_text = f"[{exp.experience_type}] {exp.description[:100]}... (similarity: {similarity:.2f})"
                exp_texts.append(exp_text)
            similar_text = "\n".join(exp_texts)
        else:
            similar_text = "No similar experiences found"
        
        # Format recent experiences
        recent_exp = memory_data['experiences']['recent']
        if recent_exp:
            recent_texts = []
            for exp in recent_exp[:3]:
                recent_text = f"[{exp.timestamp.strftime('%Y-%m-%d')}] {exp.description[:80]}..."
                recent_texts.append(recent_text)
            recent_text = "\n".join(recent_texts)
        else:
            recent_text = "No recent experiences"
        
        # Format relevant facts
        facts = memory_data['facts']
        if facts:
            fact_texts = []
            for fact in facts[:5]:  # Limit to top 5
                confidence = fact.get('confidence', 0.0)
                domain = fact.get('domain', 'general')
                fact_text = f"[{domain}] {fact['text'][:80]}... (confidence: {confidence:.2f})"
                fact_texts.append(fact_text)
            facts_text = "\n".join(fact_texts)
        else:
            facts_text = "No relevant facts found"
        
        # Format goal memories
        goal_memories = memory_data['goal_info']
        if goal_memories:
            goal_texts = []
            for goal_id, goal_data in goal_memories.items():
                progress = goal_data.get('progress', 0.0)
                description = goal_data.get('description', 'Unknown goal')
                fact_count = len(goal_data.get('related_facts', []))
                goal_text = f"[{goal_id}] {description[:60]}... (progress: {progress:.1%}, {fact_count} facts)"
                goal_texts.append(goal_text)
            goal_text = "\n".join(goal_texts)
        else:
            goal_text = "No goal-related memories"
        
        # Get conversation context
        conversation_context = context.get('topic', 'general conversation')
        
        return self.get_prompt_template().format(
            memory_summary=memory_data['summary'],
            similar_experiences=similar_text,
            recent_experiences=recent_text,
            relevant_facts=facts_text,
            goal_memories=goal_text,
            user_message=user_message,
            conversation_context=conversation_context
        )
    
    async def store_interaction_memory(
        self,
        user_message: str,
        character_response: str,
        character_state: CharacterState,
        web_search_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store the current interaction as a memory.
        
        This should be called after each interaction to build episodic memory.
        """
        try:
            from ..memory.experience_module import create_experience_from_interaction
        except ImportError:
            from memory.experience_module import create_experience_from_interaction
        
        try:
            # Create experience from interaction
            experience = await create_experience_from_interaction(
                character_id=self.character_id,
                user_message=user_message,
                character_response=character_response,
                emotional_state=character_state.agent_states.get('mood', {}).get('current_state', 'neutral'),
                web_search_data=web_search_data,
                additional_metadata={
                    'emotional_valence': self._calculate_interaction_valence(user_message, character_response),
                    'intensity': self._calculate_interaction_intensity(user_message, character_response),
                    'related_goals': [goal.get('goal_id') for goal in character_state.agent_states.get('goals', {}).get('active_goals', [])]
                }
            )
            
            # Store the experience
            success = await self.experience_module.store_experience(experience)
            
            if success:
                self.logger.debug(f"Stored interaction memory: {experience.experience_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to store interaction memory: {e}")
            return False
    
    def _calculate_interaction_valence(self, user_message: str, character_response: str) -> float:
        """Calculate emotional valence of the interaction (-1 to 1)."""
        # Simple sentiment analysis based on keywords
        positive_words = ['good', 'great', 'wonderful', 'happy', 'excited', 'love', 'amazing', 'fantastic']
        negative_words = ['bad', 'terrible', 'sad', 'angry', 'hate', 'awful', 'horrible', 'worried']
        
        text = f"{user_message} {character_response}".lower()
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        # Calculate valence based on word sentiment
        valence = (positive_count - negative_count) / max(total_words / 10, 1)
        return max(-1.0, min(1.0, valence))
    
    def _calculate_interaction_intensity(self, user_message: str, character_response: str) -> float:
        """Calculate interaction intensity (0 to 1)."""
        # Base intensity on message length and emotional indicators
        text = f"{user_message} {character_response}"
        
        # Length factor
        length_factor = min(len(text) / 500, 1.0)
        
        # Emotional intensity indicators
        intensity_words = ['very', 'extremely', 'really', 'absolutely', 'incredibly', '!', '?', 'amazing', 'terrible']
        intensity_count = sum(1 for word in intensity_words if word.lower() in text.lower())
        intensity_factor = min(intensity_count / 10, 1.0)
        
        # Combine factors
        intensity = (length_factor + intensity_factor) / 2
        return max(0.1, min(1.0, intensity))  # Minimum 0.1 to ensure some significance
    
    def close(self):
        """Clean up memory module resources."""
        try:
            self.experience_module.close()
            self.knowledge_graph.close()
            self.logger.info(f"MemoryAgent closed for character {self.character_id}")
        except Exception as e:
            self.logger.error(f"Error closing MemoryAgent: {e}")