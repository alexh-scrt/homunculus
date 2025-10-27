"""Base agent abstract class with web search capability."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging

try:
    from ..core.agent_input import AgentInput
except ImportError:
    from core.agent_input import AgentInput

try:
    from ..core.character_state import CharacterState
except ImportError:
    from core.character_state import CharacterState

try:
    from ..llm.ollama_client import OllamaClient
except ImportError:
    from llm.ollama_client import OllamaClient


class BaseAgent(ABC):
    """
    Abstract base class for all character agents.
    
    Each agent is a specialist that provides input based on its domain.
    All agents have access to web search capabilities through the LLM client.
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        character_id: str,
        llm_client: OllamaClient
    ):
        """
        Initialize base agent.
        
        Args:
            agent_id: Unique identifier for this agent instance
            agent_type: Type of agent (e.g., "personality", "mood", "goals")
            character_id: ID of the character this agent belongs to
            llm_client: LLM client with web search capabilities
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.character_id = character_id
        self.llm_client = llm_client
        self.logger = logging.getLogger(f"{__name__}.{agent_type}")
        
        # Web search preferences (can be overridden by subclasses)
        self.web_search_enabled = True
        self.web_search_threshold = 0.5  # 0-1, how likely to use web search
        self.max_search_results = 3
    
    @abstractmethod
    def consult(
        self,
        context: Dict[str, Any],
        character_state: CharacterState,
        user_message: str
    ) -> AgentInput:
        """
        Consult this agent for input on how to respond.
        
        This is where the agent uses LLM (and potentially web search) to generate 
        its perspective based on its specialized domain and current character state.
        
        Args:
            context: Current conversation context
            character_state: Current character state
            user_message: What user just said
            
        Returns:
            AgentInput with this agent's perspective
        """
        pass
    
    @abstractmethod
    def get_prompt_template(self) -> str:
        """
        Return the prompt template for this agent.
        Template should have placeholders for state variables.
        """
        pass
    
    def should_search_web(
        self,
        user_message: str,
        context: Dict[str, Any]
    ) -> bool:
        """
        Determine if this agent should use web search for this query.
        
        Default implementation uses simple heuristics, but can be overridden
        by specialized agents with domain-specific logic.
        
        Args:
            user_message: User's message
            context: Conversation context
            
        Returns:
            True if web search should be used
        """
        if not self.llm_client.web_search_enabled:
            return False
        
        # Check if agent type benefits from web search
        web_search_beneficial_agents = {
            'specialty', 'goals', 'memory'  # These agents often need current info
        }
        
        if self.agent_type not in web_search_beneficial_agents:
            return False
        
        # Check for indicators that current information would be helpful
        search_indicators = [
            'current', 'recent', 'latest', 'today', 'now', 'update',
            'what is happening', 'news', 'recent developments',
            'current events', 'what\'s new', 'breaking', 'trend',
            'how to', 'best way', 'recommend', 'advice'
        ]
        
        message_lower = user_message.lower()
        return any(indicator in message_lower for indicator in search_indicators)
    
    def search_web(
        self,
        query: str,
        max_results: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform web search and return results.
        
        Args:
            query: Search query
            max_results: Maximum results to return
            
        Returns:
            List of search results
        """
        if not self.llm_client.web_search_enabled:
            self.logger.warning("Web search requested but not enabled")
            return []
        
        try:
            max_results = max_results or self.max_search_results
            
            self.logger.info(f"Agent {self.agent_type} searching web: {query}")
            
            results = self.llm_client.tavily_tool.search(
                query=query,
                max_results=max_results
            )
            
            self.logger.info(f"Found {len(results)} search results")
            return results
            
        except Exception as e:
            self.logger.error(f"Web search failed in {self.agent_type} agent: {e}")
            return []
    
    async def _call_llm(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 200,
        use_web_search: bool = None
    ) -> str:
        """
        Shared LLM calling logic with optional web search.
        
        Args:
            prompt: The prompt to send to LLM
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            use_web_search: Whether to enable web search (None = auto-detect)
            
        Returns:
            LLM response text
        """
        try:
            if use_web_search is None:
                # Auto-detect if web search should be used based on prompt
                use_web_search = self.llm_client._should_use_web_search(prompt)
            
            if use_web_search and self.web_search_enabled:
                # Use LLM with web search capability
                result = await self.llm_client.generate_with_web_search(
                    prompt=prompt,
                    enable_search=True,
                    max_search_results=self.max_search_results,
                    temperature=temperature
                )
                
                # Log if web search was used
                if result['search_used']:
                    self.logger.info(f"Agent {self.agent_type} used web search: {result['search_query']}")
                
                return result['response']
            else:
                # Use regular LLM generation
                return await self.llm_client.generate(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
        except Exception as e:
            self.logger.error(f"LLM call failed in {self.agent_type} agent: {e}")
            # Return a fallback response
            return f"I need to think about this more. (Agent {self.agent_type} encountered an error)"
    
    def _extract_agent_state(
        self,
        character_state: CharacterState
    ) -> Dict[str, Any]:
        """
        Extract this agent's relevant state from character state.
        Each subclass should implement this to get its specific state.
        
        Args:
            character_state: Complete character state
            
        Returns:
            Dictionary of state relevant to this agent
        """
        return character_state.agent_states.get(self.agent_type, {})
    
    def _format_conversation_history(
        self,
        history: List[Dict[str, str]],
        limit: int = 3
    ) -> str:
        """
        Format conversation history for inclusion in prompts.
        
        Args:
            history: List of conversation messages
            limit: Maximum number of messages to include
            
        Returns:
            Formatted history string
        """
        if not history:
            return "First message in conversation"
        
        recent_history = history[-limit:] if len(history) > limit else history
        
        formatted_messages = []
        for msg in recent_history:
            role = msg.get('role', 'unknown')
            content = msg.get('message', '')
            formatted_messages.append(f"{role.upper()}: {content}")
        
        return "\n".join(formatted_messages)
    
    def _create_agent_input(
        self,
        content: str,
        confidence: float,
        priority: float,
        emotional_tone: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentInput:
        """
        Create standardized AgentInput with validation.
        
        Args:
            content: Main recommendation/observation
            confidence: 0-1, how certain is this agent
            priority: 0-1, how important is this input
            emotional_tone: Optional emotional tone
            metadata: Optional additional data
            
        Returns:
            Validated AgentInput instance
        """
        return AgentInput(
            agent_type=self.agent_type,
            content=content,
            confidence=max(0.0, min(1.0, confidence)),
            priority=max(0.0, min(1.0, priority)),
            emotional_tone=emotional_tone,
            metadata=metadata or {}
        )
    
    def _should_consult_web_for_domain(
        self,
        topic: str,
        user_message: str
    ) -> bool:
        """
        Determine if web search would help with domain-specific knowledge.
        Override in specialized agents for domain-specific logic.
        
        Args:
            topic: The topic being discussed
            user_message: User's message
            
        Returns:
            True if web search would be beneficial
        """
        # Default implementation - very conservative
        return False
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about this agent."""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'character_id': self.character_id,
            'web_search_enabled': self.web_search_enabled,
            'web_search_threshold': self.web_search_threshold,
            'max_search_results': self.max_search_results
        }