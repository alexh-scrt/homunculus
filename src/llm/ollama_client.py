"""Ollama client with LangChain integration and Tavily web search tool support."""

from typing import Optional, List, Dict, Any, Union
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    from langchain_community.llms import Ollama
    from langchain_core.prompts import PromptTemplate
    from langchain_core.tools import Tool
    from tavily import TavilyClient
except ImportError as e:
    raise ImportError(f"Required dependencies not installed: {e}")

try:
    from ..config.settings import get_settings
except ImportError:
    # For testing or standalone usage
    from config.settings import get_settings


class TavilySearchTool:
    """Wrapper for Tavily web search functionality."""
    
    def __init__(self, api_key: str):
        """Initialize Tavily client."""
        self.client = TavilyClient(api_key=api_key)
        self.logger = logging.getLogger(__name__)
    
    def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic"
    ) -> List[Dict[str, Any]]:
        """
        Perform web search using Tavily.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            search_depth: "basic" or "advanced" search depth
            
        Returns:
            List of search results with title, url, content, etc.
        """
        try:
            self.logger.info(f"Performing web search for: {query}")
            
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_answer=True,
                include_raw_content=False
            )
            
            results = response.get('results', [])
            self.logger.info(f"Found {len(results)} search results")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Web search failed: {e}")
            return []
    
    def get_search_summary(self, query: str) -> str:
        """Get a concise search summary for a query."""
        try:
            response = self.client.search(
                query=query,
                max_results=3,
                search_depth="basic",
                include_answer=True
            )
            
            # Return the AI-generated answer if available
            return response.get('answer', 'No summary available')
            
        except Exception as e:
            self.logger.error(f"Search summary failed: {e}")
            return f"Search failed: {str(e)}"


class OllamaClient:
    """
    Enhanced Ollama client with LangChain integration and Tavily web search.
    
    Provides both direct text generation and tool-enhanced generation
    where the LLM can decide to use web search when needed.
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        tavily_api_key: Optional[str] = None
    ):
        """Initialize Ollama client with optional Tavily integration."""
        self.settings = get_settings()
        
        # Use provided values or fall back to settings
        self.base_url = base_url or self.settings.ollama_base_url
        self.model = model or self.settings.ollama_model
        self.temperature = temperature
        
        # Initialize Ollama LLM
        self.llm = Ollama(
            base_url=self.base_url,
            model=self.model,
            temperature=self.temperature
        )
        
        # Initialize Tavily if API key is available
        self.tavily_tool = None
        self.web_search_enabled = False
        
        tavily_key = tavily_api_key or self.settings.tavily_api_key
        if tavily_key and self.settings.web_search_enabled:
            try:
                self.tavily_tool = TavilySearchTool(tavily_key)
                self.web_search_enabled = True
                self.logger.info("Tavily web search enabled")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Tavily: {e}")
        
        self.logger = logging.getLogger(__name__)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """
        Generate text from prompt with retry logic.
        
        Args:
            prompt: The input prompt
            temperature: Override default temperature
            max_tokens: Maximum tokens to generate
            stop_sequences: Sequences that stop generation
            
        Returns:
            Generated text response
        """
        try:
            # Create LLM instance with custom parameters if needed
            if temperature is not None or max_tokens is not None:
                custom_llm = Ollama(
                    base_url=self.base_url,
                    model=self.model,
                    temperature=temperature or self.temperature,
                    # Note: Ollama doesn't directly support max_tokens, this depends on model
                )
                response = custom_llm.invoke(prompt)
            else:
                response = self.llm.invoke(prompt)
            
            self.logger.debug(f"Generated response length: {len(response)} characters")
            return response
            
        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            raise
    
    def generate_with_template(
        self,
        template: str,
        variables: Dict[str, Any],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text using a template with variables.
        
        Args:
            template: Prompt template with placeholders
            variables: Dictionary of variables to fill template
            temperature: Override default temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        try:
            prompt_template = PromptTemplate.from_template(template)
            prompt = prompt_template.format(**variables)
            
            return self.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
        except Exception as e:
            self.logger.error(f"Template generation failed: {e}")
            raise
    
    def generate_with_web_search(
        self,
        prompt: str,
        enable_search: bool = True,
        max_search_results: int = 3,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate text with optional web search capability.
        
        The LLM can decide whether to use web search based on the prompt.
        This is a simplified version - in production you'd use LangChain agents.
        
        Args:
            prompt: The input prompt
            enable_search: Whether to allow web search
            max_search_results: Max search results to use
            temperature: Override default temperature
            
        Returns:
            Dict containing response, search_used, and search_results
        """
        result = {
            'response': '',
            'search_used': False,
            'search_results': [],
            'search_query': None
        }
        
        try:
            # First, check if the prompt suggests a need for current information
            should_search = self._should_use_web_search(prompt) if enable_search and self.web_search_enabled else False
            
            if should_search:
                # Extract search query from the prompt
                search_query = self._extract_search_query(prompt)
                
                if search_query:
                    self.logger.info(f"Using web search for query: {search_query}")
                    
                    # Perform web search
                    search_results = self.tavily_tool.search(
                        query=search_query,
                        max_results=max_search_results
                    )
                    
                    if search_results:
                        # Enhance prompt with search results
                        enhanced_prompt = self._enhance_prompt_with_search_results(
                            prompt, search_results
                        )
                        
                        result['response'] = self.generate(
                            enhanced_prompt,
                            temperature=temperature
                        )
                        result['search_used'] = True
                        result['search_results'] = search_results
                        result['search_query'] = search_query
                    else:
                        # Fallback to regular generation if search fails
                        result['response'] = self.generate(prompt, temperature=temperature)
                else:
                    # Couldn't extract search query, proceed normally
                    result['response'] = self.generate(prompt, temperature=temperature)
            else:
                # No search needed, proceed with regular generation
                result['response'] = self.generate(prompt, temperature=temperature)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Generation with web search failed: {e}")
            # Fallback to basic generation
            result['response'] = self.generate(prompt, temperature=temperature)
            return result
    
    def _should_use_web_search(self, prompt: str) -> bool:
        """
        Simple heuristic to determine if web search would be helpful.
        In production, this could be more sophisticated or LLM-based.
        """
        search_indicators = [
            'current', 'recent', 'latest', 'today', 'now', 'update',
            'what is happening', 'news', 'recent developments',
            'current events', 'what\'s new', 'breaking', 'trend'
        ]
        
        prompt_lower = prompt.lower()
        return any(indicator in prompt_lower for indicator in search_indicators)
    
    def _extract_search_query(self, prompt: str) -> Optional[str]:
        """
        Extract a search query from the prompt.
        This is a simple implementation - could be enhanced with NER or LLM.
        """
        # Simple approach: use the prompt as the search query, cleaned up
        # Remove common question words and make it more search-friendly
        
        query = prompt.lower()
        
        # Remove question words that don't help with search
        remove_words = ['what', 'is', 'the', 'tell', 'me', 'about', 'can', 'you', 'please']
        words = query.split()
        filtered_words = [w for w in words if w not in remove_words]
        
        search_query = ' '.join(filtered_words[:8])  # Limit to 8 words
        
        return search_query if len(search_query) > 3 else prompt[:100]
    
    def _enhance_prompt_with_search_results(
        self,
        original_prompt: str,
        search_results: List[Dict[str, Any]]
    ) -> str:
        """
        Enhance the original prompt with web search results.
        """
        # Format search results
        results_text = "Current information from web search:\n\n"
        
        for i, result in enumerate(search_results[:3], 1):
            title = result.get('title', 'No title')
            content = result.get('content', '')[:300] + "..." if len(result.get('content', '')) > 300 else result.get('content', '')
            url = result.get('url', '')
            
            results_text += f"{i}. {title}\n{content}\nSource: {url}\n\n"
        
        # Combine with original prompt
        enhanced_prompt = f"""Based on the current information provided below, please answer the following question:

{results_text}

Question: {original_prompt}

Please provide a comprehensive answer using the current information above, and indicate what sources you're drawing from."""
        
        return enhanced_prompt
    
    def test_connection(self) -> bool:
        """Test if Ollama is responding."""
        try:
            response = self.generate("Hi", temperature=0.1)
            return len(response) > 0
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
    
    def test_web_search(self) -> bool:
        """Test if web search is working."""
        if not self.web_search_enabled:
            return False
        
        try:
            results = self.tavily_tool.search("test query", max_results=1)
            return len(results) > 0
        except Exception as e:
            self.logger.error(f"Web search test failed: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get client status information."""
        return {
            'ollama_configured': True,
            'ollama_base_url': self.base_url,
            'ollama_model': self.model,
            'web_search_enabled': self.web_search_enabled,
            'tavily_configured': self.tavily_tool is not None,
            'connection_ok': self.test_connection(),
            'web_search_ok': self.test_web_search()
        }