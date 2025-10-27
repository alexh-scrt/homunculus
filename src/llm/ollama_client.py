"""Ollama client with LangChain integration and Tavily web search tool support."""

from typing import Optional, List, Dict, Any, Union
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    # Use the newer langchain-ollama package instead of langchain-community
    from langchain_ollama import ChatOllama
    from langchain_core.prompts import PromptTemplate
    from langchain_core.tools import BaseTool
    from langchain_core.messages import HumanMessage, SystemMessage
    
    # Try newer langchain-tavily package first
    try:
        from langchain_tavily import TavilySearch as TavilySearchResults
    except ImportError:
        # Fallback to older langchain-community version
        from langchain_community.tools.tavily_search import TavilySearchResults
        
except ImportError as e:
    try:
        # Fallback to community version if new one not available
        from langchain_community.llms import ChatOllama
        from langchain_core.prompts import PromptTemplate
        from langchain_core.tools import BaseTool
        from langchain_core.messages import HumanMessage, SystemMessage
        
        # Try newer langchain-tavily package first
        try:
            from langchain_tavily import TavilySearch as TavilySearchResults
        except ImportError:
            # Fallback to older langchain-community version
            from langchain_community.tools.tavily_search import TavilySearchResults
            
    except ImportError:
        raise ImportError(f"Required dependencies not installed: {e}")

try:
    from ..config.settings import get_settings
except ImportError:
    # For testing or standalone usage
    from config.settings import get_settings




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
        tavily_api_key: Optional[str] = None,
        web_search: bool = True
    ):
        """Initialize Ollama client with optional Tavily integration."""
        self.settings = get_settings()
        
        # Use provided values or fall back to settings
        self.base_url = base_url or self.settings.ollama_base_url
        self.model = model or self.settings.ollama_model
        self.temperature = temperature
        
        # Initialize logger first
        self.logger = logging.getLogger(__name__)
        
        # Web search capability (expose for agent compatibility)
        self.web_search_enabled = self.settings.web_search_enabled and web_search
        
        # Track tool usage
        self._last_tool_calls = []  # Store tool call history
        self._tools_used_this_turn = False  # Flag for current turn
        
        # Setup LLM parameters
        llm_params = {
            "base_url": self.base_url,
            "model": self.model,
            "temperature": self.temperature
        }
        
        self.llm = ChatOllama(**llm_params)
        
        # Setup tools if needed
        self.tools = [] if not web_search else self._setup_tools()
        
        # Bind tools to LLM if available
        if self.tools:
            self.llm = self.llm.bind_tools(self.tools)
            self.logger.info(f"Bound {len(self.tools)} tools to Ollama LLM")
    
    def _setup_tools(self) -> List[BaseTool]:
        """Setup tools for the LLM following the reference implementation."""
        tools = []
        
        # Add web search tool for research
        tavily_api_key = self.settings.tavily_api_key
        if tavily_api_key and self.settings.web_search_enabled:
            try:
                # Set environment variable for TavilySearch to use
                import os
                os.environ['TAVILY_API_KEY'] = tavily_api_key
                
                # Use the newer TavilySearch (imported as TavilySearchResults)
                tavily_tool = TavilySearchResults(
                    max_results=5,
                    include_answer=True,
                    include_raw_content=False,
                    include_images=False,
                    search_depth="basic",
                    include_domains=[],
                    exclude_domains=[],
                )
                
                tools.append(tavily_tool)
                self.logger.info("✅ Tavily search tool added to Ollama client")
            except Exception as e:
                self.logger.error(f"❌ Failed to setup Tavily for Ollama client: {e}")
        else:
            self.logger.warning("⚠️ TAVILY_API_KEY not set or web search disabled - web search not available")
                
        return tools
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text from prompt with tool support and retry logic.
        
        Args:
            prompt: The input prompt
            system_prompt: Optional system prompt
            temperature: Override default temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        try:
            # Reset tool usage tracking for this turn
            self._last_tool_calls = []
            self._tools_used_this_turn = False
            
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))
            
            # Create LLM instance with custom parameters if needed
            if temperature is not None:
                custom_llm = ChatOllama(
                    base_url=self.base_url,
                    model=self.model,
                    temperature=temperature
                )
                if self.tools:
                    custom_llm = custom_llm.bind_tools(self.tools)
                response = custom_llm.invoke(messages)
            else:
                response = self.llm.invoke(messages)
            
            # Handle tool calls if present
            if hasattr(response, 'tool_calls') and response.tool_calls:
                self._last_tool_calls = response.tool_calls  # Track calls
                self._tools_used_this_turn = True  # Set flag
                
                # Process tool calls
                tool_results = []
                for tool_call in response.tool_calls:
                    tool_name = tool_call['name']
                    tool_args = tool_call['args']
                    
                    # Find and execute the tool
                    tool_found = False
                    for tool in self.tools:
                        if tool.name == tool_name:
                            tool_found = True
                            try:
                                result = tool.invoke(tool_args)
                                tool_results.append(result)
                                self.logger.info(f"🔍 Ollama client used {tool_name}")
                            except Exception as e:
                                self.logger.error(f"Tool execution failed: {e}")
                                tool_results.append(f"Tool error: {str(e)}")
                            break
                    
                    # Handle unknown tools
                    if not tool_found:
                        self.logger.warning(f"LLM tried to call unknown tool: {tool_name}")
                        tool_results.append(f"Unknown tool: {tool_name}")
                
                # If we have tool results, generate a final response incorporating them
                if tool_results:
                    messages.append(response)
                    tool_message = f"Tool results: {tool_results}"
                    messages.append(HumanMessage(content=tool_message))
                    final_response = self.llm.invoke(messages)
                    final_content = final_response.content if isinstance(final_response.content, str) else str(final_response.content)
                    
                    # Ensure we have meaningful content
                    if not final_content or not final_content.strip():
                        self.logger.warning("Empty response after tool execution, creating fallback response")
                        # Create a basic response incorporating tool results
                        if tool_results and len(tool_results) > 0:
                            if isinstance(tool_results[0], dict) and 'content' in tool_results[0]:
                                final_content = tool_results[0]['content']
                            elif isinstance(tool_results[0], str):
                                final_content = tool_results[0]
                            else:
                                final_content = f"Based on my search, here's what I found: {str(tool_results[0])}"
                        else:
                            final_content = "I searched for information but need to process the results further."
                    
                    return self._strip_reasoning(final_content)
            
            # Handle case where response has content but it might be tool call JSON
            if hasattr(response, 'content'):
                # Handle different content types
                if isinstance(response.content, str):
                    response_text = response.content
                elif isinstance(response.content, list):
                    # If content is a list, try to extract text content
                    response_text = ""
                    for item in response.content:
                        if isinstance(item, str):
                            response_text += item
                        elif isinstance(item, dict) and 'text' in item:
                            response_text += item['text']
                    if not response_text:
                        response_text = str(response.content)
                else:
                    response_text = str(response.content)
            else:
                response_text = str(response)
            
            # If response looks like JSON tool call, extract meaningful content
            if isinstance(response_text, str) and response_text.strip().startswith('{') and '"name"' in response_text and '"parameters"' in response_text:
                self.logger.warning("Received raw tool call JSON as response content - this shouldn't happen")
                # Try to extract any actual content or provide fallback
                try:
                    import json
                    tool_data = json.loads(response_text)
                    if 'parameters' in tool_data and 'tool_results' in tool_data['parameters']:
                        tool_results = tool_data['parameters']['tool_results']
                        if tool_results and len(tool_results) > 0 and 'response' in tool_results[0]:
                            response_text = tool_results[0]['response']
                        else:
                            response_text = "I need to process that information further."
                    else:
                        response_text = "I'm thinking about how to respond to that."
                except:
                    response_text = "Let me think about that for a moment."
            
            self.logger.debug(f"Generated response length: {len(response_text)} characters")
            return self._strip_reasoning(str(response_text))
            
        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            raise
    
    def _strip_reasoning(self, text: str) -> str:
        """Strip reasoning blocks from response text."""
        import re
        # Remove common reasoning patterns
        reasoning_patterns = [
            r'<think>.*?</think>',
            r'<thinking>.*?</thinking>',
            r'<reasoning>.*?</reasoning>',
            r'\[thinking\].*?\[/thinking\]',
            r'\[reasoning\].*?\[/reasoning\]'
        ]
        
        cleaned_text = text
        for pattern in reasoning_patterns:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.DOTALL)
        
        return cleaned_text.strip()
    
    def generate_with_template(
        self,
        template: str,
        variables: Dict[str, Any],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text using a template with variables.
        
        Args:
            template: Prompt template with placeholders
            variables: Dictionary of variables to fill template
            system_prompt: Optional system prompt
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
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
        except Exception as e:
            self.logger.error(f"Template generation failed: {e}")
            raise
    
    def get_tool_usage_info(self) -> Dict[str, Any]:
        """Get information about tool usage in the last turn."""
        return {
            'tools_used': self._tools_used_this_turn,
            'tool_calls': self._last_tool_calls,
            'available_tools': [tool.name for tool in self.tools]
        }
    
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
        if not self.tools:
            return False
        
        try:
            # Test by attempting to use the tool directly
            for tool in self.tools:
                if "tavily" in tool.name.lower():
                    result = tool.invoke({"query": "test query"})
                    return len(result) > 0 if isinstance(result, (list, str)) else True
            return False
        except Exception as e:
            self.logger.error(f"Web search test failed: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get client status information."""
        return {
            'ollama_configured': True,
            'ollama_base_url': self.base_url,
            'ollama_model': self.model,
            'tools_enabled': len(self.tools) > 0,
            'available_tools': [tool.name for tool in self.tools],
            'tavily_configured': any("tavily" in tool.name.lower() for tool in self.tools),
            'connection_ok': self.test_connection(),
            'web_search_ok': self.test_web_search()
        }