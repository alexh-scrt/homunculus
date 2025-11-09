"""
LLM Client for Arena

This module provides a unified interface for different LLM providers,
optimized for Arena character agents. Based on the proven approach from
the talks project.

Features:
- LangChain-based Ollama integration
- Character-aware prompting
- Async and sync support
- Response cleaning and formatting
- Simple and reliable architecture

Author: Homunculus Team
"""

import os
import logging
import re
import json
import sys
import time
from typing import Dict, Any, List, Optional, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime

# LangChain imports
try:
    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.tools import BaseTool
    from langchain_tavily import TavilySearch as TavilySearchResults
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    ChatOllama = None
    HumanMessage = None
    SystemMessage = None
    BaseTool = None
    TavilySearchResults = None

# Arena settings
try:
    from ..config.arena_settings import settings
    from .dynamic_parameters import get_llm_parameters
except ImportError:
    settings = None

logger = logging.getLogger(__name__)


def strip_reasoning(text: str) -> str:
    """
    Remove reasoning blocks from LLM response text.
    
    Strips content between <think> and </think> tags (case-insensitive).
    """
    if not text:
        return text
    
    # Pattern to match <think>...</think> blocks (case-insensitive, with optional whitespace)
    pattern = r'<think\s*>.*?</think\s*>'
    
    # Remove the reasoning blocks and clean up extra whitespace
    cleaned = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'\n\s*\n+', '\n', cleaned)  # Multiple newlines to single
    cleaned = re.sub(r'  +', ' ', cleaned)        # Multiple spaces to single
    
    return cleaned.strip()


class ArenaLLMClient:
    """
    Simple LLM client for Arena agents, based on the proven talks project approach.
    
    Uses LangChain ChatOllama for reliable local LLM integration.
    """
    
    def __init__(self, model: Optional[str] = None, temperature: float = 0.7, **llm_params):
        """Initialize LLM client
        
        Args:
            model: LLM model name
            temperature: Generation temperature
            **llm_params: Additional parameters for ChatOllama
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain not available. Install with: pip install langchain langchain-ollama")
        
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.3:70b")
        self.temperature = temperature
        
        # Streaming configuration
        self.streaming_enabled = os.getenv("ENABLE_STREAMING_RESPONSE", "false").lower() == "true"
        
        # Web search configuration
        self.web_search_enabled = settings.is_web_search_enabled if settings else False
        
        # Setup LLM parameters like talks project
        default_params = {
            "model": self.model,
            "temperature": temperature
        }
        if llm_params:
            default_params.update(llm_params)
        
        try:
            self.llm = ChatOllama(**default_params)
            # Create a separate LLM instance without tools for final responses
            self.simple_llm = ChatOllama(**default_params)
            logger.info(f"Initialized Ollama client with model: {self.model} (streaming: {self.streaming_enabled})")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {e}")
            raise
        
        # Setup tools and bind to LLM (but not to simple_llm)
        self.tools = self._setup_tools()
        if self.tools:
            self.llm = self.llm.bind_tools(self.tools)
            logger.info(f"✅ Bound {len(self.tools)} tools to LLM")
        
        # Track tool usage like talks project
        self._last_tool_calls = []
        self._tools_used_this_turn = False
        
        # Character-specific prompting
        self.character_prompts = {
            "ada_lovelace": {
                "personality": "analytical, methodical, mathematically-minded, precise, pioneering",
                "speaking_style": "logical, evidence-based, often references calculations, patterns, or algorithms",
                "expertise": "mathematics, analytical engines, algorithmic thinking, systematic approaches",
                "background": "English mathematician and writer, known for work on Charles Babbage's Analytical Engine"
            },
            "captain_cosmos": {
                "personality": "adventurous, optimistic, cosmic perspective, bold, inspiring",
                "speaking_style": "inspirational, references space/cosmic themes, encouraging, grand vision",
                "expertise": "space exploration, leadership, strategic thinking, unconventional approaches",
                "background": "Intergalactic explorer and leader, experienced in navigating unknown territories"
            }
        }
        
        # Conversation history
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history = 20
    
    def get_dynamic_llm_params(self, agent_type: str, agent_name: str = "", game_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get dynamic LLM parameters based on agent type and game context"""
        try:
            return get_llm_parameters(agent_type, agent_name, game_context)
        except NameError:
            # Fallback if dynamic parameters not available
            return {
                "temperature": self.temperature,
                "max_tokens": 1000
            }
    
    def _setup_tools(self) -> List[BaseTool]:
        """Setup tools for the agent, based on talks project approach."""
        tools = []
        
        if not self.web_search_enabled:
            return tools
        
        # Add Tavily web search tool
        tavily_api_key = settings.tavily_api_key if settings else os.getenv("TAVILY_API_KEY")
        if tavily_api_key:
            try:
                # Set environment variable for Tavily
                os.environ['TAVILY_API_KEY'] = tavily_api_key
                
                tavily_tool = TavilySearchResults(
                    api_key=tavily_api_key,
                    max_results=settings.web_search_max_results if settings else 5,
                    include_answer=True,
                    include_raw_content=False,
                    include_images=False,
                    search_depth="basic",
                    include_domains=[],
                    exclude_domains=[],
                )
                tools.append(tavily_tool)
                logger.info("✅ Tavily search tool added to Arena LLM client")
            except Exception as e:
                logger.error(f"❌ Failed to setup Tavily for Arena LLM client: {e}")
        else:
            logger.warning("⚠️ TAVILY_API_KEY not set - web search disabled for Arena agents")
        
        return tools
    
    async def _process_tool_calls_loop(self, response, messages: List, llm_instance=None, max_iterations: int = 3) -> str:
        """
        Process tool calls in a loop as suggested by user.
        
        Implementation of: 
        while:
           llm call
           check for tool calls
              execute tool calls 
              add ToolMessage
        """
        if llm_instance is None:
            llm_instance = self.llm
        
        # Initialize conversation with original messages
        current_messages = messages.copy()
        current_response = response
        iterations = 0
        
        while iterations < max_iterations:
            iterations += 1
            logger.debug(f"Tool call processing iteration {iterations}")
            
            # Check for tool calls in current response
            if not hasattr(current_response, 'tool_calls') or not current_response.tool_calls:
                # No more tool calls - return the content
                return strip_reasoning(current_response.content)
            
            # Track tool usage
            self._last_tool_calls = current_response.tool_calls
            self._tools_used_this_turn = True
            logger.info(f"Processing {len(current_response.tool_calls)} tool calls in iteration {iterations}")
            
            # Execute each tool call and add ToolMessage
            for tool_call in current_response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                tool_call_id = tool_call.get('id', f"call_{iterations}_{tool_name}")
                
                # Find and execute the tool
                tool_result = None
                for tool in self.tools:
                    if tool.name == tool_name:
                        try:
                            tool_result = await tool.ainvoke(tool_args)
                            logger.info(f"🔍 Executed {tool_name}: {tool_args.get('query', 'N/A')}")
                            break
                        except Exception as e:
                            tool_result = f"Tool error: {str(e)}"
                            logger.error(f"Tool execution failed: {e}")
                            break
                
                if tool_result is None:
                    tool_result = f"Tool {tool_name} not found"
                    logger.warning(f"Tool {tool_name} not found in available tools")
                
                # Add ToolMessage to conversation (LangChain format)
                from langchain_core.messages import ToolMessage
                tool_message = ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call_id,
                    name=tool_name
                )
                current_messages.append(tool_message)
            
            # Make another LLM call with the updated conversation including tool results
            try:
                current_response = await llm_instance.ainvoke(current_messages)
                logger.debug(f"LLM call iteration {iterations} completed")
            except Exception as e:
                logger.error(f"LLM call failed in iteration {iterations}: {e}")
                # Return best response we have so far
                return strip_reasoning(current_response.content) if hasattr(current_response, 'content') else "Error processing response"
        
        # Max iterations reached - return final response
        logger.warning(f"Max tool call iterations ({max_iterations}) reached")
        return strip_reasoning(current_response.content)
    
    async def _process_tool_calls(self, response, messages: List, llm_instance=None) -> str:
        """Legacy method that now calls the enhanced loop version"""
        return await self._process_tool_calls_loop(response, messages, llm_instance)
    
    async def generate_character_response(
        self,
        character_id: str,
        character_name: str,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a character-appropriate response using tool calling with dynamic parameters."""
        
        # Reset tool usage tracking
        self._last_tool_calls = []
        self._tools_used_this_turn = False
        
        # Get dynamic LLM parameters based on context
        dynamic_params = self.get_dynamic_llm_params("character", character_name, context)
        
        # Create a new LLM instance with dynamic parameters if they differ significantly
        current_temp = getattr(self.llm, 'temperature', self.temperature)
        new_temp = dynamic_params.get('temperature', current_temp)
        
        # Check if any parameters differ significantly to warrant new LLM instance
        needs_new_llm = (
            abs(new_temp - current_temp) > 0.05 or  # Temperature difference
            dynamic_params.get('repeat_penalty') is not None or  # Has Ollama-specific params
            dynamic_params.get('top_p') is not None or
            dynamic_params.get('top_k') is not None
        )
        
        if needs_new_llm:
            # Create temporary LLM with dynamic parameters including Ollama-specific ones
            temp_llm_params = {
                "model": self.model,
                "temperature": new_temp
            }
            
            # Add Ollama-specific parameters if they exist
            for ollama_param in ['repeat_penalty', 'top_p', 'top_k', 'num_predict']:
                if ollama_param in dynamic_params and dynamic_params[ollama_param] is not None:
                    temp_llm_params[ollama_param] = dynamic_params[ollama_param]
            
            try:
                temp_llm = ChatOllama(**temp_llm_params)
                if self.tools:
                    temp_llm = temp_llm.bind_tools(self.tools)
                temp_simple_llm = ChatOllama(**temp_llm_params)
                
                param_desc = f"temp={new_temp}"
                if dynamic_params.get('repeat_penalty'):
                    param_desc += f", repeat_penalty={dynamic_params['repeat_penalty']}"
                if dynamic_params.get('top_p'):
                    param_desc += f", top_p={dynamic_params['top_p']}"
                logger.debug(f"Using dynamic parameters for {character_name}: {param_desc}")
                
            except Exception as e:
                logger.warning(f"Failed to create dynamic LLM: {e}, using default")
                temp_llm = self.llm
                temp_simple_llm = self.simple_llm
        else:
            temp_llm = self.llm
            temp_simple_llm = self.simple_llm
        
        # Build character-specific system prompt
        character_info = self.character_prompts.get(
            character_id.lower(),
            {
                "personality": "thoughtful, articulate, collaborative",
                "speaking_style": "clear and direct",
                "expertise": "problem-solving and strategic thinking",
                "background": "A knowledgeable participant in discussions"
            }
        )
        
        # Enhanced system prompt for tool usage
        tool_instructions = ""
        if self.tools:
            tool_instructions = """
You have access to web search tools. Use them when you need current information, recent developments, or real-time data to provide accurate responses. For example:
- For questions about "latest", "current", "recent" developments
- For real-time data like weather, stock prices, news
- For up-to-date research findings
- For current events and breaking news
Don't mention using tools directly - just incorporate the information naturally into your response."""

        system_prompt = f"""You are {character_name}, a character in an Arena discussion game.

Character Profile:
- Background: {character_info['background']}
- Personality: {character_info['personality']}
- Speaking style: {character_info['speaking_style']}
- Expertise: {character_info['expertise']}

Guidelines:
- Stay completely in character at all times
- Draw on your character's background and expertise
- Use your characteristic speaking style
- Provide substantive, thoughtful contributions
- Keep responses conversational and natural (2-4 sentences)
- Never break character or mention being an AI
- Be authentic to your character's unique perspective{tool_instructions}

Current Context: {json.dumps(context) if context else 'No additional context'}"""

        try:
            # Prepare messages
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt)
            ]
            
            # Generate response with potential tool calls using dynamic parameters
            response = await temp_llm.ainvoke(messages)
            
            # Process any tool calls (update _process_tool_calls to use temp_simple_llm)
            content = await self._process_tool_calls(response, messages, temp_simple_llm)
            content = content.strip()
            
            # Track conversation
            self.add_to_history("user", prompt)
            self.add_to_history("assistant", content)
            
            # Log to arena logger if not streaming
            if not self.streaming_enabled:
                from ..config.logging_config import get_arena_logger
                arena_logger = get_arena_logger()
                if arena_logger:
                    arena_logger.log_agent_response(character_name, content, character_id)
            
            tools_used = "yes" if self._tools_used_this_turn else "no"
            logger.info(f"Generated response for {character_id}: {len(content)} chars (tools_used: {tools_used})")
            return content
            
        except Exception as e:
            logger.error(f"LLM generation failed for {character_id}: {e}")
            # Return character-appropriate fallback
            if character_id.lower() == "ada_lovelace":
                return "I need to analyze this more systematically before providing a proper response."
            elif character_id.lower() == "captain_cosmos":
                return "Let me chart a course through these cosmic complexities before responding."
            else:
                return "I need a moment to gather my thoughts on this matter."
    
    def generate_character_response_streaming(
        self,
        character_id: str,
        character_name: str,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """Generate a character-appropriate response with streaming output.
        
        Note: Tool calling is disabled during streaming for simplicity.
        For topics requiring web search, use non-streaming mode.
        """
        
        # Build character-specific system prompt (without tools for streaming)
        character_info = self.character_prompts.get(
            character_id.lower(),
            {
                "personality": "thoughtful, articulate, collaborative",
                "speaking_style": "clear and direct",
                "expertise": "problem-solving and strategic thinking",
                "background": "A knowledgeable participant in discussions"
            }
        )
        
        system_prompt = f"""You are {character_name}, a character in an Arena discussion game.

Character Profile:
- Background: {character_info['background']}
- Personality: {character_info['personality']}
- Speaking style: {character_info['speaking_style']}
- Expertise: {character_info['expertise']}

Guidelines:
- Stay completely in character at all times
- Draw on your character's background and expertise
- Use your characteristic speaking style
- Provide substantive, thoughtful contributions
- Keep responses conversational and natural (2-4 sentences)
- Never break character or mention being an AI
- Be authentic to your character's unique perspective

Current Context: {json.dumps(context) if context else 'No additional context'}"""

        try:
            # Use LLM without tools for streaming (to avoid tool call complexity)
            streaming_llm = ChatOllama(model=self.model, temperature=self.temperature)
            
            # Prepare messages
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt)
            ]
            
            # Stream response
            accumulated_response = ""
            previous_token = ""
            for chunk in streaming_llm.stream(messages):
                content = chunk.content
                if content:
                    # Clean content
                    clean_content = strip_reasoning(content)
                    if clean_content:
                        # Handle spacing for better readability
                        # Add space if previous token ended with a letter/number and current starts with a letter
                        if (previous_token and 
                            len(previous_token) > 0 and 
                            len(clean_content) > 0 and
                            previous_token[-1].isalnum() and 
                            clean_content[0].isalnum() and
                            not clean_content.startswith(' ')):
                            clean_content = ' ' + clean_content
                        
                        # Add space after punctuation if next token starts with a letter
                        elif (previous_token and 
                              len(previous_token) > 0 and 
                              len(clean_content) > 0 and
                              previous_token[-1] in '.!?,:;' and 
                              clean_content[0].isalpha() and
                              not clean_content.startswith(' ')):
                            clean_content = ' ' + clean_content
                        
                        accumulated_response += clean_content
                        previous_token = clean_content
                        yield clean_content
            
            # Track conversation
            self.add_to_history("user", prompt)
            self.add_to_history("assistant", accumulated_response)
            
            logger.info(f"Generated streaming response for {character_id}: {len(accumulated_response)} chars (tools: disabled during streaming)")
            
        except Exception as e:
            logger.error(f"LLM streaming generation failed for {character_id}: {e}")
            # Return character-appropriate fallback
            if character_id.lower() == "ada_lovelace":
                fallback = "I need to analyze this more systematically before providing a proper response."
            elif character_id.lower() == "captain_cosmos":
                fallback = "Let me chart a course through these cosmic complexities before responding."
            else:
                fallback = "I need a moment to gather my thoughts on this matter."
            yield fallback
    
    def complete_sync(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Synchronous completion for non-async contexts"""
        try:
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))
            
            response = self.llm.invoke(messages)
            content = response.content
            
            # Clean response
            content = strip_reasoning(content)
            return content.strip()
            
        except Exception as e:
            logger.error(f"Sync LLM completion failed: {e}")
            return "I'm unable to respond at this moment."
    
    def add_to_history(self, role: str, content: str) -> None:
        """Add entry to conversation history."""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Trim history if too long
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "conversation_entries": len(self.conversation_history),
            "provider": "ollama"
        }