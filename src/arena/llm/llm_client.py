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
    
    async def _process_tool_calls(self, response, messages: List) -> str:
        """Process tool calls in LLM response, based on talks project approach."""
        if not hasattr(response, 'tool_calls') or not response.tool_calls:
            return strip_reasoning(response.content)
        
        # Track tool usage
        self._last_tool_calls = response.tool_calls
        self._tools_used_this_turn = True
        
        # Process each tool call
        tool_results = []
        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            
            # Find and execute the tool
            for tool in self.tools:
                if tool.name == tool_name:
                    try:
                        result = await tool.ainvoke(tool_args)
                        tool_results.append(result)
                        logger.info(f"🔍 Arena agent used {tool_name}: {tool_args.get('query', 'N/A')}")
                    except Exception as e:
                        logger.error(f"Tool execution failed: {e}")
                        tool_results.append(f"Tool error: {str(e)}")
        
        # If we have tool results, generate final response incorporating them
        if tool_results:
            # Create a comprehensive prompt that includes the tool results
            tool_info = "\n".join([f"Search result: {result}" for result in tool_results])
            
            # Create new prompt that explicitly asks for a response incorporating the tool results
            enhanced_prompt = f"""Based on the following research information, please provide a comprehensive answer:

{tool_info}

Original question: {messages[-1].content if messages else ''}

Please provide a detailed, factual response incorporating the above information."""
            
            # Create fresh conversation with the enhanced prompt using simple LLM (no tools)
            final_messages = [
                messages[0] if messages else SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content=enhanced_prompt)
            ]
            
            final_response = await self.simple_llm.ainvoke(final_messages)
            final_content = strip_reasoning(final_response.content)
            
            # If still empty, try a more direct approach
            if not final_content or len(final_content.strip()) < 10:
                logger.warning("Enhanced response still empty, trying direct synthesis")
                
                # Create a very direct synthesis prompt
                synthesis_prompt = f"Please summarize and answer based on this information: {tool_info[:1000]}"
                synthesis_messages = [
                    SystemMessage(content="You are a helpful assistant that provides clear, factual summaries."),
                    HumanMessage(content=synthesis_prompt)
                ]
                
                synthesis_response = await self.simple_llm.ainvoke(synthesis_messages)
                final_content = strip_reasoning(synthesis_response.content)
            
            logger.debug(f"Tool-enhanced response: {len(final_content)} chars")
            return final_content
        
        return strip_reasoning(response.content)
    
    async def generate_character_response(
        self,
        character_id: str,
        character_name: str,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a character-appropriate response using tool calling."""
        
        # Reset tool usage tracking
        self._last_tool_calls = []
        self._tools_used_this_turn = False
        
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
            
            # Generate response with potential tool calls
            response = await self.llm.ainvoke(messages)
            
            # Process any tool calls
            content = await self._process_tool_calls(response, messages)
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