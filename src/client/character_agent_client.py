"""Character Agent WebSocket Client.

This client connects to the Character Agent Server and provides the same interface
as the original CharacterAgent class, but communicates over WebSocket.
"""

import asyncio
import json
import logging
import os
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
import websockets
from websockets.client import WebSocketClientProtocol

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, skip loading

try:
    from ..protocol.messages import (
        BaseMessage, InitRequest, InitResponse, ChatRequest, ChatResponse,
        StatusRequest, StatusResponse, SaveRequest, SaveResponse,
        LoadRequest, LoadResponse, ResetRequest, ResetResponse,
        MemorySearchRequest, MemorySearchResponse, PingRequest, PingResponse,
        ErrorResponse, create_init_request, create_chat_request
    )
    from ..config.logging_config import get_logger
except ImportError:
    # Fallback imports for when running from different contexts
    from protocol.messages import (
        BaseMessage, InitRequest, InitResponse, ChatRequest, ChatResponse,
        StatusRequest, StatusResponse, SaveRequest, SaveResponse,
        LoadRequest, LoadResponse, ResetRequest, ResetResponse,
        MemorySearchRequest, MemorySearchResponse, PingRequest, PingResponse,
        ErrorResponse, create_init_request, create_chat_request
    )
    from config.logging_config import get_logger


class CharacterAgentClientError(Exception):
    """Exception raised by character agent client."""
    pass


class CharacterAgentClient:
    """WebSocket client for communicating with Character Agent Server.
    
    This class provides the same interface as CharacterAgent but communicates
    with a remote server instead of running the agent locally.
    """
    
    def __init__(
        self,
        server_url: str = "ws://localhost:8765",
        timeout: float = None,
        reconnect_attempts: int = 3,
        reconnect_delay: float = 1.0
    ):
        """Initialize the character agent client.
        
        Args:
            server_url: WebSocket URL of the character agent server
            timeout: Timeout for requests in seconds
            reconnect_attempts: Number of reconnection attempts
            reconnect_delay: Delay between reconnection attempts
        """
        self.server_url = server_url
        # Use AGENT_TIMEOUT from .env if available, otherwise default to 30.0
        if timeout is None:
            timeout = float(os.getenv('AGENT_TIMEOUT', '30.0'))
        self.timeout = timeout
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        
        self.websocket: Optional[WebSocketClientProtocol] = None
        self.is_connected = False
        self.character_id: Optional[str] = None
        self.character_name: Optional[str] = None
        self.character_config: Optional[Dict[str, Any]] = None
        
        # Response tracking
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self.receive_task: Optional[asyncio.Task] = None
        
        self.logger = get_logger(__name__)
        self.logger.info(f"CharacterAgentClient initialized for {server_url}")
    
    async def connect(self) -> bool:
        """Connect to the character agent server.
        
        Returns:
            True if connected successfully
        """
        if self.is_connected:
            return True
        
        for attempt in range(self.reconnect_attempts):
            try:
                self.logger.info(f"Connecting to server {self.server_url} (attempt {attempt + 1})")
                
                # Configure ping settings based on DEBUG environment variable
                is_debug = os.getenv('DEBUG', '').lower() in ('true', '1', 'yes')
                if is_debug:
                    # Disable ping when server is in debug mode
                    ping_interval = None
                    ping_timeout = None
                    self.logger.info("DEBUG mode detected - WebSocket ping disabled on client")
                else:
                    # Use standard ping settings
                    ping_interval = 30
                    ping_timeout = 10
                    self.logger.info(f"WebSocket ping configured: interval={ping_interval}s, timeout={ping_timeout}s")
                
                self.websocket = await websockets.connect(
                    self.server_url,
                    ping_interval=ping_interval,
                    ping_timeout=ping_timeout,
                    close_timeout=10
                )
                
                self.is_connected = True
                
                # Start message receiver task
                self.receive_task = asyncio.create_task(self._receive_messages())
                
                self.logger.info("Successfully connected to character agent server")
                return True
                
            except Exception as e:
                self.logger.error(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < self.reconnect_attempts - 1:
                    await asyncio.sleep(self.reconnect_delay)
                else:
                    self.logger.error("Failed to connect after all attempts")
        
        return False
    
    async def disconnect(self) -> None:
        """Disconnect from the character agent server."""
        if not self.is_connected:
            return
        
        self.logger.info("Disconnecting from character agent server")
        
        # Cancel receive task
        if self.receive_task:
            self.receive_task.cancel()
            try:
                await self.receive_task
            except asyncio.CancelledError:
                pass
        
        # Close WebSocket connection
        if self.websocket:
            await self.websocket.close()
        
        # Cancel any pending requests
        for future in self.pending_requests.values():
            if not future.done():
                future.cancel()
        self.pending_requests.clear()
        
        self.is_connected = False
        self.websocket = None
        self.logger.info("Disconnected from character agent server")
    
    async def _receive_messages(self) -> None:
        """Receive and process messages from the server."""
        try:
            while self.is_connected and self.websocket:
                try:
                    # Adjust timeout based on DEBUG mode
                    is_debug = os.getenv('DEBUG', '').lower() in ('true', '1', 'yes')
                    recv_timeout = None if is_debug else 25.0
                    
                    if recv_timeout:
                        message = await asyncio.wait_for(self.websocket.recv(), timeout=recv_timeout)
                    else:
                        message = await self.websocket.recv()
                    response = BaseMessage.from_json(message)
                    
                    # Handle response
                    if response.message_id and response.message_id in self.pending_requests:
                        future = self.pending_requests.pop(response.message_id)
                        if not future.done():
                            future.set_result(response)
                    else:
                        self.logger.warning(f"Received unexpected message: {response.message_type}")
                        
                except asyncio.TimeoutError:
                    # Only do ping check if not in debug mode (debug mode has no timeout)
                    is_debug = os.getenv('DEBUG', '').lower() in ('true', '1', 'yes')
                    if not is_debug:
                        # Check if connection is still alive with a ping
                        try:
                            await self.websocket.ping()
                            self.logger.debug("Connection alive (ping successful)")
                        except Exception:
                            self.logger.warning("Connection appears dead (ping failed)")
                            break
                        
                except websockets.exceptions.ConnectionClosed:
                    self.logger.info("Server connection closed")
                    break
                except json.JSONDecodeError as e:
                    self.logger.error(f"Invalid JSON received from server: {e}")
                except Exception as e:
                    self.logger.error(f"Error receiving message: {e}")
                    
        except asyncio.CancelledError:
            self.logger.debug("Message receiver task cancelled")
        except Exception as e:
            self.logger.error(f"Error in message receiver: {e}")
        finally:
            self.is_connected = False
            # Cancel any remaining pending requests
            for future in self.pending_requests.values():
                if not future.done():
                    future.cancel()
    
    async def _send_request(self, request: BaseMessage) -> BaseMessage:
        """Send a request and wait for response.
        
        Args:
            request: Request message to send
            
        Returns:
            Response message
            
        Raises:
            CharacterAgentClientError: If request fails
        """
        # Ensure we're connected before sending
        if not await self.ensure_connected():
            raise CharacterAgentClientError("Unable to connect to server")
        
        # Generate message ID if not set
        if not request.message_id:
            request.message_id = str(uuid.uuid4())
        
        # Create future for response
        future = asyncio.Future()
        self.pending_requests[request.message_id] = future
        
        try:
            # Send request
            await self.websocket.send(request.to_json())
            
            # Wait for response
            response = await asyncio.wait_for(future, timeout=self.timeout)
            
            # Check for error response
            if isinstance(response, ErrorResponse):
                raise CharacterAgentClientError(response.error)
            
            return response
            
        except asyncio.TimeoutError:
            self.pending_requests.pop(request.message_id, None)
            raise CharacterAgentClientError(f"Request timed out after {self.timeout} seconds")
        except websockets.exceptions.ConnectionClosed:
            self.pending_requests.pop(request.message_id, None)
            self.is_connected = False
            raise CharacterAgentClientError("Connection to server lost")
        except Exception as e:
            self.pending_requests.pop(request.message_id, None)
            raise CharacterAgentClientError(f"Request failed: {e}")
    
    async def initialize_character(
        self,
        character_id: str,
        character_config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Initialize a character on the server.
        
        Args:
            character_id: ID of character to initialize
            character_config: Optional character configuration override
            
        Returns:
            True if character initialized successfully
        """
        try:
            request = create_init_request(character_id, character_config)
            response = await self._send_request(request)
            
            if isinstance(response, InitResponse) and response.success:
                self.character_id = character_id
                self.character_name = response.character_name
                self.character_config = response.character_info
                self.logger.info(f"Character '{character_id}' initialized successfully")
                return True
            else:
                error = response.error if hasattr(response, 'error') else "Unknown error"
                self.logger.error(f"Failed to initialize character: {error}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error initializing character: {e}")
            return False
    
    async def process_message(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a user message and get character response.
        
        Args:
            user_message: User's input message
            context: Optional additional context
            
        Returns:
            Character response data
            
        Raises:
            CharacterAgentClientError: If processing fails
        """
        request = create_chat_request(user_message, context)
        response = await self._send_request(request)
        
        if isinstance(response, ChatResponse) and response.success:
            return {
                'response_text': response.response_text,
                'character_insights': response.character_insights,
                'response_metadata': response.response_metadata,
                'generation_info': response.generation_info,
                'orchestration_summary': response.orchestration_summary,
                'cognitive_summary': response.cognitive_summary,
                'character_state_summary': response.character_state_summary,
                'performance_info': response.performance_info,
                'timestamp': response.timestamp.isoformat()
            }
        else:
            error = response.error if hasattr(response, 'error') else "Unknown error"
            raise CharacterAgentClientError(f"Message processing failed: {error}")
    
    async def get_character_summary(self) -> Dict[str, Any]:
        """Get character status and summary.
        
        Returns:
            Character summary data
        """
        request = StatusRequest(include_vitals=True, include_history=False, include_performance=True)
        response = await self._send_request(request)
        
        if isinstance(response, StatusResponse) and response.success:
            return response.character_summary or {}
        else:
            error = response.error if hasattr(response, 'error') else "Unknown error"
            raise CharacterAgentClientError(f"Failed to get character summary: {error}")
    
    async def get_character_vitals(self) -> Dict[str, Any]:
        """Get character vitals (neurochemical state, mood, etc.).
        
        Returns:
            Character vitals data
        """
        request = StatusRequest(include_vitals=True, include_history=False, include_performance=False)
        response = await self._send_request(request)
        
        if isinstance(response, StatusResponse) and response.success:
            return response.vitals or {}
        else:
            error = response.error if hasattr(response, 'error') else "Unknown error"
            raise CharacterAgentClientError(f"Failed to get character vitals: {error}")
    
    async def get_conversation_history(self, limit: int = 10) -> List[Dict[str, str]]:
        """Get recent conversation history.
        
        Args:
            limit: Maximum number of messages to return
            
        Returns:
            List of conversation messages
        """
        request = StatusRequest(include_vitals=False, include_history=True, include_performance=False)
        response = await self._send_request(request)
        
        if isinstance(response, StatusResponse) and response.success:
            history = response.conversation_history or []
            return history[-limit:] if limit > 0 else history
        else:
            error = response.error if hasattr(response, 'error') else "Unknown error"
            raise CharacterAgentClientError(f"Failed to get conversation history: {error}")
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics.
        
        Returns:
            Performance statistics
        """
        request = StatusRequest(include_vitals=False, include_history=False, include_performance=True)
        response = await self._send_request(request)
        
        if isinstance(response, StatusResponse) and response.success:
            return response.performance_stats or {}
        else:
            error = response.error if hasattr(response, 'error') else "Unknown error"
            raise CharacterAgentClientError(f"Failed to get performance stats: {error}")
    
    async def save_character_state(self, filename: Optional[str] = None) -> str:
        """Save character state to file.
        
        Args:
            filename: Optional filename for save file
            
        Returns:
            Path to saved file
        """
        request = SaveRequest(filename=filename)
        response = await self._send_request(request)
        
        if isinstance(response, SaveResponse) and response.success:
            return response.filename
        else:
            error = response.error if hasattr(response, 'error') else "Unknown error"
            raise CharacterAgentClientError(f"Failed to save character state: {error}")
    
    async def load_character_state(self, filename: str) -> Dict[str, Any]:
        """Load character state from file.
        
        Args:
            filename: Path to save file
            
        Returns:
            Character info after loading
        """
        request = LoadRequest(filename=filename)
        response = await self._send_request(request)
        
        if isinstance(response, LoadResponse) and response.success:
            # Update local character info
            if response.character_info:
                self.character_id = response.character_info.get('character_id')
                self.character_name = response.character_info.get('name')
                self.character_config = response.character_info
            
            return response.character_info or {}
        else:
            error = response.error if hasattr(response, 'error') else "Unknown error"
            raise CharacterAgentClientError(f"Failed to load character state: {error}")
    
    async def reset_character_state(self, preserve_memories: bool = True) -> Dict[str, Any]:
        """Reset character state.
        
        Args:
            preserve_memories: Whether to preserve conversation memories
            
        Returns:
            Character info after reset
        """
        request = ResetRequest(preserve_memories=preserve_memories)
        response = await self._send_request(request)
        
        if isinstance(response, ResetResponse) and response.success:
            return response.character_info or {}
        else:
            error = response.error if hasattr(response, 'error') else "Unknown error"
            raise CharacterAgentClientError(f"Failed to reset character state: {error}")
    
    async def recall_past_conversations(self, query: str = "", limit: int = 5) -> List[Dict[str, Any]]:
        """Search character memories.
        
        Args:
            query: Search query
            limit: Maximum number of memories to return
            
        Returns:
            List of relevant memories
        """
        request = MemorySearchRequest(query=query, limit=limit)
        response = await self._send_request(request)
        
        if isinstance(response, MemorySearchResponse) and response.success:
            return response.memories or []
        else:
            error = response.error if hasattr(response, 'error') else "Unknown error"
            raise CharacterAgentClientError(f"Failed to search memories: {error}")
    
    async def ping_server(self) -> bool:
        """Ping the server to check connectivity.
        
        Returns:
            True if server responds to ping
        """
        try:
            request = PingRequest()
            response = await self._send_request(request)
            return isinstance(response, PingResponse) and response.success
        except Exception as e:
            self.logger.error(f"Ping failed: {e}")
            return False
    
    async def reconnect(self) -> bool:
        """Attempt to reconnect to the server.
        
        Returns:
            True if reconnection successful
        """
        if self.is_connected:
            return True
        
        self.logger.info("Attempting to reconnect to server...")
        
        # First disconnect cleanly
        await self.disconnect()
        
        # Wait a moment before reconnecting
        await asyncio.sleep(self.reconnect_delay)
        
        # Attempt to connect
        return await self.connect()
    
    async def ensure_connected(self) -> bool:
        """Ensure we're connected to the server, reconnecting if necessary.
        
        Returns:
            True if connected
        """
        if self.is_connected:
            # Test connection with a simple websocket ping instead of ping_server to avoid recursion
            try:
                if self.websocket:
                    await self.websocket.ping()
                    return True
            except Exception as e:
                self.logger.warning(f"Connection test failed: {e}, attempting reconnection")
                self.is_connected = False
        
        # Try to reconnect
        return await self.reconnect()
    
    @property
    def state(self):
        """Get a state-like object for compatibility.
        
        Note: This is a simplified implementation that fetches vitals on demand.
        For full state access, use get_character_vitals() or get_character_summary().
        """
        return RemoteCharacterState(self)
    
    def close(self):
        """Close the client connection."""
        if self.is_connected:
            asyncio.create_task(self.disconnect())


class RemoteCharacterState:
    """Wrapper to provide state-like interface for remote character."""
    
    def __init__(self, client: CharacterAgentClient):
        self.client = client
        self._cached_vitals: Optional[Dict[str, Any]] = None
        self._cache_time: Optional[datetime] = None
        self._cache_duration = 5.0  # Cache for 5 seconds
    
    async def _get_vitals(self) -> Dict[str, Any]:
        """Get vitals with caching."""
        now = datetime.now()
        
        if (self._cached_vitals is None or 
            self._cache_time is None or 
            (now - self._cache_time).total_seconds() > self._cache_duration):
            
            try:
                self._cached_vitals = await self.client.get_character_vitals()
                self._cache_time = now
            except Exception:
                # Return empty dict if fetch fails
                self._cached_vitals = {}
        
        return self._cached_vitals or {}
    
    @property
    def neurochemical_levels(self) -> Dict[str, float]:
        """Get neurochemical levels (synchronous property)."""
        # This is a simplified implementation for compatibility
        # In practice, you should use async methods
        return {}
    
    @property
    def agent_states(self) -> Dict[str, Any]:
        """Get agent states (synchronous property)."""
        # This is a simplified implementation for compatibility
        return {}
    
    async def get_neurochemical_levels(self) -> Dict[str, float]:
        """Get neurochemical levels (async version)."""
        vitals = await self._get_vitals()
        return vitals.get('neurochemical_state', {})
    
    async def get_agent_states(self) -> Dict[str, Any]:
        """Get agent states (async version)."""
        vitals = await self._get_vitals()
        return vitals.get('agent_states', {})