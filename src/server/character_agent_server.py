"""Character Agent WebSocket Server.

This server hosts character agents and provides WebSocket interface for chat clients.
Multiple clients can connect to chat with the same character instance.
"""

import asyncio
import json
import logging
import os
import signal
import sys
import traceback
import uuid
from pathlib import Path
from typing import Dict, Set, Optional, Any
import websockets
from websockets.server import WebSocketServerProtocol

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from character_agent import CharacterAgent
    from config.character_loader import CharacterLoader, CharacterConfigurationError
    from config.settings import get_settings
    from config.logging_config import setup_logging, get_logger
    from protocol.messages import (
        BaseMessage, InitRequest, InitResponse, ChatRequest, ChatResponse,
        StatusRequest, StatusResponse, SaveRequest, SaveResponse,
        LoadRequest, LoadResponse, ResetRequest, ResetResponse,
        MemorySearchRequest, MemorySearchResponse, PingRequest, PingResponse,
        ErrorResponse, create_error_response
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running from the project root and dependencies are installed.")
    sys.exit(1)


class CharacterAgentServer:
    """WebSocket server for character agents."""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        """Initialize the character agent server.
        
        Args:
            host: Host address to bind to
            port: Port to listen on
        """
        self.host = host
        self.port = port
        self.character_agent: Optional[CharacterAgent] = None
        self.character_id: Optional[str] = None
        self.connected_clients: Set[WebSocketServerProtocol] = set()
        self.character_loader = CharacterLoader()
        self.settings = get_settings()
        self.save_dir = Path("./data/saves")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        setup_logging()
        self.logger = get_logger(__name__)
        
        # Server state
        self.is_running = False
        self.server = None
        
        self.logger.info(f"CharacterAgentServer initialized on {host}:{port}")
    
    async def start_server(self, character_id: Optional[str] = None) -> None:
        """Start the WebSocket server.
        
        Args:
            character_id: Optional character to pre-load
        """
        if self.is_running:
            self.logger.warning("Server is already running")
            return
        
        try:
            # Pre-load character if specified
            if character_id:
                success = await self.load_character(character_id)
                if not success:
                    self.logger.error(f"Failed to load character '{character_id}' on startup")
                    return
                else:
                    self.logger.info(f"Pre-loaded character '{character_id}'")
            
            # Start WebSocket server
            self.logger.info(f"Starting Character Agent Server on {self.host}:{self.port}")
            
            # Configure ping settings based on DEBUG environment variable
            debug_mode = os.getenv('DEBUG', '').lower() in ('true', '1', 'yes')
            if debug_mode:
                # Disable ping when in debug mode
                ping_interval = None
                ping_timeout = None
                self.logger.info("DEBUG mode detected - WebSocket ping disabled")
            else:
                # Use 600 second ping interval in production
                ping_interval = 600
                ping_timeout = 60
                self.logger.info(f"WebSocket ping configured: interval={ping_interval}s, timeout={ping_timeout}s")
            
            self.server = await websockets.serve(
                self.handle_client,
                self.host,
                self.port,
                ping_interval=ping_interval,
                ping_timeout=ping_timeout,
                close_timeout=10   # Wait 10 seconds for close
            )
            
            self.is_running = True
            self.logger.info(f"Character Agent Server started successfully")
            
            if character_id:
                print(f"🤖 Character Agent Server running on ws://{self.host}:{self.port}")
                print(f"📝 Character '{character_id}' loaded and ready")
                print(f"🔌 Waiting for client connections...")
            else:
                print(f"🤖 Character Agent Server running on ws://{self.host}:{self.port}")
                print(f"📝 No character pre-loaded - clients can initialize characters")
                print(f"🔌 Waiting for client connections...")
            
            # Keep server running
            await self.server.wait_closed()
            
        except Exception as e:
            self.logger.error(f"Failed to start server: {e}")
            raise
    
    async def stop_server(self) -> None:
        """Stop the WebSocket server."""
        if not self.is_running:
            return
        
        self.logger.info("Stopping Character Agent Server...")
        
        # Close all client connections
        if self.connected_clients:
            close_tasks = [client.close() for client in self.connected_clients.copy()]
            await asyncio.gather(*close_tasks, return_exceptions=True)
        
        # Close server
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        # Save character state if loaded
        if self.character_agent:
            try:
                await self.auto_save_character()
            except Exception as e:
                self.logger.error(f"Error auto-saving character on shutdown: {e}")
        
        self.is_running = False
        self.logger.info("Character Agent Server stopped")
    
    async def handle_client(self, websocket: WebSocketServerProtocol) -> None:
        """Handle a new client connection.
        
        Args:
            websocket: WebSocket connection
        """
        client_id = str(uuid.uuid4())[:8]
        self.logger.info(f"New client connected: {client_id} from {websocket.remote_address}")
        
        self.connected_clients.add(websocket)
        
        try:
            async for message in websocket:
                try:
                    # Parse message
                    message_data = json.loads(message)
                    request = BaseMessage.from_json(message)
                    
                    self.logger.debug(f"Received {request.message_type} from client {client_id}")
                    
                    # Process message and get response
                    response = await self.process_message(request, client_id)
                    
                    # Send response
                    response_json = response.to_json()
                    await websocket.send(response_json)
                    
                    self.logger.debug(f"Sent {response.message_type} to client {client_id}")
                    
                except json.JSONDecodeError as e:
                    self.logger.error(f"Invalid JSON from client {client_id}: {e}")
                    error_response = create_error_response("Invalid JSON format")
                    await websocket.send(error_response.to_json())
                    
                except Exception as e:
                    self.logger.error(f"Error processing message from client {client_id}: {e}")
                    error_response = create_error_response(f"Internal server error: {str(e)}")
                    await websocket.send(error_response.to_json())
                    
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Client {client_id} disconnected")
        except Exception as e:
            self.logger.error(f"Error handling client {client_id}: {e}")
        finally:
            self.connected_clients.discard(websocket)
            self.logger.info(f"Client {client_id} connection closed")
    
    async def process_message(self, request: BaseMessage, client_id: str) -> BaseMessage:
        """Process a client message and return response.
        
        Args:
            request: Client request message
            client_id: Client identifier
            
        Returns:
            Response message
        """
        try:
            if isinstance(request, InitRequest):
                return await self.handle_init_request(request)
            
            elif isinstance(request, ChatRequest):
                return await self.handle_chat_request(request)
            
            elif isinstance(request, StatusRequest):
                return await self.handle_status_request(request)
            
            elif isinstance(request, SaveRequest):
                return await self.handle_save_request(request)
            
            elif isinstance(request, LoadRequest):
                return await self.handle_load_request(request)
            
            elif isinstance(request, ResetRequest):
                return await self.handle_reset_request(request)
            
            elif isinstance(request, MemorySearchRequest):
                return await self.handle_memory_search_request(request)
            
            elif isinstance(request, PingRequest):
                return PingResponse(message_id=request.message_id)
            
            else:
                return create_error_response(
                    f"Unknown message type: {request.message_type}",
                    message_id=request.message_id
                )
                
        except Exception as e:
            self.logger.error(f"Error processing {request.message_type}: {e}")
            return create_error_response(
                f"Error processing request: {str(e)}",
                message_id=request.message_id
            )
    
    async def handle_init_request(self, request: InitRequest) -> InitResponse:
        """Handle character initialization request."""
        try:
            success = await self.load_character(request.character_id, request.character_config)
            
            if success and self.character_agent:
                character_info = {
                    'character_id': self.character_agent.character_id,
                    'name': self.character_agent.character_name,
                    'archetype': self.character_agent.character_config.get('archetype', 'Unknown')
                }
                
                return InitResponse(
                    success=True,
                    character_name=self.character_agent.character_name,
                    character_info=character_info,
                    message_id=request.message_id
                )
            else:
                return InitResponse(
                    success=False,
                    error=f"Failed to load character '{request.character_id}'",
                    message_id=request.message_id
                )
                
        except Exception as e:
            return InitResponse(
                success=False,
                error=f"Error initializing character: {str(e)}",
                message_id=request.message_id
            )
    
    async def handle_chat_request(self, request: ChatRequest) -> ChatResponse:
        """Handle chat message request."""
        if not self.character_agent:
            return ChatResponse(
                success=False,
                error="No character loaded. Send init_request first.",
                message_id=request.message_id
            )
        
        try:
            # Process message with character agent
            result = await self.character_agent.process_message(
                user_message=request.user_message,
                context=request.context
            )
            
            return ChatResponse(
                success=True,
                response_text=result.get('response_text'),
                character_insights=result.get('character_insights'),
                response_metadata=result.get('response_metadata'),
                generation_info=result.get('generation_info'),
                orchestration_summary=result.get('orchestration_summary'),
                cognitive_summary=result.get('cognitive_summary'),
                character_state_summary=result.get('character_state_summary'),
                performance_info=result.get('performance_info'),
                message_id=request.message_id
            )
            
        except Exception as e:
            self.logger.error(f"Error processing chat message: {e}")
            return ChatResponse(
                success=False,
                error=f"Error processing message: {str(e)}",
                message_id=request.message_id
            )
    
    async def handle_status_request(self, request: StatusRequest) -> StatusResponse:
        """Handle status request."""
        if not self.character_agent:
            return StatusResponse(
                success=False,
                error="No character loaded",
                message_id=request.message_id
            )
        
        try:
            character_summary = self.character_agent.get_character_summary()
            vitals = None
            conversation_history = None
            performance_stats = None
            
            if request.include_vitals:
                vitals = {
                    'neurochemical_state': character_summary.get('neurochemical_state', {}),
                    'mood_state': character_summary.get('mood_state', {}),
                    'agent_states': character_summary.get('agent_states', {})
                }
            
            if request.include_history:
                conversation_history = self.character_agent.get_conversation_history(limit=20)
            
            if request.include_performance:
                performance_stats = self.character_agent.get_performance_stats()
            
            return StatusResponse(
                success=True,
                character_summary=character_summary,
                vitals=vitals,
                conversation_history=conversation_history,
                performance_stats=performance_stats,
                message_id=request.message_id
            )
            
        except Exception as e:
            return StatusResponse(
                success=False,
                error=f"Error getting status: {str(e)}",
                message_id=request.message_id
            )
    
    async def handle_save_request(self, request: SaveRequest) -> SaveResponse:
        """Handle save request."""
        if not self.character_agent:
            return SaveResponse(
                success=False,
                error="No character loaded",
                message_id=request.message_id
            )
        
        try:
            filename = request.filename
            if not filename:
                filename = f"{self.character_agent.character_id}_server_save.json"
            
            if not filename.endswith('.json'):
                filename += '.json'
            
            filepath = self.save_dir / filename
            
            # Save character state
            self.character_agent.save_character_state(str(filepath))
            
            return SaveResponse(
                success=True,
                filename=str(filepath),
                message_id=request.message_id
            )
            
        except Exception as e:
            return SaveResponse(
                success=False,
                error=f"Error saving character: {str(e)}",
                message_id=request.message_id
            )
    
    async def handle_load_request(self, request: LoadRequest) -> LoadResponse:
        """Handle load request."""
        try:
            filepath = Path(request.filename)
            if not filepath.is_absolute():
                filepath = self.save_dir / request.filename
            
            if not filepath.exists():
                return LoadResponse(
                    success=False,
                    error=f"Save file not found: {filepath}",
                    message_id=request.message_id
                )
            
            # Load character from save file
            self.character_agent = CharacterAgent.load_character_state(str(filepath))
            await self.character_agent.initialize()
            
            self.character_id = self.character_agent.character_id
            
            character_info = {
                'character_id': self.character_agent.character_id,
                'name': self.character_agent.character_name,
                'archetype': self.character_agent.character_config.get('archetype', 'Unknown')
            }
            
            return LoadResponse(
                success=True,
                character_info=character_info,
                message_id=request.message_id
            )
            
        except Exception as e:
            return LoadResponse(
                success=False,
                error=f"Error loading character: {str(e)}",
                message_id=request.message_id
            )
    
    async def handle_reset_request(self, request: ResetRequest) -> ResetResponse:
        """Handle reset request."""
        if not self.character_agent:
            return ResetResponse(
                success=False,
                error="No character loaded",
                message_id=request.message_id
            )
        
        try:
            await self.character_agent.reset_character_state(
                preserve_memories=request.preserve_memories
            )
            
            character_info = {
                'character_id': self.character_agent.character_id,
                'name': self.character_agent.character_name,
                'archetype': self.character_agent.character_config.get('archetype', 'Unknown')
            }
            
            return ResetResponse(
                success=True,
                character_info=character_info,
                message_id=request.message_id
            )
            
        except Exception as e:
            return ResetResponse(
                success=False,
                error=f"Error resetting character: {str(e)}",
                message_id=request.message_id
            )
    
    async def handle_memory_search_request(self, request: MemorySearchRequest) -> MemorySearchResponse:
        """Handle memory search request."""
        if not self.character_agent:
            return MemorySearchResponse(
                success=False,
                error="No character loaded",
                message_id=request.message_id
            )
        
        try:
            memories = await self.character_agent.recall_past_conversations(
                query=request.query,
                limit=request.limit
            )
            
            return MemorySearchResponse(
                success=True,
                memories=memories,
                count=len(memories),
                message_id=request.message_id
            )
            
        except Exception as e:
            return MemorySearchResponse(
                success=False,
                error=f"Error searching memories: {str(e)}",
                message_id=request.message_id
            )
    
    async def load_character(self, character_id: str, character_config: Optional[Dict[str, Any]] = None) -> bool:
        """Load a character agent.
        
        Args:
            character_id: ID of character to load
            character_config: Optional character configuration override
            
        Returns:
            True if character loaded successfully
        """
        try:
            self.logger.info(f"Loading character '{character_id}'...")
            
            # Load character configuration
            if character_config is None:
                config = self.character_loader.load_character(character_id)
            else:
                config = character_config
            
            # Create and initialize character agent
            self.character_agent = CharacterAgent(
                character_id=character_id,
                character_config=config
            )
            
            await self.character_agent.initialize()
            
            self.character_id = character_id
            self.logger.info(f"Successfully loaded character '{character_id}' ({config['name']})")
            
            return True
            
        except CharacterConfigurationError as e:
            self.logger.error(f"Character configuration error for '{character_id}': {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error loading character '{character_id}': {e}")
            return False
    
    async def auto_save_character(self) -> None:
        """Auto-save the current character state."""
        if not self.character_agent:
            return
        
        try:
            filename = f"{self.character_agent.character_id}_server_autosave.json"
            filepath = self.save_dir / filename
            
            self.character_agent.save_character_state(str(filepath))
            self.logger.info(f"Auto-saved character state to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error auto-saving character: {e}")
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get server status information."""
        return {
            'host': self.host,
            'port': self.port,
            'is_running': self.is_running,
            'connected_clients': len(self.connected_clients),
            'character_loaded': self.character_agent is not None,
            'character_id': self.character_id,
            'character_name': self.character_agent.character_name if self.character_agent else None
        }


async def main():
    """Main entry point for the character agent server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Character Agent WebSocket Server")
    parser.add_argument('--host', default='localhost', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8765, help='Port to listen on')
    parser.add_argument('--character', '-c', help='Character ID to pre-load')
    
    args = parser.parse_args()
    
    # Create server
    server = CharacterAgentServer(host=args.host, port=args.port)
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        print("\n🛑 Shutting down server...")
        asyncio.create_task(server.stop_server())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start server
        await server.start_server(character_id=args.character)
    except KeyboardInterrupt:
        print("\n🛑 Server interrupted by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        traceback.print_exc()
    finally:
        await server.stop_server()


if __name__ == "__main__":
    asyncio.run(main())