"""Simple remote chat interface for connecting to character agent server."""

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import sys

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from config.character_loader import CharacterLoader, CharacterConfigurationError
    from config.settings import get_settings
    from config.logging_config import setup_logging, get_logger, log_character_session_start, log_character_session_end, log_system_info
    from character_agent import CharacterAgent  # For fallback mode
    from client.character_agent_client import CharacterAgentClient, CharacterAgentClientError
    from cli.simple_chat_interface import SimpleChatInterface
except ImportError:
    # Fallback imports for when running from different contexts
    from ..config.character_loader import CharacterLoader, CharacterConfigurationError
    from ..config.settings import get_settings
    from ..config.logging_config import setup_logging, get_logger, log_character_session_start, log_character_session_end, log_system_info
    from ..character_agent import CharacterAgent  # For fallback mode
    from ..client.character_agent_client import CharacterAgentClient, CharacterAgentClientError
    from .simple_chat_interface import SimpleChatInterface


class SimpleRemoteChatInterface(SimpleChatInterface):
    """Simple remote chat interface that connects to character agent server."""
    
    def __init__(self, server_url: str = "ws://localhost:8765", fallback_to_local: bool = True, user_id: str = "anonymous"):
        """Initialize the simple remote chat interface.
        
        Args:
            server_url: WebSocket URL of character agent server
            fallback_to_local: Whether to fallback to local mode if server unavailable
            user_id: User ID for persistent conversations
        """
        super().__init__(user_id=user_id)
        
        # Connection settings
        self.server_url = server_url
        self.fallback_to_local = fallback_to_local
        self.is_remote_mode = False
        
        # Character agent (can be remote client or local agent)
        self.remote_client: Optional[CharacterAgentClient] = None
        
        self.logger.info(f"SimpleRemoteChatInterface initialized - server: {server_url}, fallback: {fallback_to_local}")
    
    def print_banner(self) -> None:
        """Print welcome banner with remote mode indicator."""
        print("=" * 60)
        print("    CHARACTER AGENT CHAT SYSTEM (REMOTE)")
        print("=" * 60)
        print()
        print("This system connects to a remote character agent server for")
        print("distributed conversations with AI characters.")
        print()
        print("Type '/help' for available commands or just start typing to chat!")
        print()
    
    async def connect_to_server(self) -> bool:
        """Attempt to connect to the character agent server."""
        try:
            print(f"Connecting to character agent server at {self.server_url}...")
            
            self.remote_client = CharacterAgentClient(
                server_url=self.server_url,
                timeout=None,  # Will use AGENT_TIMEOUT from .env
                reconnect_attempts=3
            )
            
            connected = await self.remote_client.connect()
            if connected:
                # Test connection with ping
                if await self.remote_client.ping_server():
                    self.is_remote_mode = True
                    print("✅ Connected to character agent server!")
                    return True
                else:
                    print("⚠️ Server not responding to ping")
                    await self.remote_client.disconnect()
            
        except Exception as e:
            self.logger.error(f"Failed to connect to server: {e}")
            print(f"❌ Failed to connect to server: {e}")
        
        return False
    
    async def run(self, character_id: Optional[str] = None) -> None:
        """Main entry point for the remote chat interface."""
        try:
            self.print_banner()
            
            # Attempt to connect to remote server
            connected_to_server = await self.connect_to_server()
            
            if not connected_to_server:
                if self.fallback_to_local:
                    print("📱 Falling back to local mode...")
                    self.is_remote_mode = False
                    await super().run(character_id)
                    return
                else:
                    print("❌ Cannot connect to server and fallback disabled")
                    return
            
            # Run in remote mode
            if not character_id:
                # Character selection
                character_id = await self.select_character()
                if not character_id:
                    print("No character selected. Exiting.")
                    return
            
            # Initialize character on server
            success = await self.remote_client.initialize_character(character_id)
            if not success:
                print("Failed to initialize character on server. Exiting.")
                return
            
            self.current_character = self.remote_client
            
            # Log character session start
            log_character_session_start(character_id, self.remote_client.character_name or character_id)
            
            # Display character loaded message
            character_info = self.remote_client.character_config or {}
            self.display_character_loaded(character_id, character_info)
            
            # Start conversation
            await self.conversation_loop()
            
        except KeyboardInterrupt:
            print("\nChat interrupted by user. Goodbye!")
            self.logger.info("Chat interrupted by user (KeyboardInterrupt)")
        except Exception as e:
            print(f"Unexpected error: {e}")
            self.logger.error(f"Unexpected error in remote chat interface: {e}", exc_info=True)
        finally:
            # Cleanup
            if self.remote_client and self.is_remote_mode:
                await self.remote_client.disconnect()
    
    def display_character_loaded(self, character_id: str, character_info: Dict[str, Any]) -> None:
        """Display character loaded message."""
        self.print_separator()
        
        if self.is_remote_mode:
            print("🌐 Connected to remote character:")
        else:
            print("📱 Loaded local character:")
        
        name = character_info.get('name', character_id)
        archetype = character_info.get('archetype', 'Unknown')
        
        print(f"You are now chatting with {name} ({archetype})")
        
        if 'description' in character_info:
            print(f"\n{character_info['description']}")
        
        print(f"\nType your message to start the conversation, or '/help' for commands.")
        self.print_separator()
    
    async def show_vitals(self) -> None:
        """Show character vitals (remote version)."""
        if not self.current_character:
            print("No active character")
            return
        
        try:
            if self.is_remote_mode:
                # Get vitals from remote server
                vitals = await self.current_character.get_character_vitals()
                
                print("\n=== CHARACTER VITALS (REMOTE) ===")
                print()
                
                # Basic info
                print(f"Character: {self.remote_client.character_name or 'Remote Character'}")
                print(f"ID: {self.remote_client.character_id}")
                print()
                
                # Neurochemical levels
                neurochemical_levels = vitals.get('neurochemical_state', {})
                if neurochemical_levels:
                    print("Neurochemical Levels:")
                    for hormone, level in neurochemical_levels.items():
                        bar = "█" * int(level * 10) + "░" * (10 - int(level * 10))
                        print(f"  {hormone.capitalize():<12}: {level:.2f} [{bar}]")
                    print()
                
                # Agent states
                agent_states = vitals.get('agent_states', {})
                if 'mood' in agent_states:
                    mood_state = agent_states['mood']
                    current_mood = mood_state.get('current_state', 'neutral')
                    intensity = mood_state.get('intensity', 0.5)
                    print(f"Mood: {current_mood.title()} (intensity: {intensity:.2f})")
                    print()
                
                # Relationship state
                relationship = vitals.get('relationship_state', {})
                if relationship:
                    trust = relationship.get('trust_level', 0)
                    rapport = relationship.get('rapport_level', 0)
                    print(f"Trust Level: {trust:.2f}")
                    print(f"Rapport Level: {rapport:.2f}")
                    print()
                
            else:
                # Use local method
                await super().show_vitals()
            
        except Exception as e:
            print(f"Error displaying vitals: {e}")
    
    async def save_character_state(self, filename: str = "") -> None:
        """Save character state (remote version)."""
        if not self.current_character:
            print("No active character")
            return
        
        try:
            if self.is_remote_mode:
                # Save on server
                server_filename = await self.current_character.save_character_state(filename)
                print(f"Character state saved on server: {server_filename}")
            else:
                # Use local method
                await super().save_character_state(filename)
                
        except Exception as e:
            print(f"Failed to save character state: {e}")
    
    async def load_character_state(self, filename: str = "") -> None:
        """Load character state (remote version)."""
        if not self.current_character:
            print("No active character")
            return
        
        try:
            if self.is_remote_mode:
                # Load from server
                success = await self.current_character.load_character_state(filename)
                if success:
                    print(f"Character state loaded from server: {filename}")
                else:
                    print(f"Failed to load character state from server")
            else:
                # Use local method
                await super().load_character_state(filename)
                
        except Exception as e:
            print(f"Failed to load character state: {e}")
    
    async def show_status(self) -> None:
        """Show character status (remote version)."""
        if not self.current_character:
            print("No active character")
            return
        
        try:
            print("\n=== CHARACTER STATUS ===")
            print()
            
            if self.is_remote_mode:
                print(f"Mode: Remote (connected to {self.server_url})")
                print(f"Character: {self.remote_client.character_name or 'Remote Character'}")
                print(f"ID: {self.remote_client.character_id}")
            else:
                print("Mode: Local (fallback)")
                state = self.current_character.state
                print(f"Character: {self.current_character.character_name}")
                print(f"ID: {self.current_character.character_id}")
                print(f"Messages: {len(state.conversation_history)}")
                print(f"Trust Level: {state.relationship_state.get('trust_level', 0):.2f}")
            
            print(f"Debug Mode: {'ON' if self.debug_mode else 'OFF'}")
            print()
            
        except Exception as e:
            print(f"Error displaying status: {e}")
    
    async def display_streaming_response(self, character_name: str, response_stream) -> str:
        """Display a streaming response - override for remote mode."""
        if self.is_remote_mode:
            # For remote mode, we simulate streaming since WebSocket streaming isn't implemented yet
            print(f"{character_name}: ", end="", flush=True)
            
            # If response_stream is a string (from simulated streaming), chunk it
            if isinstance(response_stream, str):
                chunk_size = 3  # Characters per chunk for simulation
                for i in range(0, len(response_stream), chunk_size):
                    chunk = response_stream[i:i+chunk_size]
                    print(chunk, end="", flush=True)
                    await asyncio.sleep(0.02)  # Small delay to simulate streaming
                print()  # New line after complete response
                return response_stream
            else:
                # Use parent implementation for actual streaming
                return await super().display_streaming_response(character_name, response_stream)
        else:
            # Use parent implementation when in local fallback mode
            return await super().display_streaming_response(character_name, response_stream)
    
    async def conversation_loop(self) -> None:
        """Override conversation loop to handle remote mode streaming."""
        if not self.current_character:
            return
        
        print(f"\n=== Conversation Started ===\n")
        
        while True:
            try:
                # Get user input
                user_input = input(f"You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    command_result = await self.handle_command(user_input)
                    if command_result == 'exit':
                        break
                    continue
                
                # Process message with character
                print("Thinking...")
                
                try:
                    character_name = self.current_character.character_name or "Character"
                    
                    if self.streaming_enabled:
                        if self.is_remote_mode:
                            # For remote mode, get full response then simulate streaming
                            context = {
                                'debug_mode': self.debug_mode,
                                'streaming': False  # Remote doesn't support real streaming yet
                            }
                            
                            result = await self.current_character.process_message(
                                user_message=user_input,
                                context=context
                            )
                            
                            response = result.get('response_text', 'No response generated')
                            
                            # Simulate streaming display
                            await self.display_streaming_response(character_name, response)
                            
                            # Note about remote streaming
                            if self.debug_mode:
                                print("[Remote mode: Simulated streaming - WebSocket streaming not yet implemented]")
                        
                        else:
                            # Local mode - use real streaming
                            context = {
                                'debug_mode': self.debug_mode,
                                'streaming': True
                            }
                            
                            response_stream = self.current_character.process_message_stream(
                                user_message=user_input,
                                context=context
                            )
                            
                            # Display real streaming response
                            response = await self.display_streaming_response(character_name, response_stream)
                            
                            if self.debug_mode:
                                print("[Debug info not available in streaming mode]")
                    
                    else:
                        # Non-streaming mode works the same for both remote and local
                        result = await self.current_character.process_message(
                            user_message=user_input,
                            context={'debug_mode': self.debug_mode}
                        )
                        
                        # Display character response
                        response = result.get('response_text', 'No response generated')
                        print(f"{character_name}: {response}")
                        
                        # Display debug info if enabled
                        if self.debug_mode and 'debug_info' in result:
                            self.display_debug_info(result['debug_info'])
                    
                    print()  # Add spacing
                    
                except Exception as e:
                    print(f"Error processing message: {e}")
                    continue
                
            except KeyboardInterrupt:
                if await self.confirm_exit():
                    break


async def main():
    """Main entry point for the simple remote chat interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Remote Character Agent Chat Interface")
    parser.add_argument('--server-url', default='ws://localhost:8765', help='Character agent server URL')
    parser.add_argument('--no-fallback', action='store_true', help='Disable fallback to local mode')
    parser.add_argument('--character', '-c', help='Character ID to load directly')
    
    args = parser.parse_args()
    
    interface = SimpleRemoteChatInterface(
        server_url=args.server_url,
        fallback_to_local=not args.no_fallback
    )
    
    await interface.run(character_id=args.character)


if __name__ == "__main__":
    asyncio.run(main())