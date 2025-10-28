"""Simple command-line chat interface without window management."""

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from config.character_loader import CharacterLoader, CharacterConfigurationError
    from config.settings import get_settings
    from config.logging_config import setup_logging, get_logger, log_character_session_start, log_character_session_end, log_system_info
    from character_agent import CharacterAgent
    from core.conversation_manager import ConversationManager
except ImportError:
    # Fallback imports for when running from different contexts
    from ..config.character_loader import CharacterLoader, CharacterConfigurationError
    from ..config.settings import get_settings
    from ..config.logging_config import setup_logging, get_logger, log_character_session_start, log_character_session_end, log_system_info
    from ..character_agent import CharacterAgent
    from ..core.conversation_manager import ConversationManager


class SimpleChatInterface:
    """Simple command-line chat interface without Rich components or window management."""
    
    def __init__(self, user_id: str = "anonymous"):
        """Initialize the simple chat interface."""
        # Initialize logging
        setup_logging()
        self.logger = get_logger(__name__)
        
        self.settings = get_settings()
        self.character_loader = CharacterLoader()
        self.current_character: Optional[CharacterAgent] = None
        self.debug_mode = False
        self.streaming_enabled = True  # Enable streaming by default
        
        # User context for persistent conversations
        self.user_id = user_id
        self.conversation_manager = None  # Will be initialized when needed
        
        self.logger.info(f"SimpleChatInterface initialized for user: {self.user_id}")
        log_system_info()
        
        # State management
        self.save_dir = Path("./data/saves")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize conversation manager
        self._init_conversation_manager()
        
        # Simple commands (no scroll/pane commands)
        self.commands = {
            '/help': 'Show available commands',
            '/debug': 'Toggle debug mode',
            '/stream': 'Toggle streaming mode [on/off]',
            '/vitals': 'Show character vitals',
            '/memory': 'Show recent memories or search [query]',
            '/goals': 'Show character goals', 
            '/mood': 'Show mood information',
            '/conversations': 'List all your conversations',
            '/history': 'Show conversation history [count]',
            '/save': 'Save character state [filename]',
            '/load': 'Load character state [filename]',
            '/reset': 'Reset character to initial state',
            '/export': 'Export conversation log [filename]',
            '/clear': 'Clear conversation history',
            '/status': 'Show character status',
            '/exit': 'Exit chat'
        }
    
    def print_banner(self) -> None:
        """Print welcome banner."""
        print("=" * 60)
        print("    CHARACTER AGENT CHAT SYSTEM")
        print("=" * 60)
        print()
        print("This system allows you to have natural conversations with AI characters")
        print("that have distinct personalities, memories, and emotional states.")
        print()
        print("Type '/help' for available commands or just start typing to chat!")
        print()
    
    def _init_conversation_manager(self) -> None:
        """Initialize the conversation manager."""
        try:
            self.conversation_manager = ConversationManager()
            self.logger.info("ConversationManager initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize ConversationManager: {e}")
            self.conversation_manager = None
    
    def print_separator(self) -> None:
        """Print a simple separator."""
        print("-" * 60)
    
    async def run(self, character_id: Optional[str] = None) -> None:
        """Main entry point for the simple chat interface."""
        try:
            self.print_banner()
            
            if not character_id:
                # Character selection
                character_id = await self.select_character()
                if not character_id:
                    print("No character selected. Exiting.")
                    return
            
            # Load character
            character = await self.load_character(character_id)
            if not character:
                print("Failed to load character. Exiting.")
                return
            
            self.current_character = character
            
            # Log character session start
            log_character_session_start(character_id, character.character_name)
            
            # Start conversation
            await self.conversation_loop()
            
        except KeyboardInterrupt:
            print("\nChat interrupted by user. Goodbye!")
            self.logger.info("Chat interrupted by user (KeyboardInterrupt)")
        except Exception as e:
            print(f"Unexpected error: {e}")
            self.logger.error(f"Unexpected error in chat interface: {e}", exc_info=True)
        finally:
            if self.current_character:
                await self.auto_save()
                log_character_session_end(self.current_character.character_id, self.current_character.character_name)
    
    async def select_character(self) -> Optional[str]:
        """Display character selection menu."""
        try:
            # Get available characters
            characters = self.character_loader.list_available_characters()
            
            if not characters:
                print("ERROR: No characters found in schemas directory")
                return None
            
            # Load character info
            character_info = []
            for char_id in characters:
                try:
                    info = self.character_loader.get_character_info(char_id)
                    character_info.append({
                        'id': char_id,
                        'name': info['name'],
                        'archetype': info['archetype'],
                        'description': info.get('description', 'No description available')
                    })
                except Exception as e:
                    print(f"Warning: Could not load info for {char_id}: {e}")
            
            if not character_info:
                print("ERROR: No valid characters found")
                return None
            
            # Display character options
            print("AVAILABLE CHARACTERS:")
            print()
            for i, char in enumerate(character_info, 1):
                print(f"{i}. {char['name']} ({char['archetype']})")
                print(f"   {char['description']}")
                print()
            
            # Get user selection
            while True:
                try:
                    choice = input(f"Select character (1-{len(character_info)}) or 'q' to quit: ").strip()
                    
                    if choice.lower() == 'q':
                        return None
                    
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(character_info):
                        return character_info[choice_num - 1]['id']
                    else:
                        print(f"Please enter a number between 1 and {len(character_info)}")
                        
                except ValueError:
                    print("Please enter a valid number or 'q' to quit")
                except (EOFError, KeyboardInterrupt):
                    return None
            
        except Exception as e:
            print(f"Error during character selection: {e}")
            return None
    
    async def load_character(self, character_id: str) -> Optional[CharacterAgent]:
        """Load and initialize a character agent."""
        try:
            print(f"Loading character '{character_id}'...")
            
            # Load character configuration
            config = self.character_loader.load_character(character_id)
            
            # Create character agent
            character = CharacterAgent(
                character_id=character_id,
                character_config=config
            )
            
            # Initialize the character
            await character.initialize()
            
            # Set user context for persistent conversations
            character.character_state.set_user_context(self.user_id)
            
            # Load conversation history if available
            await self._load_conversation_history(character, character_id)
            
            print(f"Successfully loaded {config['name']}!")
            self.print_separator()
            print(f"You are now chatting with {config['name']} ({config['archetype']})")
            
            if 'description' in config:
                print(f"\n{config['description']}")
            
            print(f"\nType your message to start the conversation, or '/help' for commands.")
            self.print_separator()
            
            return character
            
        except CharacterConfigurationError as e:
            print(f"Character configuration error: {e}")
            return None
        except Exception as e:
            print(f"Error loading character: {e}")
            return None
    
    async def _load_conversation_history(self, character: CharacterAgent, character_id: str) -> None:
        """Load persistent conversation history for the character."""
        if not self.conversation_manager:
            return
        
        try:
            # Check if conversation exists
            if await self.conversation_manager.conversation_exists(self.user_id, character_id):
                # Load recent messages (last 20 for working memory)
                recent_messages = await self.conversation_manager.get_recent_messages(
                    self.user_id, character_id, limit=20
                )
                
                if recent_messages:
                    # Replace the character's conversation history with loaded messages
                    character.character_state.conversation_history = recent_messages
                    print(f"📚 Loaded conversation history: {len(recent_messages)} recent messages")
                    print("🔄 Continuing previous conversation...")
                else:
                    print("👋 Starting new conversation!")
            else:
                print("👋 Starting new conversation!")
                
        except Exception as e:
            self.logger.error(f"Failed to load conversation history: {e}")
            print("⚠️ Could not load conversation history, starting fresh")
    
    async def _save_conversation_message(self, user_message: str, character_response: Optional[str] = None) -> None:
        """Save the latest messages to persistent conversation storage."""
        if not self.conversation_manager or not self.current_character:
            return
        
        try:
            character_id = self.current_character.character_id
            
            # Add user message to persistent storage
            await self.conversation_manager.add_message_to_conversation(
                self.user_id, character_id, "user", user_message
            )
            
            # Add character response if available
            if character_response:
                await self.conversation_manager.add_message_to_conversation(
                    self.user_id, character_id, "character", character_response
                )
            
        except Exception as e:
            self.logger.error(f"Failed to save conversation messages: {e}")
    
    async def conversation_loop(self) -> None:
        """Main conversation loop."""
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
                    character_name = self.current_character.character_name
                    
                    if self.streaming_enabled:
                        # Use streaming response
                        context = {
                            'debug_mode': self.debug_mode,
                            'streaming': True
                        }
                        
                        response_stream = self.current_character.process_message_stream(
                            user_message=user_input,
                            context=context
                        )
                        
                        # Display streaming response
                        response = await self.display_streaming_response(character_name, response_stream)
                        
                        # Note: debug info not available in streaming mode
                        if self.debug_mode:
                            print("[Debug info not available in streaming mode]")
                        
                    else:
                        # Use regular response
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
                    
                    # Save conversation to persistent storage
                    await self._save_conversation_message(user_input, response if 'response' in locals() else None)
                    
                    print()  # Add spacing
                    
                except Exception as e:
                    print(f"Error processing message: {e}")
                    continue
                
            except KeyboardInterrupt:
                if await self.confirm_exit():
                    break
                continue
            except EOFError:
                break
            except Exception as e:
                print(f"Conversation error: {e}")
                continue
    
    async def handle_command(self, command: str) -> Optional[str]:
        """Handle special commands."""
        parts = command.strip().split(' ', 1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        try:
            if cmd == '/exit':
                if await self.confirm_exit():
                    return 'exit'
            
            elif cmd == '/help':
                await self.show_help()
            
            elif cmd == '/debug':
                self.debug_mode = not self.debug_mode
                status = "ON" if self.debug_mode else "OFF"
                print(f"Debug mode: {status}")
            
            elif cmd == '/stream':
                await self.handle_stream_command(args)
            
            elif cmd == '/vitals':
                await self.show_vitals()
            
            elif cmd == '/memory':
                await self.search_memories(args or "recent experiences")
            
            elif cmd == '/goals':
                await self.show_goals()
            
            elif cmd == '/mood':
                await self.show_mood()
            
            elif cmd == '/conversations':
                await self.show_conversations()
            
            elif cmd == '/history':
                count = int(args) if args.isdigit() else 50
                await self.show_conversation_history(count)
            
            elif cmd == '/save':
                await self.save_character_state(args)
            
            elif cmd == '/load':
                await self.load_character_state(args)
            
            elif cmd == '/reset':
                await self.reset_character()
            
            elif cmd == '/export':
                await self.export_conversation(args)
            
            elif cmd == '/clear':
                await self.clear_conversation()
            
            elif cmd == '/status':
                await self.show_status()
            
            else:
                print(f"Unknown command: {cmd}")
                print("Type '/help' to see available commands.")
        
        except Exception as e:
            print(f"Error executing command {cmd}: {e}")
        
        return None
    
    async def confirm_exit(self) -> bool:
        """Ask for exit confirmation."""
        try:
            response = input("Are you sure you want to exit? (y/n): ").lower().strip()
            return response in ['y', 'yes', '1', 'true']
        except (EOFError, KeyboardInterrupt):
            return True
    
    async def show_help(self) -> None:
        """Show help information."""
        print("\nAVAILABLE COMMANDS:")
        print()
        for cmd, desc in self.commands.items():
            print(f"  {cmd:<12} - {desc}")
        print()
        print("EXAMPLES:")
        print("  /memory recent          - Search for recent memories")
        print("  /save my_session        - Save with custom name")
        print("  /export today           - Export with custom filename")
        print(f"  /stream on              - Enable streaming responses (currently {'ON' if self.streaming_enabled else 'OFF'})")
        print(f"  /stream off             - Disable streaming responses")
        print()
    
    async def handle_stream_command(self, args: str) -> None:
        """Handle streaming command."""
        args = args.strip().lower()
        
        if args == "on":
            self.streaming_enabled = True
            print("Streaming mode: ON - Responses will stream character by character")
        elif args == "off":
            self.streaming_enabled = False
            print("Streaming mode: OFF - Full responses will be displayed at once")
        elif args == "":
            # Toggle if no argument provided
            self.streaming_enabled = not self.streaming_enabled
            status = "ON" if self.streaming_enabled else "OFF"
            print(f"Streaming mode: {status}")
        else:
            print("Usage: /stream [on|off]")
            print(f"Current status: {'ON' if self.streaming_enabled else 'OFF'}")
    
    async def show_vitals(self) -> None:
        """Show character vitals."""
        if not self.current_character:
            print("No active character")
            return
        
        try:
            state = self.current_character.state
            
            print("\n=== CHARACTER VITALS ===")
            print()
            
            # Basic info
            print(f"Character: {self.current_character.character_name}")
            print(f"ID: {self.current_character.character_id}")
            print()
            
            # Neurochemical levels
            print("Neurochemical Levels:")
            for hormone, level in state.neurochemical_levels.items():
                bar = "█" * int(level * 10) + "░" * (10 - int(level * 10))
                print(f"  {hormone.capitalize():<12}: {level:.2f} [{bar}]")
            print()
            
            # Mood
            mood_state = state.agent_states.get('mood', {})
            current_mood = mood_state.get('current_state', 'neutral')
            intensity = mood_state.get('intensity', 0.5)
            print(f"Mood: {current_mood.title()} (intensity: {intensity:.2f})")
            print()
            
            # Relationship state
            relationship = state.relationship_state
            trust = relationship.get('trust_level', 0)
            rapport = relationship.get('rapport_level', 0)
            print(f"Trust Level: {trust:.2f}")
            print(f"Rapport Level: {rapport:.2f}")
            print()
            
            # Conversation stats
            history = state.conversation_history
            print(f"Messages Exchanged: {len(history)}")
            print()
            
        except Exception as e:
            print(f"Error displaying vitals: {e}")
    
    async def search_memories(self, query: str) -> None:
        """Search and display character memories."""
        if not self.current_character:
            print("No active character")
            return
        
        try:
            memories = await self.current_character.recall_past_conversations(
                query=query,
                limit=5
            )
            
            if not memories:
                print(f"No memories found for query: '{query}'")
                return
            
            print(f"\n=== MEMORIES FOR '{query}' ===")
            print()
            
            for i, memory in enumerate(memories, 1):
                timestamp = memory.get('timestamp', 'Unknown time')
                description = memory.get('description', 'No description')
                similarity = memory.get('similarity', 0.0)
                
                print(f"{i}. Similarity: {similarity:.3f}")
                print(f"   Time: {timestamp}")
                print(f"   Content: {description[:200]}{'...' if len(description) > 200 else ''}")
                print()
            
        except Exception as e:
            print(f"Error searching memories: {e}")
    
    async def show_goals(self) -> None:
        """Show character goals."""
        if not self.current_character:
            print("No active character")
            return
        
        try:
            goals_state = self.current_character.state.agent_states.get('goals', {})
            active_goals = goals_state.get('active_goals', [])
            
            if not active_goals:
                print("No active goals found")
                return
            
            print("\n=== CHARACTER GOALS ===")
            print()
            
            for i, goal in enumerate(active_goals, 1):
                if isinstance(goal, dict):
                    priority = goal.get('priority', 5)
                    description = goal.get('description', goal.get('goal', 'Unknown goal'))
                    status = goal.get('status', 'active')
                else:
                    priority = 5
                    description = str(goal)
                    status = 'active'
                
                print(f"{i}. Priority {priority}: {description}")
                print(f"   Status: {status}")
                print()
            
        except Exception as e:
            print(f"Error displaying goals: {e}")
    
    async def show_mood(self) -> None:
        """Show mood information."""
        if not self.current_character:
            print("No active character")
            return
        
        try:
            mood_state = self.current_character.state.agent_states.get('mood', {})
            
            print("\n=== MOOD STATUS ===")
            print()
            
            current = mood_state.get('current_state', 'neutral')
            intensity = mood_state.get('intensity', 0.5)
            energy = mood_state.get('energy_level', 0.5)
            volatility = mood_state.get('emotional_volatility', 0.5)
            duration = mood_state.get('duration', 1)
            
            print(f"Current State: {current.title()}")
            print(f"Intensity: {intensity:.2f}/1.0")
            print(f"Energy Level: {energy:.2f}/1.0")
            print(f"Volatility: {volatility:.2f}/1.0")
            print(f"Duration: {duration} interactions")
            print()
            print(f"Triggered by: {mood_state.get('triggered_by', 'unknown')}")
            print()
            
        except Exception as e:
            print(f"Error displaying mood: {e}")
    
    async def save_character_state(self, filename: str = "") -> None:
        """Save character state to file."""
        if not self.current_character:
            print("No active character")
            return
        
        try:
            if not filename:
                filename = f"{self.current_character.character_id}_autosave.json"
            
            if not filename.endswith('.json'):
                filename += '.json'
            
            filepath = self.save_dir / filename
            
            # Save state
            state_data = self.current_character.get_state_dict()
            
            with open(filepath, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)
            
            print(f"Character state saved to {filepath}")
            
        except Exception as e:
            print(f"Failed to save character state: {e}")
    
    async def load_character_state(self, filename: str = "") -> None:
        """Load character state from file."""
        if not self.current_character:
            print("No active character")
            return
        
        try:
            if not filename:
                filename = f"{self.current_character.character_id}_autosave.json"
            
            if not filename.endswith('.json'):
                filename += '.json'
            
            filepath = self.save_dir / filename
            
            if not filepath.exists():
                print(f"Save file not found: {filepath}")
                return
            
            # Load state
            with open(filepath, 'r') as f:
                state_data = json.load(f)
            
            await self.current_character.load_state_dict(state_data)
            
            print(f"Character state loaded from {filepath}")
            
        except Exception as e:
            print(f"Failed to load character state: {e}")
    
    async def reset_character(self) -> None:
        """Reset character to initial state."""
        if not self.current_character:
            print("No active character")
            return
        
        if not await self.confirm_reset():
            return
        
        try:
            character_id = self.current_character.character_id
            config = self.character_loader.load_character(character_id)
            
            # Reinitialize character
            self.current_character = CharacterAgent(
                character_id=character_id,
                character_config=config
            )
            await self.current_character.initialize()
            
            print("Character reset to initial state")
            
        except Exception as e:
            print(f"Error resetting character: {e}")
    
    async def confirm_reset(self) -> bool:
        """Confirm character reset."""
        try:
            response = input("This will reset the character to initial state. Continue? (y/n): ").lower().strip()
            return response in ['y', 'yes', '1', 'true']
        except (EOFError, KeyboardInterrupt):
            return False
    
    async def export_conversation(self, filename: str = "") -> None:
        """Export conversation log."""
        if not self.current_character:
            print("No active character")
            return
        
        try:
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = filename or f"conversation_{timestamp}.json"
            if not filename.endswith('.json'):
                filename += '.json'
            
            export_dir = Path("./data/exports")
            export_dir.mkdir(parents=True, exist_ok=True)
            filepath = export_dir / filename
            
            # Gather export data
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'character_id': self.current_character.character_id,
                'character_name': self.current_character.character_name,
                'conversation_history': self.current_character.state.conversation_history,
                'neurochemical_levels': self.current_character.state.neurochemical_levels,
                'mood_state': self.current_character.state.agent_states.get('mood', {}),
                'relationship_state': self.current_character.state.relationship_state
            }
            
            # Save export
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"Conversation exported to {filepath}")
            
        except Exception as e:
            print(f"Export failed: {e}")
    
    async def clear_conversation(self) -> None:
        """Clear conversation history."""
        if not self.current_character:
            print("No active character")
            return
        
        if not await self.confirm_clear():
            return
        
        try:
            # Clear conversation history
            self.current_character.state.conversation_history = []
            print("Conversation history cleared")
            
        except Exception as e:
            print(f"Failed to clear history: {e}")
    
    async def confirm_clear(self) -> bool:
        """Confirm conversation clear."""
        try:
            response = input("Clear conversation history? This cannot be undone. (y/n): ").lower().strip()
            return response in ['y', 'yes', '1', 'true']
        except (EOFError, KeyboardInterrupt):
            return False
    
    async def show_status(self) -> None:
        """Show character status."""
        if not self.current_character:
            print("No active character")
            return
        
        try:
            state = self.current_character.state
            
            print("\n=== CHARACTER STATUS ===")
            print()
            print(f"Character: {self.current_character.character_name}")
            print(f"ID: {self.current_character.character_id}")
            print(f"Messages: {len(state.conversation_history)}")
            print(f"Trust Level: {state.relationship_state.get('trust_level', 0):.2f}")
            print(f"Debug Mode: {'ON' if self.debug_mode else 'OFF'}")
            print(f"Streaming Mode: {'ON' if self.streaming_enabled else 'OFF'}")
            print()
            
        except Exception as e:
            print(f"Error displaying status: {e}")
    
    async def show_conversations(self) -> None:
        """Show all conversations for the current user."""
        if not self.conversation_manager:
            print("Conversation manager not available")
            return
        
        try:
            conversations = await self.conversation_manager.get_user_conversations(self.user_id)
            
            if not conversations:
                print(f"\n=== NO CONVERSATIONS FOUND ===")
                print(f"User: {self.user_id}")
                print("You haven't had any conversations yet!")
                return
            
            print(f"\n=== YOUR CONVERSATIONS ===")
            print(f"User: {self.user_id}")
            print()
            
            for conv in conversations:
                print(f"Character: {conv['character_id']}")
                print(f"Messages: {conv['message_count']}")
                print(f"Last Updated: {conv['last_updated']}")
                print("-" * 40)
            
            print(f"Total Conversations: {len(conversations)}")
            print()
            
        except Exception as e:
            print(f"Error displaying conversations: {e}")
    
    async def show_conversation_history(self, count: int = 50) -> None:
        """Show conversation history for current character."""
        if not self.conversation_manager or not self.current_character:
            print("No active character or conversation manager not available")
            return
        
        try:
            character_id = self.current_character.character_id
            history = await self.conversation_manager.load_conversation_history(self.user_id, character_id)
            
            if not history:
                print(f"\n=== NO CONVERSATION HISTORY ===")
                print("No previous messages found for this conversation.")
                return
            
            # Show last 'count' messages
            messages_to_show = history[-count:] if len(history) > count else history
            
            print(f"\n=== CONVERSATION HISTORY ===")
            print(f"Showing last {len(messages_to_show)} of {len(history)} messages")
            print(f"User: {self.user_id} | Character: {character_id}")
            print()
            
            for msg in messages_to_show:
                role_display = "You" if msg['role'] == 'user' else self.current_character.character_name
                timestamp = msg.get('timestamp', 'Unknown time')
                print(f"[{timestamp}] {role_display}: {msg['message']}")
                print()
            
        except Exception as e:
            print(f"Error displaying conversation history: {e}")
    
    async def display_streaming_response(self, character_name: str, response_stream) -> str:
        """Display a streaming response character by character."""
        print(f"{character_name}: ", end="", flush=True)
        
        accumulated_response = ""
        try:
            async for chunk in response_stream:
                if isinstance(chunk, dict):
                    # Handle streaming with metadata (from web search stream)
                    content = chunk.get('content', '')
                else:
                    # Handle plain string chunks
                    content = str(chunk)
                
                print(content, end="", flush=True)
                accumulated_response += content
                
            print()  # New line after complete response
            return accumulated_response
            
        except Exception as e:
            print(f"\n[Streaming error: {e}]")
            return accumulated_response
    
    def display_debug_info(self, debug_info: Dict[str, Any]) -> None:
        """Display debug information."""
        print("\n--- DEBUG INFO ---")
        for key, value in debug_info.items():
            print(f"{key}: {value}")
        print("--- END DEBUG ---")
    
    async def auto_save(self) -> None:
        """Automatically save character state on exit."""
        if not self.current_character:
            return
        
        try:
            await self.save_character_state()
            print("Character state auto-saved")
        except Exception as e:
            print(f"Warning: Auto-save failed: {e}")


async def main():
    """Main entry point for the simple chat interface."""
    interface = SimpleChatInterface()
    await interface.run()


if __name__ == "__main__":
    asyncio.run(main())