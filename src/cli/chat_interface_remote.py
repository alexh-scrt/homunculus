"""Enhanced interactive chat interface with remote character agent support.

This version connects to a Character Agent Server via WebSocket instead of
running the character agent locally.
"""

import asyncio
import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
import sys

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, skip loading

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from config.character_loader import CharacterLoader, CharacterConfigurationError
    from config.settings import get_settings
    from config.logging_config import setup_logging, get_logger, log_character_session_start, log_character_session_end, log_system_info
    from character_agent import CharacterAgent  # For fallback mode
    from client.character_agent_client import CharacterAgentClient, CharacterAgentClientError
    from cli.debug_view import DebugView
    from cli.vitals_display import VitalsDisplay
    from cli.layout_manager import LayoutManager
    from cli.command_handler import CommandHandler
    from cli.keyboard_handler import AsyncKeyboardHandler, ScrollAction
    from cli.pane_manager import PaneManager, PaneType
except ImportError:
    # Fallback imports for when running from different contexts
    from ..config.character_loader import CharacterLoader, CharacterConfigurationError
    from ..config.settings import get_settings
    from ..config.logging_config import setup_logging, get_logger, log_character_session_start, log_character_session_end, log_system_info
    from ..character_agent import CharacterAgent  # For fallback mode
    from ..client.character_agent_client import CharacterAgentClient, CharacterAgentClientError
    from .debug_view import DebugView
    from .vitals_display import VitalsDisplay
    from .layout_manager import LayoutManager
    from .command_handler import CommandHandler
    from .keyboard_handler import AsyncKeyboardHandler, ScrollAction
    from .pane_manager import PaneManager, PaneType


class RemoteChatInterface:
    """Enhanced interactive CLI with remote character agent support."""
    
    def __init__(
        self, 
        enable_vitals: bool = True, 
        vitals_only: bool = False,
        server_url: str = "ws://localhost:8765",
        fallback_to_local: bool = True
    ):
        """Initialize the remote chat interface.
        
        Args:
            enable_vitals: Whether to show vitals display (default True)
            vitals_only: Whether to show only vitals without chat (default False)
            server_url: WebSocket URL of character agent server
            fallback_to_local: Whether to fallback to local mode if server unavailable
        """
        # Initialize logging first
        setup_logging()
        self.logger = get_logger(__name__)
        
        self.console = Console()
        self.debug_view = DebugView(self.console)
        self.settings = get_settings()
        self.character_loader = CharacterLoader()
        
        # Connection settings
        self.server_url = server_url
        self.fallback_to_local = fallback_to_local
        self.is_remote_mode = False
        
        # Character agent (can be remote client or local agent)
        self.current_character: Optional[Union[CharacterAgentClient, CharacterAgent]] = None
        self.remote_client: Optional[CharacterAgentClient] = None
        
        self.debug_mode = False
        self.enable_vitals = enable_vitals
        self.vitals_only = vitals_only
        
        self.logger.info(f"RemoteChatInterface initialized - vitals_enabled: {enable_vitals}, vitals_only: {vitals_only}")
        self.logger.info(f"Server URL: {server_url}, fallback_to_local: {fallback_to_local}")
        log_system_info()
        
        # Enhanced components
        self.vitals_display = VitalsDisplay(self.console)
        self.layout_manager = LayoutManager(self.console)
        self.command_handler = CommandHandler(self)
        self.keyboard_handler = AsyncKeyboardHandler(self.layout_manager)
        self.pane_manager = PaneManager(self.layout_manager)
        
        # Connect pane manager to layout manager
        self.layout_manager.set_pane_manager(self.pane_manager)
        
        # State management
        self.messages: List[Dict[str, str]] = []
        self.save_dir = Path("./data/saves")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Input handling
        self._input_queue = asyncio.Queue()
        self._should_exit = False
    
    async def connect_to_server(self) -> bool:
        """Attempt to connect to the character agent server.
        
        Returns:
            True if connected successfully
        """
        try:
            self.console.print(f"[cyan]Connecting to character agent server at {self.server_url}...[/cyan]")
            
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
                    self.console.print("[green]✅ Connected to character agent server![/green]")
                    return True
                else:
                    self.console.print("[yellow]⚠️ Server not responding to ping[/yellow]")
                    await self.remote_client.disconnect()
            
        except Exception as e:
            self.logger.error(f"Failed to connect to server: {e}")
            self.console.print(f"[red]❌ Failed to connect to server: {e}[/red]")
        
        return False
    
    async def run(self, character_id: Optional[str] = None) -> None:
        """Main entry point for the remote chat interface.
        
        Args:
            character_id: Optional character ID to load directly
        """
        try:
            # Attempt to connect to remote server
            connected_to_server = await self.connect_to_server()
            
            if not connected_to_server:
                if self.fallback_to_local:
                    self.console.print("[yellow]📱 Falling back to local mode...[/yellow]")
                    await self._run_local_mode(character_id)
                    return
                else:
                    self.console.print("[red]❌ Cannot connect to server and fallback disabled[/red]")
                    return
            
            # Run in remote mode
            await self._run_remote_mode(character_id)
            
        except KeyboardInterrupt:
            self.logger.info("Chat interrupted by user (KeyboardInterrupt)")
            self.console.print("\n[yellow]Chat interrupted by user. Goodbye![/yellow]")
        except Exception as e:
            self.logger.error(f"Unexpected error in chat interface: {e}", exc_info=True)
            self.debug_view.display_error(f"Unexpected error: {e}")
        finally:
            # Cleanup
            if self.remote_client and self.is_remote_mode:
                await self.remote_client.disconnect()
            elif self.current_character and not self.is_remote_mode:
                self.logger.info("Auto-saving character state before exit")
                await self._auto_save()
                if hasattr(self.current_character, 'character_id') and hasattr(self.current_character, 'character_name'):
                    log_character_session_end(self.current_character.character_id, self.current_character.character_name)
    
    async def _run_remote_mode(self, character_id: Optional[str] = None) -> None:
        """Run chat interface in remote mode."""
        self.logger.info("Running in remote mode")
        
        if not character_id:
            self._display_welcome()
            
            # Character selection
            character_id = await self._select_character()
            if not character_id:
                self.console.print("[yellow]No character selected. Exiting.[/yellow]")
                return
        
        # Initialize character on server
        success = await self.remote_client.initialize_character(character_id)
        if not success:
            self.console.print("[red]Failed to initialize character on server. Exiting.[/red]")
            return
        
        self.current_character = self.remote_client
        
        # Log character session start
        log_character_session_start(character_id, self.remote_client.character_name or character_id)
        
        # Display character loaded message
        character_info = self.remote_client.character_config or {}
        self._display_character_loaded(character_id, character_info)
        
        # Start appropriate interface mode
        if self.vitals_only:
            self.logger.info("Starting remote vitals-only mode")
            await self._vitals_only_mode()
        elif self.enable_vitals:
            self.logger.info("Starting remote dual-pane conversation mode")
            await self._dual_pane_conversation_loop()
        else:
            self.logger.info("Starting remote fallback conversation mode")
            await self._conversation_loop()
    
    async def _run_local_mode(self, character_id: Optional[str] = None) -> None:
        """Run chat interface in local mode (fallback)."""
        self.logger.info("Running in local mode")
        self.is_remote_mode = False
        
        if not character_id:
            self._display_welcome()
            
            # Character selection
            character_id = await self._select_character()
            if not character_id:
                self.console.print("[yellow]No character selected. Exiting.[/yellow]")
                return
        
        # Load character locally
        character = await self._load_character_local(character_id)
        if not character:
            self.console.print("[red]Failed to load character. Exiting.[/red]")
            return
        
        self.current_character = character
        
        # Log character session start
        log_character_session_start(character_id, character.character_name)
        
        # Start appropriate interface mode
        if self.vitals_only:
            self.logger.info("Starting local vitals-only mode")
            await self._vitals_only_mode()
        elif self.enable_vitals:
            self.logger.info("Starting local dual-pane conversation mode")
            await self._dual_pane_conversation_loop()
        else:
            self.logger.info("Starting local fallback conversation mode")
            await self._conversation_loop()
    
    def _display_welcome(self) -> None:
        """Display welcome message and instructions."""
        welcome_text = Text()
        welcome_text.append("Welcome to the Character Agent Chat System!\n\n", style="bold green")
        welcome_text.append("This system allows you to have natural conversations with AI characters\n")
        welcome_text.append("that have distinct personalities, memories, and emotional states.\n\n")
        welcome_text.append("Each character remembers your interactions and develops over time.\n")
        welcome_text.append("Type '/help' during conversation for available commands.\n\n")
        
        if self.is_remote_mode:
            welcome_text.append("🌐 Running in REMOTE mode - connected to character agent server\n", style="cyan")
        else:
            welcome_text.append("📱 Running in LOCAL mode\n", style="yellow")
        
        panel = Panel(welcome_text, title="[bold]Character Agent System[/bold]", border_style="green")
        self.console.print(panel)
    
    def _display_character_loaded(self, character_id: str, character_info: Dict[str, Any]) -> None:
        """Display character loaded message."""
        intro_text = Text()
        
        if self.is_remote_mode:
            intro_text.append("🌐 Connected to remote character: ", style="cyan")
        else:
            intro_text.append("📱 Loaded local character: ", style="yellow")
        
        name = character_info.get('name', character_id)
        archetype = character_info.get('archetype', 'Unknown')
        
        intro_text.append(f"{name}", style="bold cyan")
        intro_text.append(f" ({archetype})\n", style="yellow")
        
        if 'description' in character_info:
            intro_text.append(f"\n{character_info['description']}\n", style="white")
        
        intro_text.append(f"\nType your message to start the conversation, or '/help' for commands.", style="dim")
        
        panel = Panel(intro_text, title="[bold]Character Ready[/bold]", border_style="cyan")
        self.console.print(panel)
    
    async def _select_character(self) -> Optional[str]:
        """Display character selection menu and get user choice.
        
        Returns:
            Selected character ID or None if cancelled
        """
        try:
            # Get available characters
            characters = self.character_loader.list_available_characters()
            
            if not characters:
                self.debug_view.display_error("No characters found in schemas directory")
                return None
            
            # Load character info for display
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
                    self.console.print(f"[yellow]Warning: Could not load info for {char_id}: {e}[/yellow]")
            
            if not character_info:
                self.debug_view.display_error("No valid characters found")
                return None
            
            # Display character selection
            self.debug_view.display_character_selection(character_info)
            
            # Get user selection
            self.console.print("\nSelect a character to chat with:")
            choices = [str(i) for i in range(1, len(character_info) + 1)]
            choices.append('q')  # Option to quit
            
            choice = Prompt.ask(
                "Enter character number (or 'q' to quit)",
                choices=choices,
                default='1'
            )
            
            if choice == 'q':
                return None
            
            selected_index = int(choice) - 1
            return character_info[selected_index]['id']
            
        except Exception as e:
            self.debug_view.display_error(f"Error during character selection: {e}")
            return None
    
    async def _load_character_local(self, character_id: str) -> Optional[CharacterAgent]:
        """Load and initialize a character agent locally (fallback mode).
        
        Args:
            character_id: ID of character to load
            
        Returns:
            Initialized CharacterAgent or None if failed
        """
        try:
            self.console.print(f"[cyan]Loading character '{character_id}' locally...[/cyan]")
            
            # Load character configuration
            config = self.character_loader.load_character(character_id)
            
            # Create character agent
            character = CharacterAgent(
                character_id=character_id,
                character_config=config
            )
            
            # Initialize the character
            await character.initialize()
            
            self.console.print(f"[green]Successfully loaded {config['name']} locally![/green]")
            return character
            
        except CharacterConfigurationError as e:
            self.debug_view.display_error(f"Character configuration error: {e}")
            return None
        except Exception as e:
            self.debug_view.display_error(f"Error loading character: {e}")
            return None
    
    async def _get_character_state(self):
        """Get character state from current character (remote or local)."""
        if self.is_remote_mode and isinstance(self.current_character, CharacterAgentClient):
            # For remote mode, we need to fetch vitals
            try:
                vitals = await self.current_character.get_character_vitals()
                # Create a simple state-like object for compatibility
                class RemoteState:
                    def __init__(self, vitals_data):
                        self.neurochemical_levels = vitals_data.get('neurochemical_state', {})
                        self.agent_states = vitals_data.get('agent_states', {})
                        self.relationship_state = vitals_data.get('relationship_state', {})
                
                return RemoteState(vitals)
            except Exception as e:
                self.logger.error(f"Error fetching remote character state: {e}")
                return None
        else:
            # Local mode
            return self.current_character.state if self.current_character else None
    
    async def _dual_pane_conversation_loop(self) -> None:
        """Enhanced conversation loop with dual-pane layout and live vitals."""
        if not self.current_character:
            return
        
        # Setup character info in layout
        if self.is_remote_mode:
            character_name = self.remote_client.character_name or "Remote Character"
            character_archetype = "Remote"
        else:
            character_name = self.current_character.character_name
            character_archetype = self.current_character.character_config.get('archetype', 'Unknown')
        
        self.layout_manager.update_character_info(character_name, character_archetype)
        
        # Initial vitals display
        state = await self._get_character_state()
        if state:
            vitals_panel = self.vitals_display.render_vitals(state)
            self.layout_manager.update_vitals(vitals_panel)
        
        # Enable keyboard handling for scrolling
        keyboard_enabled = False
        if self.keyboard_handler.is_available():
            keyboard_enabled = await self.keyboard_handler.enable_async()
            if keyboard_enabled:
                self.logger.info("Keyboard scrolling enabled")
            else:
                self.logger.warning("Failed to enable keyboard scrolling")
        
        # Enable pane management
        pane_features_enabled = False
        if self.pane_manager.is_available():
            # Set up callbacks (reuse from local interface)
            self.pane_manager.set_pane_change_callback(self._on_pane_change)
            self.pane_manager.set_scroll_callback(self._on_pane_scroll)
            
            # Enable tab navigation and mouse interaction
            tab_enabled = self.pane_manager.enable_tab_navigation()
            mouse_enabled = self.pane_manager.enable_mouse_interaction()
            
            pane_features_enabled = tab_enabled or mouse_enabled
            if pane_features_enabled:
                self.logger.info(f"Pane features enabled - Tab: {tab_enabled}, Mouse: {mouse_enabled}")
            else:
                self.logger.warning("Failed to enable pane features")
        
        # Start live display with proper input handling
        with Live(
            self.layout_manager.get_layout(),
            console=self.console,
            screen=False,  # Don't take over full screen to allow input
            refresh_per_second=2  # Reduced refresh rate for better input handling
        ) as live:
            
            # Set refresh callback for scroll operations
            self.layout_manager.set_refresh_callback(live.refresh)
            
            self.layout_manager.update_input_prompt("Type your message...")
            live.refresh()
            
            while not self._should_exit:
                try:
                    # Check for terminal resize before each interaction
                    if self.layout_manager.refresh_terminal_size():
                        # Terminal was resized, recreate layout
                        self.layout_manager.resize_layout()
                        live.update(self.layout_manager.get_layout())
                    
                    # Get user input using Rich's prompt (works better with Live)
                    live.stop()  # Temporarily stop live updates for input
                    self.console.print()  # Add some space
                    try:
                        user_input = Prompt.ask("[bold blue]💬 Message[/bold blue]")
                    except (EOFError, KeyboardInterrupt):
                        user_input = "/exit"
                    live.start()  # Resume live updates
                    
                    if not user_input.strip():
                        await asyncio.sleep(0.1)  # Small delay to prevent busy loop
                        continue
                    
                    self.logger.debug(f"User input received: {user_input[:100]}...")
                    
                    # Handle commands
                    if user_input.startswith('/'):
                        self.logger.debug(f"Processing command: {user_input}")
                        self.layout_manager.show_thinking_indicator()
                        live.refresh()
                        
                        command_result = await self.command_handler.handle(user_input)
                        if command_result == 'exit':
                            self.logger.info("Exit command received")
                            break
                        continue
                    
                    # Show thinking indicator
                    self.layout_manager.show_thinking_indicator()
                    live.refresh()
                    
                    # Process message with character
                    try:
                        self.logger.debug("Starting character message processing")
                        result = await self.current_character.process_message(
                            user_message=user_input,
                            context={'debug_mode': self.debug_mode}
                        )
                        
                        self.logger.debug(f"Received result from character: {result}")
                        
                        # Add messages to history
                        self.messages.append({'speaker': 'You', 'message': user_input})
                        character_response = result.get('response_text', 'No response generated')
                        self.logger.debug(f"Extracted character response: {character_response}")
                        
                        if character_response and character_response != 'No response generated':
                            self.messages.append({
                                'speaker': character_name,
                                'message': character_response
                            })
                            self.logger.debug(f"Added message to history, total messages: {len(self.messages)}")
                        else:
                            self.logger.error(f"Empty or missing response_text in result: {result}")
                            self.messages.append({
                                'speaker': character_name,
                                'message': "Error: No response received from character"
                            })
                        
                        # Update chat display
                        self.layout_manager.update_chat_history(self.messages)
                        self.logger.debug("Updated chat history in layout manager")
                        
                        # Ensure latest messages are visible
                        self.layout_manager.ensure_latest_visible()
                        
                        # Update vitals display with new state
                        state = await self._get_character_state()
                        if state:
                            vitals_panel = self.vitals_display.render_vitals(state)
                            self.layout_manager.update_vitals(vitals_panel)
                        
                        # Update input prompt
                        self.layout_manager.update_input_prompt("Type your message...")
                        
                        # Display debug info if enabled
                        if self.debug_mode and 'debug_info' in result:
                            # In dual-pane mode, we'll show debug info differently
                            pass  # Debug info will be shown in vitals panel
                        
                        live.refresh()
                        self.logger.debug("Live display refreshed")
                        
                    except Exception as e:
                        self.logger.error(f"Error processing message: {e}", exc_info=True)
                        self.layout_manager.show_error_message(f"Error processing message: {e}")
                        live.refresh()
                        await asyncio.sleep(2)  # Show error for 2 seconds
                        self.layout_manager.update_input_prompt("Type your message...")
                        live.refresh()
                        continue
                    
                except KeyboardInterrupt:
                    live.stop()
                    if Confirm.ask("[yellow]Are you sure you want to exit?[/yellow]"):
                        break
                    live.start()
                    continue
                except Exception as e:
                    self.layout_manager.show_error_message(f"Conversation error: {e}")
                    live.refresh()
                    await asyncio.sleep(2)
                    continue
            
            # Cleanup keyboard handling
            if keyboard_enabled:
                await self.keyboard_handler.disable_async()
                self.logger.info("Keyboard scrolling disabled")
            
            # Cleanup pane management
            if pane_features_enabled:
                self.pane_manager.disable_all()
                self.logger.info("Pane features disabled")
    
    async def _vitals_only_mode(self) -> None:
        """Vitals-only display mode for monitoring character state."""
        if not self.current_character:
            return
        
        mode_name = "Remote" if self.is_remote_mode else "Local"
        self.console.print(f"[bold cyan]{mode_name} Vitals-Only Mode - Monitoring Character State[/bold cyan]")
        self.console.print("Press Ctrl+C to exit\n")
        
        try:
            while not self._should_exit:
                # Get current vitals
                state = await self._get_character_state()
                if state:
                    vitals_panel = self.vitals_display.render_detailed_vitals(state)
                    
                    # Clear screen and show vitals
                    self.console.clear()
                    self.console.print(vitals_panel)
                
                # Wait and refresh
                await asyncio.sleep(1)  # Update every second in vitals-only mode
                
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Exiting vitals monitoring...[/yellow]")
    
    async def _conversation_loop(self) -> None:
        """Original conversation loop (fallback mode)."""
        if not self.current_character:
            return
        
        mode_name = "Remote" if self.is_remote_mode else "Local"
        self.console.print("\n" + "="*60)
        self.console.print(f"[bold green]{mode_name} Conversation Started[/bold green]")
        self.console.print("="*60 + "\n")
        
        while True:
            try:
                # Get user input
                user_input = Prompt.ask("[bold blue]You[/bold blue]")
                
                if not user_input.strip():
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    command_result = await self._handle_command(user_input)
                    if command_result == 'exit':
                        break
                    continue
                
                # Process message with character
                self.console.print("[dim]Thinking...[/dim]")
                
                try:
                    result = await self.current_character.process_message(
                        user_message=user_input,
                        context={'debug_mode': self.debug_mode}
                    )
                    
                    self.logger.debug(f"Fallback mode - received result: {result}")
                    
                    # Display character response
                    if self.is_remote_mode:
                        character_name = self.remote_client.character_name or "Remote Character"
                    else:
                        character_name = self.current_character.character_name
                    
                    response = result.get('response_text', 'No response generated')
                    self.logger.debug(f"Fallback mode - extracted response: {response}")
                    
                    self.debug_view.display_message(
                        character_name, 
                        response, 
                        style="green"
                    )
                    
                    # Display debug info if enabled
                    if self.debug_mode and 'debug_info' in result:
                        self.debug_view.display_debug_info(result['debug_info'])
                    
                except Exception as e:
                    self.debug_view.display_error(f"Error processing message: {e}")
                    continue
                
            except KeyboardInterrupt:
                if Confirm.ask("\n[yellow]Are you sure you want to exit?[/yellow]"):
                    break
                continue
            except Exception as e:
                self.debug_view.display_error(f"Conversation error: {e}")
                continue
    
    async def _handle_command(self, command: str) -> Optional[str]:
        """Handle special commands during conversation.
        
        Args:
            command: Command string starting with '/'
            
        Returns:
            'exit' if should exit conversation, None otherwise
        """
        # For now, reuse the existing command handler
        # In the future, we might want to implement remote-specific commands
        return await self.command_handler.handle(command)
    
    async def _auto_save(self) -> None:
        """Automatically save character state on exit."""
        if not self.current_character:
            return
        
        try:
            if self.is_remote_mode:
                # Save on server
                filename = await self.current_character.save_character_state()
                self.console.print(f"[dim]Character state auto-saved on server: {filename}[/dim]")
            else:
                # Save locally
                await self._save_character_state_local()
                self.console.print("[dim]Character state auto-saved locally[/dim]")
        except Exception as e:
            self.console.print(f"[yellow]Warning: Auto-save failed: {e}[/yellow]")
    
    async def _save_character_state_local(self, filename: str = "") -> None:
        """Save character state locally (for local mode)."""
        if not self.current_character or self.is_remote_mode:
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
            
            self.debug_view.display_success(f"Character state saved to {filepath}")
            
        except Exception as e:
            self.debug_view.display_error(f"Failed to save character state: {e}")
    
    def _on_pane_change(self, pane_type: PaneType) -> None:
        """Handle pane change events.
        
        Args:
            pane_type: New active pane type
        """
        try:
            self.logger.debug(f"Active pane changed to: {pane_type.value}")
            # Update terminal size for pane positions
            self.layout_manager.refresh_terminal_size()
            if self.pane_manager:
                self.pane_manager.update_pane_positions(self.layout_manager.terminal_size)
            
            # Force refresh of all panels to show new active state
            self._refresh_all_panels()
        except Exception as e:
            self.logger.error(f"Error handling pane change: {e}")
    
    def _refresh_all_panels(self) -> None:
        """Refresh all panels to update their visual state."""
        try:
            # Update conversation panel
            self.layout_manager._update_chat_history()
            
            # Update input panel
            self.layout_manager.update_input_prompt("Type your message...")
            
            # Trigger refresh
            self.layout_manager._trigger_refresh()
        except Exception as e:
            self.logger.error(f"Error refreshing panels: {e}")
    
    def _on_pane_scroll(self, pane_type: PaneType, direction: str) -> None:
        """Handle scroll events in panes.
        
        Args:
            pane_type: Pane that received scroll event
            direction: Scroll direction ("up" or "down")
        """
        try:
            if pane_type == PaneType.CONVERSATION:
                if direction == "up":
                    self.layout_manager.scroll_up(lines=3)
                elif direction == "down":
                    self.layout_manager.scroll_down(lines=3)
                self.logger.debug(f"Scrolled conversation {direction}")
        except Exception as e:
            self.logger.error(f"Error handling pane scroll: {e}")


async def main():
    """Main entry point for the remote chat interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Remote Character Agent Chat Interface")
    parser.add_argument('--server-url', default='ws://localhost:8765', help='Character agent server URL')
    parser.add_argument('--no-fallback', action='store_true', help='Disable fallback to local mode')
    parser.add_argument('--character', '-c', help='Character ID to load directly')
    parser.add_argument('--vitals-only', '-v', action='store_true', help='Show only vitals monitoring')
    parser.add_argument('--no-vitals', action='store_true', help='Disable vitals display')
    
    args = parser.parse_args()
    
    interface = RemoteChatInterface(
        enable_vitals=not args.no_vitals,
        vitals_only=args.vitals_only,
        server_url=args.server_url,
        fallback_to_local=not args.no_fallback
    )
    
    await interface.run(character_id=args.character)


if __name__ == "__main__":
    asyncio.run(main())