"""Enhanced interactive chat interface with dual-pane layout and real-time vitals."""

import asyncio
import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from config.character_loader import CharacterLoader, CharacterConfigurationError
    from config.settings import get_settings
    from config.logging_config import setup_logging, get_logger, log_character_session_start, log_character_session_end, log_system_info
    from character_agent import CharacterAgent
    from cli.debug_view import DebugView
    from cli.vitals_display import VitalsDisplay
    from cli.layout_manager import LayoutManager
    from cli.command_handler import CommandHandler
except ImportError:
    # Fallback imports for when running from different contexts
    from ..config.character_loader import CharacterLoader, CharacterConfigurationError
    from ..config.settings import get_settings
    from ..config.logging_config import setup_logging, get_logger, log_character_session_start, log_character_session_end, log_system_info
    from ..character_agent import CharacterAgent
    from .debug_view import DebugView
    from .vitals_display import VitalsDisplay
    from .layout_manager import LayoutManager
    from .command_handler import CommandHandler


class ChatInterface:
    """Enhanced interactive CLI with dual-pane layout and real-time vitals."""
    
    def __init__(self, enable_vitals: bool = True, vitals_only: bool = False):
        """Initialize the enhanced chat interface.
        
        Args:
            enable_vitals: Whether to show vitals display (default True)
            vitals_only: Whether to show only vitals without chat (default False)
        """
        # Initialize logging first
        setup_logging()
        self.logger = get_logger(__name__)
        
        self.console = Console()
        self.debug_view = DebugView(self.console)
        self.settings = get_settings()
        self.character_loader = CharacterLoader()
        self.current_character: Optional[CharacterAgent] = None
        self.debug_mode = False
        self.enable_vitals = enable_vitals
        self.vitals_only = vitals_only
        
        self.logger.info(f"ChatInterface initialized - vitals_enabled: {enable_vitals}, vitals_only: {vitals_only}")
        log_system_info()
        
        # Enhanced components
        self.vitals_display = VitalsDisplay(self.console)
        self.layout_manager = LayoutManager(self.console)
        self.command_handler = CommandHandler(self)
        
        # State management
        self.messages: List[Dict[str, str]] = []
        self.save_dir = Path("./data/saves")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Input handling
        self._input_queue = asyncio.Queue()
        self._should_exit = False
    
    async def run(self, character_id: Optional[str] = None) -> None:
        """Main entry point for the enhanced chat interface.
        
        Args:
            character_id: Optional character ID to load directly
        """
        try:
            if not character_id:
                self._display_welcome()
                
                # Character selection
                character_id = await self._select_character()
                if not character_id:
                    self.console.print("[yellow]No character selected. Exiting.[/yellow]")
                    return
            
            # Load character
            character = await self._load_character(character_id)
            if not character:
                self.console.print("[red]Failed to load character. Exiting.[/red]")
                return
            
            self.current_character = character
            
            # Log character session start
            log_character_session_start(character_id, character.character_name)
            
            # Start appropriate interface mode
            if self.vitals_only:
                self.logger.info("Starting vitals-only mode")
                await self._vitals_only_mode()
            elif self.enable_vitals:
                self.logger.info("Starting dual-pane conversation mode")
                await self._dual_pane_conversation_loop()
            else:
                self.logger.info("Starting fallback conversation mode")
                await self._conversation_loop()  # Fallback to original mode
            
        except KeyboardInterrupt:
            self.logger.info("Chat interrupted by user (KeyboardInterrupt)")
            self.console.print("\n[yellow]Chat interrupted by user. Goodbye![/yellow]")
        except Exception as e:
            self.logger.error(f"Unexpected error in chat interface: {e}", exc_info=True)
            self.debug_view.display_error(f"Unexpected error: {e}")
        finally:
            if self.current_character:
                self.logger.info("Auto-saving character state before exit")
                await self._auto_save()
                log_character_session_end(self.current_character.character_id, self.current_character.character_name)
    
    async def _get_user_input_async(self) -> str:
        """Get user input asynchronously (compatible with live display).
        
        For now, this is a simple implementation. In a full implementation,
        you might want to use a more sophisticated async input system.
        
        Returns:
            User input string
        """
        # This is a simplified version - in production you might want to use
        # libraries like aioconsole or implement proper async input handling
        loop = asyncio.get_event_loop()
        
        # Show input prompt and get input
        try:
            user_input = await loop.run_in_executor(
                None, 
                lambda: input()  # Basic input for now
            )
            return user_input
        except EOFError:
            return "/exit"
        except KeyboardInterrupt:
            return "/exit"
    
    async def _confirm_exit_async(self) -> bool:
        """Async version of exit confirmation.
        
        Returns:
            True if user confirms exit, False otherwise
        """
        try:
            loop = asyncio.get_event_loop()
            
            # Ask for confirmation
            self.console.print("\n[yellow]Are you sure you want to exit? (y/n)[/yellow]")
            response = await loop.run_in_executor(
                None,
                lambda: input().lower().strip()
            )
            return response in ['y', 'yes', '1', 'true']
        except (EOFError, KeyboardInterrupt):
            return True
    
    def _display_welcome(self) -> None:
        """Display welcome message and instructions."""
        welcome_text = Text()
        welcome_text.append("Welcome to the Character Agent Chat System!\n\n", style="bold green")
        welcome_text.append("This system allows you to have natural conversations with AI characters\n")
        welcome_text.append("that have distinct personalities, memories, and emotional states.\n\n")
        welcome_text.append("Each character remembers your interactions and develops over time.\n")
        welcome_text.append("Type '/help' during conversation for available commands.")
        
        panel = Panel(welcome_text, title="[bold]Character Agent System[/bold]", border_style="green")
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
    
    async def _load_character(self, character_id: str) -> Optional[CharacterAgent]:
        """Load and initialize a character agent.
        
        Args:
            character_id: ID of character to load
            
        Returns:
            Initialized CharacterAgent or None if failed
        """
        try:
            self.console.print(f"[cyan]Loading character '{character_id}'...[/cyan]")
            
            # Load character configuration
            config = self.character_loader.load_character(character_id)
            
            # Create character agent
            character = CharacterAgent(
                character_id=character_id,
                character_config=config
            )
            
            # Initialize the character
            await character.initialize()
            
            self.console.print(f"[green]Successfully loaded {config['name']}![/green]")
            
            # Display character introduction
            intro_text = Text()
            intro_text.append(f"You are now chatting with ", style="white")
            intro_text.append(f"{config['name']}", style="bold cyan")
            intro_text.append(f" ({config['archetype']})\n", style="yellow")
            
            if 'description' in config:
                intro_text.append(f"\n{config['description']}\n", style="white")
            
            intro_text.append(f"\nType your message to start the conversation, or '/help' for commands.", style="dim")
            
            panel = Panel(intro_text, title="[bold]Character Loaded[/bold]", border_style="cyan")
            self.console.print(panel)
            
            return character
            
        except CharacterConfigurationError as e:
            self.debug_view.display_error(f"Character configuration error: {e}")
            return None
        except Exception as e:
            self.debug_view.display_error(f"Error loading character: {e}")
            return None
    
    async def _dual_pane_conversation_loop(self) -> None:
        """Enhanced conversation loop with dual-pane layout and live vitals."""
        if not self.current_character:
            return
        
        # Setup character info in layout
        self.layout_manager.update_character_info(
            self.current_character.character_name,
            self.current_character.character_config.get('archetype', 'Unknown')
        )
        
        # Initial vitals display
        vitals_panel = self.vitals_display.render_vitals(self.current_character.state)
        self.layout_manager.update_vitals(vitals_panel)
        
        # Start live display with proper input handling
        with Live(
            self.layout_manager.get_layout(),
            console=self.console,
            screen=False,  # Don't take over full screen to allow input
            refresh_per_second=2  # Reduced refresh rate for better input handling
        ) as live:
            
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
                        
                        # Add messages to history
                        self.messages.append({'speaker': 'You', 'message': user_input})
                        character_response = result.get('response_text', 'No response generated')
                        self.logger.debug(f"Character response: {character_response[:100]}...")
                        
                        self.messages.append({
                            'speaker': self.current_character.character_name,
                            'message': character_response
                        })
                        
                        # Update chat display
                        self.layout_manager.update_chat_history(self.messages)
                        
                        # Update vitals display with new state
                        vitals_panel = self.vitals_display.render_vitals(self.current_character.state)
                        self.layout_manager.update_vitals(vitals_panel)
                        
                        # Update input prompt
                        self.layout_manager.update_input_prompt("Type your message...")
                        
                        # Display debug info if enabled
                        if self.debug_mode and 'debug_info' in result:
                            # In dual-pane mode, we'll show debug info differently
                            pass  # Debug info will be shown in vitals panel
                        
                        live.refresh()
                        
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
    
    async def _vitals_only_mode(self) -> None:
        """Vitals-only display mode for monitoring character state."""
        if not self.current_character:
            return
        
        self.console.print("[bold cyan]Vitals-Only Mode - Monitoring Character State[/bold cyan]")
        self.console.print("Press Ctrl+C to exit\n")
        
        try:
            while not self._should_exit:
                # Get current vitals
                vitals_panel = self.vitals_display.render_detailed_vitals(self.current_character.state)
                
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
        
        self.console.print("\n" + "="*60)
        self.console.print("[bold green]Conversation Started[/bold green]")
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
                    
                    # Display character response
                    character_name = self.current_character.character_name
                    response = result.get('response_text', 'No response generated')
                    
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
        parts = command.strip().split(' ', 1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        try:
            if cmd == '/exit':
                if Confirm.ask("[yellow]Are you sure you want to exit?[/yellow]"):
                    return 'exit'
            
            elif cmd == '/debug':
                self.debug_mode = not self.debug_mode
                status = "ON" if self.debug_mode else "OFF"
                self.console.print(f"[yellow]Debug mode: {status}[/yellow]")
            
            elif cmd == '/help':
                self.debug_view.display_help()
            
            elif cmd == '/save':
                await self._save_character_state(args)
            
            elif cmd == '/load':
                await self._load_character_state(args)
            
            elif cmd == '/memory':
                await self._search_memories(args or "recent experiences")
            
            elif cmd == '/reset':
                await self._reset_character()
            
            elif cmd == '/status':
                await self._display_character_status()
            
            else:
                self.debug_view.display_error(f"Unknown command: {cmd}")
                self.debug_view.display_help()
        
        except Exception as e:
            self.debug_view.display_error(f"Error executing command {cmd}: {e}")
        
        return None
    
    async def _save_character_state(self, filename: str = "") -> None:
        """Save character state to file.
        
        Args:
            filename: Optional filename. If empty, generates default name.
        """
        if not self.current_character:
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
    
    async def _load_character_state(self, filename: str = "") -> None:
        """Load character state from file.
        
        Args:
            filename: Optional filename. If empty, looks for autosave.
        """
        if not self.current_character:
            return
        
        try:
            if not filename:
                filename = f"{self.current_character.character_id}_autosave.json"
            
            if not filename.endswith('.json'):
                filename += '.json'
            
            filepath = self.save_dir / filename
            
            if not filepath.exists():
                self.debug_view.display_error(f"Save file not found: {filepath}")
                return
            
            # Load state
            with open(filepath, 'r') as f:
                state_data = json.load(f)
            
            await self.current_character.load_state_dict(state_data)
            
            self.debug_view.display_success(f"Character state loaded from {filepath}")
            
        except Exception as e:
            self.debug_view.display_error(f"Failed to load character state: {e}")
    
    async def _search_memories(self, query: str) -> None:
        """Search and display character memories.
        
        Args:
            query: Search query for memories
        """
        if not self.current_character:
            return
        
        try:
            memories = await self.current_character.recall_past_conversations(
                query=query,
                limit=5
            )
            
            if not memories:
                self.console.print(f"[yellow]No memories found for query: '{query}'[/yellow]")
                return
            
            self.console.print(f"\n[bold]Found {len(memories)} relevant memories for '{query}':[/bold]")
            
            for i, memory in enumerate(memories, 1):
                timestamp = memory.get('timestamp', 'Unknown time')
                description = memory.get('description', 'No description')
                similarity = memory.get('similarity', 0.0)
                
                sim_color = "green" if similarity > 0.8 else "yellow" if similarity > 0.6 else "red"
                
                self.console.print(f"  {i}. [{sim_color}]{similarity:.3f}[/{sim_color}] ", end="")
                self.console.print(f"[dim]{timestamp}[/dim]")
                self.console.print(f"     {description[:200]}{'...' if len(description) > 200 else ''}")
            
        except Exception as e:
            self.debug_view.display_error(f"Error searching memories: {e}")
    
    async def _reset_character(self) -> None:
        """Reset character to initial state."""
        if not self.current_character:
            return
        
        if not Confirm.ask("[red]This will reset the character to initial state. Continue?[/red]"):
            return
        
        try:
            character_id = self.current_character.character_id
            config = self.character_loader.load_character(character_id)
            
            # Reinitialize character
            self.current_character = CharacterAgent(
                character_config=config,
                settings=self.settings
            )
            await self.current_character.initialize()
            
            self.debug_view.display_success("Character reset to initial state")
            
        except Exception as e:
            self.debug_view.display_error(f"Error resetting character: {e}")
    
    async def _display_character_status(self) -> None:
        """Display current character status."""
        if not self.current_character:
            return
        
        try:
            state = self.current_character.state
            
            # Create status display
            status_text = Text()
            status_text.append(f"Character: ", style="bold")
            status_text.append(f"{self.current_character.character_name}\n", style="cyan")
            status_text.append(f"ID: ", style="bold")
            status_text.append(f"{self.current_character.character_id}\n", style="white")
            
            # Mood info
            if hasattr(state, 'agent_states') and 'mood' in state.agent_states:
                mood = state.agent_states['mood']
                status_text.append(f"Current Mood: ", style="bold")
                status_text.append(f"{mood.get('current_state', 'unknown')} ", style="green")
                status_text.append(f"(intensity: {mood.get('intensity', 0):.2f})\n", style="yellow")
            
            # Conversation count
            if hasattr(state, 'conversation_history'):
                history_len = len(state.conversation_history)
                status_text.append(f"Messages Exchanged: ", style="bold")
                status_text.append(f"{history_len}\n", style="white")
            
            panel = Panel(status_text, title="[bold]Character Status[/bold]", border_style="blue")
            self.console.print(panel)
            
        except Exception as e:
            self.debug_view.display_error(f"Error displaying status: {e}")
    
    async def _auto_save(self) -> None:
        """Automatically save character state on exit."""
        if not self.current_character:
            return
        
        try:
            await self._save_character_state()
            self.console.print("[dim]Character state auto-saved[/dim]")
        except Exception as e:
            self.console.print(f"[yellow]Warning: Auto-save failed: {e}[/yellow]")


async def main():
    """Main entry point for the chat interface."""
    interface = ChatInterface()
    await interface.run()


if __name__ == "__main__":
    asyncio.run(main())