"""Interactive chat interface for character agent system."""

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.text import Text
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from config.character_loader import CharacterLoader, CharacterConfigurationError
    from config.settings import get_settings
    from character_agent import CharacterAgent
    from cli.debug_view import DebugView
except ImportError:
    # Fallback imports for when running from different contexts
    from ..config.character_loader import CharacterLoader, CharacterConfigurationError
    from ..config.settings import get_settings
    from ..character_agent import CharacterAgent
    from .debug_view import DebugView


class ChatInterface:
    """Interactive CLI for chatting with character agents."""
    
    def __init__(self):
        """Initialize the chat interface."""
        self.console = Console()
        self.debug_view = DebugView(self.console)
        self.settings = get_settings()
        self.character_loader = CharacterLoader()
        self.current_character: Optional[CharacterAgent] = None
        self.debug_mode = False
        self.save_dir = Path("./data/saves")
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    async def run(self) -> None:
        """Main entry point for the chat interface."""
        try:
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
            
            # Main conversation loop
            await self._conversation_loop()
            
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Chat interrupted by user. Goodbye![/yellow]")
        except Exception as e:
            self.debug_view.display_error(f"Unexpected error: {e}")
        finally:
            if self.current_character:
                await self._auto_save()
    
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
                character_config=config,
                settings=self.settings
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
    
    async def _conversation_loop(self) -> None:
        """Main conversation loop."""
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
                    response = result.get('response', 'No response generated')
                    
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