"""Enhanced command handler for chat interface."""

import json
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.prompt import Confirm


class CommandHandler:
    """Handles special commands during chat sessions."""
    
    COMMANDS = {
        '/help': 'Show available commands',
        '/debug': 'Toggle debug mode',
        '/vitals': 'Show detailed vitals',
        '/memory': 'Show recent memories',
        '/goals': 'Show all goals', 
        '/mood': 'Show mood history',
        '/save': 'Save character state',
        '/load': 'Load character state',
        '/reset': 'Reset character state',
        '/export': 'Export conversation',
        '/clear': 'Clear conversation history',
        '/scroll': 'Scroll conversation (up/down/top/bottom)',
        '/autoscroll': 'Toggle auto-scroll mode',
        '/pane': 'Switch active pane or show status',
        '/status': 'Show character status',
        '/exit': 'Exit chat'
    }
    
    def __init__(self, chat_interface):
        """Initialize command handler.
        
        Args:
            chat_interface: ChatInterface instance for accessing character and methods
        """
        self.interface = chat_interface
        self.console = chat_interface.console if hasattr(chat_interface, 'console') else Console()
        
    async def handle(self, command: str) -> Optional[str]:
        """Handle command and return result or 'exit' if should exit.
        
        Args:
            command: Command string starting with '/'
            
        Returns:
            'exit' if should exit, None otherwise, or result string
        """
        parts = command.strip().split(' ', 1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        handlers = {
            '/help': self._handle_help,
            '/debug': self._handle_debug,
            '/vitals': self._handle_vitals,
            '/memory': self._handle_memory,
            '/goals': self._handle_goals,
            '/mood': self._handle_mood,
            '/save': self._handle_save,
            '/load': self._handle_load,
            '/reset': self._handle_reset,
            '/export': self._handle_export,
            '/clear': self._handle_clear,
            '/scroll': self._handle_scroll,
            '/autoscroll': self._handle_autoscroll,
            '/pane': self._handle_pane,
            '/status': self._handle_status,
            '/exit': self._handle_exit
        }
        
        handler = handlers.get(cmd)
        if handler:
            try:
                return await handler(args)
            except Exception as e:
                self.console.print(f"[red]Error executing command {cmd}: {e}[/red]")
                return None
        else:
            self.console.print(f"[red]Unknown command: {cmd}[/red]")
            self.console.print("Type /help to see available commands.")
            return None
    
    async def _handle_help(self, args: str) -> None:
        """Show help information."""
        help_content = Table(show_header=False, box=None, padding=(0, 1))
        help_content.add_column("Command", style="cyan", width=18)
        help_content.add_column("Description", style="white")
        
        for cmd, desc in self.COMMANDS.items():
            help_content.add_row(cmd, desc)
        
        # Add usage examples
        help_content.add_row("", "")
        help_content.add_row("[bold]Examples:[/bold]", "")
        help_content.add_row("/memory recent", "Search for recent memories")
        help_content.add_row("/save my_session", "Save with custom name")
        help_content.add_row("/export today", "Export with custom filename")
        
        panel = Panel(
            help_content,
            title="[bold]Available Commands[/bold]",
            border_style="blue",
            padding=(1, 2)
        )
        
        self.console.print(panel)
        return None
    
    async def _handle_debug(self, args: str) -> None:
        """Toggle debug mode."""
        if hasattr(self.interface, 'debug_mode'):
            self.interface.debug_mode = not self.interface.debug_mode
            status = "ON" if self.interface.debug_mode else "OFF"
            self.console.print(f"[yellow]Debug mode: {status}[/yellow]")
        else:
            self.console.print("[red]Debug mode not available in this interface[/red]")
        return None
    
    async def _handle_vitals(self, args: str) -> None:
        """Show detailed vitals in full screen."""
        if not hasattr(self.interface, 'current_character') or not self.interface.current_character:
            self.console.print("[red]No active character[/red]")
            return None
        
        try:
            # Get the vitals display component
            if hasattr(self.interface, 'vitals_display'):
                detailed_panel = self.interface.vitals_display.render_detailed_vitals(
                    self.interface.current_character.state
                )
                self.console.print(detailed_panel)
            else:
                # Fallback to basic state display
                await self._display_basic_vitals()
        except Exception as e:
            self.console.print(f"[red]Error displaying vitals: {e}[/red]")
        
        return None
    
    async def _display_basic_vitals(self) -> None:
        """Display basic vitals information as fallback."""
        state = self.interface.current_character.state
        
        content = Text()
        content.append("CHARACTER VITALS\n\n", style="bold cyan")
        
        # Neurochemicals
        content.append("Neurochemical Levels:\n", style="bold yellow")
        for hormone, level in state.neurochemical_levels.items():
            content.append(f"  {hormone.capitalize()}: {level:.1f}\n", style="white")
        
        # Mood
        mood_state = state.agent_states.get('mood', {})
        content.append(f"\nMood: {mood_state.get('current_state', 'unknown')}\n", style="bold yellow")
        content.append(f"Intensity: {mood_state.get('intensity', 0):.2f}\n", style="white")
        
        panel = Panel(content, title="[bold]Character Vitals[/bold]", border_style="cyan")
        self.console.print(panel)
    
    async def _handle_memory(self, args: str) -> None:
        """Show recent memories."""
        if not hasattr(self.interface, 'current_character') or not self.interface.current_character:
            self.console.print("[red]No active character[/red]")
            return None
        
        query = args.strip() or "recent experiences"
        
        try:
            # Try to use the character's memory recall method
            if hasattr(self.interface.current_character, 'recall_past_conversations'):
                memories = await self.interface.current_character.recall_past_conversations(
                    query=query,
                    limit=5
                )
                
                if not memories:
                    self.console.print(f"[yellow]No memories found for query: '{query}'[/yellow]")
                    return None
                
                self.console.print(f"\n[bold]Found {len(memories)} relevant memories for '{query}':[/bold]")
                
                for i, memory in enumerate(memories, 1):
                    timestamp = memory.get('timestamp', 'Unknown time')
                    description = memory.get('description', 'No description')
                    similarity = memory.get('similarity', 0.0)
                    
                    sim_color = "green" if similarity > 0.8 else "yellow" if similarity > 0.6 else "red"
                    
                    self.console.print(f"  {i}. [{sim_color}]{similarity:.3f}[/{sim_color}] ", end="")
                    self.console.print(f"[dim]{timestamp}[/dim]")
                    self.console.print(f"     {description[:200]}{'...' if len(description) > 200 else ''}")
            else:
                # Fallback to conversation history
                history = self.interface.current_character.state.conversation_history
                self.console.print(f"[bold]Recent Conversation History ({len(history)} messages):[/bold]")
                
                for i, msg in enumerate(history[-5:], 1):
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')[:100]
                    self.console.print(f"  {i}. [{role.upper()}] {content}...")
                
        except Exception as e:
            self.console.print(f"[red]Error retrieving memories: {e}[/red]")
        
        return None
    
    async def _handle_goals(self, args: str) -> None:
        """Show all character goals."""
        if not hasattr(self.interface, 'current_character') or not self.interface.current_character:
            self.console.print("[red]No active character[/red]")
            return None
        
        try:
            goals_state = self.interface.current_character.state.agent_states.get('goals', {})
            active_goals = goals_state.get('active_goals', [])
            
            if not active_goals:
                self.console.print("[yellow]No active goals found[/yellow]")
                return None
            
            content = Table(show_header=True, header_style="bold magenta")
            content.add_column("Priority", style="red", width=8)
            content.add_column("Goal", style="white", width=50)
            content.add_column("Status", style="green", width=12)
            
            for goal in active_goals:
                if isinstance(goal, dict):
                    priority = goal.get('priority', 5)
                    description = goal.get('description', goal.get('goal', 'Unknown goal'))
                    status = goal.get('status', 'active')
                else:
                    priority = 5
                    description = str(goal)
                    status = 'active'
                
                content.add_row(str(priority), description, status)
            
            panel = Panel(
                content,
                title="[bold]Character Goals[/bold]",
                border_style="yellow"
            )
            
            self.console.print(panel)
            
        except Exception as e:
            self.console.print(f"[red]Error displaying goals: {e}[/red]")
        
        return None
    
    async def _handle_mood(self, args: str) -> None:
        """Show mood history and current state."""
        if not hasattr(self.interface, 'current_character') or not self.interface.current_character:
            self.console.print("[red]No active character[/red]")
            return None
        
        try:
            mood_state = self.interface.current_character.state.agent_states.get('mood', {})
            
            content = Text()
            content.append("MOOD STATUS\n\n", style="bold cyan")
            
            # Current mood
            current = mood_state.get('current_state', 'neutral')
            intensity = mood_state.get('intensity', 0.5)
            energy = mood_state.get('energy_level', 0.5)
            volatility = mood_state.get('emotional_volatility', 0.5)
            duration = mood_state.get('duration', 1)
            
            content.append(f"Current State: {current.title()}\n", style="bold white")
            content.append(f"Intensity: {intensity:.2f}/1.0\n", style="yellow")
            content.append(f"Energy Level: {energy:.2f}/1.0\n", style="green")
            content.append(f"Volatility: {volatility:.2f}/1.0\n", style="red")
            content.append(f"Duration: {duration} interactions\n", style="cyan")
            
            # Mood influences
            content.append(f"\nTriggered by: {mood_state.get('triggered_by', 'unknown')}\n", style="dim")
            
            panel = Panel(
                content,
                title="[bold]Mood Analysis[/bold]",
                border_style="blue"
            )
            
            self.console.print(panel)
            
        except Exception as e:
            self.console.print(f"[red]Error displaying mood: {e}[/red]")
        
        return None
    
    async def _handle_save(self, args: str) -> None:
        """Save character state."""
        if not hasattr(self.interface, 'current_character') or not self.interface.current_character:
            self.console.print("[red]No active character[/red]")
            return None
        
        # Use the interface's save method if available
        if hasattr(self.interface, '_save_character_state'):
            await self.interface._save_character_state(args)
        else:
            # Fallback implementation
            filename = args.strip() or f"{self.interface.current_character.character_id}_manual_save.json"
            if not filename.endswith('.json'):
                filename += '.json'
            
            save_dir = Path("./data/saves")
            save_dir.mkdir(parents=True, exist_ok=True)
            filepath = save_dir / filename
            
            try:
                state_data = self.interface.current_character.get_state_dict()
                with open(filepath, 'w') as f:
                    json.dump(state_data, f, indent=2, default=str)
                
                self.console.print(f"[green]Character state saved to {filepath}[/green]")
            except Exception as e:
                self.console.print(f"[red]Failed to save: {e}[/red]")
        
        return None
    
    async def _handle_load(self, args: str) -> None:
        """Load character state."""
        if not hasattr(self.interface, 'current_character') or not self.interface.current_character:
            self.console.print("[red]No active character[/red]")
            return None
        
        # Use the interface's load method if available
        if hasattr(self.interface, '_load_character_state'):
            await self.interface._load_character_state(args)
        else:
            self.console.print("[yellow]Load functionality not available in this interface[/yellow]")
        
        return None
    
    async def _handle_reset(self, args: str) -> None:
        """Reset character state."""
        if not hasattr(self.interface, 'current_character') or not self.interface.current_character:
            self.console.print("[red]No active character[/red]")
            return None
        
        # Use the interface's reset method if available
        if hasattr(self.interface, '_reset_character'):
            await self.interface._reset_character()
        else:
            if Confirm.ask("[red]This will reset the character to initial state. Continue?[/red]"):
                self.console.print("[yellow]Reset functionality not available in this interface[/yellow]")
        
        return None
    
    async def _handle_export(self, args: str) -> None:
        """Export conversation log."""
        if not hasattr(self.interface, 'current_character') or not self.interface.current_character:
            self.console.print("[red]No active character[/red]")
            return None
        
        try:
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = args.strip() or f"conversation_{timestamp}.json"
            if not filename.endswith('.json'):
                filename += '.json'
            
            export_dir = Path("./data/exports")
            export_dir.mkdir(parents=True, exist_ok=True)
            filepath = export_dir / filename
            
            # Gather export data
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'character_id': self.interface.current_character.character_id,
                'character_name': self.interface.current_character.character_name,
                'conversation_history': self.interface.current_character.state.conversation_history,
                'neurochemical_levels': self.interface.current_character.state.neurochemical_levels,
                'mood_state': self.interface.current_character.state.agent_states.get('mood', {}),
                'relationship_state': self.interface.current_character.state.relationship_state
            }
            
            # Save export
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.console.print(f"[green]Conversation exported to {filepath}[/green]")
            
        except Exception as e:
            self.console.print(f"[red]Export failed: {e}[/red]")
        
        return None
    
    async def _handle_clear(self, args: str) -> None:
        """Clear conversation history."""
        if not hasattr(self.interface, 'current_character') or not self.interface.current_character:
            self.console.print("[red]No active character[/red]")
            return None
        
        if Confirm.ask("[yellow]Clear conversation history? This cannot be undone.[/yellow]"):
            try:
                # Clear conversation history
                self.interface.current_character.state.conversation_history = []
                
                # Clear interface messages if available
                if hasattr(self.interface, 'messages'):
                    self.interface.messages = []
                
                # Update layout if available
                if hasattr(self.interface, 'layout_manager'):
                    self.interface.layout_manager.update_chat_history([])
                
                self.console.print("[green]Conversation history cleared[/green]")
                
            except Exception as e:
                self.console.print(f"[red]Failed to clear history: {e}[/red]")
        
        return None
    
    async def _handle_status(self, args: str) -> None:
        """Show character status."""
        if not hasattr(self.interface, 'current_character') or not self.interface.current_character:
            self.console.print("[red]No active character[/red]")
            return None
        
        # Use the interface's status method if available
        if hasattr(self.interface, '_display_character_status'):
            await self.interface._display_character_status()
        else:
            # Fallback status display
            character = self.interface.current_character
            state = character.state
            
            content = Text()
            content.append(f"Character: {character.character_name}\n", style="bold cyan")
            content.append(f"ID: {character.character_id}\n", style="white")
            content.append(f"Messages: {len(state.conversation_history)}\n", style="yellow")
            content.append(f"Trust Level: {state.relationship_state.get('trust_level', 0):.2f}\n", style="green")
            
            panel = Panel(content, title="[bold]Status[/bold]", border_style="blue")
            self.console.print(panel)
        
        return None
    
    async def _handle_scroll(self, args: str) -> None:
        """Handle scroll commands."""
        if not hasattr(self.interface, 'layout_manager'):
            self.console.print("[red]Scrolling not available in this interface[/red]")
            return None
        
        direction = args.strip().lower()
        layout_manager = self.interface.layout_manager
        
        if direction in ['up', 'u']:
            if layout_manager.scroll_up():
                scroll_info = layout_manager.get_scroll_info()
                self.console.print(f"[dim]Scrolled up - showing messages {scroll_info['visible_messages']} of {scroll_info['total_messages']}[/dim]")
            else:
                self.console.print("[yellow]Already at top of conversation[/yellow]")
        
        elif direction in ['down', 'd']:
            if layout_manager.scroll_down():
                scroll_info = layout_manager.get_scroll_info()
                self.console.print(f"[dim]Scrolled down - showing messages {scroll_info['visible_messages']} of {scroll_info['total_messages']}[/dim]")
            else:
                self.console.print("[yellow]Already at bottom of conversation[/yellow]")
        
        elif direction in ['top', 'home', 'beginning']:
            layout_manager.scroll_to_top()
            self.console.print("[dim]Scrolled to top of conversation[/dim]")
        
        elif direction in ['bottom', 'end', 'latest']:
            layout_manager.scroll_to_bottom()
            self.console.print("[dim]Scrolled to bottom of conversation[/dim]")
        
        else:
            self.console.print("[red]Invalid scroll direction. Use: up, down, top, bottom[/red]")
        
        return None
    
    async def _handle_autoscroll(self, args: str) -> None:
        """Handle auto-scroll toggle."""
        if not hasattr(self.interface, 'layout_manager'):
            self.console.print("[red]Auto-scroll not available in this interface[/red]")
            return None
        
        layout_manager = self.interface.layout_manager
        new_state = layout_manager.toggle_auto_scroll()
        status = "enabled" if new_state else "disabled"
        self.console.print(f"[yellow]Auto-scroll {status}[/yellow]")
        
        if new_state:
            self.console.print("[dim]New messages will automatically scroll to bottom[/dim]")
        else:
            self.console.print("[dim]Manual scrolling mode - use /scroll commands to navigate[/dim]")
        
        return None
    
    async def _handle_pane(self, args: str) -> None:
        """Handle pane switching and status commands."""
        if not hasattr(self.interface, 'pane_manager'):
            self.console.print("[red]Pane management not available in this interface[/red]")
            return None
        
        pane_manager = self.interface.pane_manager
        direction = args.strip().lower()
        
        if direction in ['next', 'n', '']:
            # Cycle to next pane
            old_pane = pane_manager.get_active_pane()
            new_pane = pane_manager.cycle_active_pane(1)
            self.console.print(f"[green]Switched from {old_pane.value} to {new_pane.value} pane[/green]")
        
        elif direction in ['prev', 'previous', 'p']:
            # Cycle to previous pane
            old_pane = pane_manager.get_active_pane()
            new_pane = pane_manager.cycle_active_pane(-1)
            self.console.print(f"[green]Switched from {old_pane.value} to {new_pane.value} pane[/green]")
        
        elif direction in ['conversation', 'conv', 'c']:
            from cli.pane_manager import PaneType
            if pane_manager.set_active_pane(PaneType.CONVERSATION):
                self.console.print("[green]Switched to conversation pane[/green]")
        
        elif direction in ['vitals', 'v']:
            from cli.pane_manager import PaneType
            if pane_manager.set_active_pane(PaneType.VITALS):
                self.console.print("[green]Switched to vitals pane[/green]")
        
        elif direction in ['input', 'i']:
            from cli.pane_manager import PaneType
            if pane_manager.set_active_pane(PaneType.INPUT):
                self.console.print("[green]Switched to input pane[/green]")
        
        elif direction in ['status', 's']:
            # Show pane status
            status = pane_manager.get_status_info()
            active_pane = pane_manager.get_active_pane()
            self.console.print(f"[cyan]Active pane: {active_pane.value}[/cyan]")
            if status:
                self.console.print(f"[dim]{status}[/dim]")
        
        else:
            self.console.print("[red]Invalid pane command. Use: next/prev/conversation/vitals/input/status[/red]")
        
        return None
    
    async def _handle_exit(self, args: str) -> str:
        """Handle exit command."""
        if Confirm.ask("[yellow]Are you sure you want to exit?[/yellow]"):
            return 'exit'
        return None