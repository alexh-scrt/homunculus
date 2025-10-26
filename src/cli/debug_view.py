"""Debug information display for character agent interactions."""

from typing import Dict, Any, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich.progress import Progress, BarColumn, TextColumn
import json


class DebugView:
    """Handles display of debug information during character interactions."""
    
    def __init__(self, console: Optional[Console] = None):
        """Initialize debug view.
        
        Args:
            console: Rich console instance. If None, creates a new one.
        """
        self.console = console or Console()
    
    def display_debug_info(self, debug_data: Dict[str, Any]) -> None:
        """Display comprehensive debug information.
        
        Args:
            debug_data: Dictionary containing debug information
        """
        self.console.print("\n[bold yellow]═══ DEBUG INFO ═══[/bold yellow]")
        
        # Display sections
        if 'neurochemical_levels' in debug_data:
            self._display_neurochemical_levels(debug_data['neurochemical_levels'])
        
        if 'mood' in debug_data:
            self._display_mood_state(debug_data['mood'])
        
        if 'agent_inputs' in debug_data:
            self._display_agent_inputs(debug_data['agent_inputs'])
        
        if 'memory_retrieval' in debug_data:
            self._display_memory_retrieval(debug_data['memory_retrieval'])
        
        if 'cognitive_processing' in debug_data:
            self._display_cognitive_processing(debug_data['cognitive_processing'])
        
        if 'response_generation' in debug_data:
            self._display_response_generation(debug_data['response_generation'])
        
        if 'state_changes' in debug_data:
            self._display_state_changes(debug_data['state_changes'])
        
        self.console.print("[bold yellow]═══ END DEBUG ═══[/bold yellow]\n")
    
    def _display_neurochemical_levels(self, neuro_data: Dict[str, Any]) -> None:
        """Display neurochemical hormone levels."""
        table = Table(title="Neurochemical Levels", show_header=True, header_style="bold magenta")
        table.add_column("Hormone", style="cyan", width=12)
        table.add_column("Current", style="green", width=8)
        table.add_column("Baseline", style="yellow", width=8)
        table.add_column("Change", style="white", width=8)
        table.add_column("Visual", width=20)
        
        levels = neuro_data.get('current_levels', {})
        baselines = neuro_data.get('baseline_levels', {})
        changes = neuro_data.get('recent_changes', {})
        
        for hormone in ['dopamine', 'serotonin', 'oxytocin', 'endorphins', 'cortisol', 'adrenaline']:
            current = levels.get(hormone, 0)
            baseline = baselines.get(hormone, 50)
            change = changes.get(hormone, 0)
            
            # Color based on level
            if current > 70:
                level_color = "red"
            elif current > 50:
                level_color = "yellow" 
            else:
                level_color = "green"
            
            # Change indicator
            if change > 0:
                change_str = f"[green]+{change:.1f}[/green]"
            elif change < 0:
                change_str = f"[red]{change:.1f}[/red]"
            else:
                change_str = "[white]0.0[/white]"
            
            # Visual bar
            bar_length = int(current / 5)  # Scale to 20 chars max
            bar = "█" * bar_length + "░" * (20 - bar_length)
            visual = f"[{level_color}]{bar}[/{level_color}]"
            
            table.add_row(
                hormone.capitalize(),
                f"[{level_color}]{current:.1f}[/{level_color}]",
                f"{baseline:.1f}",
                change_str,
                visual
            )
        
        self.console.print(table)
    
    def _display_mood_state(self, mood_data: Dict[str, Any]) -> None:
        """Display current mood and emotional state."""
        current_state = mood_data.get('current_state', 'neutral')
        intensity = mood_data.get('intensity', 0.5)
        volatility = mood_data.get('emotional_volatility', 0.5)
        energy = mood_data.get('energy_level', 0.5)
        
        # Create mood panel
        mood_text = Text()
        mood_text.append(f"State: ", style="bold")
        mood_text.append(f"{current_state.title()}", style="cyan")
        mood_text.append(f"\nIntensity: ", style="bold")
        mood_text.append(f"{intensity:.2f}", style="green" if intensity > 0.7 else "yellow" if intensity > 0.4 else "red")
        mood_text.append(f"\nVolatility: ", style="bold")
        mood_text.append(f"{volatility:.2f}", style="red" if volatility > 0.7 else "yellow" if volatility > 0.4 else "green")
        mood_text.append(f"\nEnergy: ", style="bold")
        mood_text.append(f"{energy:.2f}", style="green" if energy > 0.7 else "yellow" if energy > 0.4 else "red")
        
        panel = Panel(mood_text, title="[bold]Mood State[/bold]", border_style="blue")
        self.console.print(panel)
    
    def _display_agent_inputs(self, agent_data: Dict[str, Any]) -> None:
        """Display inputs from all specialized agents."""
        self.console.print("\n[bold]Agent Consultation Results:[/bold]")
        
        for agent_type, agent_input in agent_data.items():
            content = agent_input.get('content', '')
            confidence = agent_input.get('confidence', 0.0)
            priority = agent_input.get('priority', 5)
            
            # Truncate long content
            display_content = content[:150] + "..." if len(content) > 150 else content
            
            # Style based on confidence
            conf_color = "green" if confidence > 0.7 else "yellow" if confidence > 0.5 else "red"
            
            self.console.print(f"  [{agent_type.upper()}] ", style="bold cyan", end="")
            self.console.print(f"(conf: [{conf_color}]{confidence:.2f}[/{conf_color}], ", end="")
            self.console.print(f"pri: {priority}) ", end="")
            self.console.print(display_content, style="white")
    
    def _display_memory_retrieval(self, memory_data: Dict[str, Any]) -> None:
        """Display memory retrieval results."""
        memories = memory_data.get('retrieved_memories', [])
        query = memory_data.get('query', 'N/A')
        
        if not memories:
            self.console.print("\n[dim]No relevant memories retrieved[/dim]")
            return
        
        self.console.print(f"\n[bold]Memory Retrieval[/bold] (query: [cyan]{query}[/cyan]):")
        
        for i, memory in enumerate(memories[:5], 1):  # Show top 5
            similarity = memory.get('similarity', 0.0)
            description = memory.get('description', '')[:100]
            timestamp = memory.get('timestamp', 'Unknown')
            
            sim_color = "green" if similarity > 0.8 else "yellow" if similarity > 0.6 else "red"
            
            self.console.print(f"  {i}. [{sim_color}]{similarity:.3f}[/{sim_color}] ", end="")
            self.console.print(f"[dim]{timestamp}[/dim] - {description}...")
    
    def _display_cognitive_processing(self, cognitive_data: Dict[str, Any]) -> None:
        """Display cognitive module synthesis results."""
        synthesis = cognitive_data.get('synthesis', {})
        conflicts = cognitive_data.get('conflicts', [])
        
        if synthesis:
            panel_content = Text()
            panel_content.append("Primary Intention: ", style="bold")
            panel_content.append(f"{synthesis.get('primary_intention', 'N/A')}\n", style="cyan")
            panel_content.append("Emotional Tone: ", style="bold")
            panel_content.append(f"{synthesis.get('emotional_tone', 'N/A')}\n", style="green")
            panel_content.append("Key Themes: ", style="bold")
            panel_content.append(f"{', '.join(synthesis.get('key_themes', []))}", style="yellow")
            
            panel = Panel(panel_content, title="[bold]Cognitive Synthesis[/bold]", border_style="green")
            self.console.print(panel)
        
        if conflicts:
            self.console.print(f"\n[bold red]Agent Conflicts Detected:[/bold red]")
            for conflict in conflicts:
                self.console.print(f"  • {conflict}", style="red")
    
    def _display_response_generation(self, response_data: Dict[str, Any]) -> None:
        """Display response generation metadata."""
        metadata = response_data.get('metadata', {})
        
        if not metadata:
            return
        
        table = Table(title="Response Generation", show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        if 'response_length' in metadata:
            table.add_row("Response Length", f"{metadata['response_length']} chars")
        
        if 'generation_time' in metadata:
            table.add_row("Generation Time", f"{metadata['generation_time']:.2f}s")
        
        if 'style_applied' in metadata:
            table.add_row("Style Applied", metadata['style_applied'])
        
        if 'personality_influence' in metadata:
            table.add_row("Personality Influence", f"{metadata['personality_influence']:.2f}")
        
        self.console.print(table)
    
    def _display_state_changes(self, state_data: Dict[str, Any]) -> None:
        """Display character state changes after interaction."""
        changes = state_data.get('changes', {})
        
        if not changes:
            self.console.print("\n[dim]No significant state changes[/dim]")
            return
        
        self.console.print(f"\n[bold]State Changes:[/bold]")
        
        for category, change_list in changes.items():
            if change_list:
                self.console.print(f"  [cyan]{category.title()}:[/cyan]")
                for change in change_list:
                    self.console.print(f"    • {change}", style="white")
    
    def display_character_selection(self, characters: list) -> None:
        """Display character selection menu.
        
        Args:
            characters: List of character info dictionaries
        """
        table = Table(title="Available Characters", show_header=True, header_style="bold magenta")
        table.add_column("ID", style="cyan", width=4)
        table.add_column("Name", style="green", width=20)
        table.add_column("Archetype", style="yellow", width=15)
        table.add_column("Description", style="white", width=50)
        
        for i, char in enumerate(characters, 1):
            name = char.get('name', 'Unknown')
            archetype = char.get('archetype', 'unknown')
            description = char.get('description', 'No description available')[:50]
            
            table.add_row(str(i), name, archetype, description)
        
        self.console.print(table)
    
    def display_message(self, speaker: str, message: str, style: str = "") -> None:
        """Display a conversation message.
        
        Args:
            speaker: Name of the speaker
            message: Message content
            style: Rich style for the panel border
        """
        panel = Panel(
            message,
            title=f"[bold]{speaker}[/bold]",
            border_style=style or "white"
        )
        self.console.print(panel)
    
    def display_help(self) -> None:
        """Display available commands and help information."""
        help_text = Text()
        help_text.append("Available Commands:\n\n", style="bold")
        help_text.append("/exit", style="cyan")
        help_text.append(" - Exit the chat\n")
        help_text.append("/debug", style="cyan") 
        help_text.append(" - Toggle debug mode on/off\n")
        help_text.append("/save", style="cyan")
        help_text.append(" - Save current character state\n")
        help_text.append("/load", style="cyan")
        help_text.append(" - Load saved character state\n")
        help_text.append("/memory [query]", style="cyan")
        help_text.append(" - Search character memories\n")
        help_text.append("/reset", style="cyan")
        help_text.append(" - Reset character to initial state\n")
        help_text.append("/help", style="cyan")
        help_text.append(" - Show this help message\n")
        
        panel = Panel(help_text, title="[bold]Help[/bold]", border_style="blue")
        self.console.print(panel)
    
    def display_error(self, message: str) -> None:
        """Display an error message.
        
        Args:
            message: Error message to display
        """
        panel = Panel(
            f"[red]{message}[/red]",
            title="[bold red]Error[/bold red]",
            border_style="red"
        )
        self.console.print(panel)
    
    def display_success(self, message: str) -> None:
        """Display a success message.
        
        Args:
            message: Success message to display  
        """
        panel = Panel(
            f"[green]{message}[/green]",
            title="[bold green]Success[/bold green]",
            border_style="green"
        )
        self.console.print(panel)