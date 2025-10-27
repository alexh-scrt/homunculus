"""Real-time vitals display for character agent interactions."""

import os
import shutil
from typing import Dict, Any, Optional, List, Tuple
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.console import Console


class VitalsDisplay:
    """Real-time display of character agent vitals."""
    
    def __init__(self, console: Optional[Console] = None):
        """Initialize vitals display.
        
        Args:
            console: Rich console instance. If None, creates a new one.
        """
        self.console = console or Console()
        self.terminal_size = self._get_terminal_size()
    
    def _get_terminal_size(self) -> Tuple[int, int]:
        """Get current terminal dimensions.
        
        Returns:
            Tuple of (width, height) of the terminal
        """
        try:
            # Try to get terminal size using shutil
            size = shutil.get_terminal_size()
            return size.columns, size.lines
        except (OSError, AttributeError):
            try:
                # Fallback to os.get_terminal_size
                size = os.get_terminal_size()
                return size.columns, size.lines
            except (OSError, AttributeError):
                # Default size if detection fails
                return 80, 24
    
    def render_vitals(self, character_state) -> Panel:
        """Render complete vitals panel.
        
        Args:
            character_state: CharacterState object with current state
            
        Returns:
            Rich Panel with formatted vitals display
        """
        content = self._build_vitals_content(character_state)
        return Panel(
            content,
            title="⚡ Agent Vitals",
            border_style="cyan",
            padding=(1, 2)
        )
    
    def _build_vitals_content(self, state) -> Table:
        """Build the complete vitals display.
        
        Args:
            state: CharacterState object
            
        Returns:
            Rich Table with formatted vitals content
        """
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Section", style="bold")
        
        # Neurochemicals section
        table.add_row(self._render_neurochemicals(state.neurochemical_levels))
        table.add_row("")  # Spacer
        
        # Mood section
        table.add_row(self._render_mood(state.agent_states.get('mood', {})))
        table.add_row("")
        
        # Goals section
        table.add_row(self._render_goals(state.agent_states.get('goals', {})))
        table.add_row("")
        
        # Knowledge section
        table.add_row(self._render_knowledge(state))
        table.add_row("")
        
        # Memory section
        table.add_row(self._render_memory(state))
        table.add_row("")
        
        # Relationship section
        table.add_row(self._render_relationship(state.relationship_state))
        
        return table
    
    def _render_neurochemicals(self, levels: Dict[str, float]) -> Text:
        """Render neurochemical levels with color-coded bars.
        
        Args:
            levels: Dictionary of hormone name to level (0-100)
            
        Returns:
            Rich Text object with formatted neurochemical display
        """
        text = Text("🧪 NEUROCHEMICALS\n", style="bold yellow")
        
        # Update terminal size
        self.terminal_size = self._get_terminal_size()
        width, height = self.terminal_size
        
        # Calculate optimal bar length based on terminal width
        # Estimate vitals panel width (roughly 40% of terminal width)
        vitals_width = int(width * 0.4)
        
        # Account for hormone name, spacing, and level numbers (roughly 25 characters)
        available_bar_width = max(6, vitals_width - 25)
        bar_length = min(available_bar_width, 15)  # Cap at 15 for readability
        
        # Order of hormones to display
        hormone_order = ['dopamine', 'serotonin', 'oxytocin', 'endorphins', 'cortisol', 'adrenaline']
        
        for hormone in hormone_order:
            level = levels.get(hormone, 50.0)
            
            # Color coding based on level
            if level > 70:
                color = "red"
            elif level > 50:
                color = "yellow"
            else:
                color = "green"
            
            # Create bar visualization with responsive length
            filled = int(level * bar_length / 100)
            bar = "█" * filled + "░" * (bar_length - filled)
            
            # Format hormone name (12 chars wide for consistency)
            hormone_name = hormone.capitalize()[:12].ljust(12)
            
            text.append(f"{hormone_name} ", style="white")
            text.append(f"{bar} ", style=color)
            text.append(f"{level:.0f}/100\n", style=color)
        
        return text
    
    def _render_mood(self, mood_state: Dict[str, Any]) -> Text:
        """Render current mood state.
        
        Args:
            mood_state: Dictionary with mood information
            
        Returns:
            Rich Text object with formatted mood display
        """
        text = Text("😊 MOOD STATE\n", style="bold yellow")
        
        current_mood = mood_state.get('current_state', 'neutral')
        intensity = mood_state.get('intensity', 0.5)
        energy = mood_state.get('energy_level', 0.5)
        
        # Emoji mapping for moods
        mood_emoji = {
            'happy': '😊', 'sad': '😢', 'angry': '😠',
            'anxious': '😰', 'excited': '🤩', 'calm': '😌',
            'frustrated': '😤', 'content': '😌', 'neutral': '😐',
            'curious': '🤔', 'surprised': '😲', 'confused': '😕',
            'proud': '😌', 'embarrassed': '😳', 'bored': '😴'
        }
        
        emoji = mood_emoji.get(current_mood.lower(), '😐')
        
        text.append(f"Current: {emoji} {current_mood.capitalize()}\n", style="white")
        text.append(f"Intensity: {intensity:.2f}\n", style="cyan")
        text.append(f"Energy: {energy:.2f}\n", style="magenta")
        
        return text
    
    def _render_goals(self, goals_state: Dict[str, Any]) -> Text:
        """Render active goals.
        
        Args:
            goals_state: Dictionary with goals information
            
        Returns:
            Rich Text object with formatted goals display
        """
        text = Text("🎯 ACTIVE GOALS\n", style="bold yellow")
        
        active_goals = goals_state.get('active_goals', [])
        
        if not active_goals:
            text.append("No active goals\n", style="dim")
        else:
            # Show top 3 goals
            for goal in active_goals[:3]:
                if isinstance(goal, dict):
                    priority = goal.get('priority', 5)
                    description = goal.get('description', goal.get('goal', 'Unknown goal'))
                else:
                    # Handle case where goals might be strings
                    priority = 5
                    description = str(goal)
                
                # Truncate long descriptions
                if len(description) > 30:
                    description = description[:27] + "..."
                
                text.append(f"• {description} (P:{priority})\n", style="white")
        
        return text
    
    def _render_knowledge(self, state) -> Text:
        """Render knowledge statistics.
        
        Args:
            state: CharacterState object
            
        Returns:
            Rich Text object with formatted knowledge display
        """
        text = Text("🧠 KNOWLEDGE\n", style="bold yellow")
        
        # Get knowledge stats from the state
        # These would come from the knowledge graph in a real implementation
        knowledge_updates = getattr(state, 'knowledge_updates', [])
        facts_count = len([k for k in knowledge_updates if k.get('type') == 'fact'])
        entities_count = len([k for k in knowledge_updates if k.get('type') == 'entity'])
        relationships_count = len([k for k in knowledge_updates if k.get('type') == 'relationship'])
        
        # If no knowledge updates yet, show default values
        if not knowledge_updates:
            facts_count = 0
            entities_count = 1  # At least knows about the user
            relationships_count = 1  # Relationship with user
        
        text.append(f"Facts learned: {facts_count}\n", style="white")
        text.append(f"Entities known: {entities_count}\n", style="white")
        text.append(f"Relationships: {relationships_count}\n", style="white")
        
        return text
    
    def _render_memory(self, state) -> Text:
        """Render memory statistics.
        
        Args:
            state: CharacterState object
            
        Returns:
            Rich Text object with formatted memory display
        """
        text = Text("💾 MEMORY\n", style="bold yellow")
        
        # Get conversation history length as episodic memory count
        conversation_history = getattr(state, 'conversation_history', [])
        episodic_count = len(conversation_history)
        
        # Get web search history count
        web_search_history = getattr(state, 'web_search_history', [])
        web_search_count = len(web_search_history)
        
        text.append(f"Episodic memories: {episodic_count}\n", style="white")
        text.append(f"Web searches: {web_search_count}\n", style="cyan")
        
        # Show most recent memory topic
        if conversation_history:
            # Get the most recent user message
            recent_messages = [msg for msg in conversation_history[-3:] if msg.get('role') == 'user']
            if recent_messages:
                recent_content = recent_messages[-1].get('content', 'general conversation')
                # Extract a topic from the content (first few words)
                topic_words = recent_content.split()[:4]
                topic = ' '.join(topic_words)
                if len(topic) > 25:
                    topic = topic[:22] + "..."
                text.append(f"Recent: \"{topic}\"\n", style="dim")
            else:
                text.append(f"Recent: \"conversation started\"\n", style="dim")
        else:
            text.append(f"Recent: \"no memories yet\"\n", style="dim")
        
        return text
    
    def _render_relationship(self, rel_state: Dict[str, Any]) -> Text:
        """Render relationship metrics.
        
        Args:
            rel_state: Dictionary with relationship information
            
        Returns:
            Rich Text object with formatted relationship display
        """
        text = Text("🤝 RELATIONSHIP\n", style="bold yellow")
        
        trust = rel_state.get('trust_level', 0.5)
        interactions = rel_state.get('interaction_count', 0)
        
        # Calculate duration based on interaction count (rough estimate)
        duration_minutes = interactions * 2  # Assume ~2 minutes per interaction
        
        # Color code trust level
        if trust > 0.7:
            trust_color = "green"
        elif trust > 0.4:
            trust_color = "yellow"
        else:
            trust_color = "red"
        
        text.append(f"Trust level: ", style="white")
        text.append(f"{trust:.2f}\n", style=trust_color)
        text.append(f"Interactions: {interactions}\n", style="white")
        text.append(f"Duration: {duration_minutes:.0f} min\n", style="white")
        
        return text
    
    def render_detailed_vitals(self, character_state) -> Panel:
        """Render detailed vitals view for full-screen display.
        
        Args:
            character_state: CharacterState object with current state
            
        Returns:
            Rich Panel with detailed vitals information
        """
        # Create a more detailed version for the /vitals command
        content = Text()
        
        content.append("═══ DETAILED CHARACTER VITALS ═══\n\n", style="bold cyan")
        
        # Detailed neurochemicals
        content.append("🧪 NEUROCHEMICAL SYSTEM\n", style="bold yellow")
        content.append("─────────────────────────\n", style="dim")
        
        levels = character_state.neurochemical_levels
        baselines = getattr(character_state, 'neurochemical_baselines', {})
        
        for hormone in ['dopamine', 'serotonin', 'oxytocin', 'endorphins', 'cortisol', 'adrenaline']:
            level = levels.get(hormone, 50.0)
            baseline = baselines.get(hormone, 50.0)
            deviation = level - baseline
            
            # Detailed bar (20 characters)
            filled = int(level / 5)
            bar = "█" * filled + "░" * (20 - filled)
            
            # Color based on level
            if level > 70:
                color = "red"
            elif level > 50:
                color = "yellow"
            else:
                color = "green"
            
            # Deviation indicator
            if deviation > 10:
                dev_indicator = "↑↑ (high)"
                dev_color = "red"
            elif deviation > 5:
                dev_indicator = "↑ (elevated)"
                dev_color = "yellow"
            elif deviation < -10:
                dev_indicator = "↓↓ (low)"
                dev_color = "blue"
            elif deviation < -5:
                dev_indicator = "↓ (reduced)"
                dev_color = "cyan"
            else:
                dev_indicator = "→ (normal)"
                dev_color = "green"
            
            content.append(f"{hormone.capitalize():12} ", style="white")
            content.append(f"{bar} ", style=color)
            content.append(f"{level:.1f}/100 ", style=color)
            content.append(f"[{dev_color}]{dev_indicator}[/{dev_color}]\n")
        
        content.append("\n")
        
        # Detailed mood information
        mood_state = character_state.agent_states.get('mood', {})
        content.append("😊 EMOTIONAL STATE\n", style="bold yellow")
        content.append("─────────────────\n", style="dim")
        content.append(f"Current Mood: {mood_state.get('current_state', 'neutral').title()}\n")
        content.append(f"Intensity: {mood_state.get('intensity', 0.5):.2f}/1.0\n")
        content.append(f"Volatility: {mood_state.get('emotional_volatility', 0.5):.2f}/1.0\n")
        content.append(f"Energy Level: {mood_state.get('energy_level', 0.5):.2f}/1.0\n")
        content.append(f"Duration: {mood_state.get('duration', 1)} interactions\n")
        
        return Panel(
            content,
            title="[bold]Detailed Character Vitals[/bold]",
            border_style="cyan",
            padding=(1, 2)
        )