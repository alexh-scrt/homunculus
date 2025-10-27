"""Layout manager for dual-pane chat interface."""

import os
import shutil
from typing import List, Dict, Any, Optional, Tuple
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.console import Console
from rich.table import Table


class LayoutManager:
    """Manages the dual-pane layout for chat interface with live vitals."""
    
    def __init__(self, console: Optional[Console] = None):
        """Initialize layout manager.
        
        Args:
            console: Rich console instance. If None, creates a new one.
        """
        self.console = console or Console()
        self.layout = Layout()
        self.messages: List[Dict[str, str]] = []
        self.terminal_size = self._get_terminal_size()
        self.scroll_offset = 0  # Number of messages to scroll up from bottom
        self.auto_scroll = True  # Whether to auto-scroll to bottom on new messages
        self._setup_layout()
    
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
    
    def _calculate_layout_ratios(self) -> Tuple[int, int, int, int]:
        """Calculate optimal layout ratios based on terminal size.
        
        Returns:
            Tuple of (chat_ratio, vitals_ratio, history_ratio, input_ratio)
        """
        width, height = self.terminal_size
        
        # Adjust ratios based on terminal size
        if width < 100:
            # Narrow terminal: prioritize chat more
            chat_ratio, vitals_ratio = 70, 30
        elif width < 120:
            # Medium terminal: balanced
            chat_ratio, vitals_ratio = 60, 40
        else:
            # Wide terminal: can afford more space for vitals
            chat_ratio, vitals_ratio = 55, 45
        
        # Adjust vertical split based on height
        if height < 20:
            # Short terminal: less space for input
            history_ratio, input_ratio = 90, 10
            input_min_size = 2
        elif height < 30:
            # Medium height: balanced
            history_ratio, input_ratio = 85, 15
            input_min_size = 3
        else:
            # Tall terminal: more space for input
            history_ratio, input_ratio = 80, 20
            input_min_size = 4
        
        return chat_ratio, vitals_ratio, history_ratio, input_ratio
    
    def _setup_layout(self) -> None:
        """Setup the dual-pane layout structure with responsive sizing."""
        # Get optimal ratios for current terminal size
        chat_ratio, vitals_ratio, history_ratio, input_ratio = self._calculate_layout_ratios()
        
        # Split into left (chat) and right (vitals) with calculated ratios
        self.layout.split_row(
            Layout(name="chat", ratio=chat_ratio),
            Layout(name="vitals", ratio=vitals_ratio)
        )
        
        # Calculate minimum input size based on terminal height
        _, height = self.terminal_size
        input_min_size = max(2, min(4, height // 8))
        
        # Split chat pane into history and input areas
        self.layout["chat"].split_column(
            Layout(name="history", ratio=history_ratio),
            Layout(name="input", ratio=input_ratio, minimum_size=input_min_size)
        )
        
        # Initialize with empty content
        self._update_chat_history()
        self._update_input_prompt("Starting chat...")
    
    def update_chat_history(self, messages: List[Dict[str, str]]) -> None:
        """Update chat history display.
        
        Args:
            messages: List of message dictionaries with 'speaker' and 'message' keys
        """
        old_count = len(self.messages)
        self.messages = messages
        
        # Auto-scroll to bottom if new messages were added and auto-scroll is enabled
        if self.auto_scroll and len(messages) > old_count:
            self.scroll_offset = 0
        
        self._update_chat_history()
    
    def add_message(self, speaker: str, message: str) -> None:
        """Add a single message to the chat history.
        
        Args:
            speaker: Name of the speaker
            message: Message content
        """
        self.messages.append({
            'speaker': speaker,
            'message': message
        })
        
        # Auto-scroll to bottom when new message is added
        if self.auto_scroll:
            self.scroll_offset = 0
            
        self._update_chat_history()
    
    def refresh_terminal_size(self) -> bool:
        """Refresh terminal size and return True if it changed.
        
        Returns:
            True if terminal size changed, False otherwise
        """
        new_size = self._get_terminal_size()
        if new_size != self.terminal_size:
            self.terminal_size = new_size
            return True
        return False
    
    def resize_layout(self) -> None:
        """Recreate layout with new terminal dimensions."""
        # Store current state
        current_messages = self.messages.copy()
        
        # Recreate layout with new dimensions
        self.layout = Layout()
        self._setup_layout()
        
        # Restore state
        self.messages = current_messages
        self._update_chat_history()
    
    def scroll_up(self, lines: int = 3) -> bool:
        """Scroll up in the chat history.
        
        Args:
            lines: Number of messages to scroll up
            
        Returns:
            True if scrolling occurred, False if at top
        """
        max_messages = self._calculate_message_display_count()
        max_scroll = max(0, len(self.messages) - max_messages)
        
        if self.scroll_offset < max_scroll:
            old_offset = self.scroll_offset
            self.scroll_offset = min(self.scroll_offset + lines, max_scroll)
            self.auto_scroll = False  # Disable auto-scroll when manually scrolling
            self._update_chat_history()
            return self.scroll_offset != old_offset
        return False
    
    def scroll_down(self, lines: int = 3) -> bool:
        """Scroll down in the chat history.
        
        Args:
            lines: Number of messages to scroll down
            
        Returns:
            True if scrolling occurred, False if at bottom
        """
        if self.scroll_offset > 0:
            old_offset = self.scroll_offset
            self.scroll_offset = max(0, self.scroll_offset - lines)
            
            # Re-enable auto-scroll if we've scrolled to the bottom
            if self.scroll_offset == 0:
                self.auto_scroll = True
                
            self._update_chat_history()
            return self.scroll_offset != old_offset
        return False
    
    def scroll_to_top(self) -> None:
        """Scroll to the top of the chat history."""
        max_messages = self._calculate_message_display_count()
        max_scroll = max(0, len(self.messages) - max_messages)
        self.scroll_offset = max_scroll
        self.auto_scroll = False
        self._update_chat_history()
    
    def scroll_to_bottom(self) -> None:
        """Scroll to the bottom of the chat history."""
        self.scroll_offset = 0
        self.auto_scroll = True
        self._update_chat_history()
    
    def toggle_auto_scroll(self) -> bool:
        """Toggle auto-scroll mode.
        
        Returns:
            New auto-scroll state
        """
        self.auto_scroll = not self.auto_scroll
        if self.auto_scroll:
            self.scroll_offset = 0
            self._update_chat_history()
        return self.auto_scroll
    
    def get_scroll_info(self) -> Dict[str, Any]:
        """Get current scroll information.
        
        Returns:
            Dictionary with scroll state information
        """
        max_messages = self._calculate_message_display_count()
        max_scroll = max(0, len(self.messages) - max_messages)
        
        return {
            'scroll_offset': self.scroll_offset,
            'max_scroll': max_scroll,
            'auto_scroll': self.auto_scroll,
            'total_messages': len(self.messages),
            'visible_messages': min(len(self.messages), max_messages),
            'at_top': self.scroll_offset >= max_scroll,
            'at_bottom': self.scroll_offset == 0
        }
    
    def _calculate_message_display_count(self) -> int:
        """Calculate how many messages to show based on terminal height.
        
        Returns:
            Number of messages that can fit in the current terminal
        """
        _, height = self.terminal_size
        
        # Estimate lines per message (speaker + content + spacing)
        # Account for panel borders and padding
        available_height = height - 8  # Reserve space for vitals, input, borders
        
        # Each message takes roughly 3-4 lines (speaker + wrapped content + spacing)
        lines_per_message = 4
        max_messages = max(3, available_height // lines_per_message)
        
        # Cap at reasonable limits
        return min(max_messages, 20)
    
    def _wrap_message_text(self, text: str, max_width: int) -> str:
        """Wrap message text to fit within available width.
        
        Args:
            text: Text to wrap
            max_width: Maximum width for wrapping
            
        Returns:
            Wrapped text
        """
        if len(text) <= max_width:
            return text
        
        # Simple word wrapping
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            word_length = len(word)
            if current_length + word_length + len(current_line) <= max_width:
                current_line.append(word)
                current_length += word_length
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = word_length
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return '\n'.join(lines)
    
    def _update_chat_history(self) -> None:
        """Update the chat history panel with responsive sizing and scrolling."""
        if not self.messages:
            # Show welcome message when no messages
            content = Text()
            content.append("Welcome to Character Chat!\n\n", style="bold green")
            content.append("Start typing to begin your conversation...", style="dim")
            
            panel = Panel(
                content,
                title="[bold]Conversation[/bold]",
                border_style="green"
            )
        else:
            # Build conversation display with scrolling support
            content = Text()
            
            # Calculate how many messages to show based on terminal size
            max_messages = self._calculate_message_display_count()
            total_messages = len(self.messages)
            
            # Calculate which messages to show based on scroll offset
            if total_messages <= max_messages:
                # All messages fit, ignore scroll offset
                visible_messages = self.messages
                start_index = 0
            else:
                # Apply scroll offset from the end
                end_index = total_messages - self.scroll_offset
                start_index = max(0, end_index - max_messages)
                visible_messages = self.messages[start_index:end_index]
            
            # Calculate available width for message content
            width, _ = self.terminal_size
            chat_ratio, vitals_ratio, _, _ = self._calculate_layout_ratios()
            chat_width = int(width * chat_ratio / 100)
            
            # Account for panel borders and padding (roughly 6 characters)
            max_content_width = max(30, chat_width - 15)  # Leave space for "Speaker: " prefix
            
            for i, msg in enumerate(visible_messages):
                speaker = msg.get('speaker', 'Unknown')
                message = msg.get('message', '')
                
                # Wrap message text to fit
                wrapped_message = self._wrap_message_text(message, max_content_width)
                
                # Style based on speaker
                if speaker.lower() == 'you':
                    speaker_style = "bold blue"
                    message_style = "white"
                else:
                    speaker_style = "bold green"
                    message_style = "white"
                
                # Add speaker name
                content.append(f"{speaker}: ", style=speaker_style)
                content.append(f"{wrapped_message}", style=message_style)
                
                # Add spacing between messages
                if i < len(visible_messages) - 1:
                    content.append("\n\n")
            
            # Create title with scroll information
            scroll_info = self.get_scroll_info()
            if total_messages > max_messages:
                title_parts = []
                title_parts.append("[bold]Conversation[/bold]")
                
                # Show which messages are visible
                visible_start = start_index + 1
                visible_end = start_index + len(visible_messages)
                title_parts.append(f"({visible_start}-{visible_end} of {total_messages})")
                
                # Show scroll indicators
                if not scroll_info['at_bottom']:
                    title_parts.append("↓")
                if not scroll_info['at_top']:
                    title_parts.append("↑")
                if not self.auto_scroll:
                    title_parts.append("[manual]")
                    
                title = " ".join(title_parts)
            else:
                title = "[bold]Conversation[/bold]"
            
            panel = Panel(
                content,
                title=title,
                border_style="green",
                padding=(1, 2)
            )
        
        self.layout["history"].update(panel)
    
    def update_vitals(self, vitals_panel: Panel) -> None:
        """Update vitals display.
        
        Args:
            vitals_panel: Rich Panel with vitals information
        """
        self.layout["vitals"].update(vitals_panel)
    
    def update_input_prompt(self, prompt_text: str) -> None:
        """Update input prompt area.
        
        Args:
            prompt_text: Text to show in the input area
        """
        self._update_input_prompt(prompt_text)
    
    def _update_input_prompt(self, prompt_text: str) -> None:
        """Internal method to update input prompt display."""
        # Create input prompt panel
        content = Text()
        content.append("You: ", style="bold blue")
        content.append(prompt_text, style="white")
        
        # Add command help
        content.append("\n", style="white")
        content.append("Commands: ", style="dim")
        content.append("/help /debug /memory /goals /vitals /save /exit", style="dim cyan")
        content.append(" | ", style="dim")
        content.append("Scroll: ↑/↓ PgUp/PgDn Home/End", style="dim yellow")
        
        panel = Panel(
            content,
            border_style="blue",
            padding=(0, 1)
        )
        
        self.layout["input"].update(panel)
    
    def update_character_info(self, character_name: str, character_archetype: str) -> None:
        """Update the chat panel title with character information.
        
        Args:
            character_name: Name of the current character
            character_archetype: Character's archetype description
        """
        # This will be reflected in the next chat history update
        self.character_name = character_name
        self.character_archetype = character_archetype
    
    def show_thinking_indicator(self) -> None:
        """Show that the character is thinking/processing."""
        thinking_text = Text()
        thinking_text.append("🤔 ", style="yellow")
        thinking_text.append("Character is thinking...", style="dim yellow")
        
        panel = Panel(
            thinking_text,
            border_style="yellow",
            padding=(0, 1)
        )
        
        self.layout["input"].update(panel)
    
    def show_error_message(self, error: str) -> None:
        """Show an error message in the input area.
        
        Args:
            error: Error message to display
        """
        error_text = Text()
        error_text.append("❌ Error: ", style="bold red")
        error_text.append(error, style="red")
        
        panel = Panel(
            error_text,
            border_style="red",
            padding=(0, 1)
        )
        
        self.layout["input"].update(panel)
    
    def show_command_result(self, result: str) -> None:
        """Show the result of a command execution.
        
        Args:
            result: Result message to display
        """
        result_text = Text()
        result_text.append("✓ ", style="green")
        result_text.append(result, style="green")
        
        panel = Panel(
            result_text,
            border_style="green",
            padding=(0, 1)
        )
        
        self.layout["input"].update(panel)
    
    def get_layout(self) -> Layout:
        """Get the current layout object.
        
        Returns:
            Rich Layout object for display
        """
        return self.layout
    
    def create_status_display(self, character_name: str, status: str) -> Panel:
        """Create a status display panel for the chat area.
        
        Args:
            character_name: Name of the character
            status: Status message
            
        Returns:
            Rich Panel with status information
        """
        content = Text()
        content.append(f"Chat with {character_name}\n", style="bold cyan")
        content.append(f"Status: {status}", style="yellow")
        
        return Panel(
            content,
            title="[bold]Character Status[/bold]",
            border_style="cyan"
        )
    
    def create_help_display(self) -> Panel:
        """Create a help display panel.
        
        Returns:
            Rich Panel with help information
        """
        content = Table(show_header=False, box=None, padding=(0, 1))
        content.add_column("Command", style="cyan", width=15)
        content.add_column("Description", style="white")
        
        commands = [
            ("/help", "Show this help message"),
            ("/debug", "Toggle debug mode on/off"),
            ("/vitals", "Show detailed character vitals"),
            ("/memory [query]", "Search character memories"),
            ("/goals", "Show character's active goals"),
            ("/mood", "Show mood history"),
            ("/save [name]", "Save character state"),
            ("/load [name]", "Load character state"),
            ("/reset", "Reset character to initial state"),
            ("/export", "Export conversation log"),
            ("/clear", "Clear conversation history"),
            ("", ""),  # Separator
            ("/scroll up", "Scroll up in conversation"),
            ("/scroll down", "Scroll down in conversation"),
            ("/scroll top", "Go to top of conversation"),
            ("/scroll bottom", "Go to bottom of conversation"),
            ("/autoscroll", "Toggle auto-scroll mode"),
            ("", ""),  # Separator
            ("/exit", "Exit the chat")
        ]
        
        for cmd, desc in commands:
            content.add_row(cmd, desc)
        
        return Panel(
            content,
            title="[bold]Available Commands[/bold]",
            border_style="blue",
            padding=(1, 2)
        )