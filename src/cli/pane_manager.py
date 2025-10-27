"""Pane management system for interactive chat interface."""

import asyncio
import threading
from enum import Enum
from typing import Optional, Callable, Dict, Any, Tuple
from dataclasses import dataclass

try:
    from pynput import mouse, keyboard as pynput_keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False


class PaneType(Enum):
    """Types of panes in the chat interface."""
    CONVERSATION = "conversation"
    VITALS = "vitals"
    INPUT = "input"


@dataclass
class PaneInfo:
    """Information about a pane."""
    pane_type: PaneType
    name: str
    description: str
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    active: bool = False


class PaneManager:
    """Manages pane selection, navigation, and mouse interactions."""
    
    def __init__(self, layout_manager=None):
        """Initialize pane manager.
        
        Args:
            layout_manager: LayoutManager instance for getting pane positions
        """
        self.layout_manager = layout_manager
        self.panes: Dict[PaneType, PaneInfo] = {}
        self.active_pane: Optional[PaneType] = PaneType.CONVERSATION
        self.mouse_enabled = False
        self.tab_navigation_enabled = False
        
        # Event handlers
        self.pane_change_callback: Optional[Callable[[PaneType], None]] = None
        self.scroll_callback: Optional[Callable[[PaneType, str], None]] = None
        
        # Mouse and keyboard listeners
        self._mouse_listener: Optional[mouse.Listener] = None
        self._key_listener: Optional[pynput_keyboard.Listener] = None
        self._stop_event = threading.Event()
        
        # Initialize panes
        self._initialize_panes()
    
    def _initialize_panes(self) -> None:
        """Initialize pane information."""
        self.panes[PaneType.CONVERSATION] = PaneInfo(
            pane_type=PaneType.CONVERSATION,
            name="Conversation",
            description="Chat history and messages",
            active=True
        )
        
        self.panes[PaneType.VITALS] = PaneInfo(
            pane_type=PaneType.VITALS,
            name="Vitals",
            description="Character status and vitals"
        )
        
        self.panes[PaneType.INPUT] = PaneInfo(
            pane_type=PaneType.INPUT,
            name="Input",
            description="Message input and commands"
        )
        
        self.active_pane = PaneType.CONVERSATION
    
    def set_pane_change_callback(self, callback: Callable[[PaneType], None]) -> None:
        """Set callback for pane change events.
        
        Args:
            callback: Function to call when active pane changes
        """
        self.pane_change_callback = callback
    
    def set_scroll_callback(self, callback: Callable[[PaneType, str], None]) -> None:
        """Set callback for scroll events in panes.
        
        Args:
            callback: Function to call when scrolling in a pane (pane_type, direction)
        """
        self.scroll_callback = callback
    
    def get_active_pane(self) -> Optional[PaneType]:
        """Get the currently active pane.
        
        Returns:
            Active pane type
        """
        return self.active_pane
    
    def set_active_pane(self, pane_type: PaneType) -> bool:
        """Set the active pane.
        
        Args:
            pane_type: Pane to make active
            
        Returns:
            True if pane was changed
        """
        if pane_type not in self.panes:
            return False
        
        old_pane = self.active_pane
        self.active_pane = pane_type
        
        # Update pane active status
        for pane in self.panes.values():
            pane.active = False
        self.panes[pane_type].active = True
        
        # Call callback if pane changed
        if old_pane != pane_type and self.pane_change_callback:
            try:
                self.pane_change_callback(pane_type)
            except Exception as e:
                print(f"Error in pane change callback: {e}")
        
        return old_pane != pane_type
    
    def cycle_active_pane(self, direction: int = 1) -> PaneType:
        """Cycle to the next/previous active pane.
        
        Args:
            direction: 1 for next, -1 for previous
            
        Returns:
            New active pane type
        """
        pane_order = [PaneType.CONVERSATION, PaneType.VITALS, PaneType.INPUT]
        
        try:
            current_index = pane_order.index(self.active_pane)
            new_index = (current_index + direction) % len(pane_order)
            new_pane = pane_order[new_index]
            
            self.set_active_pane(new_pane)
            return new_pane
        except (ValueError, IndexError):
            return self.active_pane
    
    def update_pane_positions(self, terminal_size: Tuple[int, int]) -> None:
        """Update pane positions based on terminal size.
        
        Args:
            terminal_size: (width, height) of terminal
        """
        if not self.layout_manager:
            return
        
        width, height = terminal_size
        
        # Calculate pane positions based on layout ratios
        chat_ratio, vitals_ratio, history_ratio, input_ratio = (
            self.layout_manager._calculate_layout_ratios()
        )
        
        # Chat pane (left side)
        chat_width = int(width * chat_ratio / 100)
        self.panes[PaneType.CONVERSATION].x = 0
        self.panes[PaneType.CONVERSATION].y = 0
        self.panes[PaneType.CONVERSATION].width = chat_width
        self.panes[PaneType.CONVERSATION].height = int(height * history_ratio / 100)
        
        # Input pane (bottom of chat side)
        self.panes[PaneType.INPUT].x = 0
        self.panes[PaneType.INPUT].y = self.panes[PaneType.CONVERSATION].height
        self.panes[PaneType.INPUT].width = chat_width
        self.panes[PaneType.INPUT].height = int(height * input_ratio / 100)
        
        # Vitals pane (right side)
        self.panes[PaneType.VITALS].x = chat_width
        self.panes[PaneType.VITALS].y = 0
        self.panes[PaneType.VITALS].width = int(width * vitals_ratio / 100)
        self.panes[PaneType.VITALS].height = height
    
    def get_pane_at_position(self, x: int, y: int) -> Optional[PaneType]:
        """Get the pane at the given terminal position.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            PaneType if found, None otherwise
        """
        for pane_type, pane in self.panes.items():
            if (pane.x <= x < pane.x + pane.width and 
                pane.y <= y < pane.y + pane.height):
                return pane_type
        return None
    
    def enable_tab_navigation(self) -> bool:
        """Enable Tab key navigation between panes.
        
        Returns:
            True if successfully enabled
        """
        if not PYNPUT_AVAILABLE:
            return False
        
        if self.tab_navigation_enabled:
            return True
        
        try:
            self._key_listener = pynput_keyboard.Listener(
                on_press=self._on_key_press,
                suppress=False
            )
            self._key_listener.start()
            self.tab_navigation_enabled = True
            return True
        except Exception as e:
            print(f"Failed to enable tab navigation: {e}")
            return False
    
    def enable_mouse_interaction(self) -> bool:
        """Enable mouse interaction for pane selection.
        
        Returns:
            True if successfully enabled
        """
        if not PYNPUT_AVAILABLE:
            return False
        
        if self.mouse_enabled:
            return True
        
        try:
            self._mouse_listener = mouse.Listener(
                on_click=self._on_mouse_click,
                on_scroll=self._on_mouse_scroll
            )
            self._mouse_listener.start()
            self.mouse_enabled = True
            return True
        except Exception as e:
            print(f"Failed to enable mouse interaction: {e}")
            return False
    
    def disable_all(self) -> None:
        """Disable all pane interactions."""
        self.tab_navigation_enabled = False
        self.mouse_enabled = False
        
        if self._key_listener:
            try:
                self._key_listener.stop()
            except:
                pass
            self._key_listener = None
        
        if self._mouse_listener:
            try:
                self._mouse_listener.stop()
            except:
                pass
            self._mouse_listener = None
    
    def _on_key_press(self, key) -> None:
        """Handle key press events.
        
        Args:
            key: Key that was pressed
        """
        try:
            if key == pynput_keyboard.Key.tab:
                # Cycle to next pane
                old_pane = self.active_pane
                new_pane = self.cycle_active_pane(1)
                print(f"Tab pressed: {old_pane.value} -> {new_pane.value}")  # Debug output
            elif key == pynput_keyboard.Key.up and self.active_pane == PaneType.CONVERSATION:
                if self.scroll_callback:
                    self.scroll_callback(PaneType.CONVERSATION, "up")
            elif key == pynput_keyboard.Key.down and self.active_pane == PaneType.CONVERSATION:
                if self.scroll_callback:
                    self.scroll_callback(PaneType.CONVERSATION, "down")
        except Exception as e:
            print(f"Error handling key press: {e}")
    
    def _on_mouse_click(self, x: int, y: int, button, pressed: bool) -> None:
        """Handle mouse click events.
        
        Args:
            x: X coordinate
            y: Y coordinate
            button: Mouse button
            pressed: True if pressed, False if released
        """
        if not pressed:  # Only handle button release
            return
        
        try:
            # Get terminal position (this is approximate)
            # Note: This would need more sophisticated coordinate mapping
            # for actual terminal position detection
            pane = self.get_pane_at_position(x, y)
            if pane:
                self.set_active_pane(pane)
        except Exception as e:
            print(f"Error handling mouse click: {e}")
    
    def _on_mouse_scroll(self, x: int, y: int, dx: int, dy: int) -> None:
        """Handle mouse scroll events.
        
        Args:
            x: X coordinate
            y: Y coordinate
            dx: Horizontal scroll
            dy: Vertical scroll
        """
        try:
            pane = self.get_pane_at_position(x, y)
            if pane == PaneType.CONVERSATION and self.scroll_callback:
                direction = "up" if dy > 0 else "down"
                self.scroll_callback(pane, direction)
        except Exception as e:
            print(f"Error handling mouse scroll: {e}")
    
    def get_pane_border_style(self, pane_type: PaneType) -> str:
        """Get border style for a pane based on its active status.
        
        Args:
            pane_type: Type of pane
            
        Returns:
            Border style string for Rich
        """
        if pane_type == self.active_pane:
            return "bright_blue"  # Active pane
        else:
            return "dim"  # Inactive pane
    
    def get_pane_title(self, pane_type: PaneType, base_title: str = "") -> str:
        """Get title for a pane with active indicator.
        
        Args:
            pane_type: Type of pane
            base_title: Base title text
            
        Returns:
            Title with active indicator
        """
        pane_info = self.panes.get(pane_type)
        if not pane_info:
            return base_title
        
        title_parts = []
        if base_title:
            title_parts.append(base_title)
        else:
            title_parts.append(f"[bold]{pane_info.name}[/bold]")
        
        if pane_info.active:
            title_parts.append("●")  # Active indicator
        
        return " ".join(title_parts)
    
    def create_scroll_bar(self, scroll_info: Dict[str, Any], height: int = 10) -> str:
        """Create a visual scroll bar representation.
        
        Args:
            scroll_info: Scroll information from layout manager
            height: Height of scroll bar in characters
            
        Returns:
            String representation of scroll bar
        """
        if scroll_info['total_messages'] <= scroll_info['visible_messages']:
            # No scrolling needed
            return ""
        
        total = scroll_info['total_messages']
        visible = scroll_info['visible_messages']
        offset = scroll_info['scroll_offset']
        
        # Calculate scroll bar position
        visible_ratio = visible / total
        scroll_ratio = offset / (total - visible) if total > visible else 0
        
        bar_height = max(1, int(height * visible_ratio))
        bar_position = int((height - bar_height) * scroll_ratio)
        
        # Build scroll bar
        scroll_bar = []
        for i in range(height):
            if bar_position <= i < bar_position + bar_height:
                scroll_bar.append("█")  # Solid block for thumb
            else:
                scroll_bar.append("│")  # Light vertical line for track
        
        return "".join(scroll_bar)
    
    def get_status_info(self) -> str:
        """Get status information about pane management.
        
        Returns:
            Status string
        """
        status_parts = []
        
        active_pane = self.panes.get(self.active_pane)
        if active_pane:
            status_parts.append(f"Active: {active_pane.name}")
        
        if self.tab_navigation_enabled:
            status_parts.append("Tab: On")
        
        if self.mouse_enabled:
            status_parts.append("Mouse: On")
        
        return " | ".join(status_parts)
    
    def is_available(self) -> bool:
        """Check if pane management features are available.
        
        Returns:
            True if pynput is available
        """
        return PYNPUT_AVAILABLE