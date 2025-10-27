"""Keyboard event handler for chat interface scrolling."""

import asyncio
import threading
from typing import Optional, Callable, Dict, Any
from enum import Enum

try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False


class ScrollAction(Enum):
    """Scroll action types."""
    UP = "up"
    DOWN = "down"
    PAGE_UP = "page_up"
    PAGE_DOWN = "page_down"
    HOME = "home"
    END = "end"


class KeyboardHandler:
    """Handles keyboard events for chat interface scrolling."""
    
    def __init__(self, layout_manager=None):
        """Initialize keyboard handler.
        
        Args:
            layout_manager: LayoutManager instance for scroll operations
        """
        self.layout_manager = layout_manager
        self.enabled = False
        self.scroll_callback: Optional[Callable[[ScrollAction], None]] = None
        self._keyboard_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Key mappings
        self.key_mappings: Dict[str, ScrollAction] = {
            'up': ScrollAction.UP,
            'down': ScrollAction.DOWN,
            'page up': ScrollAction.PAGE_UP,
            'page down': ScrollAction.PAGE_DOWN,
            'home': ScrollAction.HOME,
            'end': ScrollAction.END,
        }
        
        if not KEYBOARD_AVAILABLE:
            print("Warning: keyboard library not available. Keyboard scrolling disabled.")
    
    def set_scroll_callback(self, callback: Callable[[ScrollAction], None]) -> None:
        """Set callback function for scroll actions.
        
        Args:
            callback: Function to call when scroll action occurs
        """
        self.scroll_callback = callback
    
    def enable(self) -> bool:
        """Enable keyboard event handling.
        
        Returns:
            True if successfully enabled, False otherwise
        """
        if not KEYBOARD_AVAILABLE:
            return False
        
        if self.enabled:
            return True
        
        try:
            self.enabled = True
            self._stop_event.clear()
            
            # Start keyboard monitoring in a separate thread
            self._keyboard_thread = threading.Thread(
                target=self._keyboard_monitor,
                daemon=True
            )
            self._keyboard_thread.start()
            
            return True
        except Exception as e:
            print(f"Failed to enable keyboard handler: {e}")
            self.enabled = False
            return False
    
    def disable(self) -> None:
        """Disable keyboard event handling."""
        if not self.enabled:
            return
        
        self.enabled = False
        self._stop_event.set()
        
        try:
            # Unhook all keyboard events
            if KEYBOARD_AVAILABLE:
                keyboard.unhook_all()
        except Exception as e:
            print(f"Error disabling keyboard handler: {e}")
        
        # Wait for thread to finish
        if self._keyboard_thread and self._keyboard_thread.is_alive():
            self._keyboard_thread.join(timeout=1.0)
    
    def _keyboard_monitor(self) -> None:
        """Monitor keyboard events in a separate thread."""
        if not KEYBOARD_AVAILABLE:
            return
        
        try:
            # Register key event handlers
            for key, action in self.key_mappings.items():
                keyboard.add_hotkey(
                    key,
                    lambda a=action: self._handle_scroll_action(a),
                    suppress=False  # Don't suppress the key to allow normal input
                )
            
            # Keep monitoring until stop event is set
            while not self._stop_event.is_set():
                self._stop_event.wait(0.1)
                
        except Exception as e:
            print(f"Error in keyboard monitor: {e}")
        finally:
            # Clean up
            try:
                if KEYBOARD_AVAILABLE:
                    keyboard.unhook_all()
            except:
                pass
    
    def _handle_scroll_action(self, action: ScrollAction) -> None:
        """Handle a scroll action.
        
        Args:
            action: ScrollAction to perform
        """
        if not self.enabled:
            return
        
        # Call the scroll callback if set
        if self.scroll_callback:
            try:
                self.scroll_callback(action)
            except Exception as e:
                print(f"Error in scroll callback: {e}")
        
        # Directly handle scroll if layout_manager is available
        elif self.layout_manager:
            try:
                self._perform_scroll_action(action)
            except Exception as e:
                print(f"Error performing scroll action: {e}")
    
    def _perform_scroll_action(self, action: ScrollAction) -> None:
        """Perform scroll action on layout manager.
        
        Args:
            action: ScrollAction to perform
        """
        if not self.layout_manager:
            return
        
        if action == ScrollAction.UP:
            self.layout_manager.scroll_up(lines=1)
        elif action == ScrollAction.DOWN:
            self.layout_manager.scroll_down(lines=1)
        elif action == ScrollAction.PAGE_UP:
            self.layout_manager.scroll_up(lines=5)
        elif action == ScrollAction.PAGE_DOWN:
            self.layout_manager.scroll_down(lines=5)
        elif action == ScrollAction.HOME:
            self.layout_manager.scroll_to_top()
        elif action == ScrollAction.END:
            self.layout_manager.scroll_to_bottom()
    
    def is_available(self) -> bool:
        """Check if keyboard handling is available.
        
        Returns:
            True if keyboard library is available
        """
        return KEYBOARD_AVAILABLE
    
    def is_enabled(self) -> bool:
        """Check if keyboard handling is currently enabled.
        
        Returns:
            True if enabled
        """
        return self.enabled
    
    def get_key_help(self) -> Dict[str, str]:
        """Get help text for keyboard shortcuts.
        
        Returns:
            Dictionary mapping key names to descriptions
        """
        return {
            "↑ Arrow Up": "Scroll up one line",
            "↓ Arrow Down": "Scroll down one line", 
            "Page Up": "Scroll up one page",
            "Page Down": "Scroll down one page",
            "Home": "Go to top of conversation",
            "End": "Go to bottom of conversation"
        }


class AsyncKeyboardHandler(KeyboardHandler):
    """Async version of keyboard handler for better integration."""
    
    def __init__(self, layout_manager=None):
        """Initialize async keyboard handler.
        
        Args:
            layout_manager: LayoutManager instance for scroll operations
        """
        super().__init__(layout_manager)
        self._scroll_queue: Optional[asyncio.Queue] = None
        self._scroll_task: Optional[asyncio.Task] = None
    
    async def enable_async(self) -> bool:
        """Enable keyboard handling with async scroll processing.
        
        Returns:
            True if successfully enabled
        """
        if not KEYBOARD_AVAILABLE:
            return False
        
        # Create scroll queue for async processing
        self._scroll_queue = asyncio.Queue()
        
        # Set callback to add scroll actions to queue
        self.set_scroll_callback(self._queue_scroll_action)
        
        # Start async scroll processor
        self._scroll_task = asyncio.create_task(self._process_scroll_queue())
        
        # Enable keyboard monitoring
        return self.enable()
    
    async def disable_async(self) -> None:
        """Disable async keyboard handling."""
        # Cancel scroll processing task
        if self._scroll_task and not self._scroll_task.done():
            self._scroll_task.cancel()
            try:
                await self._scroll_task
            except asyncio.CancelledError:
                pass
        
        # Disable keyboard monitoring
        self.disable()
        
        # Clear scroll queue
        if self._scroll_queue:
            while not self._scroll_queue.empty():
                try:
                    self._scroll_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
    
    def _queue_scroll_action(self, action: ScrollAction) -> None:
        """Queue a scroll action for async processing.
        
        Args:
            action: ScrollAction to queue
        """
        if self._scroll_queue:
            try:
                self._scroll_queue.put_nowait(action)
            except asyncio.QueueFull:
                pass  # Drop scroll action if queue is full
    
    async def _process_scroll_queue(self) -> None:
        """Process scroll actions from the queue."""
        while True:
            try:
                # Wait for scroll action
                action = await self._scroll_queue.get()
                
                # Perform scroll action
                if self.layout_manager:
                    self._perform_scroll_action(action)
                
                # Mark task as done
                self._scroll_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error processing scroll action: {e}")