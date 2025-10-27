## CLI Chat Interface Design with Rich UI

### Architecture Overview

The CLI will feature a **dual-pane layout** using the Rich library:

**Left Pane (60% width)**: Interactive chat window
- User input prompt at bottom
- Scrollable conversation history
- Command support (`/help`, `/debug`, `/memory`, etc.)

**Right Pane (40% width)**: Real-time Agent Vitals Display
- **Neurohormonal Levels** (bar charts with color coding)
- **Mood State** (current emotional state + intensity)
- **Energy Level** (affects response verbosity)
- **Knowledge Stats** (facts learned, relationships formed)
- **Memory Status** (recent memories, episodic count)
- **Active Goals** (current priorities)
- **Relationship Metrics** (trust level, interaction count)

### Visual Layout Design

```
┌─────────────────────────────────────┬───────────────────────────────────┐
│ Chat with Marcus Rivera             │ ⚡ Agent Vitals                   │
│ (Playful - Elementary Teacher)      │                                   │
├─────────────────────────────────────┤ 🧪 NEUROCHEMICALS                 │
│                                     │ Dopamine    ████████░░ 85/100    │
│ Marcus: Hey! What's going on? 😄    │ Serotonin   ██████░░░░ 65/100    │
│                                     │ Oxytocin    ███████░░░ 72/100    │
│ You: Just working on a project.    │ Cortisol    ███░░░░░░░ 32/100    │
│                                     │ Endorphins  █████░░░░░ 55/100    │
│ Marcus: Ooh, what kind? Tell me    │ Adrenaline  ████░░░░░░ 48/100    │
│ everything! Is it something cool?   │                                   │
│                                     │ 😊 MOOD STATE                     │
│                                     │ Current: Happy                    │
│                                     │ Intensity: 0.78                   │
│                                     │ Energy: 0.82                      │
│                                     │                                   │
│                                     │ 🎯 ACTIVE GOALS                   │
│                                     │ • Make conversation fun (P:9)     │
│                                     │ • Build genuine connection (P:8)  │
│                                     │                                   │
│                                     │ 🧠 KNOWLEDGE                      │
│                                     │ Facts learned: 3                  │
│                                     │ Entities known: 5                 │
│                                     │ Relationships: 2                  │
│                                     │                                   │
│                                     │ 💾 MEMORY                         │
│                                     │ Episodic memories: 12             │
│                                     │ Recent: "project discussion"      │
│                                     │                                   │
│                                     │ 🤝 RELATIONSHIP                   │
│                                     │ Trust level: 0.65                 │
│                                     │ Interactions: 8                   │
│                                     │ Duration: 5 min                   │
├─────────────────────────────────────┴───────────────────────────────────┤
│ You: _                                                                  │
│ Commands: /help /debug /memory /goals /save /exit                      │
└─────────────────────────────────────────────────────────────────────────┘
```

## Implementation Design

### File Structure

```
src/cli/
├── __init__.py
├── chat_interface.py          # Main chat interface (ENHANCED)
├── debug_view.py              # Debug display utilities (ENHANCED)
├── vitals_display.py          # NEW: Real-time vitals panel
├── layout_manager.py          # NEW: Manages dual-pane layout
└── command_handler.py         # NEW: Command processing

scripts/
└── run_chat.py               # Entry point (ENHANCED)
```

### Key Components

#### 1. **VitalsDisplay** (New Component)

```python
from rich.panel import Panel
from rich.table import Table
from rich.progress import BarColumn, Progress
from rich.text import Text

class VitalsDisplay:
    """Real-time display of character agent vitals."""
    
    def __init__(self, console: Console):
        self.console = console
        
    def render_vitals(self, character_state: CharacterState) -> Panel:
        """Render complete vitals panel."""
        content = self._build_vitals_content(character_state)
        return Panel(
            content,
            title="⚡ Agent Vitals",
            border_style="cyan",
            padding=(1, 2)
        )
    
    def _build_vitals_content(self, state: CharacterState) -> Table:
        """Build the complete vitals display."""
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
        """Render neurochemical levels with color-coded bars."""
        text = Text("🧪 NEUROCHEMICALS\n", style="bold yellow")
        
        for hormone, level in levels.items():
            # Color coding based on level
            if level > 70:
                color = "red"
            elif level > 50:
                color = "yellow"
            else:
                color = "green"
            
            # Create bar visualization
            filled = int(level / 10)
            bar = "█" * filled + "░" * (10 - filled)
            
            text.append(f"{hormone.capitalize():12} ", style="white")
            text.append(f"{bar} ", style=color)
            text.append(f"{level:.0f}/100\n", style=color)
        
        return text
    
    def _render_mood(self, mood_state: Dict[str, Any]) -> Text:
        """Render current mood state."""
        text = Text("😊 MOOD STATE\n", style="bold yellow")
        
        current_mood = mood_state.get('current_state', 'neutral')
        intensity = mood_state.get('intensity', 0.5)
        energy = mood_state.get('energy_level', 0.5)
        
        # Emoji mapping for moods
        mood_emoji = {
            'happy': '😊', 'sad': '😢', 'angry': '😠',
            'anxious': '😰', 'excited': '🤩', 'calm': '😌',
            'frustrated': '😤', 'content': '😌', 'neutral': '😐'
        }
        
        emoji = mood_emoji.get(current_mood.lower(), '😐')
        
        text.append(f"Current: {emoji} {current_mood.capitalize()}\n", style="white")
        text.append(f"Intensity: {intensity:.2f}\n", style="cyan")
        text.append(f"Energy: {energy:.2f}\n", style="magenta")
        
        return text
    
    def _render_goals(self, goals_state: Dict[str, Any]) -> Text:
        """Render active goals."""
        text = Text("🎯 ACTIVE GOALS\n", style="bold yellow")
        
        active_goals = goals_state.get('active_goals', [])
        
        if not active_goals:
            text.append("No active goals\n", style="dim")
        else:
            for goal in active_goals[:3]:  # Show top 3
                priority = goal.get('priority', 5)
                description = goal.get('description', 'Unknown goal')
                # Truncate long descriptions
                if len(description) > 35:
                    description = description[:32] + "..."
                text.append(f"• {description} (P:{priority})\n", style="white")
        
        return text
    
    def _render_knowledge(self, state: CharacterState) -> Text:
        """Render knowledge statistics."""
        text = Text("🧠 KNOWLEDGE\n", style="bold yellow")
        
        # These would come from the knowledge graph
        facts_count = len(state.knowledge_base.get('facts', []))
        entities_count = len(state.knowledge_base.get('entities', []))
        relationships_count = len(state.knowledge_base.get('relationships', []))
        
        text.append(f"Facts learned: {facts_count}\n", style="white")
        text.append(f"Entities known: {entities_count}\n", style="white")
        text.append(f"Relationships: {relationships_count}\n", style="white")
        
        return text
    
    def _render_memory(self, state: CharacterState) -> Text:
        """Render memory statistics."""
        text = Text("💾 MEMORY\n", style="bold yellow")
        
        episodic_count = len(state.episodic_memory)
        
        text.append(f"Episodic memories: {episodic_count}\n", style="white")
        
        # Show most recent memory topic
        if state.episodic_memory:
            recent = state.episodic_memory[-1]
            topic = recent.get('topic', 'general conversation')
            if len(topic) > 30:
                topic = topic[:27] + "..."
            text.append(f"Recent: \"{topic}\"\n", style="dim")
        
        return text
    
    def _render_relationship(self, rel_state: Dict[str, Any]) -> Text:
        """Render relationship metrics."""
        text = Text("🤝 RELATIONSHIP\n", style="bold yellow")
        
        trust = rel_state.get('trust_level', 0.5)
        interactions = rel_state.get('interaction_count', 0)
        duration = rel_state.get('total_duration_minutes', 0)
        
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
        text.append(f"Duration: {duration:.0f} min\n", style="white")
        
        return text
```

#### 2. **LayoutManager** (New Component)

```python
from rich.layout import Layout
from rich.live import Live

class LayoutManager:
    """Manages the dual-pane layout."""
    
    def __init__(self, console: Console):
        self.console = console
        self.layout = Layout()
        self._setup_layout()
        
    def _setup_layout(self):
        """Setup the dual-pane layout."""
        # Split into left (chat) and right (vitals)
        self.layout.split_row(
            Layout(name="chat", ratio=60),
            Layout(name="vitals", ratio=40)
        )
        
        # Split chat pane into history and input
        self.layout["chat"].split_column(
            Layout(name="history", ratio=90),
            Layout(name="input", ratio=10, minimum_size=3)
        )
    
    def update_chat_history(self, messages: List[Dict[str, str]]):
        """Update chat history display."""
        # Render messages with proper formatting
        pass
    
    def update_vitals(self, vitals_panel: Panel):
        """Update vitals display."""
        self.layout["vitals"].update(vitals_panel)
    
    def update_input_prompt(self, prompt_text: str):
        """Update input prompt area."""
        self.layout["input"].update(Panel(prompt_text, style="blue"))
```

#### 3. **Enhanced ChatInterface**

```python
class ChatInterface:
    """Enhanced interactive CLI with dual-pane display."""
    
    def __init__(self):
        self.console = Console()
        self.vitals_display = VitalsDisplay(self.console)
        self.layout_manager = LayoutManager(self.console)
        self.command_handler = CommandHandler(self)
        # ... existing initialization
        
    async def _conversation_loop_with_vitals(self):
        """Main conversation loop with live vitals update."""
        
        with Live(
            self.layout_manager.layout,
            console=self.console,
            screen=True,
            refresh_per_second=4
        ) as live:
            
            while True:
                # Get user input
                user_input = await self._get_user_input()
                
                if user_input.startswith('/'):
                    result = await self.command_handler.handle(user_input)
                    if result == 'exit':
                        break
                    continue
                
                # Process message
                result = await self.current_character.process_message(
                    user_message=user_input,
                    context={'debug_mode': self.debug_mode}
                )
                
                # Update chat history
                self.messages.append({
                    'speaker': 'You',
                    'message': user_input
                })
                self.messages.append({
                    'speaker': self.current_character.character_name,
                    'message': result['response']
                })
                
                # Update displays
                self.layout_manager.update_chat_history(self.messages)
                
                # Update vitals in real-time
                vitals_panel = self.vitals_display.render_vitals(
                    self.current_character.state
                )
                self.layout_manager.update_vitals(vitals_panel)
                
                live.refresh()
```

#### 4. **Command Handler** (New Component)

```python
class CommandHandler:
    """Handles special commands during chat."""
    
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
        '/exit': 'Exit chat'
    }
    
    def __init__(self, chat_interface: ChatInterface):
        self.interface = chat_interface
        
    async def handle(self, command: str) -> Optional[str]:
        """Handle command and return 'exit' if should exit."""
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
            '/exit': self._handle_exit
        }
        
        handler = handlers.get(cmd)
        if handler:
            return await handler(args)
        else:
            self.interface.console.print(
                f"[red]Unknown command: {cmd}[/red]\n"
                f"Type /help to see available commands."
            )
            return None
    
    async def _handle_vitals(self, args: str) -> None:
        """Show detailed vitals in full screen."""
        # Display expanded vitals view
        pass
    
    async def _handle_memory(self, args: str) -> None:
        """Show recent memories."""
        # Query and display episodic memories
        pass
    
    # ... other handlers
```

## Detailed Implementation Plan for Claude Code

### Phase 1: Core Vitals Display Components (Day 1)

**Files to Create:**

1. **`src/cli/vitals_display.py`** - Complete vitals rendering
   - Implement `VitalsDisplay` class
   - Add all rendering methods for each vitals section
   - Implement color coding and progress bars
   - Add dynamic updates support

2. **`src/cli/layout_manager.py`** - Layout management
   - Implement `LayoutManager` class
   - Setup dual-pane layout with Rich
   - Add methods for updating each pane
   - Implement smooth refresh logic

3. **`src/cli/command_handler.py`** - Command processing
   - Implement `CommandHandler` class
   - Add all command handlers
   - Implement command validation
   - Add help system

**Testing:**
```bash
# Test vitals display with mock data
python -c "from src.cli.vitals_display import VitalsDisplay; # test code"
```

### Phase 2: Enhanced Chat Interface (Day 2)

**Files to Modify:**

4. **`src/cli/chat_interface.py`** - Major enhancements
   - Integrate `VitalsDisplay`
   - Integrate `LayoutManager`
   - Integrate `CommandHandler`
   - Replace simple conversation loop with dual-pane version
   - Add real-time vitals updates
   - Implement message history management

5. **`src/cli/debug_view.py`** - Enhanced debug display
   - Add detailed agent consultation visualization
   - Add timeline view of state changes
   - Add memory retrieval visualization
   - Implement agent input comparison view

**Testing:**
```bash
# Test enhanced interface with test character
python scripts/run_chat.py --character m-playful --debug
```

### Phase 3: Character Activation System (Day 3)

**Files to Create/Modify:**

6. **`scripts/activate_character.py`** - New activation script
   ```python
   # Usage: python scripts/activate_character.py m-playful
   # Simplified character activation for quick testing
   ```

7. **`src/character_agent.py`** - Enhancements
   - Add vitals snapshot method
   - Add state export for display
   - Optimize state updates for live display

8. **`scripts/run_chat.py`** - Enhanced entry point
   - Add `--activate` flag for quick character launch
   - Add `--vitals-only` mode (vitals display without chat)
   - Add `--replay` mode (replay saved conversation)

**Testing:**
```bash
# Test quick activation
python scripts/activate_character.py m-playful

# Test with different modes
python scripts/run_chat.py --activate m-playful --debug
python scripts/run_chat.py --activate m-playful --vitals-only
```

### Phase 4: Advanced Vitals Features (Day 4)

**Features to Implement:**

9. **Real-time State Change Animations**
   - Add smooth transitions for hormone level changes
   - Add mood shift animations
   - Implement sparklines for trending data

10. **Historical Vitals View**
   - Add `/history` command to show vitals over time
   - Implement charts for hormone trends
   - Add mood timeline visualization

11. **Vitals Export**
   - Add ability to export vitals data as JSON
   - Add screenshot capability for vitals panel
   - Implement vitals logging for analysis

**Files:**
- `src/cli/vitals_display.py` - Add animation support
- `src/cli/vitals_history.py` - NEW: Historical visualization
- `src/cli/vitals_exporter.py` - NEW: Export functionality

### Phase 5: Testing & Refinement (Day 5)

**Testing Scenarios:**

12. **Character Activation Tests**
   ```python
   # Test all 8 characters
   for char in ['m-playful', 'f-serious', 'm-sarcastic', ...]:
       # Activate and verify vitals display
       # Chat for 10 messages
       # Verify state changes are reflected
   ```

13. **Performance Tests**
   - Test with rapid message sending
   - Test with long conversations (100+ messages)
   - Measure refresh rate impact on responsiveness

14. **Visual Polish**
   - Adjust column widths for optimal readability
   - Fine-tune color schemes
   - Optimize for different terminal sizes

15. **Documentation**
   - Create usage guide with screenshots
   - Document all commands
   - Add troubleshooting section

## Summary: Key Implementation Steps

### For Claude Code to implement:

**Step 1: Create Foundation**
```bash
# Create new files
touch src/cli/vitals_display.py
touch src/cli/layout_manager.py
touch src/cli/command_handler.py
touch src/cli/vitals_history.py
touch scripts/activate_character.py
```

**Step 2: Implement VitalsDisplay**
- Copy the `VitalsDisplay` class design above
- Implement all rendering methods
- Add color coding logic
- Test with mock CharacterState

**Step 3: Implement LayoutManager**
- Setup Rich Layout with dual panes
- Implement update methods
- Add Live display support
- Test layout rendering

**Step 4: Enhance ChatInterface**
- Integrate new components
- Replace conversation loop
- Add real-time updates
- Test full integration

**Step 5: Add Command System**
- Implement CommandHandler
- Add all command handlers
- Test each command
- Add help documentation

**Step 6: Create Activation Script**
- Simple CLI: `activate m-playful`
- Load character and start chat
- Auto-enable vitals display

**Step 7: Polish & Test**
- Test with all 8 characters
- Optimize performance
- Add visual polish
- Write documentation

## Expected Outcome

When complete, users will be able to:

```bash
# Activate character with rich vitals display
$ activate m-playful

# Or use full script
$ python scripts/run_chat.py --activate m-playful --debug
```

And see:
- **Left side**: Natural conversation with Marcus
- **Right side**: Real-time vitals showing his dopamine spiking when you laugh at his jokes, oxytocin rising as trust builds, cortisol staying low due to low neuroticism
- **Commands**: Type `/memory` to see what he remembers, `/goals` to see his hidden agenda, `/vitals` for detailed breakdown

This creates an **observable, debuggable character system** where you can literally watch the character's internal state evolve during conversation.