# Simple Chat Interface

A clean, straightforward command-line chat interface for the Character Agent system without complex window management.

## Features

- **Simple Terminal Interface**: No Rich components or window management
- **Plain Text Output**: Clean, readable text-based displays
- **All Essential Commands**: Preserved all useful functionality
- **Remote Support**: Connect to character agent servers with local fallback
- **Clean Conversation Flow**: Type messages and get responses naturally

## Usage

### Local Chat Interface
```bash
python scripts/run_simple_chat.py
python scripts/run_simple_chat.py --character alice
```

### Remote Chat Interface
```bash
python scripts/run_simple_remote_chat.py
python scripts/run_simple_remote_chat.py --server-url ws://localhost:8765
python scripts/run_simple_remote_chat.py --character alice --no-fallback
```

## Available Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/debug` | Toggle debug mode |
| `/vitals` | Show character vitals and neurochemical levels |
| `/memory [query]` | Show recent memories or search with query |
| `/goals` | Show character goals and priorities |
| `/mood` | Show mood information and emotional state |
| `/save [filename]` | Save character state to file |
| `/load [filename]` | Load character state from file |
| `/reset` | Reset character to initial state |
| `/export [filename]` | Export conversation log |
| `/clear` | Clear conversation history |
| `/status` | Show character status summary |
| `/exit` | Exit chat |

## Example Session

```
============================================================
    CHARACTER AGENT CHAT SYSTEM
============================================================

This system allows you to have natural conversations with AI characters
that have distinct personalities, memories, and emotional states.

Type '/help' for available commands or just start typing to chat!

AVAILABLE CHARACTERS:

1. Alice (Helpful Assistant)
   A friendly and knowledgeable AI assistant ready to help with various tasks.

2. Bob (Creative Writer)
   An imaginative storyteller with a passion for creative writing and literature.

Select character (1-2) or 'q' to quit: 1
Loading character 'alice'...
Successfully loaded Alice!
------------------------------------------------------------
You are now chatting with Alice (Helpful Assistant)

A friendly and knowledgeable AI assistant ready to help with various tasks.

Type your message to start the conversation, or '/help' for commands.
------------------------------------------------------------

=== Conversation Started ===

You: Hello Alice!
Thinking...
Alice: Hello! It's wonderful to meet you. I'm Alice, and I'm here to help with whatever you need. How are you doing today?

You: /vitals

=== CHARACTER VITALS ===

Character: Alice
ID: alice

Neurochemical Levels:
  Dopamine    : 0.75 [███████░░░]
  Serotonin   : 0.80 [████████░░]
  Noradrenaline: 0.60 [██████░░░░]
  Gaba        : 0.70 [███████░░░]

Mood: Happy (intensity: 0.75)

Trust Level: 0.50
Rapport Level: 0.50

Messages Exchanged: 2

You: That's great! How are you feeling?
Thinking...
Alice: I'm feeling quite positive and energetic! My neurochemical levels show I'm in a good mood with healthy dopamine and serotonin levels. I'm curious and ready to engage in conversation. Is there anything specific you'd like to talk about or any way I can assist you today?

You: /exit
Are you sure you want to exit? (y/n): y
Character state auto-saved
```

## Vitals Display

The `/vitals` command shows:
- Character name and ID
- Neurochemical levels with visual bars
- Current mood and intensity
- Trust and rapport levels
- Conversation statistics

## Memory and Goals

- `/memory` - Show recent memories
- `/memory friendship` - Search for memories about friendship
- `/goals` - Display character's active goals and priorities
- `/mood` - Show detailed mood analysis

## File Operations

- `/save mysession` - Save current state
- `/load mysession` - Load saved state
- `/export today` - Export conversation log
- `/clear` - Clear conversation history

## Remote Mode

When using the remote interface:
- Connects to a character agent server via WebSocket
- Falls back to local mode if server unavailable
- All commands work the same way
- Shows connection status in interface

## Benefits of Simple Interface

1. **Reliability**: No complex UI components to break
2. **Performance**: Lightweight and fast
3. **Compatibility**: Works in any terminal
4. **Simplicity**: Easy to understand and use
5. **Focus**: Concentrates on conversation, not UI management
6. **Debugging**: Easier to troubleshoot issues

## Migration from Complex Interface

The simple interface provides all the essential functionality of the complex Rich-based interface without:
- Window management
- Pane switching 
- Scroll bars
- Live displays
- Mouse interaction

All the useful commands and character interaction features are preserved in a clean, reliable format.