# Arena CLI Guide

## Overview

Arena CLI is a command-line interface for managing competitive AI agent training games, tournaments, and analytics.

## Installation

```bash
# Install Arena and dependencies
pip install -r requirements.txt

# Make CLI executable
chmod +x arena_cli.py
```

## Quick Start

```bash
# Start a new game
python arena_cli.py start game1 -a alice bob charlie -m 100

# Watch the game
python arena_cli.py watch game1 --follow

# View statistics
python arena_cli.py stats

# Enter interactive mode
python arena_cli.py interactive
```

## Commands

### Game Management

#### `start` - Start a new game
```bash
python arena_cli.py start <game_id> -a <agents...> [options]

Options:
  -a, --agents      List of agent IDs (required)
  -m, --max-turns   Maximum number of turns (default: 100)
  --mode            Game mode: competitive/cooperative/mixed
  
Example:
  python arena_cli.py start game1 -a agent1 agent2 agent3 -m 150 --mode competitive
```

#### `stop` - Stop a running game
```bash
python arena_cli.py stop <game_id>

Example:
  python arena_cli.py stop game1
```

#### `save` - Save game state
```bash
python arena_cli.py save <game_id> [-n <name>]

Options:
  -n, --name    Save name (optional)
  
Example:
  python arena_cli.py save game1 -n "checkpoint_turn_50"
```

#### `load` - Load a saved game
```bash
python arena_cli.py load <save_id>

Example:
  python arena_cli.py load checkpoint_turn_50
```

#### `list` - List games
```bash
python arena_cli.py list [--status <status>]

Options:
  --status    Filter by status: active/completed/all
  
Example:
  python arena_cli.py list --status active
```

#### `watch` - Watch a game in progress
```bash
python arena_cli.py watch <game_id> [--follow]

Options:
  --follow    Follow game in real-time
  
Example:
  python arena_cli.py watch game1 --follow
```

### Agent Management

#### `agent-create` - Create a new agent
```bash
python arena_cli.py agent-create <agent_id> -n <name> [options]

Options:
  -n, --name      Agent display name (required)
  -t, --type      Agent type: llm/character/narrator/judge
  --profile       Character profile JSON file
  
Example:
  python arena_cli.py agent-create alpha -n "Agent Alpha" -t character --profile profiles/alpha.json
```

#### `agents` - List available agents
```bash
python arena_cli.py agents [--type <type>]

Options:
  --type    Filter by agent type
  
Example:
  python arena_cli.py agents --type character
```

#### `agent-info` - Show agent information
```bash
python arena_cli.py agent-info <agent_id>

Example:
  python arena_cli.py agent-info alpha
```

#### `agent-stats` - Show agent statistics
```bash
python arena_cli.py agent-stats <agent_id>

Example:
  python arena_cli.py agent-stats alpha
```

### Tournament Management

#### `tournament` - Create a tournament
```bash
python arena_cli.py tournament <tournament_id> -a <agents...> [options]

Options:
  -a, --agents    List of participating agents (required)
  -f, --format    Tournament format: single-elim/double-elim/round-robin/swiss
  
Example:
  python arena_cli.py tournament summer2024 -a alice bob charlie david -f single-elim
```

#### `tournament-status` - Show tournament status
```bash
python arena_cli.py tournament-status <tournament_id>

Example:
  python arena_cli.py tournament-status summer2024
```

#### `bracket` - Show tournament bracket
```bash
python arena_cli.py bracket <tournament_id>

Example:
  python arena_cli.py bracket summer2024
```

### Replay & Analysis

#### `replays` - List available replays
```bash
python arena_cli.py replays [--limit <n>]

Options:
  --limit    Number of replays to show (default: 10)
  
Example:
  python arena_cli.py replays --limit 20
```

#### `replay` - Play a game replay
```bash
python arena_cli.py replay <replay_id> [-s <speed>]

Options:
  -s, --speed    Playback speed (default: 1.0)
  
Example:
  python arena_cli.py replay game1 -s 2.0
```

#### `analyze` - Analyze a game replay
```bash
python arena_cli.py analyze <replay_id>

Example:
  python arena_cli.py analyze game1
```

### Statistics & Analytics

#### `stats` - Show overall statistics
```bash
python arena_cli.py stats [--period <period>]

Options:
  --period    Time period: day/week/month/all
  
Example:
  python arena_cli.py stats --period week
```

#### `leaderboard` - Show agent leaderboard
```bash
python arena_cli.py leaderboard [options]

Options:
  --metric    Ranking metric: wins/score/performance
  --limit     Number of agents to show
  
Example:
  python arena_cli.py leaderboard --metric performance --limit 10
```

#### `export` - Export game data
```bash
python arena_cli.py export <output> [options]

Options:
  --format    Export format: json/csv/excel/html
  --games     Game IDs to export
  
Example:
  python arena_cli.py export report.html --format html --games game1 game2
```

### Configuration

#### `config` - Show configuration
```bash
python arena_cli.py config [<key>]

Example:
  python arena_cli.py config
  python arena_cli.py config log_level
```

#### `config-set` - Set configuration value
```bash
python arena_cli.py config-set <key> <value>

Example:
  python arena_cli.py config-set max_parallel_games 10
  python arena_cli.py config-set auto_save true
```

#### `config-reset` - Reset configuration
```bash
python arena_cli.py config-reset

Example:
  python arena_cli.py config-reset
```

## Interactive Mode

Enter interactive mode for continuous command execution:

```bash
python arena_cli.py interactive
```

In interactive mode:
- Type commands without the `arena_cli.py` prefix
- Use `help` to see available commands
- Use `quit` or `exit` to leave

Example session:
```
arena> start game1 -a alice bob charlie
Game game1 started successfully!
arena> watch game1
...
arena> stats
...
arena> quit
Goodbye!
```

## Configuration File

Arena stores configuration in `~/.arena/config.json`:

```json
{
  "data_dir": "~/.arena/data",
  "log_level": "INFO",
  "auto_save": true,
  "auto_save_interval": 10,
  "max_parallel_games": 5,
  "default_max_turns": 100,
  "theme": "default"
}
```

## Environment Variables

- `ARENA_CONFIG`: Path to configuration file
- `ARENA_DATA_DIR`: Data directory path
- `ARENA_LOG_LEVEL`: Logging level (DEBUG/INFO/WARNING/ERROR)

## Output Formats

### Table Output
Games, agents, and statistics are displayed in formatted tables:
```
ID     | Status | Turn | Players | Phase
-------|--------|------|---------|-------
game1  | Active | 45   | 3       | MID
game2  | Saved  | 100  | 4       | FINAL
```

### Progress Indicators
Long-running operations show progress:
```
Processing [████████████░░░░░░░] 75.0% 
```

### Color Coding
- 🟢 Green: Success messages
- 🔴 Red: Error messages
- 🟡 Yellow: Warnings
- 🔵 Blue: Information
- 🥇 Gold: Winners/champions

## Advanced Usage

### Batch Operations
Process multiple games:
```bash
# Export multiple games
python arena_cli.py export batch.json --games game1 game2 game3

# Run tournament with many agents
python arena_cli.py tournament mega -a $(cat agents.txt)
```

### Scripting
Use Arena CLI in scripts:
```bash
#!/bin/bash
# Run daily tournament
DATE=$(date +%Y%m%d)
python arena_cli.py tournament daily_$DATE -a alice bob charlie -f round-robin
python arena_cli.py export results_$DATE.html --format html
```

### Monitoring
Watch multiple games:
```bash
# In separate terminals
python arena_cli.py watch game1 --follow
python arena_cli.py watch game2 --follow
```

## Troubleshooting

### Common Issues

1. **Game won't start**
   - Check agent IDs exist
   - Verify no game with same ID is running
   - Check log files for errors

2. **Can't load save**
   - Verify save file exists
   - Check save compatibility
   - Ensure game isn't already active

3. **Export fails**
   - Check output path is writable
   - Verify format is supported
   - Ensure enough disk space

### Debug Mode
Enable debug logging:
```bash
python arena_cli.py --verbose start game1 -a alice bob
```

### Log Files
Logs are stored in:
- `~/.arena/logs/arena.log` - Main log
- `~/.arena/logs/games/` - Per-game logs

## Examples

### Complete Game Workflow
```bash
# Create agents
python arena_cli.py agent-create alice -n "Alice" -t character
python arena_cli.py agent-create bob -n "Bob" -t character

# Start game
python arena_cli.py start test_game -a alice bob -m 50

# Watch progress
python arena_cli.py watch test_game --follow

# Save checkpoint
python arena_cli.py save test_game -n checkpoint1

# View stats
python arena_cli.py stats
python arena_cli.py leaderboard

# Analyze replay
python arena_cli.py analyze test_game

# Export results
python arena_cli.py export test_game.html --format html
```

### Tournament Example
```bash
# Create tournament
python arena_cli.py tournament winter2024 -a alice bob charlie david eve frank -f single-elim

# Check bracket
python arena_cli.py bracket winter2024

# Monitor status
python arena_cli.py tournament-status winter2024

# Export results
python arena_cli.py export tournament_results.json --format json
```

## API Integration

Arena CLI can be used programmatically:

```python
from src.arena.cli import ArenaCLI

async def run_game():
    cli = ArenaCLI()
    await cli.game_commands.start_game(
        "api_game",
        ["agent1", "agent2"],
        max_turns=100,
        mode="competitive"
    )
```

## Support

For issues and questions:
- GitHub Issues: https://github.com/homunculus/arena
- Documentation: https://docs.homunculus.ai/arena

## License

Part of the Homunculus Project
Copyright (c) 2024 Homunculus Team