# AI Agent Survival Arena - HOWTO Guide

![AI Agent Survival Arena](./img/arena.png)

## Overview

Arena is a competitive AI agent training system that enables multi-agent games with elimination mechanics, tournaments, and comprehensive analytics. This guide covers everything you need to get Arena up and running.

## Prerequisites

- Python 3.13+
- Docker and Docker Compose (for optional services)
- Git

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd homunculus

# Install dependencies
pip install -r requirements.txt

# Verify installation
python arena_cli.py --version
```

### 2. Quick Test

Run a simple 3-agent game to verify everything works:

```bash
# Create test agents
python arena_cli.py agent-create alice -n "Alice" -t character
python arena_cli.py agent-create bob -n "Bob" -t character  
python arena_cli.py agent-create charlie -n "Charlie" -t character

# Start a quick test game
python arena_cli.py start test_game -a alice bob charlie -m 50

# Watch the game progress
python arena_cli.py watch test_game --follow

# View results when complete
python arena_cli.py stats
python arena_cli.py leaderboard
```

## Installation Options

### Option 1: Minimal Setup (Recommended for Testing)

For quick testing without external dependencies:

```bash
# Install basic dependencies
pip install pydantic asyncio

# Set development mode (skips LLM and Kafka)
export ARENA_DEV_MODE=true

# Run Arena CLI
python arena_cli.py interactive
```

### Option 2: Full Setup with Services

For production use with all features:

```bash
# Install all dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env.arena
# Edit .env.arena with your API keys and settings
```

## Docker Services

Arena can use several optional services. Start them in this order:

### 1. Start Infrastructure Services

```bash
# Start Kafka (for message bus)
docker-compose up -d kafka zookeeper

# Start PostgreSQL (for persistence)
docker-compose up -d postgres

# Start Redis (for caching)
docker-compose up -d redis
```

### 2. Start Arena Services

```bash
# Start Arena backend services
docker-compose up -d arena-orchestrator arena-analytics

# Start web interface (optional)
docker-compose up -d arena-web
```

### 3. Verify Services

```bash
# Check all services are running
docker-compose ps

# Check logs
docker-compose logs arena-orchestrator
```

## Configuration

### Environment Variables

Create `.env.arena` file:

```bash
# Core settings
ARENA_DATA_DIR=/path/to/arena/data
ARENA_LOG_LEVEL=INFO

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/arena

# Message Bus
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# LLM Configuration
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here

# Development mode (skip external services)
ARENA_DEV_MODE=false
```

### Configuration File

Arena stores settings in `~/.arena/config.json`:

```json
{
  "data_dir": "~/.arena/data",
  "log_level": "INFO",
  "auto_save": true,
  "auto_save_interval": 10,
  "max_parallel_games": 5,
  "default_max_turns": 100,
  "database_url": "sqlite:///arena.db",
  "kafka_enabled": false
}
```

## CLI Commands Reference

### Game Management

```bash
# Start a new game
python arena_cli.py start <game_id> -a <agent1> <agent2> <agent3> [options]
  --max-turns 100        # Maximum turns (default: 100)
  --mode competitive     # Game mode: competitive/cooperative/mixed

# Stop a running game
python arena_cli.py stop <game_id>

# Save game state
python arena_cli.py save <game_id> -n "save_name"

# Load saved game
python arena_cli.py load <save_id>

# List games
python arena_cli.py list --status all  # all/active/completed

# Watch game in progress
python arena_cli.py watch <game_id> --follow
```

### Agent Management

```bash
# Create agent
python arena_cli.py agent-create <agent_id> -n "Name" -t character
  --profile profile.json  # Character profile file

# List agents
python arena_cli.py agents --type character

# Show agent info
python arena_cli.py agent-info <agent_id>

# Show agent statistics
python arena_cli.py agent-stats <agent_id>
```

### Tournament Management

```bash
# Create tournament
python arena_cli.py tournament <tournament_id> -a <agents...> -f single-elim
  -f single-elim         # single-elim/double-elim/round-robin/swiss

# Check tournament status
python arena_cli.py tournament-status <tournament_id>

# Show bracket
python arena_cli.py bracket <tournament_id>
```

### Analytics & Replays

```bash
# View overall statistics
python arena_cli.py stats --period week

# Show leaderboard
python arena_cli.py leaderboard --metric performance --limit 10

# List replays
python arena_cli.py replays --limit 20

# Play replay
python arena_cli.py replay <replay_id> -s 2.0  # 2x speed

# Analyze replay
python arena_cli.py analyze <replay_id>

# Export data
python arena_cli.py export results.json --format json --games game1 game2
```

### Configuration

```bash
# Show configuration
python arena_cli.py config

# Set configuration value
python arena_cli.py config-set max_parallel_games 10

# Reset configuration
python arena_cli.py config-reset
```

## Testing Arena

### Unit Tests

```bash
# Run all tests
pytest tests/

# Run specific test suites
pytest tests/arena/test_phase1_infrastructure.py -v
pytest tests/arena/test_phase3_message_bus.py -v
pytest tests/arena/test_phase7_persistence.py -v

# Run with coverage
pytest --cov=src/arena tests/
```

### Integration Tests

```bash
# Test CLI functionality
python arena_cli.py agent-create test_agent -n "Test Agent" -t character
python arena_cli.py start integration_test -a test_agent -m 10
python arena_cli.py watch integration_test
```

### Performance Tests

```bash
# Run performance benchmarks
python tests/performance/benchmark_message_bus.py
python tests/performance/benchmark_scoring.py
```

## Common Workflows

### Workflow 1: Single Game

```bash
# 1. Create agents
python arena_cli.py agent-create alpha -n "Agent Alpha" -t character
python arena_cli.py agent-create beta -n "Agent Beta" -t character
python arena_cli.py agent-create gamma -n "Agent Gamma" -t character

# 2. Start game
python arena_cli.py start my_game -a alpha beta gamma -m 100

# 3. Monitor progress
python arena_cli.py watch my_game --follow

# 4. Save checkpoint (optional)
python arena_cli.py save my_game -n "checkpoint_50"

# 5. View results
python arena_cli.py stats
python arena_cli.py agent-stats alpha

# 6. Analyze replay
python arena_cli.py analyze my_game
```

### Workflow 2: Tournament

```bash
# 1. Create multiple agents
for i in {1..8}; do
  python arena_cli.py agent-create agent_$i -n "Agent $i" -t character
done

# 2. Create tournament
python arena_cli.py tournament summer_championship \
  -a agent_1 agent_2 agent_3 agent_4 agent_5 agent_6 agent_7 agent_8 \
  -f single-elim

# 3. Monitor tournament
python arena_cli.py tournament-status summer_championship
python arena_cli.py bracket summer_championship

# 4. Export results
python arena_cli.py export tournament_results.html --format html
```

### Workflow 3: Agent Development

```bash
# 1. Create development agent
python arena_cli.py agent-create dev_agent -n "Development Agent" -t character

# 2. Run multiple test games
for i in {1..5}; do
  python arena_cli.py start test_$i -a dev_agent alpha beta -m 50
done

# 3. Analyze performance
python arena_cli.py agent-stats dev_agent
python arena_cli.py leaderboard

# 4. Export detailed analytics
python arena_cli.py export dev_analysis.json --format json
```

## Running with Docker

### Full Docker Setup

```bash
# Build Arena image
docker build -t arena:latest .

# Run with docker-compose
docker-compose up -d

# Execute CLI commands in container
docker-compose exec arena python arena_cli.py interactive

# View logs
docker-compose logs -f arena
```

### Docker Compose Services

The `docker-compose.yml` includes:

1. **Infrastructure Services:**
   ```bash
   docker-compose up -d zookeeper kafka postgres redis
   ```

2. **Arena Services:**
   ```bash
   docker-compose up -d arena-orchestrator arena-analytics arena-web
   ```

3. **Development:**
   ```bash
   docker-compose up -d arena-dev  # Development environment
   ```

## Troubleshooting

### Common Issues

#### 1. "No module named 'src.arena'"
```bash
# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# Or run from project root
cd /path/to/homunculus
python arena_cli.py --help
```

#### 2. "Agent not found" errors
```bash
# List available agents
python arena_cli.py agents

# Create missing agents
python arena_cli.py agent-create missing_agent -n "Missing Agent" -t character
```

#### 3. "Kafka connection failed"
```bash
# Check if Kafka is running
docker-compose ps kafka

# Start Kafka services
docker-compose up -d zookeeper kafka

# Or disable Kafka for testing
python arena_cli.py config-set kafka_enabled false
```

#### 4. Database connection issues
```bash
# Check PostgreSQL
docker-compose ps postgres

# Use SQLite for testing
python arena_cli.py config-set database_url "sqlite:///arena.db"
```

#### 5. Permission errors
```bash
# Create Arena data directory
mkdir -p ~/.arena/data
chmod 755 ~/.arena

# Fix ownership
sudo chown -R $USER:$USER ~/.arena
```

### Debug Mode

Enable verbose logging:

```bash
# Set debug level
export ARENA_LOG_LEVEL=DEBUG

# Run with verbose output
python arena_cli.py --verbose start test_game -a alice bob charlie

# Check logs
tail -f ~/.arena/logs/arena.log
```

### Log Files

Arena logs are stored in:
- `~/.arena/logs/arena.log` - Main application log
- `~/.arena/logs/games/` - Individual game logs
- `~/.arena/logs/tournaments/` - Tournament logs

## Performance Tuning

### For High-Throughput Games

```bash
# Increase parallel games
python arena_cli.py config-set max_parallel_games 20

# Reduce auto-save frequency
python arena_cli.py config-set auto_save_interval 50

# Enable caching
python arena_cli.py config-set redis_enabled true
```

### For Large Tournaments

```bash
# Use batch processing
python arena_cli.py config-set batch_processing true

# Increase worker threads
export ARENA_WORKER_THREADS=10

# Use faster storage
python arena_cli.py config-set storage_backend "memory"
```

## Development Mode

For development and testing without external services:

```bash
# Enable development mode
export ARENA_DEV_MODE=true

# Skip LLM calls
export ARENA_SKIP_LLM=true

# Use in-memory storage
python arena_cli.py config-set storage_backend "memory"

# Start development server
python arena_cli.py interactive
```

## Production Deployment

### Preparation

1. **Environment Setup:**
   ```bash
   # Production environment file
   cp .env.example .env.production
   # Configure with production values
   ```

2. **Database Setup:**
   ```bash
   # Initialize production database
   python -c "from src.arena.persistence import DatabaseManager; DatabaseManager().create_tables()"
   ```

3. **Service Dependencies:**
   ```bash
   # Start all services
   docker-compose -f docker-compose.prod.yml up -d
   ```

### Monitoring

```bash
# Check system status
python arena_cli.py stats
docker-compose ps

# Monitor logs
docker-compose logs -f --tail=100

# Check resource usage
docker stats
```

## API Integration

Arena can be used programmatically:

```python
from src.arena.cli import ArenaCLI
import asyncio

async def run_tournament():
    cli = ArenaCLI()
    
    # Create tournament
    await cli.tournament_commands.create_tournament(
        "api_tournament",
        ["agent1", "agent2", "agent3", "agent4"],
        "single-elim"
    )
    
    # Monitor status
    status = await cli.tournament_commands.show_status("api_tournament")
    print(f"Tournament status: {status}")

# Run
asyncio.run(run_tournament())
```

## Support

### Documentation
- Full CLI Reference: `docs/ARENA_CLI_GUIDE.md`
- Implementation Details: `design/arena_implementation_progress.md`
- API Documentation: `docs/api/`

### Getting Help
- Use `python arena_cli.py --help` for command help
- Use `python arena_cli.py interactive` then `help` for interactive help
- Check logs in `~/.arena/logs/`

### Reporting Issues
- Check logs for error details
- Include command that failed
- Include environment information:
  ```bash
  python arena_cli.py config
  docker-compose ps
  ```

---

**Arena is ready for competitive AI training! 🏆**

Start with the Quick Test section above, then explore the full capabilities using the CLI commands and workflows described in this guide.