# Getting Started with Homunculus

Welcome to Homunculus! This guide will help you get the system up and running quickly.

## Overview

Homunculus is an AI-powered character agent system with two main components:
- **Core System**: Character conversations with memory, personality, and neurochemical simulation
- **Arena System**: Multi-agent game environment for character competitions

## Prerequisites

Before you begin, ensure you have the following installed:

- **Docker & Docker Compose** (v3.8+)
- **Python 3.11+**
- **Git**
- **NVIDIA Docker Runtime** (for GPU acceleration with Ollama)

### Hardware Requirements

- **Minimum**: 8GB RAM, 4GB free disk space
- **Recommended**: 16GB+ RAM, 50GB+ free disk space (for Ollama models)
- **GPU**: NVIDIA GPU with 8GB+ VRAM for optimal performance

## Quick Start (Core System Only)

If you just want to chat with characters and don't need the Arena features:

### 1. Clone and Setup

```bash
git clone https://github.com/yourusername/homunculus.git
cd homunculus

# Create environment file
cp .env.example .env
# Edit .env with your preferred settings (optional)
```

### 2. Start Core Services

```bash
# Start only core services (Ollama, ChromaDB, Neo4j, Redis)
docker-compose -f docker-compose.unified.yml up -d ollama chromadb neo4j redis
```

This will:
- Start Ollama and automatically download the `llama3.3:70b` model
- Start ChromaDB for memory storage
- Start Neo4j for knowledge graphs
- Start Redis for caching

**Note**: The first startup will take 10-20 minutes as Ollama downloads the 70GB language model.

### 3. Install Python Dependencies

```bash
# Using Poetry (recommended)
poetry install

# Or using pip
pip install -r requirements.txt
```

### 4. Initialize Databases

```bash
python scripts/setup_databases.py
```

### 5. Start Chatting!

```bash
# Interactive character selection
python scripts/run_chat.py

# Or chat with a specific character
python scripts/run_chat.py --character ada_lovelace

# Enable debug mode to see agent thinking
python scripts/run_chat.py --character zen_master --debug
```

## Full Setup (Core + Arena)

If you want to use the Arena system for multi-agent games:

### 1. Clone and Setup

```bash
git clone https://github.com/yourusername/homunculus.git
cd homunculus

# Create environment files
cp .env.example .env
cp .env.arena.example .env.arena
# Edit both files with your preferred settings
```

### 2. Start All Services

```bash
# Start all services including Kafka, PostgreSQL, etc.
docker-compose -f docker-compose.unified.yml up -d
```

This starts:
- **Core services**: Ollama, ChromaDB, Neo4j, Redis
- **Arena services**: Kafka, Zookeeper, PostgreSQL
- **Optional**: Kafka UI for monitoring (only in dev mode)

### 3. Verify Services

```bash
# Check all services are running
docker-compose -f docker-compose.unified.yml ps

# Check service health
docker-compose -f docker-compose.unified.yml logs --tail=50
```

### 4. Install Python Dependencies

```bash
poetry install
# or
pip install -r requirements.txt
```

### 5. Initialize All Databases

```bash
python scripts/setup_databases.py
```

### 6. Run Arena CLI

```bash
# Start a game with specific characters
python arena_cli.py start game1 -a ada_lovelace captain_cosmos

# List active games
python arena_cli.py list --status active

# Stop a game
python arena_cli.py stop game1

# View game details
python arena_cli.py watch game1

# Run with different modes
python arena_cli.py start game2 -a zen_master tech_enthusiast --mode cooperative

# List all available character agents  
python arena_cli.py agents

# Get detailed info about a character
python arena_cli.py agent-info ada_lovelace
```

## Arena Agent Management

The Arena CLI automatically loads all available Homunculus characters and provides commands to manage them:

```bash
# List all characters
python arena_cli.py agents

# Filter by type
python arena_cli.py agents --type character

# Get detailed character information
python arena_cli.py agent-info ada_lovelace

# View character statistics
python arena_cli.py agent-stats zen_master
```

The Arena now shows real character data instead of placeholder agents:
- **Real Names**: Ada Lovelace, Captain Cosmos, Zen Master Kiku, etc.
- **Character Profiles**: Age, occupation, personality traits, background
- **Performance Tracking**: Games played, wins, win rate (starts at 0%)
- **Dynamic Loading**: Automatically discovers characters from `schemas/characters/`

## Available Characters

Homunculus includes 15 distinct character personalities:

### Analytical & Technical
- **Ada Lovelace** (`ada_lovelace`) - Brilliant mathematician and programmer
- **Alex CodeWalker** (`tech_enthusiast`) - Passionate technologist

### Wisdom & Teaching  
- **Zen Master Kiku** (`zen_master`) - Peaceful meditation teacher
- **Professor Elena Bright** (`friendly_teacher`) - Warm educator

### Adventure & Creativity
- **Captain Cosmos** (`captain_cosmos`) - Enthusiastic space explorer
- **Luna Starweaver** (`creative_artist`) - Passionate artist

### Expertise
- **Archmage Grimbold** (`grumpy_wizard`) - Cantankerous but brilliant wizard

### Personality Archetypes
- **Marcus Rivera** (`m-playful`) - Playful elementary teacher
- **Zoe Kim** (`f-playful`) - Whimsical graphic designer  
- **David Okonkwo** (`m-sarcastic`) - Sarcastic software engineer
- **Rachel Stern** (`f-sarcastic`) - Sharp attorney
- **Dr. James Morrison** (`m-serious`) - Contemplative philosophy professor
- **Dr. Anita Patel** (`f-serious`) - Intense surgeon
- **Tyler "TJ" Johnson** (`m-dumb`) - Humorous personal trainer
- **Brittany "Britt" Cooper** (`f-dumb`) - Sweet but ditzy influencer

## Chat Commands

While chatting with any character, you can use these commands:

- `/exit` - End the conversation
- `/debug` - Toggle debug mode (see agent thinking)
- `/save [filename]` - Save character state
- `/load [filename]` - Load saved character state
- `/memory [query]` - Search character's memories
- `/reset` - Reset character to initial state
- `/status` - Show character's current mood and stats
- `/help` - Show all available commands

## Service Ports

When running, these services will be available:

- **Ollama API**: http://localhost:11434
- **ChromaDB**: http://localhost:8000
- **Neo4j Browser**: http://localhost:7474 (user: `neo4j`, pass: `homunculus123`)
- **Redis**: localhost:6379
- **Kafka** (Arena): localhost:9092
- **PostgreSQL** (Arena): localhost:5432
- **Kafka UI** (dev mode): http://localhost:8080

## Troubleshooting

### Services Won't Start

```bash
# Check Docker is running
docker --version
docker-compose --version

# Check service logs
docker-compose -f docker-compose.unified.yml logs [service-name]

# Restart specific service
docker-compose -f docker-compose.unified.yml restart [service-name]
```

### Ollama Model Issues

```bash
# Check if model is downloaded
docker exec ollama ollama list

# Manually pull model if needed
docker exec ollama ollama pull llama3.3:70b

# Check Ollama logs
docker logs ollama
```

### Python Dependency Issues

```bash
# Clean install
pip uninstall -r requirements.txt -y
pip install -r requirements.txt

# Or with Poetry
poetry env remove python
poetry install
```

### Database Connection Issues

```bash
# Reset all databases
docker-compose -f docker-compose.unified.yml down -v
docker-compose -f docker-compose.unified.yml up -d
python scripts/setup_databases.py
```

### Setup Script Issues

If you see errors about incorrect method signatures:

```bash
# These errors indicate outdated interface usage and are now fixed
✗ ChromaDB: 'dict' object has no attribute 'experience_id'
✗ Neo4j: 'KnowledgeGraphModule' object has no attribute 'initialize'
```

**Solution**: The setup script has been updated to use the correct APIs. Make sure you're using the latest version.

### Neo4j Relationship Warnings

Neo4j may show warnings about missing relationship types - this is normal for fresh installations:

```
warn: relationship type does not exist. The relationship type `RELATES_TO` does not exist
```

These relationships are created automatically as the system is used.

### Arena CLI Issues

Common issues that have been resolved:

```bash
# ✅ FIXED: CharacterAgent.__init__() got an unexpected keyword argument 'agent_id'
# ✅ FIXED: TypeError: Cannot instantiate typing.Literal  
# ✅ FIXED: ImportError: LangGraph is required for orchestration
# ✅ FIXED: Games terminating immediately after starting
```

**Current Status**: ✅ **FULLY WORKING!** The Arena CLI is now operating with complete LangGraph orchestration in real mode. All core features are functional with proper state machine execution, turn management, and game completion.

**Arena Features Working**:
- ✅ Real character loading (all 15 Homunculus characters)
- ✅ Full LangGraph state machine orchestration
- ✅ Complete game flow with all nodes (start, setup, turns, scoring, elimination, phase transitions)
- ✅ Turn-based gameplay with round-robin speaker selection
- ✅ Multiple game modes (competitive, cooperative, mixed)
- ✅ Proper game termination and winner determination
- ✅ Event streaming and detailed execution logging
- ✅ Checkpoint management and state recovery
- ✅ Configurable recursion limits for complex games

**Recent Fixes**:
- ✅ **RESOLVED**: LangGraph execution hang due to infinite recursion
- ✅ **ADDED**: Safety termination limits and improved game-over logic
- ✅ **IMPROVED**: Event streaming with proper monitoring and timeout handling
- ✅ **ENHANCED**: Configuration system with `arena.yml` for easy customization
- ✅ **IMPLEMENTED**: Configurable recursion limits following best practices from talks project

### Performance Issues

If characters respond slowly:

1. **Check GPU utilization**: `nvidia-smi`
2. **Reduce model size**: Edit Ollama config to use `llama3.1:8b` instead
3. **Check memory**: Characters use ~100MB RAM each
4. **Disable debug mode**: Debug output slows responses

## Development Mode

For development with additional monitoring tools:

```bash
# Start with Kafka UI for monitoring
docker-compose -f docker-compose.unified.yml --profile dev up -d
```

This adds:
- **Kafka UI**: Monitor Kafka topics and messages at http://localhost:8080

## Configuration

### Arena Configuration (`arena.yml`)

The Arena system uses a comprehensive configuration file inspired by the talks project pattern:

```yaml
# Core orchestration settings
orchestration:
  recursion_limit: 250        # LangGraph recursion limit
  max_turns: 100             # Maximum turns per game
  min_agents: 2              # Minimum agents required
  checkpoint_frequency: 5     # Checkpoint every N turns

# Game termination settings  
termination:
  safety_turn_limit: 50      # Hard safety limit
  auto_terminate:
    max_turns_reached: true   # End at max turns
    single_survivor: true     # End with one agent
    safety_limit_reached: true # End at safety limit

# Elimination mechanics
elimination:
  enabled: true              # Enable elimination
  min_turn_threshold: 20     # Min turns before elimination
  frequency: 10              # Check every N turns
```

Key benefits:
- **Configurable recursion limits**: Prevent LangGraph hangs
- **Flexible termination**: Multiple safety mechanisms  
- **Easy customization**: Edit `arena.yml` without code changes
- **Production ready**: Proven patterns from talks project

### Environment Variables

Edit `.env` to customize:

```bash
# Model configuration
OLLAMA_NUM_GPU=1                    # Number of GPUs to use
OLLAMA_NUM_THREADS=8               # CPU threads for Ollama

# Database ports
NEO4J_HTTP_PORT=7474
NEO4J_BOLT_PORT=7687  
CHROMA_PORT=8000
REDIS_PORT=6379

# Memory limits
REDIS_MAX_MEMORY=512mb
```

For Arena features, edit `.env.arena`:

```bash
# PostgreSQL configuration
POSTGRES_DB=arena_db
POSTGRES_USER=arena_user  
POSTGRES_PASSWORD=arena_pass
POSTGRES_PORT=5432

# Kafka configuration  
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
```

### Character Customization

Create new characters by adding YAML files to `schemas/characters/`:

```yaml
character_id: "my_character"
name: "My Character"
archetype: "custom"

demographics:
  age: 25
  gender: "non-binary"
  occupation: "Artist"

initial_agent_states:
  personality:
    big_five:
      openness: 0.9
      conscientiousness: 0.6
      extraversion: 0.7
      agreeableness: 0.8
      neuroticism: 0.3
  # ... see existing character files for full schema
```

## Next Steps

- **Explore Characters**: Try conversations with different personality types
- **Read the Documentation**: Check out the full README.md for architecture details
- **Join Arena Games**: Use the Arena CLI for multi-character competitions
- **Customize**: Create your own characters or modify existing ones
- **Contribute**: Help improve the system on GitHub

## Support

- **GitHub Issues**: Report bugs or request features
- **Documentation**: See README.md for detailed architecture info
- **Character Schemas**: Check `schemas/characters/` for examples

---

**Happy Chatting!** 🎭

*Remember: These aren't chatbots with prompts—they're artificial humans with psychology.*