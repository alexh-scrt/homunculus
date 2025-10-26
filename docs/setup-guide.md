# Setup Guide - Homunculus Character Agent System

This guide provides detailed setup instructions for the Homunculus character agent system.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Database Setup](#database-setup)
- [Configuration](#configuration)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Software

1. **Python 3.11 or higher**
   ```bash
   python --version  # Should be 3.11+
   ```

2. **Docker and Docker Compose**
   ```bash
   docker --version
   docker-compose --version
   ```

3. **Ollama with a compatible model**
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull the recommended model
   ollama pull llama3.3:70b
   ```

### System Requirements

- **RAM**: Minimum 16GB (32GB recommended for llama3.3:70b)
- **Storage**: At least 50GB free space for models and data
- **CPU**: Modern multi-core processor (8+ cores recommended)
- **GPU**: Optional but recommended (NVIDIA GPU with CUDA support)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd homunculus
```

### 2. Install Dependencies

#### Option A: Using Poetry (Recommended)

```bash
# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

#### Option B: Using pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
# Test basic imports
python -c "import src.config.settings; print('✓ Basic imports working')"

# List available characters
python scripts/run_chat.py --list-characters
```

## Database Setup

### 1. Start Database Services

```bash
# Start Neo4j and Redis using Docker Compose
docker-compose up -d

# Verify services are running
docker-compose ps
```

Expected output:
```
       Name                     Command               State                          Ports
------------------------------------------------------------------------------------------------
homunculus_neo4j_1    docker-entrypoint.sh neo4j      Up      0.0.0.0:7474->7474/tcp, 0.0.0.0:7687->7687/tcp
homunculus_redis_1    docker-entrypoint.sh redis ...   Up      0.0.0.0:6379->6379/tcp
```

### 2. Initialize Databases

```bash
# Run the setup script
python scripts/setup_databases.py
```

Expected output:
```
Character Agent System - Database Setup
==================================================
Current Configuration:
==================================================
Ollama URL: http://localhost:11434
Ollama Model: llama3.3:70b
ChromaDB Directory: ./data/chroma_db
Neo4j URI: bolt://localhost:7687
Neo4j User: neo4j
Neo4j Password: ************************
==================================================

Testing Ollama connection...
✓ Ollama: Connection successful

Testing ChromaDB connection...
✓ ChromaDB: Experience stored successfully
✓ ChromaDB: Experience retrieved successfully

Testing Neo4j connection...
✓ Neo4j: Connection established
✓ Neo4j: Entity stored successfully
✓ Neo4j: Query executed successfully

Setup Summary:
==================================================
✓ Ollama (LLM) (required)
✓ ChromaDB (Memory) (required)
✓ Neo4j (Knowledge Graph) (required)
==================================================
🎉 All required services are working!
```

## Configuration

### 1. Environment Configuration

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit the `.env` file with your settings:

```bash
# Core Configuration
ENVIRONMENT=development
DEBUG=true

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.3:70b

# Database Configuration
CHROMA_PERSIST_DIRECTORY=./data/chroma_db
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

# Redis Configuration (optional for current version)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Optional: Web Search Integration
TAVILY_API_KEY=your_tavily_api_key_here

# Performance Tuning
DEFAULT_TEMPERATURE=0.7
DEFAULT_MAX_TOKENS=500
AGENT_CONSULTATION_MAX_TOKENS=200
MAX_CONVERSATION_HISTORY=20

# Neurochemical System
HORMONE_DECAY_INTERVAL_SECONDS=30
ENABLE_HORMONE_DECAY=true

# Directory Configuration
CHARACTER_SCHEMAS_DIR=schemas/characters
DATA_DIR=./data
SAVES_DIR=./data/saves
```

### 2. Neo4j Password Setup

Set the Neo4j password:

```bash
# Access Neo4j browser interface
open http://localhost:7474

# Default credentials:
# Username: neo4j
# Password: neo4j

# Change password to something secure and update .env file
```

### 3. Directory Structure

Ensure the following directories exist:

```bash
mkdir -p data/chroma_db
mkdir -p data/logs
mkdir -p data/saves
```

## Verification

### 1. Run System Tests

```bash
# Run the complete test suite
python -m pytest -v

# Test specific components
python -m pytest tests/test_cli/ -v           # CLI tests
python -m pytest tests/test_integration/ -v   # Integration tests

# Test character validation
python scripts/run_chat.py --list-characters
```

### 2. Test Character Interaction

```bash
# Start interactive chat
python scripts/run_chat.py

# Or chat with a specific character
python scripts/run_chat.py --character ada_lovelace --debug
```

### 3. Performance Verification

```bash
# Run performance tests
python -m pytest tests/test_integration/test_performance.py -v

# Check character loading performance
time python scripts/run_chat.py --list-characters
```

## Troubleshooting

### Common Issues

#### 1. "Connection refused" to Ollama

**Symptoms**: `ConnectionError` when trying to connect to Ollama

**Solutions**:
```bash
# Check if Ollama is running
ps aux | grep ollama

# Start Ollama service
ollama serve

# Verify model is available
ollama list

# Test direct connection
curl http://localhost:11434/api/tags
```

#### 2. Docker Services Not Starting

**Symptoms**: `docker-compose up -d` fails or services show as unhealthy

**Solutions**:
```bash
# Check Docker daemon
docker info

# Check port conflicts
netstat -tulpn | grep -E '(6379|7687|7474)'

# Restart Docker services
docker-compose down
docker-compose up -d

# Check logs
docker-compose logs neo4j
docker-compose logs redis
```

#### 3. Import Errors

**Symptoms**: `ModuleNotFoundError` or import issues

**Solutions**:
```bash
# Ensure virtual environment is activated
poetry shell  # or source venv/bin/activate

# Reinstall dependencies
poetry install  # or pip install -r requirements.txt

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

#### 4. Character Loading Issues

**Symptoms**: "Character not found" or YAML parsing errors

**Solutions**:
```bash
# Validate character files
python -c "
import yaml
from pathlib import Path
for f in Path('schemas/characters').glob('*.yaml'):
    try:
        with open(f) as file:
            yaml.safe_load(file)
        print(f'✓ {f.name}')
    except Exception as e:
        print(f'✗ {f.name}: {e}')
"

# Check character directory
ls -la schemas/characters/

# Test character validation
python -m pytest tests/test_integration/test_character_conversations.py::TestCharacterValidation::test_character_loader_validation -v
```

#### 5. Memory/Performance Issues

**Symptoms**: Slow responses, high memory usage, or system freezing

**Solutions**:
```bash
# Check system resources
htop  # or top on some systems

# Monitor Docker resource usage
docker stats

# Reduce model size (if needed)
# Edit .env file:
# OLLAMA_MODEL=llama3.2:3b  # Smaller, faster model

# Clear ChromaDB data (if corrupted)
rm -rf ./data/chroma_db/*
python scripts/setup_databases.py
```

### Performance Optimization

#### For Lower-End Systems

```bash
# Use smaller model
OLLAMA_MODEL=llama3.2:3b

# Reduce conversation history
MAX_CONVERSATION_HISTORY=10

# Reduce token limits
DEFAULT_MAX_TOKENS=250
AGENT_CONSULTATION_MAX_TOKENS=100
```

#### For High-End Systems

```bash
# Use larger model for better quality
OLLAMA_MODEL=llama3.3:70b

# Increase limits for richer interactions
MAX_CONVERSATION_HISTORY=50
DEFAULT_MAX_TOKENS=800
AGENT_CONSULTATION_MAX_TOKENS=300
```

### Getting Help

If you continue to experience issues:

1. **Check the logs**:
   ```bash
   tail -f data/logs/homunculus.log
   ```

2. **Run diagnostics**:
   ```bash
   python scripts/setup_databases.py
   ```

3. **Verify environment**:
   ```bash
   python -c "
   from src.config.settings import get_settings
   settings = get_settings()
   print('Settings loaded successfully')
   print(f'Ollama URL: {settings.ollama_base_url}')
   print(f'Model: {settings.ollama_model}')
   "
   ```

4. **Check system resources**:
   ```bash
   # Available RAM
   free -h
   
   # Available disk space
   df -h
   
   # CPU usage
   htop
   ```

5. **Create an issue** on the project repository with:
   - Your system specifications
   - Error messages
   - Steps to reproduce the issue
   - Output from diagnostic commands

## Next Steps

Once setup is complete:

1. **Explore the characters**: `python scripts/run_chat.py --list-characters`
2. **Have conversations**: Try different characters and notice their distinct personalities
3. **Use debug mode**: Add `--debug` to see the internal decision-making process
4. **Experiment with settings**: Adjust parameters in `.env` to see how they affect behavior
5. **Create your own character**: Follow the character creation guide in the main README

## Maintenance

### Regular Maintenance Tasks

```bash
# Update dependencies (monthly)
poetry update  # or pip install -r requirements.txt --upgrade

# Clean up old data (as needed)
rm -rf ./data/chroma_db/*
rm -rf ./data/saves/*

# Restart databases (weekly)
docker-compose restart

# Check for model updates
ollama list
ollama pull llama3.3:70b  # Re-pull latest version
```

### Backup Important Data

```bash
# Backup character states
cp -r ./data/saves ~/backups/homunculus-saves-$(date +%Y%m%d)

# Backup custom characters
cp -r ./schemas/characters ~/backups/homunculus-characters-$(date +%Y%m%d)

# Backup conversation logs
cp -r ./data/logs ~/backups/homunculus-logs-$(date +%Y%m%d)
```