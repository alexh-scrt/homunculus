# Agent Infrastructure Docker Compose

A complete Docker Compose setup for running AI agents with LLM serving, vector storage, graph database, and caching capabilities.

## Services

### 1. Ollama (LLM Server)
- **Port**: 11434
- **Purpose**: Serves large language models via API
- **GPU Support**: Configured for NVIDIA GPU acceleration
- **Models**: Pre-configured to pull llama3.2 and mistral (customizable)

### 2. Ollama-Init
- **Purpose**: Initializes Ollama with required models on first run
- **Shared Volume**: Uses the same volume as Ollama to persist models
- **Loading**: Not only pulls models but also loads them into memory via generate endpoint for faster first-use

### 3. ChromaDB (Vector Database)
- **Port**: 8000
- **Purpose**: Stores embeddings and performs similarity search
- **Persistence**: Data persisted to local volume
- **API**: RESTful API for vector operations

### 4. Neo4j (Graph Database)
- **Ports**: 
  - 7474 (HTTP Browser Interface)
  - 7687 (Bolt Protocol)
- **Purpose**: Stores knowledge graphs and relationships
- **Default Credentials**: neo4j/password123 (⚠️ CHANGE THIS!)

### 5. Redis (Cache & Message Broker)
- **Port**: 6379
- **Purpose**: Caching, session storage, pub/sub messaging
- **Persistence**: AOF (Append Only File) enabled
- **Memory Policy**: LRU eviction with 512MB max memory

## Quick Start

### Prerequisites
- Docker 20.10+
- Docker Compose 2.0+
- (Optional) NVIDIA GPU with Docker GPU support for Ollama

### Installation

1. **Clone/Copy the docker-compose.yml file**

2. **Configure environment variables**
   ```bash
   # Copy the example env file
   cp .env.example .env
   
   # Edit .env and update values (especially NEO4J_PASSWORD!)
   nano .env
   ```
   
   Key variables to configure:
   - `NEO4J_PASSWORD` - **Change this!** Default: `change_this_password_123`
   - `OLLAMA_DEFAULT_MODEL` - Default model to use (default: `llama3.2`)
   - `REDIS_MAX_MEMORY` - Redis memory limit (default: `512mb`)
   - `OLLAMA_NUM_GPU` - Number of GPUs to use (default: `1`)

3. **Start all services**
   ```bash
   docker-compose up -d
   ```

4. **Check service status**
   ```bash
   docker-compose ps
   docker-compose logs -f
   ```

### Without GPU Support

If you don't have NVIDIA GPU, remove the `deploy` section from both ollama services:

```yaml
# Remove these lines:
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

## Service Endpoints

| Service       | Endpoint               | Purpose           |
| ------------- | ---------------------- | ----------------- |
| Ollama        | http://localhost:11434 | LLM API           |
| ChromaDB      | http://localhost:8000  | Vector DB API     |
| Neo4j Browser | http://localhost:7474  | Graph DB UI       |
| Neo4j Bolt    | bolt://localhost:7687  | Graph DB Protocol |
| Redis         | localhost:6379         | Cache/Broker      |

## Customizing Ollama Models

Edit the `ollama-init` service command section to pull and load different models:

```yaml
command:
  - -c
  - |
    ollama serve &
    sleep 5
    
    # Pull models
    ollama pull llama3.2:latest
    ollama pull mistral:latest
    ollama pull codellama:latest  # Add more models
    
    # Load models into memory
    curl -s http://localhost:11434/api/generate -d '{
      "model": "llama3.2:latest",
      "prompt": "Hi",
      "stream": false
    }' > /dev/null
    
    curl -s http://localhost:11434/api/generate -d '{
      "model": "codellama:latest",
      "prompt": "Hi",
      "stream": false
    }' > /dev/null
    
    wait
```

Available models: https://ollama.ai/library

## Environment Variables

All services are configured via the `.env` file. Here are the key variables:

### Neo4j
- `NEO4J_USER` - Database username (default: `neo4j`)
- `NEO4J_PASSWORD` - Database password (**CHANGE THIS!**)
- `NEO4J_URI` - Connection URI (default: `bolt://localhost:7687`)
- `NEO4J_HTTP_PORT` - Web interface port (default: `7474`)
- `NEO4J_BOLT_PORT` - Bolt protocol port (default: `7687`)

### Redis
- `REDIS_HOST` - Redis hostname (default: `localhost`)
- `REDIS_PORT` - Redis port (default: `6379`)
- `REDIS_MAX_MEMORY` - Maximum memory allocation (default: `512mb`)

### ChromaDB
- `CHROMA_HOST` - ChromaDB hostname (default: `localhost`)
- `CHROMA_PORT` - ChromaDB port (default: `8000`)
- `CHROMA_PERSIST_DIRECTORY` - Data directory (default: `/chroma/chroma`)

### Ollama
- `OLLAMA_HOST` - API endpoint (default: `http://localhost:11434`)
- `OLLAMA_DEFAULT_MODEL` - Default model name (default: `llama3.2`)
- `OLLAMA_NUM_GPU` - Number of GPUs to use (default: `1`)
- `OLLAMA_NUM_THREADS` - CPU threads for inference (default: `8`)

### Agent Configuration
- `AGENT_LOG_LEVEL` - Logging level (default: `INFO`)
- `AGENT_MAX_ITERATIONS` - Maximum iterations per task (default: `10`)
- `AGENT_TIMEOUT` - Task timeout in seconds (default: `300`)

You can override any of these in your `.env` file.

## Usage Examples

### Python - Ollama
```python
import requests

response = requests.post('http://localhost:11434/api/generate',
    json={
        'model': 'llama3.2',
        'prompt': 'Why is the sky blue?'
    })
print(response.json())
```

### Python - ChromaDB
```python
import chromadb

client = chromadb.HttpClient(host='localhost', port=8000)
collection = client.create_collection("my_collection")

collection.add(
    documents=["This is a document", "This is another document"],
    ids=["id1", "id2"]
)
```

### Python - Neo4j
```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password123")
)

with driver.session() as session:
    result = session.run("MATCH (n) RETURN count(n)")
    print(result.single()[0])
```

### Python - Redis
```python
import redis

r = redis.Redis(host='localhost', port=6379, decode_responses=True)
r.set('key', 'value')
print(r.get('key'))
```

## Health Checks

All services include health checks. Monitor with:

```bash
docker-compose ps
```

Healthy services will show "(healthy)" status.

## Data Persistence

All data is persisted in Docker volumes:
- `ollama-data`: LLM models
- `chroma-data`: Vector embeddings
- `neo4j-data`: Graph database
- `redis-data`: Cache data

### Backup volumes
```bash
docker run --rm -v ollama-data:/data -v $(pwd):/backup alpine tar czf /backup/ollama-backup.tar.gz /data
```

### Clean up volumes (⚠️ Destroys data)
```bash
docker-compose down -v
```

## Troubleshooting

### Ollama not responding
```bash
docker-compose logs ollama
docker-compose restart ollama
```

### ChromaDB connection errors
```bash
curl http://localhost:8000/api/v1/heartbeat
```

### Neo4j authentication issues
- Default: neo4j/password123
- Reset by removing neo4j-data volume and recreating

### Redis memory issues
Adjust `maxmemory` in docker-compose.yml redis command

## Resource Requirements

- **Minimum**: 8GB RAM, 20GB disk
- **Recommended**: 16GB RAM, 50GB disk, NVIDIA GPU
- **Ollama with large models**: 32GB RAM recommended

## Security Notes

⚠️ **IMPORTANT**: This configuration is for development/local use:
1. Change Neo4j default password
2. Add authentication to ChromaDB for production
3. Use Redis password protection in production
4. Configure proper network isolation
5. Use TLS/SSL for production endpoints

## Stopping Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (deletes data)
docker-compose down -v
```

## Network

All services communicate via the `agent-network` bridge network, allowing:
- Service discovery by container name
- Isolated networking
- Inter-service communication

Example: Agents can access Ollama at `http://ollama:11434` from within the network.