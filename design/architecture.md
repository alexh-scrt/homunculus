# Agent Infrastructure Architecture

## 🏗️ System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Your AI Agents                          │
│                 (agent_example.py)                          │
└──────┬──────────┬──────────┬──────────┬────────────────────┘
       │          │          │          │
       │          │          │          │
   ┌───▼───┐  ┌──▼───┐  ┌───▼────┐  ┌─▼──────┐
   │Ollama │  │Chroma│  │ Neo4j  │  │ Redis  │
   │ :11434│  │ :8000│  │:7474   │  │ :6379  │
   │       │  │      │  │:7687   │  │        │
   └───────┘  └──────┘  └────────┘  └────────┘
       │          │          │          │
       └──────────┴──────────┴──────────┘
                  │
            agent-network
```

## 📦 Components

### 1. Ollama (LLM Server)
- **Purpose**: Serves large language models
- **Port**: 11434
- **Models**: llama3.2, mistral (auto-loaded)
- **Volume**: `ollama-data`
- **Use Case**: Text generation, embeddings, reasoning

### 2. Ollama-Init (Initialization)
- **Purpose**: Downloads and pre-loads models
- **Runs Once**: On first startup
- **Shares**: Same volume as Ollama
- **Benefit**: Models ready immediately

### 3. ChromaDB (Vector Database)
- **Purpose**: Semantic search, embeddings storage
- **Port**: 8000
- **Volume**: `chroma-data`
- **Use Case**: RAG, similarity search, document retrieval

### 4. Neo4j (Graph Database)
- **Purpose**: Knowledge graphs, relationships
- **Ports**: 7474 (UI), 7687 (Bolt)
- **Volumes**: `neo4j-data`, `neo4j-logs`, `neo4j-import`, `neo4j-plugins`
- **Use Case**: Entity relationships, graph queries, knowledge management

### 5. Redis (Cache & Broker)
- **Purpose**: Caching, session storage, pub/sub
- **Port**: 6379
- **Volume**: `redis-data`
- **Use Case**: Response caching, rate limiting, task queues

## 🔄 Data Flow Example

```
User Query
    │
    ▼
┌─────────────────┐
│   Your Agent    │
└────┬────────────┘
     │
     ├─1─► Redis (Check cache)
     │     Cache Hit? Return ✓
     │
     ├─2─► ChromaDB (Find similar docs)
     │     Retrieve relevant context
     │
     ├─3─► Neo4j (Query knowledge graph)
     │     Get related entities
     │
     ├─4─► Ollama (Generate response)
     │     With augmented context
     │
     └─5─► Redis (Store result)
           Cache for future queries
```

## 🎯 Agent Patterns

### Pattern 1: Simple Query
```python
# Direct LLM query
response = infra.query_llm("What is machine learning?")
```

### Pattern 2: RAG (Retrieval Augmented Generation)
```python
# 1. Search vector DB
docs = infra.similarity_search("knowledge", query)

# 2. Add context to prompt
prompt = f"Context: {docs}\n\nQuestion: {query}"

# 3. Generate with context
response = infra.query_llm(prompt)
```

### Pattern 3: Graph-Enhanced RAG
```python
# 1. Search vectors
vector_context = infra.similarity_search("docs", query)

# 2. Query graph
graph_context = infra.query_graph(
    "MATCH (n)-[r]->(m) WHERE n.content CONTAINS $keyword",
    {"keyword": query}
)

# 3. Combine and generate
response = infra.query_llm(combined_context + query)
```

### Pattern 4: Cached Agent
```python
# 1. Check cache first
cache_key = f"query:{hash(query)}"
if infra.cache_exists(cache_key):
    return infra.cache_get(cache_key)

# 2. Process query (RAG + Graph + LLM)
result = process_query(query)

# 3. Cache result
infra.cache_set(cache_key, result, expire=3600)
```

## 🔧 Configuration Matrix

| Service  | Config File | Key Settings            | Tunable |
| -------- | ----------- | ----------------------- | ------- |
| Ollama   | .env        | MODEL, NUM_GPU, THREADS | ✅       |
| ChromaDB | .env        | PORT, PERSIST_DIR       | ✅       |
| Neo4j    | .env        | PASSWORD, MEMORY        | ✅       |
| Redis    | .env        | MEMORY, PORT            | ✅       |

## 🚀 Scaling Considerations

### Horizontal Scaling
- **Ollama**: Run multiple instances, load balance
- **ChromaDB**: Distributed mode (enterprise)
- **Neo4j**: Clustering (enterprise)
- **Redis**: Redis Cluster or Sentinel

### Vertical Scaling
- **Ollama**: More GPU memory, larger models
- **Neo4j**: Increase heap size (2-4GB recommended)
- **Redis**: Increase max memory
- **ChromaDB**: More disk space

## 🔐 Security Layers

```
┌─────────────────────────────────────────┐
│  Layer 4: Application (Your Agents)    │
│  - Input validation                     │
│  - Rate limiting                        │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│  Layer 3: Authentication                │
│  - Neo4j credentials (.env)             │
│  - Redis password (optional)            │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│  Layer 2: Network                       │
│  - agent-network isolation              │
│  - Exposed ports (configurable)         │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│  Layer 1: Infrastructure                │
│  - Docker container isolation           │
│  - Volume encryption (host level)       │
└─────────────────────────────────────────┘
```

## 📊 Resource Requirements

### Minimum (Development)
- **CPU**: 4 cores
- **RAM**: 8GB
- **Disk**: 20GB
- **GPU**: Optional (CPU fallback)

### Recommended (Production)
- **CPU**: 8+ cores
- **RAM**: 16-32GB
- **Disk**: 100GB SSD
- **GPU**: NVIDIA with 8GB+ VRAM

### Per Service
| Service      | CPU     | RAM       | Disk     |
| ------------ | ------- | --------- | -------- |
| Ollama (GPU) | 2 cores | 4-8GB     | 10GB+    |
| Ollama (CPU) | 4 cores | 8-16GB    | 10GB+    |
| ChromaDB     | 1 core  | 2GB       | Variable |
| Neo4j        | 2 cores | 2-4GB     | Variable |
| Redis        | 1 core  | 512MB-2GB | 1GB      |

## 🔍 Monitoring

### Health Checks
All services include health checks:
```bash
docker-compose ps  # Check health status
```

### Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f ollama
docker-compose logs -f neo4j
```

### Metrics
```bash
# Container stats
docker stats

# Service-specific
curl http://localhost:11434/api/tags        # Ollama models
curl http://localhost:8000/api/v1/heartbeat # ChromaDB health
```

## 🛠️ Troubleshooting

### Common Issues

**Ollama not loading models**
- Check: `docker-compose logs ollama-init`
- Solution: Increase `sleep 5` to `sleep 10` in init script

**Neo4j authentication fails**
- Check: Password in `.env` matches connection string
- Solution: Reset by removing `neo4j-data` volume

**Out of memory**
- Check: `docker stats`
- Solution: Adjust memory limits in `.env`

**Port conflicts**
- Check: `netstat -tulpn | grep [PORT]`
- Solution: Change ports in `.env`

## 🎓 Best Practices

1. **Environment Variables**: Always use `.env`, never hardcode
2. **Volumes**: Regular backups of all data volumes
3. **Security**: Change default passwords before production
4. **Monitoring**: Set up log aggregation for production
5. **Updates**: Regular `docker-compose pull` for patches
6. **Testing**: Use `make test` after any changes
7. **Documentation**: Keep track of custom configurations

## 📚 Additional Resources

- [Ollama Documentation](https://github.com/ollama/ollama)
- [ChromaDB Documentation](https://docs.trychroma.com)
- [Neo4j Documentation](https://neo4j.com/docs)
- [Redis Documentation](https://redis.io/docs)
- [Docker Compose Documentation](https://docs.docker.com/compose)