# Quick Start Guide

## 📦 What's Included

Your agent infrastructure package contains:

```
├── docker-compose.yml    # Main orchestration file
├── .env                  # Environment configuration (with defaults)
├── .env.example          # Environment template
├── README.md             # Full documentation
├── setup.sh              # Automated setup script
├── Makefile              # Easy management commands
├── agent_example.py      # Python integration example
└── requirements.txt      # Python dependencies
```

## 🚀 Quick Start (3 Steps)

### Option A: Using the Setup Script (Recommended)

```bash
# 1. Make setup script executable
chmod +x setup.sh

# 2. Run setup (pulls images, creates .env, starts services)
./setup.sh

# 3. Done! Services are running
```

### Option B: Using Make Commands

```bash
# 1. Setup everything
make setup

# That's it! Or use individual commands:
make up      # Start services
make status  # Check status
make logs    # View logs
make down    # Stop services
```

### Option C: Manual Setup

```bash
# 1. Configure environment (IMPORTANT: Change NEO4J_PASSWORD!)
cp .env.example .env
nano .env

# 2. Start services
docker-compose up -d

# 3. Check status
docker-compose ps
```

## 🔐 Security Note

**IMPORTANT:** Edit `.env` and change `NEO4J_PASSWORD` before running in any non-local environment!

```bash
nano .env
# Change: NEO4J_PASSWORD=change_this_password_123
# To:     NEO4J_PASSWORD=your_secure_password_here
```

## 📊 Access Your Services

Once running, access at:

| Service           | URL                    | Credentials    |
| ----------------- | ---------------------- | -------------- |
| **Ollama API**    | http://localhost:11434 | None           |
| **ChromaDB**      | http://localhost:8000  | None           |
| **Neo4j Browser** | http://localhost:7474  | See .env file  |
| **Neo4j Bolt**    | bolt://localhost:7687  | See .env file  |
| **Redis**         | localhost:6379         | None (default) |

## 🧪 Test Your Setup

```bash
# Option 1: Use Make
make test

# Option 2: Run Python example
pip install -r requirements.txt
python agent_example.py

# Option 3: Manual tests
curl http://localhost:11434/api/tags              # Ollama
curl http://localhost:8000/api/v1/heartbeat       # ChromaDB
curl http://localhost:7474                        # Neo4j
redis-cli ping                                    # Redis
```

## 📝 Configuration

All configuration is in `.env`:

```bash
# LLM Configuration
OLLAMA_DEFAULT_MODEL=llama3.2
OLLAMA_NUM_GPU=1

# Database Passwords
NEO4J_PASSWORD=change_this_password_123  # ⚠️ CHANGE THIS!

# Resource Limits
REDIS_MAX_MEMORY=512mb
```

## 🔧 Common Commands

```bash
# Start services
docker-compose up -d
# or
make up

# View logs
docker-compose logs -f
# or
make logs

# Stop services
docker-compose down
# or
make down

# Restart a specific service
docker-compose restart ollama

# Check service health
docker-compose ps
# or
make status
```

## 🐍 Python Integration

```python
from agent_example import AgentInfrastructure

# Automatically loads from .env
infra = AgentInfrastructure()

# Query LLM
response = infra.query_llm("What is AI?")

# Store in vector DB
infra.store_embeddings("my_collection", ["document text"])

# Search similar documents
results = infra.similarity_search("my_collection", "query")

# Create knowledge graph
node_id = infra.create_knowledge_node("Concept", {"name": "AI"})

# Cache data
infra.cache_set("key", {"data": "value"})
```

## 🛑 Stopping & Cleanup

```bash
# Stop services (keeps data)
docker-compose down
# or
make down

# Stop and delete ALL data (⚠️ destructive)
docker-compose down -v
# or
make clean-all
```

## 🆘 Troubleshooting

**Services not starting?**
```bash
docker-compose logs [service-name]
# Example: docker-compose logs ollama
```

**Port already in use?**
Edit `.env` and change the port numbers:
```bash
NEO4J_HTTP_PORT=7475
CHROMA_PORT=8001
```

**No GPU?**
Remove the `deploy` sections from `docker-compose.yml` for ollama and ollama-init services.

**Need to reset everything?**
```bash
make clean-all  # Removes all data
make setup      # Fresh start
```

## 📚 Next Steps

1. ✅ Change Neo4j password in `.env`
2. ✅ Run `make test` to verify all services
3. ✅ Try `python agent_example.py`
4. ✅ Read `README.md` for detailed documentation
5. ✅ Customize models in `docker-compose.yml`

## 💡 Pro Tips

- Use `make help` to see all available commands
- Models persist between restarts (stored in volumes)
- Neo4j browser has a great query interface
- Check `agent_example.py` for integration patterns
- All services communicate via `agent-network`

## 🎯 What's Pre-configured

✅ Ollama with llama3.2 and mistral (auto-loaded into memory)  
✅ ChromaDB with persistence  
✅ Neo4j with optimized memory settings  
✅ Redis with LRU caching  
✅ Health checks for all services  
✅ Shared network for service communication  
✅ Volume persistence for all data  

---

**Ready to build agents?** Start with `./setup.sh` and you're good to go! 🚀