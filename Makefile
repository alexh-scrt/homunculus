.PHONY: help setup up down restart logs status clean clean-all test env

# Default target
help:
	@echo "Agent Infrastructure - Docker Compose Management"
	@echo "================================================"
	@echo ""
	@echo "Available commands:"
	@echo "  make setup      - Initial setup (create .env, pull images, start services)"
	@echo "  make up         - Start all services"
	@echo "  make down       - Stop all services"
	@echo "  make restart    - Restart all services"
	@echo "  make logs       - Show logs (Ctrl+C to exit)"
	@echo "  make status     - Show service status"
	@echo "  make test       - Test all service connections"
	@echo "  make clean      - Stop services and remove containers"
	@echo "  make clean-all  - Stop services and remove containers AND volumes (⚠️  deletes data)"
	@echo "  make env        - Create .env from .env.example"
	@echo ""

# Initial setup
setup: env
	@echo "🚀 Setting up Agent Infrastructure..."
	@chmod +x setup.sh
	@./setup.sh

# Create .env file
env:
	@if [ ! -f .env ]; then \
		echo "📝 Creating .env file..."; \
		cp .env.example .env; \
		echo "✓ Created .env file"; \
		echo "⚠️  Please edit .env and change NEO4J_PASSWORD!"; \
	else \
		echo "✓ .env file already exists"; \
	fi

# Start services
up:
	@echo "🚀 Starting all services..."
	@docker-compose up -d
	@echo "✓ Services started"
	@make status

# Stop services
down:
	@echo "🛑 Stopping all services..."
	@docker-compose down
	@echo "✓ Services stopped"

# Restart services
restart:
	@echo "🔄 Restarting all services..."
	@docker-compose restart
	@echo "✓ Services restarted"
	@make status

# Show logs
logs:
	@docker-compose logs -f

# Show status
status:
	@echo "📊 Service Status:"
	@docker-compose ps
	@echo ""
	@echo "💡 Tips:"
	@echo "   • Ollama:    http://localhost:11434"
	@echo "   • ChromaDB:  http://localhost:8000"
	@echo "   • Neo4j:     http://localhost:7474"
	@echo "   • Redis:     localhost:6379"

# Test connections
test:
	@echo "🔍 Testing service connections..."
	@echo ""
	@echo "Testing Ollama..."
	@curl -s http://localhost:11434/api/tags > /dev/null && echo "✓ Ollama is responding" || echo "❌ Ollama is not responding"
	@echo ""
	@echo "Testing ChromaDB..."
	@curl -s http://localhost:8000/api/v1/heartbeat > /dev/null && echo "✓ ChromaDB is responding" || echo "❌ ChromaDB is not responding"
	@echo ""
	@echo "Testing Neo4j..."
	@curl -s http://localhost:7474 > /dev/null && echo "✓ Neo4j is responding" || echo "❌ Neo4j is not responding"
	@echo ""
	@echo "Testing Redis..."
	@docker exec redis redis-cli ping > /dev/null && echo "✓ Redis is responding" || echo "❌ Redis is not responding"
	@echo ""

# Clean (remove containers)
clean:
	@echo "🧹 Cleaning up (removing containers)..."
	@docker-compose down
	@echo "✓ Cleanup complete (volumes preserved)"

# Clean all (remove containers AND volumes)
clean-all:
	@echo "⚠️  WARNING: This will delete all data in volumes!"
	@read -p "Are you sure? (yes/no): " confirm && [ "$$confirm" = "yes" ] || exit 1
	@echo "🧹 Cleaning up everything (removing containers and volumes)..."
	@docker-compose down -v
	@echo "✓ Complete cleanup done (all data deleted)"

# Install Python dependencies
install-deps:
	@echo "📦 Installing Python dependencies..."
	@pip install -r requirements.txt
	@echo "✓ Dependencies installed"

# Run Python example
run-example: install-deps
	@echo "🤖 Running agent example..."
	@python agent_example.py