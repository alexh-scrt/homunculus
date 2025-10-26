#!/bin/bash

# Agent Infrastructure Setup Script
# This script helps you set up the Docker Compose environment

set -e

echo "🚀 Agent Infrastructure Setup"
echo "=============================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    echo "   Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✓ Docker is installed"
echo "✓ Docker Compose is installed"
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "✓ Created .env file"
        echo ""
        echo "⚠️  IMPORTANT: Please edit .env and change the following:"
        echo "   - NEO4J_PASSWORD (currently set to 'change_this_password_123')"
        echo ""
        read -p "Press Enter to continue or Ctrl+C to exit and edit .env first..."
    else
        echo "❌ .env.example not found. Please ensure all files are present."
        exit 1
    fi
else
    echo "✓ .env file already exists"
fi

echo ""
echo "🔍 Checking GPU support..."

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    GPU_SUPPORT=true
else
    echo "⚠️  No NVIDIA GPU detected"
    echo "   Ollama will run on CPU (slower performance)"
    GPU_SUPPORT=false
    
    read -p "Do you want to disable GPU support in docker-compose.yml? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing GPU configuration from docker-compose.yml..."
        # Create backup
        cp docker-compose.yml docker-compose.yml.backup
        # Remove GPU sections (this is a simple approach, might need adjustment)
        echo "Note: You may need to manually remove the 'deploy' sections from docker-compose.yml"
    fi
fi

echo ""
echo "📦 Starting services..."
echo ""

# Pull images first
echo "Pulling Docker images (this may take a while)..."
docker-compose pull

echo ""
echo "🚀 Starting all services..."
docker-compose up -d

echo ""
echo "⏳ Waiting for services to be healthy..."
sleep 10

# Check service health
echo ""
echo "🔍 Checking service status..."
docker-compose ps

echo ""
echo "✅ Setup complete!"
echo ""
echo "📊 Service Endpoints:"
echo "   • Ollama:    http://localhost:11434"
echo "   • ChromaDB:  http://localhost:8000"
echo "   • Neo4j UI:  http://localhost:7474"
echo "   • Neo4j Bolt: bolt://localhost:7687"
echo "   • Redis:     localhost:6379"
echo ""
echo "🔐 Neo4j Credentials:"
echo "   Username: $(grep NEO4J_USER .env | cut -d '=' -f2)"
echo "   Password: $(grep NEO4J_PASSWORD .env | cut -d '=' -f2)"
echo ""
echo "📝 Next steps:"
echo "   1. Check logs: docker-compose logs -f"
echo "   2. Test the setup: python agent_example.py"
echo "   3. Access Neo4j UI: http://localhost:7474"
echo ""
echo "🛑 To stop all services: docker-compose down"
echo "🗑️  To remove all data: docker-compose down -v"