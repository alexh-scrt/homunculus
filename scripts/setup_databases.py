#!/usr/bin/env python3
"""Setup script for initializing databases and testing connections."""

import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from config.settings import get_settings
    from memory.experience_module import ExperienceModule
    from memory.knowledge_graph_module import KnowledgeGraphModule
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running from the project root and dependencies are installed.")
    sys.exit(1)


async def test_chroma_connection():
    """Test ChromaDB connection and setup."""
    print("Testing ChromaDB connection...")
    try:
        from core.experience import Experience
        from datetime import datetime
        
        experience_module = ExperienceModule(character_id='test_setup')
        
        # Create a proper Experience object for testing
        test_experience = Experience(
            experience_id='test_setup_exp',
            character_id='test_setup',
            timestamp=datetime.now(),
            experience_type='test',
            description='Test experience for database setup',
            participants=['test_user'],
            emotional_state='neutral',
            emotional_valence=0.0,
            intensity=0.5
        )
        
        await experience_module.store_experience(test_experience)
        print("✓ ChromaDB: Experience stored successfully")
        
        # Test retrieval using similar experiences method
        results = await experience_module.retrieve_similar_experiences(
            "test setup", n_results=1
        )
        
        if results and len(results) > 0:
            print("✓ ChromaDB: Experience retrieved successfully")
        else:
            print("⚠ ChromaDB: No experiences retrieved (might be normal)")
        
        # Cleanup test data
        try:
            await experience_module.delete_experience('test_setup_exp')
            print("✓ ChromaDB: Test data cleaned up")
        except Exception:
            pass  # Ignore cleanup errors
            
        return True
        
    except Exception as e:
        print(f"✗ ChromaDB: Connection failed - {e}")
        return False


async def test_neo4j_connection():
    """Test Neo4j connection and setup."""
    print("\nTesting Neo4j connection...")
    try:
        settings = get_settings()
        
        if not all([settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password]):
            print("⚠ Neo4j: Configuration missing (check .env file)")
            return False
        
        kg_module = KnowledgeGraphModule(character_id='test_setup')
        
        # KnowledgeGraphModule doesn't have initialize() method, but the connection is tested in __init__
        if not kg_module.driver:
            print("✗ Neo4j: Driver initialization failed")
            return False
        
        print("✓ Neo4j: Connection established")
        
        # Test storing a fact (this is what the module actually supports)
        test_fact = {
            'content': 'Test fact for database setup',
            'domain': 'test',
            'confidence': 0.9,
            'source_type': 'test'
        }
        
        await kg_module.store_fact(
            fact_text=test_fact['content'],
            source=test_fact['source_type'],
            confidence=test_fact['confidence'],
            domain=test_fact['domain']
        )
        print("✓ Neo4j: Fact stored successfully")
        
        # Test getting knowledge stats
        stats = await kg_module.get_knowledge_stats()
        if stats:
            print("✓ Neo4j: Stats query executed successfully")
        
        # Cleanup - the module manages its own data, no need to manually clean up test facts
        print("✓ Neo4j: Test completed")
        
        kg_module.close()
        return True
        
    except Exception as e:
        print(f"✗ Neo4j: Connection failed - {e}")
        print("  Make sure Neo4j is running and credentials are correct")
        return False


async def test_ollama_connection():
    """Test Ollama connection."""
    print("\nTesting Ollama connection...")
    try:
        from llm.ollama_client import OllamaClient
        
        settings = get_settings()
        client = OllamaClient(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model
        )
        
        # Test simple generation
        response = await client.generate("Say 'Hello, world!' if you can hear me.", temperature=0.1)
        
        if response and len(response.strip()) > 0:
            print("✓ Ollama: Connection successful")
            print(f"  Response: {response.strip()[:100]}...")
            return True
        else:
            print("✗ Ollama: No response received")
            return False
            
    except Exception as e:
        print(f"✗ Ollama: Connection failed - {e}")
        print("  Make sure Ollama is running and the model is available")
        return False


def display_configuration():
    """Display current configuration."""
    print("Current Configuration:")
    print("=" * 50)
    
    try:
        settings = get_settings()
        
        print(f"Ollama URL: {settings.ollama_base_url}")
        print(f"Ollama Model: {settings.ollama_model}")
        print(f"ChromaDB Directory: {settings.chroma_persist_directory}")
        print(f"Neo4j URI: {settings.neo4j_uri}")
        print(f"Neo4j User: {settings.neo4j_user}")
        print(f"Neo4j Password: {'*' * len(settings.neo4j_password) if settings.neo4j_password else 'Not set'}")
        
        if hasattr(settings, 'tavily_api_key') and settings.tavily_api_key:
            print(f"Tavily API Key: {'*' * 20}...{settings.tavily_api_key[-4:]}")
        else:
            print("Tavily API Key: Not set (optional)")
        
    except Exception as e:
        print(f"Error reading configuration: {e}")
    
    print("=" * 50)


async def main():
    """Main setup function."""
    print("Character Agent System - Database Setup")
    print("=" * 50)
    
    display_configuration()
    
    # Test all connections
    results = []
    
    # Test Ollama (required)
    ollama_ok = await test_ollama_connection()
    results.append(("Ollama (LLM)", ollama_ok, True))
    
    # Test ChromaDB (required)
    chroma_ok = await test_chroma_connection()
    results.append(("ChromaDB (Memory)", chroma_ok, True))
    
    # Test Neo4j (required)
    neo4j_ok = await test_neo4j_connection()
    results.append(("Neo4j (Knowledge Graph)", neo4j_ok, True))
    
    # Summary
    print("\nSetup Summary:")
    print("=" * 50)
    
    all_required_ok = True
    for service, status, required in results:
        status_icon = "✓" if status else "✗"
        req_text = "(required)" if required else "(optional)"
        print(f"{status_icon} {service} {req_text}")
        
        if required and not status:
            all_required_ok = False
    
    print("=" * 50)
    
    if all_required_ok:
        print("🎉 All required services are working!")
        print("You can now run the chat system with: python scripts/run_chat.py")
    else:
        print("⚠️  Some required services are not working.")
        print("Please check the error messages above and fix any issues.")
        print("\nCommon solutions:")
        print("- Make sure Docker is running for databases")
        print("- Check that Ollama is running with the correct model")
        print("- Verify your .env file has correct credentials")
    
    print(f"\nFor help, see the README.md file or run: python scripts/run_chat.py --help")


if __name__ == "__main__":
    asyncio.run(main())