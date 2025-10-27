#!/usr/bin/env python3
"""Test script to reproduce the live system error."""

import sys
from pathlib import Path
import asyncio
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Set up logging to see what's happening
logging.basicConfig(level=logging.DEBUG)

async def test_live_system_reproduction():
    """Try to reproduce the exact error from the live system."""
    print("🔍 Reproducing Live System Error")
    print("=" * 50)
    
    try:
        # Import the same way the live system would
        from agents.memory_agent import MemoryAgent
        from core.character_state import CharacterState
        from memory.knowledge_graph_module import extract_concepts_from_text
        
        print("✅ Imports successful")
        
        # Create a memory agent like the live system
        from llm.ollama_client import OllamaClient
        llm_client = OllamaClient()
        memory_agent = MemoryAgent("memory_agent", "test_character", llm_client)
        print("✅ MemoryAgent created")
        
        # Create a basic character state
        from datetime import datetime
        character_state = CharacterState("test_character", datetime.now())
        character_state.agent_states = {
            'goals': {'active_goals': []}
        }
        
        # Test the exact call chain that would happen in the live system
        user_message = "Hi Marcus! Would you like to meet your creator?"
        print(f"🧪 Testing with message: {user_message}")
        
        # Extract concepts first (this is what the memory agent does)
        concepts = await extract_concepts_from_text(user_message)
        print(f"📝 Extracted concepts: {concepts}")
        
        # Call the memory agent consult method (this is what gets called in the real system)
        print("🔍 Calling memory agent consult...")
        context = {'topic': 'general conversation'}
        memories = await memory_agent.consult(
            context, character_state, user_message
        )
        print(f"✅ Retrieved memories successfully: {type(memories).__name__}")
        
    except Exception as e:
        print(f"❌ Error reproduced: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("✅ Test completed without errors")
    return True

if __name__ == "__main__":
    success = asyncio.run(test_live_system_reproduction())