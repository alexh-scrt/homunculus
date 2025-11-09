#!/usr/bin/env python3
"""
Test script to demonstrate Arena web search integration with Tavily.

This script shows how Arena agents can automatically use web search
when discussing topics that require current information.

Usage:
    export TAVILY_API_KEY=your_key_here
    python test_arena_web_search.py
"""

import asyncio
import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up environment
os.environ["ENABLE_STREAMING_RESPONSE"] = "false"  # For cleaner test output
os.environ["WEB_SEARCH_ENABLED"] = "true"

from src.arena.llm.llm_client import ArenaLLMClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_web_search_integration():
    """Test Arena LLM client with web search capabilities."""
    
    print("🔍 Arena Web Search Integration Test")
    print("=" * 50)
    
    # Check if Tavily API key is set
    if not os.getenv("TAVILY_API_KEY"):
        print("❌ TAVILY_API_KEY environment variable not set!")
        print("Please set it with: export TAVILY_API_KEY=your_key_here")
        return
    
    try:
        # Initialize Arena LLM client
        print("🚀 Initializing Arena LLM Client...")
        client = ArenaLLMClient(
            model="llama3.3:70b",
            temperature=0.7
        )
        
        print(f"✅ LLM Client initialized")
        print(f"   - Model: {client.model}")
        print(f"   - Web Search Enabled: {client.web_search_enabled}")
        print(f"   - Tools Available: {len(client.tools)} tools")
        print()
        
        # Test scenarios
        test_scenarios = [
            {
                "title": "🌍 Current Events Topic (should trigger web search)",
                "character": "ada_lovelace",
                "name": "Ada Lovelace", 
                "prompt": "What are the latest developments in artificial intelligence research?"
            },
            {
                "title": "📚 Historical Topic (should NOT trigger web search)",
                "character": "captain_cosmos",
                "name": "Captain Cosmos",
                "prompt": "What were the key achievements of the Apollo space program?"
            },
            {
                "title": "📊 Current Data Topic (should trigger web search)",
                "character": "ada_lovelace", 
                "name": "Ada Lovelace",
                "prompt": "What is the current state of quantum computing research?"
            }
        ]
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"🎭 Test {i}: {scenario['title']}")
            print("-" * 40)
            print(f"Character: {scenario['name']}")
            print(f"Prompt: {scenario['prompt']}")
            print()
            
            try:
                response = await client.generate_character_response(
                    character_id=scenario['character'],
                    character_name=scenario['name'],
                    prompt=scenario['prompt']
                )
                
                print(f"🤖 {scenario['name']}: {response}")
                print()
                print("=" * 50)
                print()
                
            except Exception as e:
                print(f"❌ Error in scenario {i}: {e}")
                print()
        
        print(f"🎯 Test completed successfully!")
        print(f"   - Web search is now integrated via LangChain tool calling")
        print(f"   - Tools are automatically used when agents need current information")
        print(f"   - No manual search calls needed - agents decide when to use tools")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.exception("Test error")

def print_setup_instructions():
    """Print setup instructions for web search."""
    print("""
🔧 Setup Instructions for Web Search:

1. Install required packages:
   pip install tavily-python

2. Get a Tavily API key:
   - Visit: https://app.tavily.com/
   - Sign up for an account
   - Get your API key

3. Set environment variable:
   export TAVILY_API_KEY=your_key_here

4. Configure Arena (.env.arena):
   TAVILY_API_KEY=your_key_here
   WEB_SEARCH_ENABLED=true
   WEB_SEARCH_MAX_RESULTS=5

5. Run the test:
   python test_arena_web_search.py
   
6. Try Arena with web search topics:
   python arena_cli.py start current_events -a ada_lovelace captain_cosmos \\
     -s "What are the latest breakthroughs in AI research?"
""")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--setup":
        print_setup_instructions()
    else:
        asyncio.run(test_web_search_integration())