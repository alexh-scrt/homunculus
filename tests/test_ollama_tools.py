#!/usr/bin/env python3
"""Test script for Ollama client with Tavily tool integration."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from llm.ollama_client import OllamaClient
from config.settings import get_settings

def test_ollama_tools():
    """Test the Ollama client with tools."""
    settings = get_settings()
    
    print("🧪 Testing Ollama Client with Tool Integration")
    print("=" * 50)
    
    # Test 1: Initialize client
    print("\n1. Initializing Ollama client...")
    client = OllamaClient()
    
    status = client.get_status()
    print(f"   ✅ Client configured: {status['ollama_configured']}")
    print(f"   🔧 Tools enabled: {status['tools_enabled']}")
    print(f"   🌐 Tavily configured: {status['tavily_configured']}")
    print(f"   📡 Available tools: {status['available_tools']}")
    
    # Test 2: Basic generation
    print("\n2. Testing basic text generation...")
    try:
        response = client.generate("Hello, how are you?", temperature=0.7)
        print(f"   ✅ Basic generation works: {response[:100]}...")
    except Exception as e:
        print(f"   ❌ Basic generation failed: {e}")
        return
    
    # Test 3: Tool usage info
    print("\n3. Checking tool usage info...")
    tool_info = client.get_tool_usage_info()
    print(f"   🔧 Tools used in last turn: {tool_info['tools_used']}")
    print(f"   📊 Tool calls: {len(tool_info['tool_calls'])}")
    
    # Test 4: Test with a prompt that might trigger web search
    print("\n4. Testing with search-triggering prompt...")
    try:
        search_prompt = "What are the latest developments in AI technology today?"
        response = client.generate(search_prompt, temperature=0.7)
        print(f"   ✅ Search prompt handled: {response[:150]}...")
        
        tool_info = client.get_tool_usage_info()
        if tool_info['tools_used']:
            print(f"   🔍 Tools were used! Tool calls: {len(tool_info['tool_calls'])}")
        else:
            print("   📝 No tools used (this is normal if Ollama doesn't decide to search)")
            
    except Exception as e:
        print(f"   ❌ Search prompt failed: {e}")
    
    # Test 5: Template generation
    print("\n5. Testing template generation...")
    try:
        template = "You are a {role}. Please {task} about {topic}."
        variables = {
            'role': 'helpful assistant',
            'task': 'provide information',
            'topic': 'renewable energy'
        }
        response = client.generate_with_template(template, variables, temperature=0.7)
        print(f"   ✅ Template generation works: {response[:100]}...")
    except Exception as e:
        print(f"   ❌ Template generation failed: {e}")
    
    # Test 6: Connection tests
    print("\n6. Testing connections...")
    connection_ok = client.test_connection()
    web_search_ok = client.test_web_search()
    print(f"   📡 Ollama connection: {'✅ OK' if connection_ok else '❌ Failed'}")
    print(f"   🌐 Web search: {'✅ OK' if web_search_ok else '❌ Failed'}")
    
    print("\n🎉 Testing completed!")
    print(f"\nFinal Status:")
    print(f"   - Ollama Model: {status['ollama_model']}")
    print(f"   - Tools Available: {len(status['available_tools'])}")
    print(f"   - Tavily Ready: {status['tavily_configured']}")

if __name__ == "__main__":
    test_ollama_tools()