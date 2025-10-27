#!/usr/bin/env python3
"""Test script for knowledge graph module fix."""

import sys
from pathlib import Path
import asyncio

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from memory.knowledge_graph_module import KnowledgeGraphModule

async def test_knowledge_graph_fix():
    """Test the knowledge graph module with various edge cases."""
    print("🧪 Testing Knowledge Graph Module Fix")
    print("=" * 50)
    
    # Test 1: Normal case with valid concepts
    print("\n1. Testing with valid concepts...")
    kg = KnowledgeGraphModule("test_character")
    
    try:
        result = await kg.retrieve_related_facts(
            concepts=["technology", "science"],
            limit=5
        )
        print(f"   ✅ Valid concepts: Retrieved {len(result)} facts")
    except Exception as e:
        print(f"   ❌ Valid concepts failed: {e}")
    
    # Test 2: Empty concepts list (should be handled by early return)
    print("\n2. Testing with empty concepts list...")
    try:
        result = await kg.retrieve_related_facts(
            concepts=[],
            limit=5
        )
        print(f"   ✅ Empty concepts: Retrieved {len(result)} facts (expected 0)")
    except Exception as e:
        print(f"   ❌ Empty concepts failed: {e}")
    
    # Test 3: None concepts (should be handled by early return)
    print("\n3. Testing with None concepts...")
    try:
        result = await kg.retrieve_related_facts(
            concepts=None,
            limit=5
        )
        print(f"   ✅ None concepts: Retrieved {len(result)} facts (expected 0)")
    except Exception as e:
        print(f"   ❌ None concepts failed: {e}")
    
    # Test 4: Concepts with whitespace and empty strings
    print("\n4. Testing with whitespace/empty concepts...")
    try:
        result = await kg.retrieve_related_facts(
            concepts=["", "  ", "technology", "   science   ", ""],
            limit=5
        )
        print(f"   ✅ Whitespace concepts: Retrieved {len(result)} facts")
    except Exception as e:
        print(f"   ❌ Whitespace concepts failed: {e}")
    
    # Test 5: Concepts with special characters that could cause injection
    print("\n5. Testing with special characters...")
    try:
        result = await kg.retrieve_related_facts(
            concepts=["tech'nology", "sci\"ence", "data-science"],
            limit=5
        )
        print(f"   ✅ Special characters: Retrieved {len(result)} facts")
    except Exception as e:
        print(f"   ❌ Special characters failed: {e}")
    
    # Test 6: With domains
    print("\n6. Testing with domains...")
    try:
        result = await kg.retrieve_related_facts(
            concepts=["technology"],
            domains=["science", "general"],
            limit=5
        )
        print(f"   ✅ With domains: Retrieved {len(result)} facts")
    except Exception as e:
        print(f"   ❌ With domains failed: {e}")
    
    # Close the module
    kg.close()
    print("\n🎉 Testing completed!")

if __name__ == "__main__":
    asyncio.run(test_knowledge_graph_fix())