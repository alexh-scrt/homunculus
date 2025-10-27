#!/usr/bin/env python3
"""Debug test script for knowledge graph module."""

import sys
from pathlib import Path
import asyncio
import traceback

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from memory.knowledge_graph_module import KnowledgeGraphModule

async def test_knowledge_graph_debug():
    """Debug the knowledge graph module to find the exact error location."""
    print("🔍 Debugging Knowledge Graph Module")
    print("=" * 50)
    
    kg = KnowledgeGraphModule("test_character")
    
    try:
        print("\n1. Testing retrieve_related_facts with detailed error tracking...")
        
        # Test with empty list that might cause the error
        print("   Testing with empty concepts...")
        result = await kg.retrieve_related_facts(concepts=[])
        print(f"   Empty concepts result: {len(result)} facts")
        
        # Test with list containing empty strings
        print("   Testing with empty string concepts...")
        result = await kg.retrieve_related_facts(concepts=["", "  "])
        print(f"   Empty string concepts result: {len(result)} facts")
        
        # Test with None list
        print("   Testing with None concepts...")
        result = await kg.retrieve_related_facts(concepts=None)
        print(f"   None concepts result: {len(result)} facts")
        
        # Test normal case
        print("   Testing with valid concepts...")
        result = await kg.retrieve_related_facts(concepts=["test"])
        print(f"   Valid concepts result: {len(result)} facts")
        
    except Exception as e:
        print(f"❌ Error in retrieve_related_facts: {e}")
        print("Full traceback:")
        traceback.print_exc()
    
    try:
        print("\n2. Testing other methods that might have the issue...")
        
        # Test find_concept_connections
        print("   Testing find_concept_connections...")
        result = await kg.find_concept_connections("test")
        print(f"   Concept connections result: {result}")
        
    except Exception as e:
        print(f"❌ Error in find_concept_connections: {e}")
        print("Full traceback:")
        traceback.print_exc()
    
    try:
        # Test get_knowledge_stats
        print("   Testing get_knowledge_stats...")
        result = await kg.get_knowledge_stats()
        print(f"   Knowledge stats result keys: {list(result.keys())}")
        
    except Exception as e:
        print(f"❌ Error in get_knowledge_stats: {e}")
        print("Full traceback:")
        traceback.print_exc()
    
    # Close the module
    kg.close()
    print("\n🔍 Debug test completed!")

if __name__ == "__main__":
    asyncio.run(test_knowledge_graph_debug())