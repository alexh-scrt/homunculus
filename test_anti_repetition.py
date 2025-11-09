#!/usr/bin/env python3
"""
Test script for the anti-repetition system

This script tests the core components of the anti-repetition mechanism
borrowed from the talks project.
"""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from arena.utils.redundancy_checker import RedundancyChecker
from arena.utils.entailment_detector import EntailmentDetector, EntailmentType
from arena.utils.topic_extractor import TopicExtractor
from arena.controllers.progression_controller import ProgressionController, ProgressionConfig


async def test_redundancy_detection():
    """Test redundancy detection with arena-style messages"""
    print("🔍 Testing Redundancy Detection...")
    
    redundancy_checker = RedundancyChecker(similarity_threshold=0.85)
    
    # Messages that simulate the repetitive pattern from the logs
    messages = [
        "Building on the ideas presented by others, I believe AI and renewable energy sectors are poised to create tremendous value, with some estimates suggesting a $1 trillion opportunity.",
        "As we continue exploring AI and renewable energy, it's clear these sectors could optimize energy demand and infrastructure, creating significant growth opportunities.",
        "I'd like to build on the ideas regarding AI and renewable energy potential. Companies like Tesla and oil & gas firms are leveraging their expertise to provide reliable energy for AI data centers."
    ]
    
    # Test first message (should not be redundant)
    is_redundant_1 = redundancy_checker.is_redundant(messages[0], [])
    print(f"Message 1 redundant: {is_redundant_1} (expected: False)")
    
    # Test second message (should be redundant) 
    is_redundant_2 = redundancy_checker.is_redundant(messages[1], [messages[0]])
    print(f"Message 2 redundant: {is_redundant_2} (expected: True)")
    
    # Test third message (should be redundant)
    is_redundant_3 = redundancy_checker.is_redundant(messages[2], messages[:2])
    print(f"Message 3 redundant: {is_redundant_3} (expected: True)")
    
    print("✅ Redundancy detection test completed\n")


def test_entailment_detection():
    """Test entailment detection for meaningful contributions"""
    print("🧠 Testing Entailment Detection...")
    
    detector = EntailmentDetector()
    
    # Test messages with different entailment levels
    test_cases = [
        {
            "text": "Building on AI and energy ideas, these sectors could create value.",
            "expected_entailments": 0,  # Vague, no specific entailments
            "description": "Vague statement"
        },
        {
            "text": "If we invest in AI-powered grid optimization, then we can reduce energy waste by 30% within 3 years.",
            "expected_entailments": 1,  # Clear implication
            "description": "Specific prediction"
        },
        {
            "text": "This means we need to build dedicated data centers, which would require $50B investment, therefore creating 100,000 jobs.",
            "expected_entailments": 2,  # Multiple entailments
            "description": "Multiple implications"
        }
    ]
    
    for i, case in enumerate(test_cases):
        entailments = detector.detect(case["text"])
        count = len([e for e in entailments if e["confidence"] > 0.6])
        print(f"Case {i+1} ({case['description']}): {count} entailments (expected: {case['expected_entailments']})")
        for ent in entailments:
            print(f"  - {ent['type'].value}: {ent['match'][:50]}... (conf: {ent['confidence']:.2f})")
    
    print("✅ Entailment detection test completed\n")


def test_topic_extraction():
    """Test topic extraction for arena discussions"""
    print("🏷️ Testing Topic Extraction...")
    
    extractor = TopicExtractor()
    
    # Test topic extraction from arena-style message
    text = "Our AI-powered renewable energy platform would leverage machine learning algorithms to optimize solar panel efficiency while reducing infrastructure costs through blockchain-based energy trading."
    
    topics = extractor.extract_topics(text)
    print(f"Extracted topics: {topics}")
    
    # Test tension detection
    recent_topics = [
        {"ai_technology", "innovation"},
        {"energy", "sustainability"},
        {"finance", "business_model"}
    ]
    
    tensions = extractor.detect_tensions(text, recent_topics)
    print(f"Detected tensions: {tensions}")
    
    # Test cycle detection
    repetitive_messages = [
        "AI and energy sectors could create trillion-dollar opportunities",
        "Building on AI and energy ideas, these sectors show great potential", 
        "The intersection of AI and renewable energy presents massive market opportunities"
    ]
    
    cycle_result = extractor.detect_topic_cycles(repetitive_messages)
    print(f"Cycle detection: {cycle_result}")
    
    print("✅ Topic extraction test completed\n")


async def test_progression_controller():
    """Test the full progression control system"""
    print("🎮 Testing Progression Controller...")
    
    config = ProgressionConfig(
        cycles_threshold=2,
        max_consequence_tests=2,
        redundancy_threshold=0.85,
        enable_progression=True
    )
    
    controller = ProgressionController(config)
    
    # Simulate repetitive agent responses
    responses = [
        ("jobs", "I believe AI and renewable energy sectors could create a trillion-dollar opportunity through optimized infrastructure."),
        ("gates", "Building on Jobs' idea, AI and energy integration could optimize demand and create significant value."),
        ("musk", "As Jobs and Gates mentioned, AI and renewable energy sectors present tremendous growth opportunities."),
        ("bezos", "I agree with the others - AI and energy convergence could create trillion-dollar companies."),
    ]
    
    for agent, response in responses:
        result = await controller.process_agent_response(agent, response)
        print(f"\nAgent {agent}:")
        print(f"  Allow response: {result.get('allow_response', True)}")
        print(f"  Interventions: {len(result.get('interventions', []))}")
        
        for intervention in result.get('interventions', []):
            print(f"    - {intervention['type']}: {intervention.get('message', 'No message')[:80]}...")
    
    # Check final status
    status = controller.get_progression_status()
    print(f"\nFinal status:")
    print(f"  Total turns: {status['turn']}")
    print(f"  Saturated tensions: {status['saturated_tensions']}")
    print(f"  Needs intervention: {status['needs_intervention']}")
    print(f"  Metrics: {status['metrics']}")
    
    print("✅ Progression controller test completed\n")


async def main():
    """Run all tests"""
    print("🚀 Testing Arena Anti-Repetition System\n")
    print("=" * 60)
    
    try:
        await test_redundancy_detection()
        test_entailment_detection() 
        test_topic_extraction()
        await test_progression_controller()
        
        print("=" * 60)
        print("🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())