#!/usr/bin/env python3
"""
Test script for enhanced anti-paraphrasing system

This script tests that the enhanced validation system properly detects and rejects
the kind of paraphrasing responses shown in the user's example, while allowing
substantive responses with meaningful entailments.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import asyncio
from arena.utils.redundancy_checker import RedundancyChecker
from arena.utils.entailment_detector import EntailmentDetector
from arena.controllers.progression_controller import ProgressionController, ProgressionConfig, ArenaProgressionOrchestrator


async def test_paraphrasing_detection():
    """Test detection of the paraphrasing patterns from user's example"""
    print("🔍 Testing Anti-Paraphrasing Detection")
    print("=" * 60)
    
    # Initialize components with aggressive settings
    config = ProgressionConfig(
        redundancy_threshold=0.82,  # Lowered for more sensitivity
        enable_progression=True
    )
    
    redundancy_checker = RedundancyChecker(similarity_threshold=0.82, lookback_window=5)
    entailment_detector = EntailmentDetector()
    orchestrator = ArenaProgressionOrchestrator(config)
    
    # Actual paraphrasing examples from user's conversation
    jobs_response = """As we explore ideas for creating the next trillion-dollar company, it's essential to consider emerging sectors that have shown tremendous potential for growth. The research suggests that fields like AI, healthcare, and renewable energy are poised to drive innovation and valuation in the coming years. I'd like to build on this by highlighting the intersection of technology and healthcare, specifically in areas like computational biology and genomics. By leveraging advancements in AI and machine learning, we can unlock new opportunities for personalized medicine, disease prevention, and targeted therapies."""
    
    gates_response = """As we delve into the realm of creating the next trillion-dollar company, it's crucial to focus on emerging sectors that have demonstrated significant growth potential. Building on Jobs' insight, I'd like to emphasize the importance of AI software, robotics, and electric vehicles as key areas of interest. According to recent research, these sectors are driven by technological advancements and market demand, making them prime targets for investment and innovation."""
    
    musk_response = """As we explore the realm of creating the next trillion-dollar company, I'd like to build on Gates' insight regarding emerging sectors with significant growth potential. The research highlights three key areas: AI, sustainability, and retail, which are poised to drive substantial economic growth. Notably, AI's advancements in healthcare and data centers are projected to reach $10.6 trillion by 2035, while sustainability and retail innovation could also create new trillion-dollar companies."""
    
    bezos_response = """As we delve into creating the next trillion-dollar company, I'd like to build on Musk's insight regarding emerging sectors with significant growth potential. The research highlights three key areas: AI, sustainability, and retail, each with substantial prospects for innovation and disruption. Notably, AI's advancements in healthcare and data centers are projected to reach $10.6 trillion by 2035, presenting a compelling opportunity for investment and development."""
    
    responses = [
        ("Jobs", jobs_response),
        ("Gates", gates_response), 
        ("Musk", musk_response),
        ("Bezos", bezos_response)
    ]
    
    print("📋 Testing Conversation Sequence:")
    print("-" * 40)
    
    previous_responses = []
    
    for i, (speaker, response) in enumerate(responses):
        print(f"\n🎭 {speaker} (Turn {i+1}):")
        
        # Test redundancy detection
        is_redundant = redundancy_checker.is_redundant(response, previous_responses)
        
        # Test entailment detection
        entailments = entailment_detector.detect(response)
        has_entailment = len(entailments) > 0
        
        # Test progression controller validation
        controller_result = await orchestrator.process_turn(speaker, response, {
            "turn": i + 1,
            "phase": "mid",
            "game_context": {}
        })
        
        allowed = controller_result.get("allow_response", True)
        reason = controller_result.get("reason", "")
        
        print(f"   📊 Redundancy: {'❌ REDUNDANT' if is_redundant else '✅ UNIQUE'}")
        print(f"   🧠 Entailments: {'✅ FOUND' if has_entailment else '❌ MISSING'} ({len(entailments)} total)")
        print(f"   🚦 Controller: {'🚫 BLOCKED' if not allowed else '✅ ALLOWED'}")
        
        if not allowed:
            print(f"   💬 Reason: {reason}")
            feedback = controller_result.get("feedback", "")
            if feedback:
                print(f"   📝 Feedback: {feedback[:100]}...")
        
        if has_entailment:
            entailment_types = [ent['type'].value for ent in entailments]
            print(f"   🔍 Entailment Types: {entailment_types}")
        
        previous_responses.append(response)
        print()
    
    print("=" * 60)


async def test_good_vs_bad_responses():
    """Test that good responses pass while bad ones are blocked"""
    print("🎯 Testing Response Quality Validation")
    print("=" * 60)
    
    config = ProgressionConfig(redundancy_threshold=0.82)
    orchestrator = ArenaProgressionOrchestrator(config)
    
    # Seed the conversation with a typical startup discussion
    seed_responses = [
        "We should focus on emerging technologies like AI and healthcare to build the next trillion-dollar company.",
        "AI and healthcare are indeed promising areas with significant growth potential for innovative startups."
    ]
    
    for j, response in enumerate(seed_responses):
        await orchestrator.process_turn("Seed", response, {"turn": j, "phase": "early"})
    
    test_cases = [
        {
            "name": "🚫 Bad: Pure Paraphrasing",
            "response": "Building on the previous insights, AI and healthcare represent tremendous opportunities for creating innovative companies with substantial growth potential in emerging technology sectors.",
            "should_pass": False
        },
        {
            "name": "🚫 Bad: Topic Repetition Without Entailments", 
            "response": "I agree that AI and healthcare are important areas. These sectors offer significant opportunities for companies looking to create innovative solutions.",
            "should_pass": False
        },
        {
            "name": "✅ Good: Adds Concrete Predictions",
            "response": "If we focus on AI-powered diagnostic tools, by 2027 we could capture 15% of the $50B medical imaging market through automated radiology analysis.",
            "should_pass": True
        },
        {
            "name": "✅ Good: Specific Implementation Strategy",
            "response": "To implement this healthcare AI approach, we need to first secure FDA approval for our diagnostic algorithm, then partner with 3 major hospital networks for pilot testing.",
            "should_pass": True
        },
        {
            "name": "✅ Good: Addresses Risks/Challenges",
            "response": "However, the challenge with healthcare AI is regulatory compliance - if we don't have robust validation data, the FDA will reject our application and delay market entry by 18 months.",
            "should_pass": True
        },
        {
            "name": "✅ Good: Genuinely New Topic",
            "response": "Instead of traditional healthcare AI, what if we focused on quantum computing for financial modeling? This could revolutionize risk assessment in ways that haven't been explored.",
            "should_pass": True
        }
    ]
    
    print("📋 Testing Different Response Types:")
    print("-" * 50)
    
    for i, test_case in enumerate(test_cases, start=3):
        speaker = f"Agent{i}"
        response = test_case["response"]
        should_pass = test_case["should_pass"]
        name = test_case["name"]
        
        result = await orchestrator.process_turn(speaker, response, {
            "turn": i,
            "phase": "mid", 
            "game_context": {}
        })
        
        allowed = result.get("allow_response", True)
        reason = result.get("reason", "")
        
        status = "✅ PASSED" if allowed else "🚫 BLOCKED"
        expected = "✅ Expected" if (allowed == should_pass) else "❌ UNEXPECTED"
        
        print(f"\n{name}")
        print(f"   Result: {status} | {expected}")
        
        if not allowed:
            print(f"   Reason: {reason}")
        
        if allowed != should_pass:
            print(f"   ⚠️ MISMATCH: Expected {'PASS' if should_pass else 'BLOCK'}, got {'PASS' if allowed else 'BLOCK'}")
    
    print("\n" + "=" * 60)


def test_similarity_thresholds():
    """Test different similarity thresholds"""
    print("🔬 Testing Similarity Thresholds")
    print("=" * 60)
    
    # Original paraphrasing text
    original = "We should focus on AI and healthcare to build trillion-dollar companies in emerging sectors."
    
    paraphrases = [
        ("Mild paraphrase", "AI and healthcare represent opportunities to create trillion-dollar businesses in emerging technology sectors."),
        ("Strong paraphrase", "Building on this insight, artificial intelligence and medical technology offer tremendous potential for developing billion-dollar enterprises."),
        ("Buzzword variant", "Leveraging AI innovation and healthcare disruption, we can unlock substantial value creation in high-growth market segments."),
        ("Different approach", "Instead of AI, what if we focused on quantum computing for financial risk modeling?")
    ]
    
    thresholds = [0.75, 0.82, 0.90]
    
    print("📊 Similarity Analysis:")
    print("-" * 40)
    
    for threshold in thresholds:
        print(f"\n🎛️ Threshold: {threshold}")
        checker = RedundancyChecker(similarity_threshold=threshold)
        
        for name, paraphrase in paraphrases:
            is_redundant = checker.is_redundant(paraphrase, [original])
            details = checker.get_similarity_details(paraphrase, [original])
            similarity = details.get("max_similarity", 0.0)
            
            status = "🚫 BLOCKED" if is_redundant else "✅ PASSED"
            print(f"   {name:20} | Sim: {similarity:.3f} | {status}")


def show_enhancement_summary():
    """Show summary of anti-paraphrasing enhancements"""
    print("\n🚀 Anti-Paraphrasing Enhancement Summary")
    print("=" * 60)
    
    enhancements = [
        "🔍 Lower similarity threshold (0.85 → 0.82) catches subtler paraphrasing",
        "🧠 Mandatory entailment detection prevents empty responses",
        "🔄 Validation loop forces agents to regenerate blocked responses",
        "📝 Specific feedback guides agents toward substantive content",
        "🎯 Enhanced text normalization removes common filler phrases",
        "📊 Three-tier rejection system (redundant+no entailment, redundant, no entailment)",
        "🚫 Buzzword filtering removes meaningless business jargon",
        "✅ Allows responses with genuine implications, predictions, actions"
    ]
    
    for enhancement in enhancements:
        print(f"   {enhancement}")
    
    print("\n📈 Expected Impact:")
    print("   • Agents can no longer just rephrase 'AI + healthcare + renewable energy'")
    print("   • Responses must include concrete implications or genuinely new topics")
    print("   • Validation loop ensures quality before acceptance")
    print("   • System guides agents toward more substantive contributions")


async def main():
    """Run all anti-paraphrasing tests"""
    print("🎯 Testing Enhanced Anti-Paraphrasing System")
    
    try:
        await test_paraphrasing_detection()
        await test_good_vs_bad_responses()
        test_similarity_thresholds()
        show_enhancement_summary()
        
        print(f"\n{'='*60}")
        print("🎉 All anti-paraphrasing tests completed!")
        print("Enhanced validation system ready to prevent repetitive discussions.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))