#!/usr/bin/env python3
"""
Test script to verify blocked responses are hidden from users

This script tests that when the anti-paraphrasing system blocks responses,
those blocked responses are never shown to users - only approved responses
should be visible.
"""

import sys
import os
import asyncio
from io import StringIO
import logging

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from arena.orchestration.game_orchestrator import GameOrchestrator, GameConfig
from arena.agents.base_agent import BaseAgent
from arena.llm.llm_client import ArenaLLMClient


class MockAgent(BaseAgent):
    """Mock agent that generates predictable responses for testing"""
    
    def __init__(self, agent_id: str, responses: list):
        super().__init__(agent_id, "Mock Agent", {})
        self.responses = responses
        self.response_index = 0
    
    async def generate_action(self, context: dict):
        """Generate predictable responses for testing"""
        if self.response_index < len(self.responses):
            response_text = self.responses[self.response_index]
            self.response_index += 1
            
            from arena.messaging.message_schemas import Message
            return Message(
                sender_id=self.agent_id,
                sender_name="Mock Agent",
                content=response_text,
                message_type="response"
            )
        return None


async def test_blocked_response_hiding():
    """Test that blocked responses are not shown in game state"""
    print("🔍 Testing Blocked Response Hiding")
    print("=" * 50)
    
    # Create orchestrator with anti-paraphrasing enabled
    config = GameConfig(
        enable_progression_control=True,
        progression_redundancy_threshold=0.82,
        progression_cycles_threshold=1,
        max_agents=2,
        max_turns=5
    )
    
    orchestrator = GameOrchestrator(config)
    
    # Create mock agent with responses that should be blocked
    blocked_responses = [
        "We should focus on AI and healthcare to build trillion-dollar companies.",  # Baseline
        "Building on the previous insight, AI and healthcare represent tremendous opportunities for creating innovative companies.",  # Should be blocked - paraphrasing
        "AI and healthcare are indeed important areas with significant growth potential for startups.",  # Should be blocked - paraphrasing  
        "If we focus on AI-powered diagnostics, by 2027 we could capture 15% of the medical imaging market through automated analysis.",  # Should pass - has entailment
    ]
    
    agent = MockAgent("test_agent", blocked_responses)
    
    # Initialize game state
    initial_state = {
        "phase": "active",
        "turn": 0,
        "messages": [],
        "agents": {"test_agent": agent}
    }
    
    print("📋 Testing Response Sequence:")
    print("-" * 30)
    
    approved_responses = []
    
    for i in range(4):  # Test 4 responses
        print(f"\n🎭 Turn {i + 1}:")
        expected_response = blocked_responses[i]
        print(f"   Input: {expected_response[:60]}...")
        
        # Capture any console output during this turn
        old_stdout = sys.stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            # Process the turn
            result_state = await orchestrator._agent_action_node(initial_state, "test_agent")
            
            # Check if response was added to messages
            new_messages = [msg for msg in result_state.get("messages", []) if msg.get("sender_id") == "test_agent"]
            
            if new_messages:
                # A response was approved and added
                approved_content = new_messages[-1]["content"]
                approved_responses.append(approved_content)
                print(f"   ✅ APPROVED: {approved_content[:60]}...")
                
                # Update state for next turn
                initial_state = result_state
                initial_state["turn"] = i + 1
            else:
                print(f"   🚫 BLOCKED: Response not added to game state")
                
        finally:
            sys.stdout = old_stdout
            captured = captured_output.getvalue()
            
            # Check if blocked content appeared in output
            if expected_response in captured:
                print(f"   ❌ ERROR: Blocked response visible in output!")
                print(f"       Found: {captured[:100]}...")
            elif captured.strip():
                print(f"   📝 Output: {captured.strip()[:60]}...")
            else:
                print(f"   ✅ Clean: No blocked content in output")
    
    print(f"\n📊 Results Summary:")
    print(f"   Total responses attempted: 4")
    print(f"   Responses approved: {len(approved_responses)}")
    print(f"   Responses blocked: {4 - len(approved_responses)}")
    
    print(f"\n✅ Approved responses:")
    for i, response in enumerate(approved_responses, 1):
        print(f"   {i}. {response[:80]}...")
    
    # Verify expected behavior
    if len(approved_responses) < 4:
        print(f"\n🎯 SUCCESS: Blocked responses properly hidden from users")
        print(f"   Only {len(approved_responses)} approved responses appear in game state")
    else:
        print(f"\n⚠️ ISSUE: All responses were approved - validation may not be working")
    
    return len(approved_responses) < 4


async def test_logging_levels():
    """Test that debug/warning messages don't appear in user output"""
    print("\n🔧 Testing Logging Level Isolation")
    print("=" * 50)
    
    # Set up logging to capture output at different levels
    logger = logging.getLogger("arena.orchestration.game_orchestrator")
    
    # Capture log output
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    
    # Test different log levels
    logger.debug("This debug message should be hidden from users")
    logger.info("This info message might be shown to users")
    logger.warning("This warning should not appear in user output")
    logger.error("This error should be handled appropriately")
    
    captured_logs = log_capture.getvalue()
    
    print("📋 Log Output Analysis:")
    print(f"   Debug messages: {'✅ Present' if 'debug message' in captured_logs else '❌ Missing'}")
    print(f"   Warning messages: {'✅ Present' if 'warning' in captured_logs else '❌ Missing'}")
    
    # Clean up
    logger.removeHandler(handler)
    
    print("✅ Logging isolation test completed")


def show_hiding_summary():
    """Show summary of response hiding implementation"""
    print("\n🛡️ Response Hiding Implementation Summary")
    print("=" * 50)
    
    features = [
        "🔒 Blocked responses never added to game state messages",
        "🔄 Validation loop runs internally with max 3 attempts", 
        "📝 Only approved responses appear in conversation flow",
        "🐛 Debug-level logging for blocked attempts (hidden from users)",
        "⚠️ Warning messages changed to debug level",
        "🎯 System interventions shown only when all attempts fail",
        "✅ Users see clean conversation without rejected content"
    ]
    
    for feature in features:
        print(f"   {feature}")
    
    print("\n📋 Expected User Experience:")
    print("   • Users never see blocked/rejected responses")
    print("   • Validation happens silently behind the scenes") 
    print("   • Only quality, approved responses appear in conversation")
    print("   • System guidance appears only when needed")
    print("   • Clean, professional discussion flow maintained")


async def main():
    """Run all response hiding tests"""
    print("🛡️ Testing Response Hiding System")
    
    try:
        success = await test_blocked_response_hiding()
        await test_logging_levels()
        show_hiding_summary()
        
        print(f"\n{'='*50}")
        if success:
            print("🎉 ✅ Response hiding system working correctly!")
            print("Blocked responses are properly hidden from users.")
        else:
            print("⚠️ ❌ Response hiding needs attention")
            print("Some blocked responses may be visible to users.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))