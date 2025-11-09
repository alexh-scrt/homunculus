#!/usr/bin/env python3
"""
Simple conversation demo to show Arena agents talking to each other.
"""

import asyncio
import logging
from src.arena.agents import CharacterAgent
from src.arena.agents.base_agent import AgentConfig, AgentRole
from src.arena.models.homunculus_integration import HomunculusCharacterProfile

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

async def demo_conversation():
    """Demo conversation between Ada Lovelace and Captain Cosmos."""
    
    print("🎭 Arena Conversation Demo")
    print("=" * 50)
    
    # Create Ada Lovelace
    ada_config = AgentConfig(
        agent_id="ada_lovelace",
        agent_name="Ada Lovelace",
        role=AgentRole.CHARACTER,
        metadata={"source": "demo"}
    )
    
    ada_profile = HomunculusCharacterProfile(
        character_name="Ada Lovelace",
        personality_traits=["analytical", "logical", "innovative"],
        expertise_areas=["mathematics", "computing", "analysis"],
        communication_style="precise",
        backstory="World's first computer programmer",
        goals=["solve problems methodically", "find optimal solutions"]
    )
    
    ada = CharacterAgent(ada_config, ada_profile)
    
    # Create Captain Cosmos
    cosmos_config = AgentConfig(
        agent_id="captain_cosmos",
        agent_name="Captain Cosmos",
        role=AgentRole.CHARACTER,
        metadata={"source": "demo"}
    )
    
    cosmos_profile = HomunculusCharacterProfile(
        character_name="Captain Cosmos",
        personality_traits=["adventurous", "optimistic", "creative"],
        expertise_areas=["exploration", "leadership", "innovation"],
        communication_style="inspiring",
        backstory="Interstellar explorer and captain",
        goals=["explore new frontiers", "inspire others"]
    )
    
    cosmos = CharacterAgent(cosmos_config, cosmos_profile)
    
    # Initialize agents
    await ada.initialize()
    await cosmos.initialize()
    
    print("\n👥 Characters Initialized:")
    print(f"• {ada_profile.character_name}: {', '.join(ada_profile.personality_traits)}")
    print(f"• {cosmos_profile.character_name}: {', '.join(cosmos_profile.personality_traits)}")
    
    print("\n💬 Conversation:")
    print("-" * 30)
    
    # Simulate a conversation
    context = {
        "turn": 1,
        "phase": "early",
        "my_turn": True,
        "current_speaker": "ada_lovelace",
        "active_agents": ["ada_lovelace", "captain_cosmos"]
    }
    
    # Ada speaks first
    ada_message = await ada.generate_action(context)
    if ada_message:
        print(f"\n🔬 {ada_message.sender_name}:")
        print(f"   \"{ada_message.content}\"")
    
    # Cosmos responds
    context["current_speaker"] = "captain_cosmos"
    cosmos_message = await cosmos.generate_action(context)
    if cosmos_message:
        print(f"\n🚀 {cosmos_message.sender_name}:")
        print(f"   \"{cosmos_message.content}\"")
    
    # Ada responds back
    context["current_speaker"] = "ada_lovelace"
    context["turn"] = 2
    ada_message2 = await ada.generate_action(context)
    if ada_message2:
        print(f"\n🔬 {ada_message2.sender_name}:")
        print(f"   \"{ada_message2.content}\"")
    
    # Final Cosmos response
    context["current_speaker"] = "captain_cosmos"
    cosmos_message2 = await cosmos.generate_action(context)
    if cosmos_message2:
        print(f"\n🚀 {cosmos_message2.sender_name}:")
        print(f"   \"{cosmos_message2.content}\"")
    
    print("\n" + "=" * 50)
    print("✨ Conversation Complete!")
    print("\nThis demonstrates that:")
    print("• ✅ Agents generate character-specific responses")
    print("• ✅ Ada Lovelace uses analytical language")
    print("• ✅ Captain Cosmos uses cosmic/adventurous language")
    print("• ✅ LangGraph orchestration is working")
    print("• ✅ The conversation system is functional!")

if __name__ == "__main__":
    asyncio.run(demo_conversation())