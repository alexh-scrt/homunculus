#!/usr/bin/env python3
"""
Test script for dynamic LLM parameters

This script tests the enhanced LLM parameter system that borrows
concepts from the talks project for better creativity and content richness.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from arena.llm.dynamic_parameters import DynamicParameterManager, AgentRole, GamePhase
from arena.config.arena_settings import ArenaSettings


def test_parameter_optimization():
    """Test the dynamic parameter optimization system"""
    print("🎯 Testing Dynamic LLM Parameter System")
    print("=" * 60)
    
    # Initialize manager
    manager = DynamicParameterManager()
    
    # Test different scenarios
    test_scenarios = [
        {
            "name": "Character Agent - Early Game",
            "role": AgentRole.CHARACTER,
            "context": {"phase": "early", "turn": 5, "current_speaker": "jobs"}
        },
        {
            "name": "Character Agent - Late Game High Competition",
            "role": AgentRole.CHARACTER,
            "context": {
                "phase": "late", 
                "turn": 35, 
                "current_speaker": "gates",
                "scores": {"jobs": 15.2, "gates": 15.1, "musk": 15.3, "bezos": 14.8}  # Close scores
            }
        },
        {
            "name": "Character Agent - Stale Discussion",
            "role": AgentRole.CHARACTER,
            "context": {
                "phase": "mid",
                "turn": 20,
                "current_speaker": "musk",
                "stale_discussion": True,
                "needs_intervention": True
            }
        },
        {
            "name": "Underperforming Agent",
            "role": AgentRole.CHARACTER,
            "context": {
                "phase": "mid",
                "turn": 25,
                "current_speaker": "bezos",
                "scores": {"jobs": 18.5, "gates": 17.2, "musk": 16.8, "bezos": 12.1}  # Bezos behind
            }
        },
        {
            "name": "Narrator Agent",
            "role": AgentRole.NARRATOR,
            "context": {"phase": "mid", "turn": 15}
        },
        {
            "name": "Judge Agent",
            "role": AgentRole.JUDGE,
            "context": {"phase": "final", "turn": 45}
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n📊 {scenario['name']}:")
        params = manager.get_parameters(scenario['role'], scenario.get('context', {}))
        
        print(f"   Temperature: {params.temperature:.3f}")
        print(f"   Max Tokens:  {params.max_tokens}")
        if params.frequency_penalty:
            print(f"   Freq Penalty: {params.frequency_penalty:.3f}")
        if params.presence_penalty:
            print(f"   Pres Penalty: {params.presence_penalty:.3f}")


def test_settings_comparison():
    """Compare old vs new parameter settings"""
    print("\n🔄 Parameter Comparison: Old vs New")
    print("=" * 60)
    
    settings = ArenaSettings()
    
    print("🔸 Base LLM Settings:")
    print(f"   Anthropic Temperature: {settings.anthropic_temperature} (was 0.7)")
    print(f"   OpenAI Temperature:    {settings.openai_temperature} (was 0.7)")
    
    print("\n🔸 Role-Specific Settings:")
    print(f"   Character Temperature: {settings.character_agent_temperature} (new)")
    print(f"   Character Max Tokens:  {settings.character_agent_max_tokens} (new)")
    print(f"   Narrator Temperature:  {settings.narrator_temperature} (new)")
    print(f"   Judge Temperature:     {settings.judge_temperature} (new)")
    
    print("\n🔸 Advanced Features:")
    print(f"   Dynamic Temperature:   {settings.use_dynamic_temperature} (new)")
    print(f"   Late Game Boost:      {settings.creativity_boost_late_game} (new)")
    print(f"   Diverse Sampling:     {settings.enable_diverse_response_sampling} (new)")
    print(f"   Variety Factor:       {settings.response_variety_factor} (new)")


def test_creative_scenarios():
    """Test scenarios that should boost creativity"""
    print("\n🎨 Creativity Boost Scenarios")
    print("=" * 60)
    
    manager = DynamicParameterManager()
    
    # Base character parameters
    base_params = manager.get_parameters(AgentRole.CHARACTER, {"phase": "early"})
    print(f"Base Character Temperature: {base_params.temperature:.3f}")
    
    # Late game scenario
    late_game_params = manager.get_parameters(
        AgentRole.CHARACTER, 
        {"phase": "final", "turn": 48}
    )
    print(f"Late Game Temperature:     {late_game_params.temperature:.3f} (+{late_game_params.temperature - base_params.temperature:.3f})")
    
    # High competition
    competition_params = manager.get_parameters(
        AgentRole.CHARACTER,
        {
            "phase": "late",
            "scores": {"jobs": 20.1, "gates": 20.0, "musk": 19.9, "bezos": 20.2}  # Very close
        }
    )
    print(f"High Competition Temp:     {competition_params.temperature:.3f} (+{competition_params.temperature - base_params.temperature:.3f})")
    
    # Stale discussion
    stale_params = manager.get_parameters(
        AgentRole.CHARACTER,
        {"phase": "mid", "stale_discussion": True}
    )
    print(f"Stale Discussion Temp:     {stale_params.temperature:.3f} (+{stale_params.temperature - base_params.temperature:.3f})")
    if stale_params.frequency_penalty:
        print(f"   + Frequency Penalty:    {stale_params.frequency_penalty:.3f}")


def test_token_optimization():
    """Test token allocation optimization"""
    print("\n📝 Token Allocation Optimization")
    print("=" * 60)
    
    manager = DynamicParameterManager()
    
    scenarios = [
        ("Character - Early",  AgentRole.CHARACTER, {"phase": "early"}),
        ("Character - Late",   AgentRole.CHARACTER, {"phase": "late"}),
        ("Character - Final",  AgentRole.CHARACTER, {"phase": "final"}),
        ("Narrator - Mid",     AgentRole.NARRATOR,  {"phase": "mid"}),
        ("Judge - Scoring",    AgentRole.JUDGE,     {"phase": "mid"}),
    ]
    
    for name, role, context in scenarios:
        params = manager.get_parameters(role, context)
        print(f"{name:20} {params.max_tokens:4d} tokens")


def show_improvement_summary():
    """Show summary of improvements made"""
    print("\n✅ Summary of LLM Parameter Improvements")
    print("=" * 60)
    
    improvements = [
        "🌡️  Increased base temperature: 0.7 → 0.85 (matches talks project)",
        "🎭 Role-specific temperatures: Characters=0.9, Narrator=0.75, Judge=0.6",
        "📝 Optimized token limits: Characters=1200, Narrator=800, Judge=600",
        "🎯 Context-aware adjustments: Late game boost, competition response",
        "🔄 Dynamic parameter adaptation: Based on game phase and situation",
        "📊 Underperformance boost: Extra creativity for struggling agents",
        "🚫 Anti-repetition penalties: Frequency/presence penalties when needed",
        "🎨 Diversity sampling: Encourages vocabulary and response variety"
    ]
    
    for improvement in improvements:
        print(f"   {improvement}")
    
    print("\n📈 Expected Impact:")
    print("   • More creative and diverse agent responses")
    print("   • Reduced repetition in discussions")
    print("   • Context-appropriate response styles")
    print("   • Better adaptation to competitive pressure")
    print("   • More engaging late-game dynamics")


def main():
    """Run all tests"""
    print("🚀 Testing Enhanced LLM Parameters for Arena")
    
    try:
        test_parameter_optimization()
        test_settings_comparison()
        test_creative_scenarios()
        test_token_optimization()
        show_improvement_summary()
        
        print(f"\n{'='*60}")
        print("🎉 All tests completed successfully!")
        print("Enhanced creativity and content richness systems are ready.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())