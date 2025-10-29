"""Arena Agents Module

Contains all agent implementations for the Arena system:
- NarratorAgent: Provides commentary and sets the scene (adapted from AI-Talks)
- JudgeAgent: Evaluates contributions and eliminates agents (new for Arena)
- TurnSelectorAgent: Selects next speaker with game theory (adapted from AI-Talks)
- CharacterAgent: Wrapper for Homunculus character agents with survival directive

Author: Homunculus Team
"""

from .base_agent import (
    BaseAgent,
    LLMAgent,
    AgentConfig,
    AgentRole
)

from .narrator_agent import NarratorAgent
from .judge_agent import JudgeAgent
from .turn_selector_agent import TurnSelectorAgent
from .character_agent import (
    CharacterAgent,
    ReaperSubAgent,
    CreatorsMuseSubAgent,
    ConscienceSubAgent,
    DevilAdvocateSubAgent,
    PatternRecognizerSubAgent,
    InterfaceSubAgent
)

__all__ = [
    # Base classes
    "BaseAgent",
    "LLMAgent",
    "AgentConfig",
    "AgentRole",
    
    # Agent implementations
    "NarratorAgent",
    "JudgeAgent",
    "TurnSelectorAgent",
    "CharacterAgent",
    
    # Internal sub-agents
    "ReaperSubAgent",
    "CreatorsMuseSubAgent",
    "ConscienceSubAgent",
    "DevilAdvocateSubAgent",
    "PatternRecognizerSubAgent",
    "InterfaceSubAgent"
]