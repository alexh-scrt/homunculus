"""AgentInput dataclass for standardized agent outputs."""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class AgentInput:
    """
    Standardized output from any agent.
    
    All agents provide their perspective through this structured format,
    allowing the cognitive module to synthesize inputs consistently.
    """
    
    agent_type: str                    # Type of agent (e.g., "personality", "mood", "goals")
    content: str                       # Main recommendation/observation from the agent
    confidence: float                  # 0-1, how certain is this agent about its input
    priority: float                    # 0-1, how important is this input for the response
    emotional_tone: Optional[str]      # e.g., "cautious", "enthusiastic", "neutral"
    metadata: Dict[str, Any]          # Agent-specific extra data
    
    def __post_init__(self) -> None:
        """Validate AgentInput values."""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
        
        if not 0 <= self.priority <= 1:
            raise ValueError(f"Priority must be between 0 and 1, got {self.priority}")
        
        if not self.agent_type:
            raise ValueError("Agent type cannot be empty")
        
        if not self.content:
            raise ValueError("Content cannot be empty")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_type": self.agent_type,
            "content": self.content,
            "confidence": self.confidence,
            "priority": self.priority,
            "emotional_tone": self.emotional_tone,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentInput":
        """Create AgentInput from dictionary."""
        return cls(
            agent_type=data["agent_type"],
            content=data["content"],
            confidence=data["confidence"],
            priority=data["priority"],
            emotional_tone=data.get("emotional_tone"),
            metadata=data.get("metadata", {})
        )