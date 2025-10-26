"""Experience dataclass for episodic memory storage."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib


@dataclass
class Experience:
    """
    A single memorable experience for the character.
    
    Stored in ChromaDB with vector embedding for semantic search.
    Contains both interaction data and any web search results that
    were part of the experience.
    """
    
    experience_id: str
    character_id: str
    timestamp: datetime
    
    # Content
    experience_type: str              # 'conversation', 'interaction', 'learned_fact', 'event', 'web_search'
    description: str                  # Natural language description
    participants: List[str]           # Who was involved (character_ids or 'human')
    emotional_state: str              # Mood at the time
    location: Optional[str] = None
    
    # Emotional context
    emotional_impact: Dict[str, float] = field(default_factory=dict)  # Hormone changes caused
    emotional_valence: float = 0.0    # -1 (negative) to 1 (positive)
    intensity: float = 0.5            # 0-1, how significant/memorable
    
    # Cognitive context
    related_goals: List[str] = field(default_factory=list)           # goal_ids affected
    knowledge_gained: List[str] = field(default_factory=list)        # new facts learned
    relationship_changes: Dict[str, float] = field(default_factory=dict)  # character_id -> trust delta
    
    # Web search integration
    web_search_triggered: bool = False                               # Was web search used
    web_search_query: Optional[str] = None                          # What was searched
    web_search_results: List[Dict[str, Any]] = field(default_factory=list)  # Search results
    web_knowledge_gained: List[str] = field(default_factory=list)    # Facts from web search
    
    # Metadata
    tags: List[str] = field(default_factory=list)                   # Searchable tags
    embedding: Optional[List[float]] = None                         # Vector embedding (generated)
    retrieval_count: int = 0                                        # How often recalled
    last_retrieved: Optional[datetime] = None
    
    def __post_init__(self) -> None:
        """Validate Experience values."""
        if not -1 <= self.emotional_valence <= 1:
            raise ValueError(f"Emotional valence must be between -1 and 1, got {self.emotional_valence}")
        
        if not 0 <= self.intensity <= 1:
            raise ValueError(f"Intensity must be between 0 and 1, got {self.intensity}")
        
        if self.web_search_triggered and not self.web_search_query:
            raise ValueError("Web search query is required when web search was triggered")
    
    def to_searchable_text(self) -> str:
        """Convert to text for embedding generation."""
        parts = [
            f"Experience at {self.timestamp.isoformat()}",
            f"Participants: {', '.join(self.participants)}",
            f"Description: {self.description}",
            f"Emotional state: {self.emotional_state}",
            f"Impact: {self.emotional_valence:.2f} intensity",
        ]
        
        if self.knowledge_gained:
            parts.append(f"Knowledge: {', '.join(self.knowledge_gained)}")
        
        if self.web_knowledge_gained:
            parts.append(f"Web knowledge: {', '.join(self.web_knowledge_gained)}")
        
        if self.web_search_query:
            parts.append(f"Searched for: {self.web_search_query}")
        
        if self.tags:
            parts.append(f"Tags: {', '.join(self.tags)}")
        
        return "\n".join(parts)
    
    def add_web_search_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        knowledge_extracted: List[str]
    ) -> None:
        """Add web search results to this experience."""
        self.web_search_triggered = True
        self.web_search_query = query
        self.web_search_results = results
        self.web_knowledge_gained.extend(knowledge_extracted)
        
        # Add web search tag
        if "web_search" not in self.tags:
            self.tags.append("web_search")
    
    def increment_retrieval(self) -> None:
        """Update retrieval statistics."""
        self.retrieval_count += 1
        self.last_retrieved = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "experience_id": self.experience_id,
            "character_id": self.character_id,
            "timestamp": self.timestamp.isoformat(),
            "experience_type": self.experience_type,
            "description": self.description,
            "participants": self.participants,
            "location": self.location,
            "emotional_state": self.emotional_state,
            "emotional_impact": self.emotional_impact,
            "emotional_valence": self.emotional_valence,
            "intensity": self.intensity,
            "related_goals": self.related_goals,
            "knowledge_gained": self.knowledge_gained,
            "relationship_changes": self.relationship_changes,
            "web_search_triggered": self.web_search_triggered,
            "web_search_query": self.web_search_query,
            "web_search_results": self.web_search_results,
            "web_knowledge_gained": self.web_knowledge_gained,
            "tags": self.tags,
            "embedding": self.embedding,
            "retrieval_count": self.retrieval_count,
            "last_retrieved": self.last_retrieved.isoformat() if self.last_retrieved else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experience":
        """Create Experience from dictionary."""
        # Convert timestamp strings back to datetime objects
        timestamp = datetime.fromisoformat(data["timestamp"])
        last_retrieved = None
        if data.get("last_retrieved"):
            last_retrieved = datetime.fromisoformat(data["last_retrieved"])
        
        return cls(
            experience_id=data["experience_id"],
            character_id=data["character_id"],
            timestamp=timestamp,
            experience_type=data["experience_type"],
            description=data["description"],
            participants=data["participants"],
            location=data.get("location"),
            emotional_state=data["emotional_state"],
            emotional_impact=data.get("emotional_impact", {}),
            emotional_valence=data.get("emotional_valence", 0.0),
            intensity=data.get("intensity", 0.5),
            related_goals=data.get("related_goals", []),
            knowledge_gained=data.get("knowledge_gained", []),
            relationship_changes=data.get("relationship_changes", {}),
            web_search_triggered=data.get("web_search_triggered", False),
            web_search_query=data.get("web_search_query"),
            web_search_results=data.get("web_search_results", []),
            web_knowledge_gained=data.get("web_knowledge_gained", []),
            tags=data.get("tags", []),
            embedding=data.get("embedding"),
            retrieval_count=data.get("retrieval_count", 0),
            last_retrieved=last_retrieved
        )
    
    @staticmethod
    def generate_id(character_id: str, timestamp: datetime) -> str:
        """Generate unique experience ID."""
        content = f"{character_id}_{timestamp.isoformat()}"
        return f"exp_{hashlib.md5(content.encode()).hexdigest()[:12]}"