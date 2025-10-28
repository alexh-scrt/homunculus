"""CharacterState dataclass for complete character state management."""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime


@dataclass
class CharacterState:
    """
    Complete state of a character at any moment.
    
    This is the single source of truth for character state - no prompt engineering needed.
    Aligns with the character schema defined in schemas/characters/*.yaml files.
    """
    
    character_id: str
    last_updated: datetime
    
    # User context for persistent conversations
    user_id: str = "anonymous"
    conversation_id: str = ""
    
    # Core character configuration (loaded from YAML, mostly static)
    name: str = ""
    archetype: str = ""
    demographics: Dict[str, Any] = field(default_factory=dict)
    
    # Agent-specific states (dynamic, updated during conversations)
    agent_states: Dict[str, Any] = field(default_factory=dict)
    
    # Neurochemical levels (0-100 scale, highly dynamic)
    neurochemical_levels: Dict[str, float] = field(default_factory=lambda: {
        'dopamine': 50.0,      # Achievement, novelty, reward-seeking
        'serotonin': 50.0,     # Status, confidence, well-being
        'oxytocin': 50.0,      # Connection, trust, bonding
        'endorphins': 50.0,    # Pleasure, comfort, pain relief
        'cortisol': 50.0,      # Stress, threat response, anxiety
        'adrenaline': 50.0     # Acute danger, excitement, fight-or-flight
    })
    
    # Conversation memory (working memory for current session)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    
    # Relationship state with current user/human
    relationship_state: Dict[str, Any] = field(default_factory=lambda: {
        'trust_level': 0.5,                    # 0-1, how much character trusts user
        'interaction_count': 0,                # Total interactions with this user
        'last_interaction_quality': 'neutral', # 'positive', 'neutral', 'negative'
        'shared_experiences': 0,               # Count of meaningful shared moments
        'connection_strength': 0.0             # 0-1, overall relationship strength
    })
    
    # Web search and learning tracking
    web_search_history: List[Dict[str, Any]] = field(default_factory=list)
    knowledge_updates: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Initialize default agent states if not provided."""
        if not self.agent_states:
            self.agent_states = {
                'personality': {
                    'big_five': {
                        'openness': 0.5,
                        'conscientiousness': 0.5,
                        'extraversion': 0.5,
                        'agreeableness': 0.5,
                        'neuroticism': 0.5
                    },
                    'behavioral_traits': [],
                    'core_values': []
                },
                'specialty': {
                    'domain': 'general',
                    'expertise_level': 0.5,
                    'subdomain_knowledge': []
                },
                'skills': {
                    'intelligence': {
                        'analytical': 0.5,
                        'creative': 0.5,
                        'practical': 0.5
                    },
                    'emotional_intelligence': 0.5,
                    'physical_capability': 0.5,
                    'problem_solving': 0.5
                },
                'mood': {
                    'current_state': 'neutral',
                    'intensity': 0.5,
                    'duration': 1,
                    'baseline_setpoint': 0.5,
                    'emotional_volatility': 0.5,
                    'triggered_by': 'initialization'
                },
                'communication_style': {
                    'verbal_pattern': 'moderate',
                    'social_comfort': 'neutral',
                    'listening_preference': 0.5,
                    'body_language': 'neutral',
                    'quirks': []
                },
                'goals': {
                    'active_goals': [],
                    'completed_goals': [],
                    'abandoned_goals': []
                },
                'development': {
                    'arc_stage': 'introduction',
                    'growth_areas': [],
                    'key_experiences': [],
                    'changed_beliefs': []
                }
            }
    
    def set_user_context(self, user_id: str) -> None:
        """Set user context and generate conversation ID."""
        self.user_id = user_id
        self.conversation_id = self._generate_conversation_id()
    
    def _generate_conversation_id(self) -> str:
        """Generate conversation ID from user and character IDs."""
        safe_user_id = "".join(c for c in self.user_id if c.isalnum() or c in '-_').lower()
        safe_character_id = "".join(c for c in self.character_id if c.isalnum() or c in '-_').lower()
        return f"{safe_user_id}-{safe_character_id}"
    
    def add_to_history(self, role: str, message: str) -> None:
        """Add message to conversation history (working memory only)."""
        self.conversation_history.append({
            'role': role,  # 'user' or 'character'
            'message': message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 20 messages for working memory
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
    
    def update_neurochemical_level(self, hormone: str, new_level: float) -> None:
        """Update a specific hormone level with bounds checking."""
        if hormone not in self.neurochemical_levels:
            raise ValueError(f"Unknown hormone: {hormone}")
        
        # Clamp to 0-100 range
        self.neurochemical_levels[hormone] = max(0.0, min(100.0, new_level))
    
    def apply_neurochemical_changes(self, changes: Dict[str, float]) -> None:
        """Apply multiple hormone changes at once."""
        for hormone, delta in changes.items():
            if hormone in self.neurochemical_levels:
                current_level = self.neurochemical_levels[hormone]
                new_level = current_level + delta
                self.update_neurochemical_level(hormone, new_level)
    
    def update_mood_from_hormones(self) -> None:
        """Recalculate mood based on current hormone levels."""
        dopamine = self.neurochemical_levels['dopamine']
        cortisol = self.neurochemical_levels['cortisol']
        oxytocin = self.neurochemical_levels['oxytocin']
        serotonin = self.neurochemical_levels['serotonin']
        
        # Determine mood state based on hormone pattern
        if cortisol > 70:
            new_state = "anxious"
            intensity = min(1.0, cortisol / 100)
        elif dopamine > 70 and oxytocin > 60:
            new_state = "happy"
            intensity = min(1.0, (dopamine + oxytocin) / 200)
        elif dopamine > 70:
            new_state = "excited"
            intensity = min(1.0, dopamine / 100)
        elif oxytocin > 70:
            new_state = "content"
            intensity = min(1.0, oxytocin / 100)
        elif serotonin > 70:
            new_state = "confident"
            intensity = min(1.0, serotonin / 100)
        elif dopamine < 40 and cortisol > 50:
            new_state = "tired"
            intensity = min(1.0, (100 - dopamine + cortisol) / 200)
        else:
            new_state = "neutral"
            intensity = 0.5
        
        # Update mood state
        current_mood = self.agent_states['mood']
        
        # Update duration if same state, reset if different
        if new_state == current_mood.get('current_state'):
            duration = current_mood.get('duration', 0) + 1
        else:
            duration = 1
        
        self.agent_states['mood'].update({
            'current_state': new_state,
            'intensity': intensity,
            'duration': duration,
            'triggered_by': 'neurochemical_change'
        })
    
    def add_web_search_record(
        self,
        query: str,
        results: List[Dict[str, Any]],
        triggered_by: str = "conversation"
    ) -> None:
        """Record a web search that was performed."""
        search_record = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'result_count': len(results),
            'triggered_by': triggered_by,
            'results_summary': [
                {
                    'title': result.get('title', ''),
                    'url': result.get('url', ''),
                    'snippet': result.get('content', '')[:200] + "..." if len(result.get('content', '')) > 200 else result.get('content', '')
                }
                for result in results[:3]  # Store only top 3 result summaries
            ]
        }
        
        self.web_search_history.append(search_record)
        
        # Keep only last 20 searches
        if len(self.web_search_history) > 20:
            self.web_search_history = self.web_search_history[-20:]
    
    def add_knowledge_update(self, source: str, knowledge: List[str]) -> None:
        """Record new knowledge gained."""
        if knowledge:
            update_record = {
                'timestamp': datetime.now().isoformat(),
                'source': source,  # 'conversation', 'web_search', 'inference'
                'knowledge_items': knowledge
            }
            
            self.knowledge_updates.append(update_record)
            
            # Keep only last 50 knowledge updates
            if len(self.knowledge_updates) > 50:
                self.knowledge_updates = self.knowledge_updates[-50:]
    
    def get_recent_context(self, limit: int = 5) -> List[Dict[str, str]]:
        """Get recent conversation history for context."""
        return self.conversation_history[-limit:] if self.conversation_history else []
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            'character_id': self.character_id,
            'last_updated': self.last_updated.isoformat(),
            'name': self.name,
            'archetype': self.archetype,
            'demographics': self.demographics,
            'agent_states': self.agent_states,
            'neurochemical_levels': self.neurochemical_levels,
            'conversation_history': self.conversation_history,
            'relationship_state': self.relationship_state,
            'web_search_history': self.web_search_history,
            'knowledge_updates': self.knowledge_updates
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CharacterState":
        """Deserialize from dictionary."""
        # Convert timestamp string back to datetime
        last_updated = datetime.fromisoformat(data['last_updated'])
        
        return cls(
            character_id=data['character_id'],
            last_updated=last_updated,
            name=data.get('name', ''),
            archetype=data.get('archetype', ''),
            demographics=data.get('demographics', {}),
            agent_states=data.get('agent_states', {}),
            neurochemical_levels=data.get('neurochemical_levels', {}),
            conversation_history=data.get('conversation_history', []),
            relationship_state=data.get('relationship_state', {}),
            web_search_history=data.get('web_search_history', []),
            knowledge_updates=data.get('knowledge_updates', [])
        )