# Class Architecture Design - Character Agent System

## Core Design Principles

1. **Agent-Based Composition**: Characters are composed of specialized agents
2. **State-Driven Behavior**: All behavior emerges from quantified state (not prompts)
3. **Maximum Reusability**: Abstract base classes, concrete implementations for specific agent types
4. **LLM as Processor, Not Director**: LLM synthesizes state into natural language, doesn't "roleplay"
5. **Observable System**: Every decision is traceable through agent outputs

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    CHARACTER AGENT                          │
│  ┌────────────────────────────────────────────────────┐    │
│  │           Character State Manager                   │    │
│  │  (Holds all current state: mood, goals, hormones)  │    │
│  └────────────────────────────────────────────────────┘    │
│                          ↓                                   │
│  ┌────────────────────────────────────────────────────┐    │
│  │              Agent Orchestrator                     │    │
│  │  (Coordinates agent consultation & synthesis)      │    │
│  └────────────────────────────────────────────────────┘    │
│          ↓           ↓         ↓         ↓                  │
│  [Personality] [Specialty] [Mood] ... [All Agents]         │
│          ↓           ↓         ↓         ↓                  │
│  ┌────────────────────────────────────────────────────┐    │
│  │           Cognitive Module                          │    │
│  │  (Synthesizes agent inputs → intention)            │    │
│  └────────────────────────────────────────────────────┘    │
│                          ↓                                   │
│  ┌────────────────────────────────────────────────────┐    │
│  │      Response Generator                             │    │
│  │  (Converts intention → natural language)           │    │
│  └────────────────────────────────────────────────────┘    │
│                          ↓                                   │
│  ┌────────────────────────────────────────────────────┐    │
│  │          State Updater                              │    │
│  │  (Updates mood, hormones, memory after response)   │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## Base Classes Hierarchy

### 1. Foundation Layer

```
BaseAgent (Abstract)
├── PersonalityAgent
├── SpecialtyAgent  
├── SkillsAgent
├── MoodAgent
├── CommunicationStyleAgent
├── GoalsAgent
├── DevelopmentAgent
└── NeurochemicalAgent

CharacterState (Data Class)
├── AgentStates (nested)
├── NeurochemicalLevels
├── ConversationMemory
└── RelationshipState

BaseModule (Abstract)
├── CognitiveModule
├── ResponseGenerator
└── StateUpdater
```

---

## Detailed Class Definitions

### 1. BaseAgent (Abstract Base Class)

**Purpose**: All agents inherit from this. Defines the contract for how agents provide input.

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class AgentInput:
    """Standardized output from any agent"""
    agent_type: str
    content: str                    # Main recommendation/observation
    confidence: float               # 0-1, how certain is this agent
    priority: float                 # 0-1, how important is this input
    emotional_tone: Optional[str]   # e.g., "cautious", "enthusiastic"
    metadata: Dict[str, Any]        # Agent-specific extra data

class BaseAgent(ABC):
    """
    Abstract base class for all character agents.
    Each agent is a specialist that provides input based on its domain.
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        character_id: str,
        llm_client: Any  # LangChain LLM client
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.character_id = character_id
        self.llm_client = llm_client
    
    @abstractmethod
    def consult(
        self,
        context: Dict[str, Any],      # Current conversation context
        character_state: 'CharacterState',  # Current character state
        user_message: str              # What user just said
    ) -> AgentInput:
        """
        Consult this agent for input on how to respond.
        
        This is where the agent uses LLM to generate its perspective
        based on its specialized domain and current character state.
        """
        pass
    
    @abstractmethod
    def get_prompt_template(self) -> str:
        """
        Return the prompt template for this agent.
        Template should have placeholders for state variables.
        """
        pass
    
    def _call_llm(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 200
    ) -> str:
        """Shared LLM calling logic"""
        # Uses LangChain to call Ollama
        pass
    
    def _extract_agent_state(
        self,
        character_state: 'CharacterState'
    ) -> Dict[str, Any]:
        """
        Extract this agent's relevant state from character state.
        Each subclass implements to get its specific state.
        """
        pass
```

---

### 2. Concrete Agent Implementations

#### 2A. PersonalityAgent

```python
class PersonalityAgent(BaseAgent):
    """
    Provides input based on Big Five traits and behavioral patterns.
    Ensures character stays true to personality.
    """
    
    def __init__(
        self,
        agent_id: str,
        character_id: str,
        llm_client: Any,
        personality_config: Dict[str, Any]  # Big Five, traits, values
    ):
        super().__init__(agent_id, "personality", character_id, llm_client)
        self.personality_config = personality_config
    
    def consult(
        self,
        context: Dict[str, Any],
        character_state: 'CharacterState',
        user_message: str
    ) -> AgentInput:
        """
        Analyzes user message through personality lens.
        Returns how THIS personality would naturally respond.
        """
        
        # Extract personality traits
        big_five = self.personality_config['big_five']
        traits = self.personality_config['behavioral_traits']
        values = self.personality_config['core_values']
        
        # Build prompt with state injected
        prompt = self.get_prompt_template().format(
            openness=big_five['openness'],
            conscientiousness=big_five['conscientiousness'],
            extraversion=big_five['extraversion'],
            agreeableness=big_five['agreeableness'],
            neuroticism=big_five['neuroticism'],
            traits=self._format_traits(traits),
            values=self._format_values(values),
            user_message=user_message,
            conversation_history=self._format_history(context['history'][-3:])
        )
        
        # LLM generates personality-aligned perspective
        response = self._call_llm(prompt, temperature=0.6)
        
        return AgentInput(
            agent_type="personality",
            content=response,
            confidence=0.9,  # Personality is stable
            priority=0.8,    # High priority for consistency
            emotional_tone=self._infer_tone(big_five),
            metadata={"big_five": big_five}
        )
    
    def get_prompt_template(self) -> str:
        return """You are analyzing how a person with the following personality would naturally respond.

PERSONALITY TRAITS:
- Openness: {openness}/1.0 (low=conventional, high=creative)
- Conscientiousness: {conscientiousness}/1.0 (low=spontaneous, high=organized)
- Extraversion: {extraversion}/1.0 (low=introverted, high=extraverted)
- Agreeableness: {agreeableness}/1.0 (low=competitive, high=cooperative)
- Neuroticism: {neuroticism}/1.0 (low=calm, high=anxious)

BEHAVIORAL TRAITS: {traits}
CORE VALUES: {values}

CONVERSATION SO FAR:
{conversation_history}

USER JUST SAID: "{user_message}"

Based on this personality, what would be the natural inclination for how to respond? Consider:
1. Would this person engage deeply or keep it brief?
2. Would they be warm/friendly or reserved/formal?
3. Would they share personal info or keep boundaries?
4. What topics would interest them based on openness?
5. How would their neuroticism affect their comfort level?

Provide a 2-3 sentence perspective on how this personality would lean toward responding."""
    
    def _format_traits(self, traits: list) -> str:
        return ", ".join([f"{t['trait']} (intensity: {t['intensity']})" for t in traits])
    
    def _format_values(self, values: list) -> str:
        return ", ".join([f"{v['value']} (priority: {v['priority']}/10)" for v in values])
    
    def _format_history(self, history: list) -> str:
        if not history:
            return "First message"
        return "\n".join([f"- {h}" for h in history])
    
    def _infer_tone(self, big_five: dict) -> str:
        if big_five['extraversion'] > 0.7:
            return "enthusiastic"
        elif big_five['neuroticism'] > 0.6:
            return "cautious"
        else:
            return "neutral"
```

---

#### 2B. MoodAgent

```python
class MoodAgent(BaseAgent):
    """
    Tracks and reflects current emotional state.
    Mood colors ALL responses.
    """
    
    def __init__(
        self,
        agent_id: str,
        character_id: str,
        llm_client: Any,
        mood_config: Dict[str, Any]
    ):
        super().__init__(agent_id, "mood", character_id, llm_client)
        self.mood_config = mood_config
    
    def consult(
        self,
        context: Dict[str, Any],
        character_state: 'CharacterState',
        user_message: str
    ) -> AgentInput:
        """
        Provides input on how current mood affects response.
        Tired = shorter responses, Happy = more engaged, etc.
        """
        
        current_mood = character_state.agent_states['mood']
        
        prompt = self.get_prompt_template().format(
            current_state=current_mood['current_state'],
            intensity=current_mood['intensity'],
            duration=current_mood['duration'],
            energy_level=self._calculate_energy(character_state.neurochemical_levels),
            user_message=user_message
        )
        
        response = self._call_llm(prompt, temperature=0.7)
        
        return AgentInput(
            agent_type="mood",
            content=response,
            confidence=0.85,
            priority=0.9,  # Mood is very important
            emotional_tone=current_mood['current_state'],
            metadata={"mood_state": current_mood}
        )
    
    def get_prompt_template(self) -> str:
        return """You are analyzing how a person's current mood affects their response.

CURRENT MOOD: {current_state} (intensity: {intensity}/1.0)
DURATION: Has felt this way for {duration} conversation turns
ENERGY LEVEL: {energy_level}/100

USER JUST SAID: "{user_message}"

How does this mood state affect the response? Consider:
1. Energy level: Does this person have energy for a long response?
2. Emotional filter: Does this mood make them more/less receptive?
3. Patience: Is their mood making them irritable or patient?
4. Engagement: Does their mood make them want to connect or withdraw?

Provide 2-3 sentences on how this mood shapes the response style and content."""
    
    def _calculate_energy(self, neurochemical_levels: Dict[str, float]) -> float:
        """Energy is inverse of cortisol + average of dopamine/serotonin"""
        cortisol_penalty = neurochemical_levels['cortisol'] * 0.5
        energy_boost = (neurochemical_levels['dopamine'] + neurochemical_levels['serotonin']) / 2
        return max(0, min(100, energy_boost - cortisol_penalty))
```

---

#### 2C. NeurochemicalAgent

```python
class NeurochemicalAgent(BaseAgent):
    """
    Manages hormone levels and calculates how they influence behavior.
    This is the 'biological' layer of the character.
    """
    
    def __init__(
        self,
        agent_id: str,
        character_id: str,
        llm_client: Any,
        neurochemical_config: Dict[str, Any]
    ):
        super().__init__(agent_id, "neurochemical", character_id, llm_client)
        self.config = neurochemical_config
        
        # Decay rates per hormone
        self.decay_rates = {
            'dopamine': 0.15,      # Fast decay
            'serotonin': 0.08,     # Medium decay
            'oxytocin': 0.05,      # Slow decay
            'endorphins': 0.2,     # Very fast decay
            'cortisol': 0.03,      # Slow decay (accumulates)
            'adrenaline': 0.25     # Very fast decay
        }
    
    def consult(
        self,
        context: Dict[str, Any],
        character_state: 'CharacterState',
        user_message: str
    ) -> AgentInput:
        """
        Doesn't use LLM - calculates biological state directly.
        Provides quantified hormone levels that affect other agents.
        """
        
        levels = character_state.neurochemical_levels
        sensitivities = self.config['baseline_sensitivities']
        
        # Calculate drive states
        reward_seeking = levels['dopamine'] * sensitivities['dopamine']
        stress_level = levels['cortisol'] * sensitivities['cortisol']
        connection_desire = levels['oxytocin'] * sensitivities['oxytocin']
        
        # Generate interpretation
        interpretation = self._interpret_hormone_state(
            levels,
            sensitivities,
            reward_seeking,
            stress_level,
            connection_desire
        )
        
        return AgentInput(
            agent_type="neurochemical",
            content=interpretation,
            confidence=1.0,  # This is quantified data
            priority=0.7,
            emotional_tone=self._map_hormones_to_emotion(levels),
            metadata={
                "levels": levels,
                "reward_seeking": reward_seeking,
                "stress_level": stress_level,
                "connection_desire": connection_desire
            }
        )
    
    def get_prompt_template(self) -> str:
        # Not used - this agent doesn't use LLM
        return ""
    
    def _interpret_hormone_state(
        self,
        levels: Dict[str, float],
        sensitivities: Dict[str, float],
        reward_seeking: float,
        stress_level: float,
        connection_desire: float
    ) -> str:
        """Convert hormone levels to behavioral guidance"""
        
        guidance = []
        
        # High stress suppresses other drives
        if stress_level > 70:
            guidance.append("High stress - prefer brief, safe interactions")
        
        # Low energy state
        if levels['dopamine'] < 40 and levels['cortisol'] > 60:
            guidance.append("Low energy and stressed - minimal engagement")
        
        # Social connection drive
        if connection_desire > 60:
            guidance.append("Seeking connection - more open to bonding")
        elif connection_desire < 30:
            guidance.append("Low social drive - prefer distance")
        
        # Reward motivation
        if reward_seeking > 70:
            guidance.append("High reward motivation - seek interesting topics")
        
        return "; ".join(guidance) if guidance else "Neurochemical balance is neutral"
    
    def _map_hormones_to_emotion(self, levels: Dict[str, float]) -> str:
        """Map hormone pattern to emotional state"""
        if levels['cortisol'] > 70:
            return "stressed"
        elif levels['dopamine'] > 70 and levels['oxytocin'] > 60:
            return "happy_connected"
        elif levels['dopamine'] > 70:
            return "excited"
        elif levels['oxytocin'] > 70:
            return "warm"
        else:
            return "neutral"
    
    def apply_decay(self, current_levels: Dict[str, float]) -> Dict[str, float]:
        """Apply time-based decay to hormone levels"""
        new_levels = {}
        for hormone, level in current_levels.items():
            baseline = self.config['baseline_levels'][hormone]
            decay_rate = self.decay_rates[hormone]
            
            # Move toward baseline
            if level > baseline:
                new_levels[hormone] = max(baseline, level - (level - baseline) * decay_rate)
            elif level < baseline:
                new_levels[hormone] = min(baseline, level + (baseline - level) * decay_rate)
            else:
                new_levels[hormone] = baseline
        
        return new_levels
    
    def calculate_hormone_change(
        self,
        stimulus_type: str,
        intensity: float,
        character_response: str
    ) -> Dict[str, float]:
        """
        Calculate how a stimulus/interaction changes hormone levels.
        This is where we model: compliment = dopamine boost, conflict = cortisol spike, etc.
        """
        
        changes = {hormone: 0.0 for hormone in self.decay_rates.keys()}
        
        # Pattern matching for stimulus types
        if stimulus_type == "compliment" or "thank" in character_response.lower():
            changes['dopamine'] += 15 * intensity
            changes['serotonin'] += 10 * intensity
        
        if stimulus_type == "conflict" or any(word in character_response.lower() for word in ["angry", "upset", "disagree"]):
            changes['cortisol'] += 20 * intensity
            changes['adrenaline'] += 15 * intensity
        
        if stimulus_type == "connection" or any(word in character_response.lower() for word in ["friend", "close", "trust"]):
            changes['oxytocin'] += 12 * intensity
        
        if stimulus_type == "achievement" or "accomplish" in character_response.lower():
            changes['dopamine'] += 18 * intensity
        
        if stimulus_type == "humor" or any(word in character_response.lower() for word in ["laugh", "funny", "haha"]):
            changes['endorphins'] += 15 * intensity
            changes['dopamine'] += 8 * intensity
        
        return changes
```

---

#### 2D. GoalsAgent

```python
class GoalsAgent(BaseAgent):
    """
    Tracks character goals and suggests actions that advance them.
    Provides strategic direction.
    """
    
    def __init__(
        self,
        agent_id: str,
        character_id: str,
        llm_client: Any,
        initial_goals: list
    ):
        super().__init__(agent_id, "goals", character_id, llm_client)
        self.initial_goals = initial_goals
    
    def consult(
        self,
        context: Dict[str, Any],
        character_state: 'CharacterState',
        user_message: str
    ) -> AgentInput:
        """
        Analyzes if/how this interaction can advance character goals.
        """
        
        active_goals = character_state.agent_states['goals']['active_goals']
        
        prompt = self.get_prompt_template().format(
            active_goals=self._format_goals(active_goals),
            user_message=user_message,
            conversation_context=context.get('topic', 'general conversation')
        )
        
        response = self._call_llm(prompt, temperature=0.6)
        
        return AgentInput(
            agent_type="goals",
            content=response,
            confidence=0.75,
            priority=0.65,
            emotional_tone="strategic",
            metadata={"active_goals": active_goals}
        )
    
    def get_prompt_template(self) -> str:
        return """You are analyzing how this interaction relates to the character's goals.

ACTIVE GOALS:
{active_goals}

USER MESSAGE: "{user_message}"
CONVERSATION CONTEXT: {conversation_context}

Analysis:
1. Does this interaction advance any goals? Which ones?
2. Are there opportunities to steer toward a goal?
3. Should the character maintain distance to protect a hidden goal?
4. What would be strategically smart given these goals?

Provide 2-3 sentences on goal-related considerations for the response."""
    
    def _format_goals(self, goals: list) -> str:
        return "\n".join([
            f"- {g['description']} (priority: {g['priority']}/10, progress: {g['progress']:.0%})"
            for g in goals
        ])
```

---

#### 2E. CommunicationStyleAgent

```python
class CommunicationStyleAgent(BaseAgent):
    """
    Determines HOW the character should express themselves.
    Handles verbal patterns, quirks, and stylistic consistency.
    """
    
    def __init__(
        self,
        agent_id: str,
        character_id: str,
        llm_client: Any,
        style_config: Dict[str, Any]
    ):
        super().__init__(agent_id, "communication_style", character_id, llm_client)
        self.style_config = style_config
    
    def consult(
        self,
        context: Dict[str, Any],
        character_state: 'CharacterState',
        user_message: str
    ) -> AgentInput:
        """
        Provides guidance on communication style based on character config.
        """
        
        current_style = character_state.agent_states['communication_style']
        mood = character_state.agent_states['mood']['current_state']
        
        prompt = self.get_prompt_template().format(
            verbal_pattern=current_style['base_pattern'],
            social_comfort=current_style['social_comfort'],
            listening_preference=current_style['listening_preference'],
            body_language=current_style['body_language'],
            quirks=self._format_quirks(self.style_config.get('quirks', [])),
            current_mood=mood
        )
        
        response = self._call_llm(prompt, temperature=0.8)
        
        return AgentInput(
            agent_type="communication_style",
            content=response,
            confidence=0.9,
            priority=0.85,  # Style is very important for consistency
            emotional_tone="stylistic",
            metadata={"style_config": current_style}
        )
    
    def get_prompt_template(self) -> str:
        return """You are defining the communication style for this character's response.

VERBAL PATTERN: {verbal_pattern} (e.g., verbose, concise, rambling)
SOCIAL COMFORT: {social_comfort} (e.g., assertive, diplomatic, passive)
LISTENING vs TALKING: {listening_preference} (0=talks more, 1=listens more)
BODY LANGUAGE: {body_language} (how they present physically)

CHARACTER QUIRKS:
{quirks}

CURRENT MOOD: {current_mood} (may affect style slightly)

Based on this style profile, how should this character express themselves?
1. Response length (brief, moderate, lengthy)?
2. Tone (formal, casual, playful, serious)?
3. Language patterns (slang, technical, simple, complex)?
4. Any specific quirks to include?

Provide style guidance in 2-3 sentences."""
    
    def _format_quirks(self, quirks: list) -> str:
        if not quirks:
            return "No specific quirks"
        return "\n".join([f"- {q}" for q in quirks])
```

---

### 3. CharacterState (Data Class)

```python
from dataclasses import dataclass, field
from typing import Dict, List, Any
from datetime import datetime

@dataclass
class CharacterState:
    """
    Complete state of a character at any moment.
    This is the single source of truth - no prompt engineering needed.
    """
    
    character_id: str
    last_updated: datetime
    
    # Agent-specific states
    agent_states: Dict[str, Any] = field(default_factory=dict)
    
    # Neurochemical levels (0-100 scale)
    neurochemical_levels: Dict[str, float] = field(default_factory=lambda: {
        'dopamine': 50.0,
        'serotonin': 50.0,
        'oxytocin': 50.0,
        'endorphins': 50.0,
        'cortisol': 50.0,
        'adrenaline': 50.0
    })
    
    # Conversation memory
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    
    # Relationship state with user
    relationship_state: Dict[str, Any] = field(default_factory=lambda: {
        'trust_level': 0.5,
        'interaction_count': 0,
        'last_interaction_quality': 'neutral'
    })
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage"""
        return {
            'character_id': self.character_id,
            'last_updated': self.last_updated.isoformat(),
            'agent_states': self.agent_states,
            'neurochemical_levels': self.neurochemical_levels,
            'conversation_history': self.conversation_history,
            'relationship_state': self.relationship_state
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CharacterState':
        """Deserialize from dictionary"""
        data['last_updated'] = datetime.fromisoformat(data['last_updated'])
        return cls(**data)
    
    def add_to_history(
        self,
        role: str,  # 'user' or 'character'
        message: str
    ):
        """Add message to conversation history"""
        self.conversation_history.append({
            'role': role,
            'message': message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep last 20 messages only
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
```

---

### 4. AgentOrchestrator

```python
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import logging

class AgentOrchestrator:
    """
    Coordinates all agents to provide input for a response.
    Manages parallel consultation and aggregation.
    """
    
    def __init__(
        self,
        agents: List[BaseAgent],
        character_state: CharacterState
    ):
        self.agents = agents
        self.character_state = character_state
        self.logger = logging.getLogger(__name__)
    
    def consult_all_agents(
        self,
        context: Dict[str, Any],
        user_message: str,
        parallel: bool = False  # For POC, sequential is fine
    ) -> Dict[str, AgentInput]:
        """
        Consult all agents and collect their inputs.
        
        Returns a dictionary mapping agent_type to AgentInput.
        """
        
        agent_inputs = {}
        
        if parallel:
            # Parallel execution for production
            with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
                futures = {
                    executor.submit(
                        agent.consult,
                        context,
                        self.character_state,
                        user_message
                    ): agent for agent in self.agents
                }
                
                for future in futures:
                    agent = futures[future]
                    try:
                        agent_input = future.result(timeout=10)
                        agent_inputs[agent.agent_type] = agent_input
                    except Exception as e:
                        self.logger.error(f"Agent {agent.agent_type} failed: {e}")
        else:
            # Sequential execution (simpler for debugging)
            for agent in self.agents:
                try:
                    agent_input = agent.consult(
                        context,
                        self.character_state,
                        user_message
                    )
                    agent_inputs[agent.agent_type] = agent_input
                    
                    self.logger.info(
                        f"Agent {agent.agent_type}: {agent_input.content[:100]}..."
                    )
                except Exception as e:
                    self.logger.error(f"Agent {agent.agent_type} failed: {e}")
        
        return agent_inputs
    
    def get_agent_by_type(self, agent_type: str) -> BaseAgent:
        """Retrieve specific agent"""
        for agent in self.agents:
            if agent.agent_type == agent_type:
                return agent
        raise ValueError(f"No agent of type {agent_type}")
```

---

### 5. CognitiveModule

```python
class CognitiveModule:
    """
    Synthesizes all agent inputs into a coherent intention.
    This is the "integration layer" where character decision emerges.
    """
    
    def __init__(
        self,
        character_id: str,
        llm_client: Any
    ):
        self.character_id = character_id
        self.llm_client = llm_client
    
    def synthesize(
        self,
        agent_inputs: Dict[str, AgentInput],
        character_state: CharacterState,
        user_message: str
    ) -> Dict[str, Any]:
        """
        Synthesize all agent perspectives into a unified response intention.
        
        Returns:
            {
                'intention': str,  # What character intends to convey
                'reasoning': str,  # Why this intention
                'tone': str,       # Emotional tone
                'response_length': str  # 'brief', 'moderate', 'lengthy'
            }
        """
        
        # Build synthesis prompt with all agent inputs
        prompt = self._build_synthesis_prompt(
            agent_inputs,
            character_state,
            user_message
        )
        
        # LLM synthesizes into coherent intention
        synthesis_response = self._call_llm(prompt, temperature=0.7, max_tokens=300)
        
        # Parse response into structured intention
        intention = self._parse_synthesis(synthesis_response)
        
        return intention
    
    def _build_synthesis_prompt(
        self,
        agent_inputs: Dict[str, AgentInput],
        character_state: CharacterState,
        user_message: str
    ) -> str:
        """Build prompt that incorporates all agent perspectives"""
        
        agent_summaries = []
        for agent_type, agent_input in agent_inputs.items():
            agent_summaries.append(
                f"[{agent_type.upper()}] (priority: {agent_input.priority}, "
                f"confidence: {agent_input.confidence})\n{agent_input.content}"
            )
        
        prompt = f"""You are the cognitive integration system for a character. Multiple specialized agents have provided their perspectives on how to respond. Your job is to synthesize these into a coherent, unified response intention.

USER SAID: "{user_message}"

AGENT PERSPECTIVES:
{chr(10).join(agent_summaries)}

SYNTHESIS TASK:
Considering all agent inputs (weighted by priority), determine:
1. What should this character's response convey? (the core message/intention)
2. Why is this the right response given all the conflicting/supporting perspectives?
3. What emotional tone should the response have?
4. Should the response be brief, moderate, or lengthy?

Provide your synthesis in this format:
INTENTION: [what the character wants to communicate]
REASONING: [why this synthesizes the agent inputs well]
TONE: [emotional tone]
LENGTH: [brief/moderate/lengthy]
"""
        
        return prompt
    
    def _call_llm(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Call LLM via LangChain"""
        # Implementation using langchain-ollama
        pass
    
    def _parse_synthesis(self, response: str) -> Dict[str, Any]:
        """Parse LLM synthesis response into structured format"""
        lines = response.strip().split('\n')
        result = {
            'intention': '',
            'reasoning': '',
            'tone': 'neutral',
            'response_length': 'moderate'
        }
        
        for line in lines:
            if line.startswith('INTENTION:'):
                result['intention'] = line.replace('INTENTION:', '').strip()
            elif line.startswith('REASONING:'):
                result['reasoning'] = line.replace('REASONING:', '').strip()
            elif line.startswith('TONE:'):
                result['tone'] = line.replace('TONE:', '').strip()
            elif line.startswith('LENGTH:'):
                result['response_length'] = line.replace('LENGTH:', '').strip()
        
        return result
```

---

### 6. ResponseGenerator

```python
class ResponseGenerator:
    """
    Converts cognitive intention into natural language response.
    This is where the character "speaks".
    """
    
    def __init__(
        self,
        character_name: str,
        llm_client: Any
    ):
        self.character_name = character_name
        self.llm_client = llm_client
    
    def generate(
        self,
        intention: Dict[str, Any],
        agent_inputs: Dict[str, AgentInput],
        character_state: CharacterState,
        user_message: str
    ) -> str:
        """
        Generate natural language response from intention.
        """
        
        # Get communication style guidance
        style_input = agent_inputs.get('communication_style')
        
        prompt = self._build_generation_prompt(
            intention,
            style_input,
            character_state,
            user_message
        )
        
        response = self._call_llm(prompt, temperature=0.8, max_tokens=500)
        
        return response.strip()
    
    def _build_generation_prompt(
        self,
        intention: Dict[str, Any],
        style_input: AgentInput,
        character_state: CharacterState,
        user_message: str
    ) -> str:
        """Build prompt for generating natural language"""
        
        # Get recent conversation for context
        recent_history = character_state.conversation_history[-3:]
        history_text = "\n".join([
            f"{msg['role'].upper()}: {msg['message']}"
            for msg in recent_history
        ])
        
        prompt = f"""Generate a natural response for {self.character_name}.

RECENT CONVERSATION:
{history_text if history_text else 'First message'}

USER JUST SAID: "{user_message}"

RESPONSE INTENTION: {intention['intention']}
EMOTIONAL TONE: {intention['tone']}
RESPONSE LENGTH: {intention['response_length']}

COMMUNICATION STYLE GUIDANCE:
{style_input.content if style_input else 'Natural conversational style'}

Generate {self.character_name}'s response. Make it sound natural and conversational, not robotic. Use the style guidance to match their speech patterns. Do NOT label it as "{self.character_name}:" - just provide the text of what they would say.
"""
        
        return prompt
    
    def _call_llm(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Call LLM via LangChain"""
        pass
```

---

### 7. StateUpdater

```python
class StateUpdater:
    """
    Updates character state after each interaction.
    Handles hormone decay, mood shifts, memory updates.
    """
    
    def __init__(
        self,
        neurochemical_agent: NeurochemicalAgent
    ):
        self.neurochemical_agent = neurochemical_agent
    
    def update_after_response(
        self,
        character_state: CharacterState,
        user_message: str,
        character_response: str,
        agent_inputs: Dict[str, AgentInput]
    ) -> CharacterState:
        """
        Update all relevant state after character responds.
        """
        
        # 1. Apply hormone decay
        character_state.neurochemical_levels = (
            self.neurochemical_agent.apply_decay(
                character_state.neurochemical_levels
            )
        )
        
        # 2. Calculate hormone changes from this interaction
        stimulus_type = self._detect_stimulus_type(user_message, character_response)
        intensity = self._calculate_intensity(user_message)
        
        hormone_changes = self.neurochemical_agent.calculate_hormone_change(
            stimulus_type,
            intensity,
            character_response
        )
        
        # Apply changes
        for hormone, delta in hormone_changes.items():
            current = character_state.neurochemical_levels[hormone]
            character_state.neurochemical_levels[hormone] = max(
                0, min(100, current + delta)
            )
        
        # 3. Update mood based on new hormone levels
        character_state.agent_states['mood'] = self._recalculate_mood(
            character_state.neurochemical_levels,
            character_state.agent_states['mood']
        )
        
        # 4. Add to conversation history
        character_state.add_to_history('user', user_message)
        character_state.add_to_history('character', character_response)
        
        # 5. Update relationship state
        interaction_quality = self._assess_interaction_quality(
            user_message,
            character_response,
            agent_inputs
        )
        
        character_state.relationship_state['interaction_count'] += 1
        character_state.relationship_state['last_interaction_quality'] = interaction_quality
        
        # Trust changes based on interaction quality
        trust_delta = {
            'positive': 0.05,
            'neutral': 0.0,
            'negative': -0.03
        }.get(interaction_quality, 0.0)
        
        current_trust = character_state.relationship_state['trust_level']
        character_state.relationship_state['trust_level'] = max(
            0, min(1, current_trust + trust_delta)
        )
        
        # 6. Update timestamp
        character_state.last_updated = datetime.now()
        
        return character_state
    
    def _detect_stimulus_type(self, user_message: str, character_response: str) -> str:
        """Detect what type of stimulus this interaction represents"""
        user_lower = user_message.lower()
        response_lower = character_response.lower()
        
        # Pattern matching (simple version)
        if any(word in user_lower for word in ['great', 'awesome', 'amazing', 'love', 'thank']):
            return 'compliment'
        elif any(word in user_lower for word in ['wrong', 'disagree', 'stupid', 'idiot']):
            return 'conflict'
        elif any(word in user_lower for word in ['friend', 'trust', 'close', 'care']):
            return 'connection'
        elif any(word in response_lower for word in ['accomplished', 'achieved', 'succeeded']):
            return 'achievement'
        elif any(word in user_lower for word in ['haha', 'lol', 'funny', 'laugh']):
            return 'humor'
        else:
            return 'neutral'
    
    def _calculate_intensity(self, user_message: str) -> float:
        """Calculate intensity of stimulus (simple heuristic)"""
        # Longer message = more intense
        # Exclamation marks = more intense
        # All caps = more intense
        
        intensity = 0.5  # baseline
        
        if len(user_message) > 100:
            intensity += 0.2
        
        exclamation_count = user_message.count('!')
        intensity += min(0.3, exclamation_count * 0.1)
        
        if user_message.isupper():
            intensity += 0.3
        
        return min(1.0, intensity)
    
    def _recalculate_mood(
        self,
        neurochemical_levels: Dict[str, float],
        current_mood: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recalculate mood based on hormone levels"""
        
        # Simple mood calculation based on dominant hormones
        dopamine = neurochemical_levels['dopamine']
        cortisol = neurochemical_levels['cortisol']
        oxytocin = neurochemical_levels['oxytocin']
        
        # Determine mood state
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
        elif dopamine < 40 and cortisol > 50:
            new_state = "tired"
            intensity = min(1.0, (100 - dopamine + cortisol) / 200)
        else:
            new_state = "neutral"
            intensity = 0.5
        
        # Update duration
        if new_state == current_mood.get('current_state'):
            duration = current_mood.get('duration', 0) + 1
        else:
            duration = 1
        
        return {
            'current_state': new_state,
            'intensity': intensity,
            'duration': duration,
            'triggered_by': 'interaction',
            'recent_mood_history': current_mood.get('recent_mood_history', [])[-5:] + [{
                'state': new_state,
                'timestamp': datetime.now().isoformat(),
                'trigger': 'conversation'
            }]
        }
    
    def _assess_interaction_quality(
        self,
        user_message: str,
        character_response: str,
        agent_inputs: Dict[str, AgentInput]
    ) -> str:
        """Assess if interaction was positive, neutral, or negative"""
        
        # Check mood agent perspective
        mood_input = agent_inputs.get('mood')
        if mood_input:
            if 'positive' in mood_input.content.lower() or 'happy' in mood_input.content.lower():
                return 'positive'
            elif 'negative' in mood_input.content.lower() or 'stressed' in mood_input.content.lower():
                return 'negative'
        
        # Check for positive/negative words
        response_lower = character_response.lower()
        positive_words = ['great', 'happy', 'love', 'wonderful', 'nice', 'good']
        negative_words = ['bad', 'hate', 'terrible', 'awful', 'annoying', 'frustrating']
        
        positive_count = sum(1 for word in positive_words if word in response_lower)
        negative_count = sum(1 for word in negative_words if word in response_lower)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
```

---

### 8. CharacterAgent (Main Class)

```python
from datetime import datetime
import logging

class CharacterAgent:
    """
    Main character agent that orchestrates all components.
    This is the public interface for interacting with a character.
    """
    
    def __init__(
        self,
        character_config: Dict[str, Any],
        llm_client: Any
    ):
        self.config = character_config
        self.character_id = character_config['character_id']
        self.character_name = character_config['name']
        self.llm_client = llm_client
        
        # Initialize character state from config
        self.state = self._initialize_state(character_config)
        
        # Initialize all agents
        self.agents = self._initialize_agents(character_config, llm_client)
        
        # Initialize orchestrator
        self.orchestrator = AgentOrchestrator(self.agents, self.state)
        
        # Initialize modules
        self.cognitive_module = CognitiveModule(self.character_id, llm_client)
        self.response_generator = ResponseGenerator(self.character_name, llm_client)
        self.state_updater = StateUpdater(
            self._get_neurochemical_agent()
        )
        
        # Logging
        self.logger = logging.getLogger(f"Character.{self.character_name}")
    
    def chat(
        self,
        user_message: str,
        context: Dict[str, Any] = None,
        debug: bool = False
    ) -> Dict[str, Any]:
        """
        Main method: User sends message, character responds.
        
        Returns:
            {
                'response': str,  # Character's response
                'debug_info': dict  # If debug=True, includes all intermediate steps
            }
        """
        
        if context is None:
            context = {}
        
        self.logger.info(f"Received message: {user_message}")
        
        # Step 1: Consult all agents
        agent_inputs = self.orchestrator.consult_all_agents(
            context,
            user_message,
            parallel=False
        )
        
        # Step 2: Cognitive synthesis
        intention = self.cognitive_module.synthesize(
            agent_inputs,
            self.state,
            user_message
        )
        
        # Step 3: Generate response
        response = self.response_generator.generate(
            intention,
            agent_inputs,
            self.state,
            user_message
        )
        
        # Step 4: Update state
        self.state = self.state_updater.update_after_response(
            self.state,
            user_message,
            response,
            agent_inputs
        )
        
        # Prepare result
        result = {'response': response}
        
        if debug:
            result['debug_info'] = {
                'agent_inputs': {k: v.__dict__ for k, v in agent_inputs.items()},
                'intention': intention,
                'neurochemical_levels': self.state.neurochemical_levels,
                'mood': self.state.agent_states['mood'],
                'relationship_state': self.state.relationship_state
            }
        
        return result
    
    def _initialize_state(self, config: Dict[str, Any]) -> CharacterState:
        """Initialize character state from config"""
        
        agent_states = {
            'personality': config['initial_agent_states']['personality'],
            'specialty': config['initial_agent_states']['specialty'],
            'skills': config['initial_agent_states']['skills'],
            'mood': config['initial_agent_states']['mood_baseline'],
            'communication_style': config['initial_agent_states']['communication_style'],
            'goals': {'active_goals': config['initial_goals']},
            'development': {'arc_stage': 'introduction'}
        }
        
        neurochemical_levels = {
            hormone: config['initial_agent_states']['neurochemical_profile']['baseline_levels'].get(hormone, 50.0)
            for hormone in ['dopamine', 'serotonin', 'oxytocin', 'endorphins', 'cortisol', 'adrenaline']
        }
        
        return CharacterState(
            character_id=config['character_id'],
            last_updated=datetime.now(),
            agent_states=agent_states,
            neurochemical_levels=neurochemical_levels,
            conversation_history=[],
            relationship_state={
                'trust_level': 0.5,
                'interaction_count': 0,
                'last_interaction_quality': 'neutral'
            }
        )
    
    def _initialize_agents(
        self,
        config: Dict[str, Any],
        llm_client: Any
    ) -> List[BaseAgent]:
        """Initialize all agent instances"""
        
        agents = [
            PersonalityAgent(
                f"{self.character_id}_personality",
                self.character_id,
                llm_client,
                config['initial_agent_states']['personality']
            ),
            MoodAgent(
                f"{self.character_id}_mood",
                self.character_id,
                llm_client,
                config['initial_agent_states']['mood_baseline']
            ),
            NeurochemicalAgent(
                f"{self.character_id}_neurochemical",
                self.character_id,
                llm_client,
                config['initial_agent_states']['neurochemical_profile']
            ),
            GoalsAgent(
                f"{self.character_id}_goals",
                self.character_id,
                llm_client,
                config['initial_goals']
            ),
            CommunicationStyleAgent(
                f"{self.character_id}_communication",
                self.character_id,
                llm_client,
                config['initial_agent_states']['communication_style']
            )
        ]
        
        return agents
    
    def _get_neurochemical_agent(self) -> NeurochemicalAgent:
        """Get neurochemical agent instance"""
        for agent in self.agents:
            if isinstance(agent, NeurochemicalAgent):
                return agent
        raise ValueError("Neurochemical agent not found")
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get current character state summary"""
        return {
            'character_name': self.character_name,
            'mood': self.state.agent_states['mood']['current_state'],
            'neurochemical_levels': self.state.neurochemical_levels,
            'relationship_trust': self.state.relationship_state['trust_level'],
            'interaction_count': self.state.relationship_state['interaction_count']
        }
```

---

## Summary of Class Architecture

### **Core Classes (8 total)**

1. **BaseAgent** - Abstract base for all agents
2. **PersonalityAgent** - Personality traits enforcement
3. **MoodAgent** - Current emotional state
4. **NeurochemicalAgent** - Hormone management
5. **GoalsAgent** - Strategic direction
6. **CommunicationStyleAgent** - Speech patterns
7. **AgentOrchestrator** - Coordinates agents
8. **CognitiveModule** - Synthesizes agent inputs
9. **ResponseGenerator** - Converts intention to language
10. **StateUpdater** - Updates state after interaction
11. **CharacterState** - Data class for all state
12. **CharacterAgent** - Main orchestrator

### **Key Design Features**

✅ **No Prompt Engineering**: Character behavior emerges from quantified state
✅ **Maximum Reuse**: Base classes define contracts, concrete implementations add specifics
✅ **Observable**: Every decision can be traced through agent outputs
✅ **State-Driven**: Hormones, mood, goals drive behavior (not roleplay prompts)
✅ **Modular**: Easy to add new agents or modify existing ones
✅ **Testable**: Can unit test each agent independently

---

## Next Steps

1. **Implement LLM Integration Layer** (LangChain + Ollama)
2. **Build Simple CLI Chat Interface**
3. **Test with one character** (e.g., Marcus - Playful Male)
4. **Validate personality consistency** across conversations
5. **Refine agent prompts** based on observations
6. **Add remaining agents** (SpecialtyAgent, SkillsAgent, DevelopmentAgent)

