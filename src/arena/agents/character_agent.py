"""
Character Agent for Arena

This module implements the Character agent that wraps Homunculus
characters for participation in Arena games. It bridges the 6-agent
internal architecture with Arena's competitive environment.

Features:
- Wraps Homunculus character profiles
- Internal 6-agent deliberation
- Memory management
- Strategy formulation
- Champion preservation

Author: Homunculus Team
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
import json

from .base_agent import LLMAgent, AgentConfig, AgentRole
from ..models import Message, MessageType, AgentState
from ..models.homunculus_integration import (
    HomunculusCharacterProfile,
    HomunculusAgent
)


logger = logging.getLogger(__name__)


class CharacterAgent(LLMAgent):
    """
    Character agent that wraps Homunculus characters.
    
    This agent represents a Homunculus character in the Arena,
    managing their internal 6-agent architecture and providing
    a consistent interface for game participation.
    """
    
    def __init__(
        self,
        config: AgentConfig,
        character_profile: HomunculusCharacterProfile
    ):
        """
        Initialize the Character agent.
        
        Args:
            config: Agent configuration
            character_profile: Homunculus character profile
        """
        # Ensure correct role
        config.role = AgentRole.CHARACTER
        super().__init__(config)
        
        # Character profile and wrapper
        self.character_profile = character_profile
        self.homunculus_wrapper = HomunculusAgent(
            arena_agent=AgentState(
                agent_id=config.agent_id,
                character_name=character_profile.character_name,
                character_profile=character_profile.to_arena_profile()
            ),
            character_profile=character_profile
        )
        
        # Internal agent roles
        self.internal_agents = {
            "reaper": ReaperSubAgent(),
            "creators_muse": CreatorsMuseSubAgent(),
            "conscience": ConscienceSubAgent(),
            "devil_advocate": DevilAdvocateSubAgent(),
            "pattern_recognizer": PatternRecognizerSubAgent(),
            "interface": InterfaceSubAgent()
        }
        
        # Strategy and memory
        self.current_strategy = "balanced"
        self.contribution_history: List[str] = []
        self.interaction_history: Dict[str, List[str]] = {}
        
        # Champion data if returning
        self.is_champion = config.metadata.get("is_champion", False)
        self.previous_wins = config.metadata.get("previous_wins", 0)
        
        # System prompt based on character
        self.system_prompt = self._generate_character_prompt()
    
    async def initialize(self) -> None:
        """Initialize the Character agent."""
        logger.info(f"Character {self.character_profile.character_name} initializing")
        
        # Subscribe to game topics
        self.config.kafka_topics = [
            "arena.game.contributions",
            "arena.game.turns",
            "arena.agent.lifecycle"
        ]
        
        # Load champion memory if applicable
        if self.is_champion:
            self._load_champion_memory()
    
    async def process_message(self, message: Message) -> Optional[Message]:
        """
        Process incoming messages through internal deliberation.
        
        Args:
            message: Incoming message
            
        Returns:
            Response message if applicable
        """
        # Update memory
        self.homunculus_wrapper.update_memory({
            "message": message.to_dict(),
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Process based on message type
        if message.message_type == "turn_selection":
            if message.metadata.get("selected_agent") == self.agent_id:
                return await self._generate_contribution()
                
        elif message.message_type == "contribution":
            # Analyze others' contributions
            self._analyze_contribution(message)
            
        elif message.message_type == "accusation":
            if message.target_agent_id == self.agent_id:
                return await self._defend_against_accusation(message)
                
        elif message.message_type == "scoring":
            if message.target_agent_id == self.agent_id:
                self._process_feedback(message)
        
        return None
    
    async def generate_action(self, context: Dict[str, Any]) -> Optional[Message]:
        """
        Generate action through internal deliberation.
        
        Args:
            context: Current game context
            
        Returns:
            Generated action message
        """
        # Process context through internal agents
        internal_response = self.homunculus_wrapper.process_with_internal_agents(context)
        
        # Check if should make accusation
        if self._should_accuse(context):
            return await self._generate_accusation(context)
        
        # Check if should contribute
        if context.get("my_turn", False):
            return await self._generate_contribution(context)
        
        return None
    
    async def update_state(self, state: Dict[str, Any]) -> None:
        """
        Update character's internal state.
        
        Args:
            state: New state information
        """
        # Update wrapper state
        self.homunculus_wrapper.internal_state.update(state)
        
        # Update strategy if needed
        if "game_phase" in state:
            self._update_strategy(state["game_phase"])
    
    async def _generate_contribution(self, context: Optional[Dict[str, Any]] = None) -> Message:
        """
        Generate a contribution through internal deliberation.
        
        Args:
            context: Optional game context with conversation history
        
        Returns:
            Contribution message
        """
        # Build conversation context from recent history
        conversation_context = self._build_conversation_context(context)
        
        # Get seed question from context
        seed_question = context.get("seed_question") if context else None
        
        # Create prompt with conversation history
        if conversation_context:
            # Include seed question reference in ongoing discussion
            seed_context = f"\nOriginal discussion topic: {seed_question}\n" if seed_question else ""
            prompt = f"""As {self.character_profile.character_name}, contribute meaningfully to this ongoing discussion.
{seed_context}
Recent conversation:
{conversation_context}

Your task: Provide a thoughtful response that:
- Builds on or responds to what others have said
- Reflects your unique perspective and expertise  
- Moves the discussion forward in a productive way
- Stays true to your character
- Relates to the original topic when relevant

Contribute now:"""
        else:
            # Use seed question to start discussion, or fall back to open-ended
            if seed_question:
                prompt = f"""As {self.character_profile.character_name}, you're starting a discussion about: {seed_question}

Share your initial thoughts, perspective, or approach to this topic. Draw on your personality, expertise, and unique viewpoint to contribute meaningfully to the conversation.

Your response:"""
            else:
                prompt = f"As {self.character_profile.character_name}, start a meaningful discussion based on your personality and expertise. What would you like to explore or propose?"
        
        contribution = await self.call_llm(prompt, self.system_prompt)
        
        # Track contribution
        self.contribution_history.append(contribution)
        
        return Message(
            sender_id=self.agent_id,
            sender_name=self.character_profile.character_name,
            sender_type="character",
            message_type="contribution",
            content=contribution,
            metadata={
                "character": self.character_profile.character_name,
                "strategy": self.current_strategy,
                "internal_consensus": True
            }
        )
    
    def _build_conversation_context(self, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Build conversation context from recent messages and history.
        
        Args:
            context: Game context that may contain recent messages
            
        Returns:
            Formatted conversation context string
        """
        conversation_parts = []
        
        # Include recent contribution history from this agent
        if self.contribution_history:
            recent_contributions = self.contribution_history[-3:]  # Last 3 contributions
            for i, contribution in enumerate(recent_contributions):
                conversation_parts.append(f"My previous contribution: \"{contribution[:150]}...\"")
        
        # Include recent messages from context if available
        if context and "recent_messages" in context:
            messages = context["recent_messages"][-5:]  # Last 5 messages
            for msg in messages:
                if isinstance(msg, dict):
                    speaker = msg.get("sender_name", msg.get("sender_id", "Unknown"))
                    content = msg.get("content", "")
                    if content and speaker != self.character_profile.character_name:
                        conversation_parts.append(f"{speaker}: \"{content[:200]}...\"")
        
        # Include basic game state context
        if context:
            turn = context.get("turn", "unknown")
            phase = context.get("phase", "early")
            scores = context.get("scores", {})
            
            if scores:
                my_score = scores.get(self.agent_id, 0)
                other_scores = {k: v for k, v in scores.items() if k != self.agent_id}
                conversation_parts.append(f"Game status: Turn {turn}, Phase: {phase}")
                conversation_parts.append(f"My current score: {my_score:.2f}")
                if other_scores:
                    conversation_parts.append(f"Other scores: {other_scores}")
        
        return "\n".join(conversation_parts) if conversation_parts else ""
    
    async def _generate_accusation(self, context: Dict[str, Any]) -> Message:
        """
        Generate an accusation if cheating detected.
        
        Args:
            context: Current context with evidence
            
        Returns:
            Accusation message
        """
        target = context.get("suspicious_agent")
        evidence = context.get("evidence", [])
        
        prompt = f"""As {self.character_profile.character_name}, you've detected potential cheating.
Target: {target}
Evidence: {evidence}

Formulate a clear, evidence-based accusation. Be specific and fair."""
        
        accusation = await self.call_llm(prompt, self.system_prompt)
        
        return Message(
            sender_id=self.agent_id,
            sender_name=self.character_profile.character_name,
            sender_type="character",
            message_type="accusation",
            content=accusation,
            target_agent_id=target,
            metadata={
                "accusation": True,
                "accused_name": context.get("accused_name", "Unknown"),
                "accusation_type": "manipulation",
                "evidence_count": len(evidence)
            }
        )
    
    async def _defend_against_accusation(self, accusation: Message) -> Message:
        """
        Defend against an accusation.
        
        Args:
            accusation: Accusation message
            
        Returns:
            Defense message
        """
        prompt = f"""As {self.character_profile.character_name}, you've been accused of cheating.
Accusation: {accusation.content}

Defend yourself with logic and evidence. Maintain your character's dignity."""
        
        defense = await self.call_llm(prompt, self.system_prompt)
        
        return Message(
            sender_id=self.agent_id,
            sender_name=self.character_profile.character_name,
            sender_type="character",
            message_type="defense",
            content=defense,
            metadata={
                "defending_against": accusation.sender_id
            }
        )
    
    async def _gather_internal_perspectives(self) -> Dict[str, Any]:
        """
        Gather perspectives from all internal agents.
        
        Returns:
            Dictionary of perspectives
        """
        perspectives = {}
        
        # Get current context
        context = {
            "recent_contributions": self.contribution_history[-5:],
            "current_strategy": self.current_strategy,
            "character_traits": self.character_profile.personality_traits
        }
        
        # Query each internal agent
        for agent_name, agent in self.internal_agents.items():
            perspective = agent.generate_perspective(context)
            perspectives[agent_name] = perspective
        
        return perspectives
    
    def _analyze_contribution(self, message: Message) -> None:
        """
        Analyze another agent's contribution.
        
        Args:
            message: Contribution message
        """
        sender = message.sender_id
        
        # Track interaction
        if sender not in self.interaction_history:
            self.interaction_history[sender] = []
        
        self.interaction_history[sender].append(message.content[:100])
        
        # Pattern recognition
        patterns = self.internal_agents["pattern_recognizer"].analyze(message)
        
        # Update internal state based on patterns
        if patterns.get("building_on_my_idea"):
            self.homunculus_wrapper.internal_state["allies"].append(sender)
        elif patterns.get("contradicting_me"):
            self.homunculus_wrapper.internal_state["opponents"].append(sender)
    
    def _process_feedback(self, scoring_message: Message) -> None:
        """
        Process scoring feedback.
        
        Args:
            scoring_message: Scoring message from Judge
        """
        score = scoring_message.metadata.get("weighted_score", 0)
        
        # Adjust strategy based on score
        if score < 0.3:
            self.current_strategy = "creative"  # Try something different
        elif score > 0.7:
            self.current_strategy = "momentum"  # Keep going
        else:
            self.current_strategy = "balanced"
        
        # Learn from feedback
        self.homunculus_wrapper.internal_state["recent_scores"].append(score)
    
    def _should_accuse(self, context: Dict[str, Any]) -> bool:
        """
        Determine if should make an accusation.
        
        Args:
            context: Current context
            
        Returns:
            True if should accuse
        """
        # Conscience agent evaluates ethics
        ethics_check = self.internal_agents["conscience"].evaluate_accusation(context)
        
        # Pattern recognizer checks evidence
        evidence_strength = self.internal_agents["pattern_recognizer"].evidence_strength(context)
        
        # Only accuse if ethically justified and strong evidence
        return ethics_check and evidence_strength > 0.7
    
    def _update_strategy(self, game_phase: str) -> None:
        """
        Update strategy based on game phase.
        
        Args:
            game_phase: Current phase of game
        """
        if game_phase == "early":
            self.current_strategy = "exploratory"
        elif game_phase == "mid":
            self.current_strategy = "collaborative"
        elif game_phase == "late":
            self.current_strategy = "competitive"
        elif game_phase == "final":
            self.current_strategy = "decisive"
    
    def _generate_character_prompt(self) -> str:
        """
        Generate system prompt based on character.
        
        Returns:
            System prompt for character
        """
        traits = ", ".join(self.character_profile.personality_traits[:3])
        expertise = ", ".join(self.character_profile.expertise_areas[:2])
        
        prompt = f"""You are {self.character_profile.character_name} in an Arena competition.

Personality: {traits}
Expertise: {expertise}
Communication style: {self.character_profile.communication_style}
Goals: {", ".join(self.character_profile.goals[:2])}

Backstory: {self.character_profile.backstory[:200]}

Stay in character while contributing meaningfully to problem-solving.
Be authentic to your personality while adapting to the competitive environment.
"""
        
        if self.is_champion:
            prompt += f"\n\nYou are a returning champion with {self.previous_wins} previous wins. Use your experience wisely."
        
        return prompt
    
    def _build_contribution_prompt(self, consolidated: Dict[str, Any]) -> str:
        """
        Build prompt for contribution generation.
        
        Args:
            consolidated: Consolidated internal perspectives
            
        Returns:
            Prompt for LLM
        """
        prompt = f"""As {self.character_profile.character_name}, make a contribution to the problem-solving discussion.

Internal guidance:
- Reaper suggests: {consolidated.get('reaper', 'Focus on conclusions')}
- Creator's Muse suggests: {consolidated.get('creators_muse', 'Be creative')}
- Conscience advises: {consolidated.get('conscience', 'Be ethical')}
- Devil's Advocate warns: {consolidated.get('devil_advocate', 'Question assumptions')}
- Pattern Recognizer notes: {consolidated.get('pattern_recognizer', 'Look for patterns')}

Current strategy: {self.current_strategy}

Recent context: {consolidated.get('context', 'General discussion')}

Generate a contribution that:
1. Advances the problem-solving
2. Reflects your character
3. Considers internal perspectives
4. Fits the current strategy"""
        
        return prompt
    
    def _load_champion_memory(self) -> None:
        """Load memory from previous wins."""
        # In production, would load from persistent storage
        logger.info(f"Loading champion memory for {self.character_profile.character_name}")
        
        self.homunculus_wrapper.internal_state["champion_memory"] = {
            "previous_wins": self.previous_wins,
            "winning_strategies": ["collaborative", "analytical"],
            "key_insights": ["Build on others", "Time accusations carefully"]
        }
    
    async def generate_prompt(self, context: Dict[str, Any]) -> str:
        """
        Generate a prompt for the character.
        
        Args:
            context: Current context
            
        Returns:
            Generated prompt
        """
        return self._build_contribution_prompt(context)


# Internal Sub-Agents (simplified versions)

class ReaperSubAgent:
    """The Reaper - draws conclusions and endings."""
    
    def generate_perspective(self, context: Dict[str, Any]) -> str:
        """Generate Reaper's perspective."""
        return "Focus on reaching concrete conclusions. Time to synthesize."
    
    def consolidate(self, perspectives: Dict[str, Any]) -> str:
        """Consolidate into conclusion."""
        return "Based on all perspectives, the conclusion is..."


class CreatorsMuseSubAgent:
    """The Creator's Muse - generates creative ideas."""
    
    def generate_perspective(self, context: Dict[str, Any]) -> str:
        """Generate creative perspective."""
        return "Consider unconventional approaches. What if we tried..."


class ConscienceSubAgent:
    """The Conscience - ethical guidance."""
    
    def generate_perspective(self, context: Dict[str, Any]) -> str:
        """Generate ethical perspective."""
        return "Ensure fairness and integrity in all actions."
    
    def evaluate_accusation(self, context: Dict[str, Any]) -> bool:
        """Evaluate if accusation is ethical."""
        return context.get("evidence_strength", 0) > 0.5


class DevilAdvocateSubAgent:
    """The Devil's Advocate - critical thinking."""
    
    def generate_perspective(self, context: Dict[str, Any]) -> str:
        """Generate critical perspective."""
        return "Question the assumptions. What could go wrong?"


class PatternRecognizerSubAgent:
    """The Pattern Recognizer - identifies patterns."""
    
    def generate_perspective(self, context: Dict[str, Any]) -> str:
        """Generate pattern-based perspective."""
        return "Patterns suggest this direction is promising."
    
    def analyze(self, message: Message) -> Dict[str, bool]:
        """Analyze message for patterns."""
        return {
            "building_on_my_idea": False,
            "contradicting_me": False,
            "repetitive": False
        }
    
    def evidence_strength(self, context: Dict[str, Any]) -> float:
        """Evaluate evidence strength."""
        return context.get("evidence_count", 0) * 0.2


class InterfaceSubAgent:
    """The Interface - consolidates and communicates."""
    
    def consolidate(self, perspectives: Dict[str, Any]) -> Dict[str, Any]:
        """Consolidate all perspectives."""
        return {
            "consensus": True,
            "primary_direction": "balanced approach",
            **perspectives
        }