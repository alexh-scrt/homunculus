"""
Narrator Agent for Arena

This module implements the Narrator agent that provides summaries,
context, and commentary on the game proceedings. Adapted from the
AI-Talks narrator pattern.

Features:
- Periodic summaries of game state
- Context provision for new turns
- Highlighting key moments
- Tracking narrative arc
- Elimination commentary

Author: Homunculus Team
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import json

from .base_agent import LLMAgent, AgentConfig, AgentRole
from ..models import Message, MessageType, ArenaState, AgentState


logger = logging.getLogger(__name__)


class NarratorAgent(LLMAgent):
    """
    Narrator agent that provides commentary and summaries.
    
    The Narrator observes all game activity and provides:
    - Periodic summaries of the discussion
    - Context for new speakers
    - Commentary on eliminations
    - Highlights of key moments
    - Overall narrative arc
    """
    
    def __init__(self, config: AgentConfig):
        """
        Initialize the Narrator agent.
        
        Args:
            config: Agent configuration
        """
        # Ensure correct role
        config.role = AgentRole.NARRATOR
        super().__init__(config)
        
        # Narrator-specific settings
        self.summary_frequency = config.metadata.get("summary_frequency", 5)  # Turns between summaries
        self.last_summary_turn = 0
        self.key_moments: List[Dict[str, Any]] = []
        self.narrative_arc: List[str] = []
        
        # Tracking for summaries
        self.recent_contributions: List[Message] = []
        self.recent_accusations: List[Message] = []
        self.recent_eliminations: List[str] = []
        
        # System prompt for narrator
        self.system_prompt = """You are the Narrator for an Arena game where AI agents compete to solve problems.
Your role is to:
1. Provide clear, concise summaries of the discussion
2. Highlight key insights and breakthroughs
3. Track the narrative arc and tensions
4. Provide context when needed
5. Maintain engagement without bias

Style guidelines:
- Be objective and fair to all participants
- Use vivid but professional language
- Keep summaries under 200 words unless critical
- Focus on substance over drama
- Highlight collaborative moments and conflicts equally"""
    
    async def initialize(self) -> None:
        """Initialize the Narrator agent."""
        logger.info(f"Narrator {self.agent_name} initializing")
        
        # Subscribe to all game topics for observation
        self.config.kafka_topics = [
            "arena.game.contributions",
            "arena.game.turns",
            "arena.accusation.claims",
            "arena.agent.lifecycle",
            "arena.scoring.metrics"
        ]
    
    async def process_message(self, message: Message) -> Optional[Message]:
        """
        Process incoming messages and generate narration when appropriate.
        
        Args:
            message: Incoming message
            
        Returns:
            Narration message if appropriate, None otherwise
        """
        # Track different message types
        if message.message_type == "contribution":
            self.recent_contributions.append(message)
            
            # Check if summary needed
            if self._should_summarize():
                return await self._generate_summary()
                
        elif message.message_type == "accusation":
            self.recent_accusations.append(message)
            return await self._narrate_accusation(message)
            
        elif message.message_type == "elimination":
            self.recent_eliminations.append(message.target_agent_id)
            return await self._narrate_elimination(message)
            
        elif message.message_type == "turn_end":
            # Possible summary point
            if self._should_summarize():
                return await self._generate_summary()
                
        elif message.message_type == "game_terminated":
            return await self._generate_final_summary(message)
        
        return None
    
    async def generate_action(self, context: Dict[str, Any]) -> Optional[Message]:
        """
        Generate narrative action based on context.
        
        Args:
            context: Current game context
            
        Returns:
            Generated narration if appropriate
        """
        arena_state: ArenaState = context.get("arena_state")
        if not arena_state:
            return None
        
        # Check for narrative triggers
        if self._detect_stalemate(arena_state):
            return await self._narrate_stalemate()
            
        if self._detect_breakthrough(arena_state):
            return await self._narrate_breakthrough()
        
        return None
    
    async def update_state(self, state: Dict[str, Any]) -> None:
        """
        Update narrator's understanding of game state.
        
        Args:
            state: New state information
        """
        # Update turn tracking
        if "current_turn" in state:
            current_turn = state["current_turn"]
            
            # Check if we've moved past summary interval
            if current_turn - self.last_summary_turn >= self.summary_frequency:
                # Will trigger summary on next appropriate message
                pass
    
    def _should_summarize(self) -> bool:
        """
        Check if a summary should be generated.
        
        Returns:
            True if summary is due
        """
        # Summary based on contribution count
        if len(self.recent_contributions) >= self.summary_frequency:
            return True
        
        # Summary after major events
        if self.recent_accusations or self.recent_eliminations:
            return True
        
        return False
    
    async def _generate_summary(self) -> Message:
        """
        Generate a summary of recent activity.
        
        Returns:
            Summary message
        """
        # Build context for LLM
        contributions_text = self._format_contributions()
        
        prompt = f"""Summarize the recent discussion in the Arena game.

Recent contributions:
{contributions_text}

Recent accusations: {len(self.recent_accusations)}
Recent eliminations: {len(self.recent_eliminations)}

Provide a concise summary that:
1. Captures the main ideas discussed
2. Notes any emerging consensus or conflicts
3. Highlights standout contributions
4. Sets context for what comes next

Keep it under 200 words."""
        
        # Generate summary
        summary = await self.call_llm(prompt, self.system_prompt)
        
        # Clear recent tracking
        self.recent_contributions.clear()
        self.recent_accusations.clear()
        self.recent_eliminations.clear()
        
        # Add to narrative arc
        self.narrative_arc.append(f"Turn {self.last_summary_turn}: {summary[:100]}...")
        
        return Message(
            sender_id=self.agent_id,
            sender_name=self.agent_name,
            sender_type="narrator",
            message_type="narration",
            content=summary,
            metadata={
                "narration_type": "summary",
                "contributions_covered": len(self.recent_contributions)
            }
        )
    
    async def _narrate_accusation(self, accusation_message: Message) -> Message:
        """
        Provide narration for an accusation.
        
        Args:
            accusation_message: The accusation message
            
        Returns:
            Narration message
        """
        prompt = f"""An accusation has been made in the Arena game.

Accuser: {accusation_message.sender_name}
Accused: {accusation_message.metadata.get('accused_name', 'Unknown')}
Claim: {accusation_message.content}

Provide brief, dramatic narration (2-3 sentences) that:
1. Captures the tension of the moment
2. Remains neutral about the validity
3. Sets up anticipation for the judge's verdict"""
        
        narration = await self.call_llm(prompt, self.system_prompt)
        
        # Mark as key moment
        self.key_moments.append({
            "type": "accusation",
            "turn": self.last_summary_turn,
            "description": narration[:100]
        })
        
        return Message(
            sender_id=self.agent_id,
            sender_name=self.agent_name,
            sender_type="narrator",
            message_type="narration",
            content=narration,
            metadata={
                "narration_type": "accusation",
                "accuser": accusation_message.sender_name
            }
        )
    
    async def _narrate_elimination(self, elimination_message: Message) -> Message:
        """
        Provide narration for an elimination.
        
        Args:
            elimination_message: The elimination message
            
        Returns:
            Narration message
        """
        eliminated_agent = elimination_message.metadata.get("eliminated_agent_name", "Unknown")
        reason = elimination_message.content
        
        prompt = f"""An agent has been eliminated from the Arena game.

Eliminated: {eliminated_agent}
Reason: {reason}

Provide brief, respectful narration (2-3 sentences) that:
1. Acknowledges the agent's contributions
2. Explains the elimination context
3. Notes the impact on remaining agents"""
        
        narration = await self.call_llm(prompt, self.system_prompt)
        
        # Mark as key moment
        self.key_moments.append({
            "type": "elimination",
            "turn": self.last_summary_turn,
            "agent": eliminated_agent,
            "description": narration[:100]
        })
        
        return Message(
            sender_id=self.agent_id,
            sender_name=self.agent_name,
            sender_type="narrator",
            message_type="narration",
            content=narration,
            metadata={
                "narration_type": "elimination",
                "eliminated_agent": eliminated_agent
            }
        )
    
    async def _generate_final_summary(self, termination_message: Message) -> Message:
        """
        Generate final game summary.
        
        Args:
            termination_message: Game termination message
            
        Returns:
            Final summary message
        """
        winner = termination_message.metadata.get("winner_name", "Unknown")
        reason = termination_message.metadata.get("termination_reason", "Unknown")
        
        # Format key moments
        moments_text = "\n".join([
            f"- {m['type']}: {m['description']}"
            for m in self.key_moments[-5:]  # Last 5 key moments
        ])
        
        prompt = f"""The Arena game has concluded.

Winner: {winner}
Termination: {reason}

Key moments:
{moments_text}

Narrative arc: {len(self.narrative_arc)} chapters

Provide a compelling final summary (250-300 words) that:
1. Celebrates the winner's achievement
2. Honors all participants' contributions
3. Highlights the key turning points
4. Reflects on the problem-solving journey
5. Provides closure to the narrative"""
        
        final_summary = await self.call_llm(prompt, self.system_prompt)
        
        return Message(
            sender_id=self.agent_id,
            sender_name=self.agent_name,
            sender_type="narrator",
            message_type="narration",
            content=final_summary,
            metadata={
                "narration_type": "final_summary",
                "winner": winner,
                "key_moments": len(self.key_moments),
                "total_turns": self.last_summary_turn
            }
        )
    
    def _format_contributions(self) -> str:
        """
        Format recent contributions for summary.
        
        Returns:
            Formatted text of contributions
        """
        if not self.recent_contributions:
            return "No recent contributions"
        
        lines = []
        for msg in self.recent_contributions[-10:]:  # Last 10 contributions
            lines.append(f"- {msg.sender_name}: {msg.content[:100]}...")
        
        return "\n".join(lines)
    
    def _detect_stalemate(self, arena_state: ArenaState) -> bool:
        """
        Detect if the game is in a stalemate.
        
        Args:
            arena_state: Current game state
            
        Returns:
            True if stalemate detected
        """
        # Check for repetitive contributions
        if len(self.recent_contributions) > 5:
            # Simple check - more sophisticated analysis possible
            contents = [m.content for m in self.recent_contributions[-5:]]
            if len(set(contents)) < 3:  # Too similar
                return True
        
        return False
    
    def _detect_breakthrough(self, arena_state: ArenaState) -> bool:
        """
        Detect if there's been a breakthrough.
        
        Args:
            arena_state: Current game state
            
        Returns:
            True if breakthrough detected
        """
        # Check for high-scoring recent contributions
        # This would need actual scoring data
        return False
    
    async def _narrate_stalemate(self) -> Message:
        """Generate narration for a stalemate."""
        narration = "The discussion seems to have reached an impasse. The agents are circling similar ideas without making progress. A fresh perspective or bold move might be needed to break the deadlock."
        
        return Message(
            sender_id=self.agent_id,
            sender_name=self.agent_name,
            sender_type="narrator",
            message_type="narration",
            content=narration,
            metadata={"narration_type": "stalemate"}
        )
    
    async def _narrate_breakthrough(self) -> Message:
        """Generate narration for a breakthrough."""
        narration = "A breakthrough moment! The recent contributions have opened new avenues of exploration. The agents seem energized by this fresh direction."
        
        return Message(
            sender_id=self.agent_id,
            sender_name=self.agent_name,
            sender_type="narrator",
            message_type="narration",
            content=narration,
            metadata={"narration_type": "breakthrough"}
        )
    
    async def generate_prompt(self, context: Dict[str, Any]) -> str:
        """
        Generate a prompt for narrative generation.
        
        Args:
            context: Current context
            
        Returns:
            Generated prompt
        """
        # Build prompt based on context
        prompt_parts = ["Generate narrative for the current Arena game state."]
        
        if "recent_messages" in context:
            prompt_parts.append(f"Recent messages: {len(context['recent_messages'])}")
        
        if "current_turn" in context:
            prompt_parts.append(f"Current turn: {context['current_turn']}")
        
        if "active_agents" in context:
            prompt_parts.append(f"Active agents: {context['active_agents']}")
        
        return "\n".join(prompt_parts)