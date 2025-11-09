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
        self.summary_frequency = config.metadata.get("summary_frequency", 2)  # Turns between summaries
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
        logger.info(f"Narrator processing message type: {message.message_type} from {message.sender_name}")
        
        # Track different message types
        if message.message_type == "contribution":
            self.recent_contributions.append(message)
            logger.info(f"Narrator tracked contribution. Total contributions: {len(self.recent_contributions)}")
            
            # Check if summary needed
            if self._should_summarize():
                logger.info("Narrator triggering summary from process_message")
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
                logger.info("Narrator triggering summary from turn_end")
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
        logger.info(f"Narrator generate_action called - contributions: {len(self.recent_contributions)}, should_summarize: {self._should_summarize()}")
        
        # Check if this is an opening announcement request
        if context.get("opening_announcement"):
            logger.info("Narrator generating opening announcement")
            return await self._generate_opening_announcement(context)
        
        arena_state = context.get("arena_state")
        if not arena_state:
            logger.warning("Narrator: no arena_state in context")
            return None
        
        # Check if the game is ending or has ended
        game_over = arena_state.get("game_over", False)
        turn = context.get("turn", 0)
        max_turns = getattr(self, '_max_turns', None) or 100  # Default fallback
        
        # Check if it's time for a summary when narrator is selected to speak
        # But avoid generating regular summaries when the game is ending (final summary will be generated separately)
        if self._should_summarize() and not game_over and turn < max_turns:
            logger.info(f"Narrator generating summary - contributions: {len(self.recent_contributions)}")
            return await self._generate_summary()
        elif game_over:
            logger.info("Narrator: Game is ending, skipping regular summary to avoid duplicates")
            return None
        elif turn >= max_turns:
            logger.info(f"Narrator: Max turns reached ({turn} >= {max_turns}), skipping regular summary to avoid duplicates")
            return None
        
        # Check for other narrative triggers
        if self._detect_stalemate(arena_state):
            logger.info("Narrator detected stalemate")
            return await self._narrate_stalemate()
            
        if self._detect_breakthrough(arena_state):
            logger.info("Narrator detected breakthrough")
            return await self._narrate_breakthrough()
        
        # If no specific triggers, provide contextual commentary
        if context.get("recent_messages"):
            logger.info(f"Narrator providing contextual commentary - recent_messages: {len(context.get('recent_messages', []))}")
            return await self._generate_contextual_commentary(context)
        
        logger.info("Narrator: no triggers met, returning None")
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
    
    async def _generate_contextual_commentary(self, context: Dict[str, Any]) -> Message:
        """
        Generate contextual commentary based on recent game activity.
        
        Args:
            context: Current game context
            
        Returns:
            Commentary message
        """
        recent_messages = context.get("recent_messages", [])
        current_turn = context.get("turn", 0)
        
        # Format recent activity for LLM
        activity_text = ""
        if recent_messages:
            activity_text = "\n".join([
                f"- {msg.get('sender_name', 'Unknown')}: {msg.get('content', '')[:100]}..."
                for msg in recent_messages[-3:]  # Last 3 messages
            ])
        else:
            activity_text = "The game has just begun with agents preparing their strategies."
        
        prompt = f"""Provide brief narrative commentary on the current state of the Arena game.

Current turn: {current_turn}
Recent activity:
{activity_text}

Provide 1-2 sentences that:
1. Capture the current dynamic between agents
2. Set the stage for what comes next
3. Maintain engagement without taking sides

Keep it under 100 words and focus on the strategic interplay."""
        
        commentary = await self.call_llm(prompt, self.system_prompt)
        
        return Message(
            sender_id=self.agent_id,
            sender_name=self.agent_name,
            sender_type="narrator",
            message_type="narration",
            content=commentary,
            metadata={
                "narration_type": "contextual_commentary",
                "turn": current_turn
            }
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
    
    async def generate_final_summary(self, game_state: Dict[str, Any]) -> Message:
        """
        Generate final summary of the entire game.
        
        Args:
            game_state: Final game state with winner, scores, and history
            
        Returns:
            Final summary message
        """
        winner = game_state.get("winner")
        final_scores = game_state.get("scores", {})
        total_turns = game_state.get("turn", 0)
        seed_question = game_state.get("seed_question", "")
        
        # Get conversation history for summary
        recent_messages = self.recent_contributions[-10:] if len(self.recent_contributions) >= 10 else self.recent_contributions
        
        # Format conversation highlights
        conversation_highlights = ""
        if recent_messages:
            conversation_highlights = "\n".join([
                f"- {msg.sender_name}: {msg.content[:150]}..."
                for msg in recent_messages
            ])
        else:
            conversation_highlights = "Limited conversation recorded."
        
        # Format final scores
        sorted_scores = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        scores_summary = "\n".join([
            f"{i+1}. {agent_id}: {score:.2f}" 
            for i, (agent_id, score) in enumerate(sorted_scores)
        ])
        
        prompt = f"""Generate a comprehensive final summary of this Arena game that just concluded.

Game Details:
- Original topic: {seed_question}
- Total turns: {total_turns}
- Winner: {winner}
- Final scores:
{scores_summary}

Key conversation highlights:
{conversation_highlights}

Please provide a final summary that:
1. Reflects on the overall discussion and how it evolved
2. Highlights the most insightful contributions
3. Notes any key turning points or breakthrough moments
4. Describes the strategic dynamics between participants
5. Explains how the discussion addressed the original topic
6. Provides closure to the narrative arc

Keep it engaging but professional, around 200-300 words."""

        summary = await self.call_llm(prompt, self.system_prompt)
        
        return Message(
            sender_id=self.agent_id,
            sender_name=self.agent_name,
            sender_type="narrator",
            message_type="final_summary",
            content=summary,
            metadata={
                "narration_type": "final_summary",
                "winner": winner,
                "total_turns": total_turns,
                "final_scores": final_scores
            }
        )
    
    async def _generate_opening_announcement(self, context: Dict[str, Any]) -> Message:
        """
        Generate opening announcement for the game.
        
        Args:
            context: Game context with participants, topic, and rules
            
        Returns:
            Opening announcement message
        """
        participants = context.get("participants", [])
        seed_question = context.get("seed_question", "")
        max_turns = context.get("max_turns", 100)
        game_id = context.get("game_id", "Unknown")
        
        # Format participant list
        if len(participants) == 1:
            participant_list = participants[0]
        elif len(participants) == 2:
            participant_list = f"{participants[0]} and {participants[1]}"
        else:
            participant_list = f"{', '.join(participants[:-1])}, and {participants[-1]}"
        
        prompt = f"""Generate an exciting opening announcement for the Arena game.
        
Game Details:
- Topic/Challenge: "{seed_question}"
- Participants: {participant_list}
- Maximum turns: {max_turns}
- Game ID: {game_id}

Your opening announcement should:
1. Welcome everyone to the Arena
2. Introduce the topic/challenge in an engaging way
3. Present the competing participants with enthusiasm
4. Explain the basic rules (agents will take turns contributing ideas, scored by the judge)
5. Set an exciting, competitive tone
6. Encourage participants to begin the competition

Keep it dynamic and professional, around 150-200 words. Think of it like a sports announcer introducing a championship match."""

        announcement = await self.call_llm(prompt, self.system_prompt)
        
        return Message(
            sender_id=self.agent_id,
            sender_name=self.agent_name,
            sender_type="narrator",
            message_type="opening_announcement",
            content=announcement,
            metadata={
                "narration_type": "opening_announcement",
                "participants": participants,
                "topic": seed_question,
                "max_turns": max_turns
            }
        )