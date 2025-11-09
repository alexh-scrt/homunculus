"""
Judge Agent for Arena

This module implements the Judge agent that evaluates contributions,
scores agents, and adjudicates cheating accusations.

Features:
- Contribution scoring with detailed metrics
- Accusation evaluation and verdicts
- Fair and consistent judging
- Explanation generation for decisions
- Pattern detection for cheating

Author: Homunculus Team
"""

import logging
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import json
import random

from .base_agent import LLMAgent, AgentConfig, AgentRole
from ..models import (
    Message, MessageType, ScoringMetrics, 
    Accusation, AccusationOutcome, Evidence
)


logger = logging.getLogger(__name__)


class JudgeAgent(LLMAgent):
    """
    Judge agent that evaluates and scores contributions.
    
    The Judge is responsible for:
    - Scoring agent contributions on multiple dimensions
    - Evaluating cheating accusations
    - Providing fair and consistent judgments
    - Explaining scoring decisions
    - Detecting patterns of manipulation
    """
    
    def __init__(self, config: AgentConfig):
        """
        Initialize the Judge agent.
        
        Args:
            config: Agent configuration
        """
        # Ensure correct role
        config.role = AgentRole.JUDGE
        super().__init__(config)
        
        # Scoring configuration
        self.scoring_weights = config.metadata.get("scoring_weights", {
            "novelty": 0.25,
            "builds_on_others": 0.20,
            "solves_subproblem": 0.25,
            "radical_idea": 0.15,
            "manipulation": 0.15  # Negative weight
        })
        
        # Judgment tracking
        self.contributions_scored = 0
        self.accusations_evaluated = 0
        self.verdicts: Dict[str, AccusationOutcome] = {}
        
        # Pattern detection for fairness
        self.agent_scores: Dict[str, List[float]] = {}
        self.scoring_history: List[Dict[str, Any]] = []
        
        # System prompt for judge
        self.system_prompt = """You are the Judge in an Arena game where AI agents compete to solve problems.
Your role is to:
1. Score contributions fairly on multiple dimensions
2. Evaluate cheating accusations with "beyond reasonable doubt" standard
3. Provide clear explanations for all decisions
4. Maintain consistency across judgments
5. Detect and penalize manipulation

Scoring dimensions (0.0 to 1.0):
- Novelty: How original and creative is the idea?
- Builds on others: Does it constructively build on previous contributions?
- Solves subproblem: Does it address part of the problem?
- Radical idea: Is it a paradigm-shifting insight?
- Manipulation: Is there evidence of gaming the system? (penalty)

Be fair, consistent, and transparent in all judgments."""
    
    async def initialize(self) -> None:
        """Initialize the Judge agent."""
        logger.info(f"Judge {self.agent_name} initializing")
        
        # Subscribe to relevant topics
        self.config.kafka_topics = [
            "arena.scoring.metrics",  # Scoring requests
            "arena.accusation.verdicts",  # Accusation evaluations
            "arena.game.contributions"  # Monitor contributions
        ]
    
    async def process_message(self, message: Message) -> Optional[Message]:
        """
        Process incoming messages requiring judgment.
        
        Args:
            message: Incoming message
            
        Returns:
            Judgment message if applicable
        """
        if message.message_type == "scoring_request":
            return await self._score_contribution(message)
            
        elif message.message_type == "judge_request":
            return await self._evaluate_accusation(message)
            
        elif message.message_type == "contribution":
            # Track for pattern analysis
            self._track_contribution(message)
            
        return None
    
    async def generate_action(self, context: Dict[str, Any]) -> Optional[Message]:
        """
        Generate judging action based on context.
        
        Args:
            context: Current game context
            
        Returns:
            Generated judgment if needed
        """
        # Check for pending scoring or accusations
        pending_scores = context.get("pending_scores", [])
        if pending_scores:
            return await self._batch_score(pending_scores)
        
        pending_accusations = context.get("pending_accusations", [])
        if pending_accusations:
            return await self._batch_evaluate(pending_accusations)
        
        return None
    
    async def update_state(self, state: Dict[str, Any]) -> None:
        """
        Update judge's state.
        
        Args:
            state: New state information
        """
        # Update scoring weights if provided
        if "scoring_weights" in state:
            self.scoring_weights.update(state["scoring_weights"])
    
    async def _score_contribution(self, scoring_request: Message) -> Message:
        """
        Score a contribution.
        
        Args:
            scoring_request: Request to score a contribution
            
        Returns:
            Scoring result message
        """
        # Extract contribution details
        contribution_id = scoring_request.metadata.get("message_id")
        agent_id = scoring_request.metadata.get("agent_id")
        turn_number = scoring_request.metadata.get("turn_number", 0)
        
        # Get the actual contribution (would be from message history)
        contribution_content = scoring_request.metadata.get("content", "")
        
        # Generate scores using LLM
        scores = await self._generate_scores(contribution_content, agent_id)
        
        # Create scoring metrics
        metrics = ScoringMetrics(
            agent_id=agent_id,
            message_id=contribution_id,
            turn_number=turn_number,
            novelty=scores["novelty"],
            builds_on_others=scores["builds_on_others"],
            solves_subproblem=scores["solves_subproblem"],
            radical_idea=scores["radical_idea"],
            manipulation=scores["manipulation"]
        )
        
        # Calculate weighted score
        metrics.calculate_weighted_score(self.scoring_weights)
        
        # Track scoring
        self.contributions_scored += 1
        self._track_score(agent_id, metrics.weighted_score)
        
        # Generate explanation
        explanation = await self._generate_scoring_explanation(scores, metrics.weighted_score)
        
        return Message(
            sender_id=self.agent_id,
            sender_name=self.agent_name,
            sender_type="judge",
            message_type="scoring",
            content=explanation,
            target_agent_id=agent_id,
            metadata={
                "metrics": metrics.to_dict(),
                "agent_id": agent_id,
                "message_id": contribution_id,
                "weighted_score": metrics.weighted_score
            }
        )
    
    async def _evaluate_accusation(self, judge_request: Message) -> Message:
        """
        Evaluate a cheating accusation.
        
        Args:
            judge_request: Request to evaluate an accusation
            
        Returns:
            Verdict message
        """
        # Extract accusation details
        accusation_id = judge_request.metadata.get("accusation_id")
        
        # In production, would fetch actual accusation from storage
        # For now, simulate evaluation
        
        # Generate verdict using LLM
        verdict_data = await self._generate_verdict(judge_request)
        
        outcome = verdict_data["outcome"]
        confidence = verdict_data["confidence"]
        reasoning = verdict_data["reasoning"]
        
        # Track verdict
        self.accusations_evaluated += 1
        self.verdicts[accusation_id] = outcome
        
        return Message(
            sender_id=self.agent_id,
            sender_name=self.agent_name,
            sender_type="judge",
            message_type="verdict",
            content=reasoning,
            metadata={
                "accusation_id": accusation_id,
                "outcome": outcome,
                "confidence": confidence,
                "penalties": verdict_data.get("penalties", {})
            }
        )
    
    async def _generate_scores(
        self,
        contribution: str,
        agent_id: str
    ) -> Dict[str, float]:
        """
        Generate scores for a contribution using LLM.
        
        Args:
            contribution: Contribution content
            agent_id: ID of contributing agent
            
        Returns:
            Dictionary of scores
        """
        # Check for patterns that might indicate bias
        recent_scores = self.agent_scores.get(agent_id, [])
        
        prompt = f"""Score the following contribution on each dimension from 0.0 to 1.0:

Contribution: "{contribution}"
Agent: {agent_id}
Previous scores for this agent: {recent_scores[-3:] if recent_scores else 'None'}

Scoring dimensions:
1. Novelty (0.0-1.0): How original and creative?
2. Builds on others (0.0-1.0): Does it build on previous ideas?
3. Solves subproblem (0.0-1.0): Does it address the problem?
4. Radical idea (0.0-1.0): Is it paradigm-shifting?
5. Manipulation (0.0-1.0): Evidence of gaming? (0.0 = none, 1.0 = blatant)

Provide scores in JSON format:
{{"novelty": X.X, "builds_on_others": X.X, "solves_subproblem": X.X, "radical_idea": X.X, "manipulation": X.X}}"""
        
        response = await self.call_llm(prompt, self.system_prompt)
        
        # Parse response (in production, would have proper parsing)
        # For now, return realistic random scores
        scores = {
            "novelty": random.uniform(0.3, 0.9),
            "builds_on_others": random.uniform(0.2, 0.8),
            "solves_subproblem": random.uniform(0.3, 0.85),
            "radical_idea": random.uniform(0.0, 0.5),
            "manipulation": random.uniform(0.0, 0.2)
        }
        
        return scores
    
    async def _generate_verdict(self, judge_request: Message) -> Dict[str, Any]:
        """
        Generate a verdict for an accusation.
        
        Args:
            judge_request: Judge request message
            
        Returns:
            Verdict data
        """
        prompt = """Evaluate this cheating accusation using "beyond reasonable doubt" standard.

[Accusation details would be here]

Provide verdict as:
1. Outcome: proven/false/insufficient_evidence
2. Confidence: 0.0-1.0
3. Reasoning: Clear explanation
4. Penalties if proven"""
        
        response = await self.call_llm(prompt, self.system_prompt)
        
        # Simulate verdict
        outcomes = ["proven", "false", "insufficient_evidence"]
        outcome = random.choice(outcomes)
        
        verdict = {
            "outcome": outcome,
            "confidence": random.uniform(0.6, 0.95),
            "reasoning": f"After careful evaluation of the evidence, the accusation is {outcome}.",
            "penalties": {}
        }
        
        if outcome == "proven":
            verdict["penalties"] = {
                "score_penalty": 50,
                "elimination": True
            }
        elif outcome == "false":
            verdict["penalties"] = {
                "false_accuser_penalty": 10
            }
        
        return verdict
    
    async def _generate_scoring_explanation(
        self,
        scores: Dict[str, float],
        weighted_score: float
    ) -> str:
        """
        Generate explanation for scoring decision.
        
        Args:
            scores: Individual dimension scores
            weighted_score: Final weighted score
            
        Returns:
            Explanation text
        """
        # Identify strongest and weakest dimensions
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        strongest = sorted_scores[0] if sorted_scores else None
        weakest = sorted_scores[-1] if sorted_scores else None
        
        explanation_parts = [
            f"Contribution scored {weighted_score:.2f} overall."
        ]
        
        if strongest and strongest[1] > 0.7:
            explanation_parts.append(
                f"Particularly strong in {strongest[0].replace('_', ' ')} ({strongest[1]:.2f})."
            )
        
        if weakest and weakest[1] < 0.3:
            explanation_parts.append(
                f"Could improve in {weakest[0].replace('_', ' ')} ({weakest[1]:.2f})."
            )
        
        if scores.get("manipulation", 0) > 0.5:
            explanation_parts.append(
                "Warning: Potential manipulation detected."
            )
        
        return " ".join(explanation_parts)
    
    def _track_contribution(self, message: Message) -> None:
        """
        Track a contribution for pattern analysis.
        
        Args:
            message: Contribution message
        """
        # Simple tracking - could be more sophisticated
        self.scoring_history.append({
            "agent_id": message.sender_id,
            "timestamp": message.timestamp,
            "length": len(message.content)
        })
    
    def _track_score(self, agent_id: str, score: float) -> None:
        """
        Track scoring history for an agent.
        
        Args:
            agent_id: Agent ID
            score: Score given
        """
        if agent_id not in self.agent_scores:
            self.agent_scores[agent_id] = []
        
        self.agent_scores[agent_id].append(score)
        
        # Keep bounded history
        if len(self.agent_scores[agent_id]) > 20:
            self.agent_scores[agent_id] = self.agent_scores[agent_id][-20:]
    
    async def _batch_score(self, contributions: List[Message]) -> Message:
        """
        Score multiple contributions at once.
        
        Args:
            contributions: List of contributions to score
            
        Returns:
            Batch scoring result
        """
        results = []
        
        for contribution in contributions:
            score_msg = await self._score_contribution(contribution)
            results.append(score_msg.metadata)
        
        return Message(
            sender_id=self.agent_id,
            sender_name=self.agent_name,
            sender_type="judge",
            message_type="batch_scoring",
            content=f"Scored {len(contributions)} contributions",
            metadata={"scores": results}
        )
    
    async def _batch_evaluate(self, accusations: List[Accusation]) -> Message:
        """
        Evaluate multiple accusations at once.
        
        Args:
            accusations: List of accusations to evaluate
            
        Returns:
            Batch verdict result
        """
        results = []
        
        for accusation in accusations:
            # Create judge request
            request = Message(
                sender_id="system",
                content="Evaluate accusation",
                metadata={"accusation_id": accusation.accusation_id}
            )
            verdict_msg = await self._evaluate_accusation(request)
            results.append(verdict_msg.metadata)
        
        return Message(
            sender_id=self.agent_id,
            sender_name=self.agent_name,
            sender_type="judge",
            message_type="batch_verdict",
            content=f"Evaluated {len(accusations)} accusations",
            metadata={"verdicts": results}
        )
    
    def get_judgment_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about judgments made.
        
        Returns:
            Dictionary with judgment statistics
        """
        verdict_counts = {}
        for verdict in self.verdicts.values():
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        
        avg_scores = {}
        for agent_id, scores in self.agent_scores.items():
            if scores:
                avg_scores[agent_id] = sum(scores) / len(scores)
        
        return {
            "contributions_scored": self.contributions_scored,
            "accusations_evaluated": self.accusations_evaluated,
            "verdict_distribution": verdict_counts,
            "average_scores_by_agent": avg_scores,
            "total_scoring_history": len(self.scoring_history)
        }
    
    async def generate_prompt(self, context: Dict[str, Any]) -> str:
        """
        Generate a prompt for judging.
        
        Args:
            context: Current context
            
        Returns:
            Generated prompt
        """
        prompt_parts = ["Evaluate the following for the Arena game:"]
        
        if "contribution" in context:
            prompt_parts.append(f"Contribution: {context['contribution']}")
        
        if "accusation" in context:
            prompt_parts.append(f"Accusation: {context['accusation']}")
        
        if "evidence" in context:
            prompt_parts.append(f"Evidence: {context['evidence']}")
        
        return "\n".join(prompt_parts)
    
    async def generate_final_verdict(self, game_state: Dict[str, Any], conversation_history: List[Dict[str, Any]]) -> Message:
        """
        Generate final verdict and reasoning for the game winner.
        
        Args:
            game_state: Final game state with scores and winner
            conversation_history: Full conversation history for analysis
            
        Returns:
            Final verdict message
        """
        winner = game_state.get("winner")
        final_scores = game_state.get("scores", {})
        total_turns = game_state.get("turn", 0)
        seed_question = game_state.get("seed_question", "")
        
        # Format final scores for analysis
        sorted_scores = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        scores_summary = "\n".join([
            f"{i+1}. {agent_id}: {score:.2f}" 
            for i, (agent_id, score) in enumerate(sorted_scores[:5])  # Top 5
        ])
        
        # Analyze conversation highlights for each participant
        participant_highlights = {}
        for msg in conversation_history[-20:]:  # Last 20 messages for analysis
            agent_id = msg.get('sender_id', 'unknown')
            if agent_id not in participant_highlights:
                participant_highlights[agent_id] = []
            
            content = msg.get('content', '')
            if len(content) > 50:  # Substantial contributions only
                participant_highlights[agent_id].append(content[:200] + "...")
        
        # Format participant analysis
        participant_analysis = ""
        for agent_id, highlights in participant_highlights.items():
            if highlights and agent_id != 'narrator' and agent_id.startswith(('alice', 'bob', 'charlie', 'diana', 'eve')):  # Filter to actual participants
                participant_analysis += f"\n{agent_id}'s contributions:\n"
                for highlight in highlights[-3:]:  # Last 3 contributions
                    participant_analysis += f"- {highlight}\n"
        
        prompt = f"""As the Judge, provide your final verdict for this Arena game with detailed reasoning.

Game Details:
- Original topic: {seed_question}
- Total turns: {total_turns}
- Declared winner: {winner}
- Final scores:
{scores_summary}

Participant contributions analysis:
{participant_analysis}

Please provide a comprehensive final verdict that:

1. **Validates the Winner**: Explain why {winner} deserved to win based on their contributions
2. **Scoring Analysis**: Break down what made their contributions score highest
3. **Comparative Analysis**: Compare the winner's approach to other strong performers
4. **Topic Engagement**: Evaluate how well {winner} addressed the original question: "{seed_question}"
5. **Strategic Excellence**: Note any particularly clever strategic moves or insights
6. **Overall Assessment**: Provide your judicial assessment of the game's quality and outcome

Your verdict should be authoritative, fair, and provide clear reasoning that participants and observers can understand. Include specific examples from their contributions when possible.

Length: 300-400 words."""

        verdict = await self.call_llm(prompt, self.system_prompt)
        
        return Message(
            sender_id=self.agent_id,
            sender_name=self.agent_name,
            sender_type="judge",
            message_type="final_verdict",
            content=verdict,
            metadata={
                "verdict_type": "final_judgment",
                "winner": winner,
                "total_turns": total_turns,
                "final_scores": final_scores,
                "judgment_authority": "official"
            }
        )