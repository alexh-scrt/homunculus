"""
Comprehensive tests for Arena Phase 5: Game Theory & Scoring

Tests all game theory components including:
- Scoring algorithms
- Elimination mechanics
- Coalition detection
- Reputation system
- Game strategies
- Leaderboard

Author: Homunculus Team  
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Any

# Add src to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Import game theory components
from arena.game_theory.scoring_engine import (
    ScoringEngine, MultidimensionalScorer, ScoringContext
)
from arena.game_theory.elimination_mechanics import (
    EliminationEngine, FairElimination, EliminationContext, 
    EliminationCandidate
)
from arena.game_theory.coalition_detection import (
    CoalitionDetector, CollaborationPattern, ManipulationDetector
)
from arena.game_theory.reputation_system import (
    ReputationEngine, ReputationEvent, ReputationFactor
)
from arena.game_theory.game_strategies import (
    TitForTat, AdaptiveStrategy, Action, GameState
)
from arena.game_theory.leaderboard import (
    Leaderboard, EloRating, RankingAlgorithm
)

# Import models
from arena.models import Message, AgentState, AccusationOutcome


class TestScoringEngine:
    """Test scoring engine functionality."""
    
    def test_multidimensional_scorer_initialization(self):
        """Test multidimensional scorer setup."""
        scorer = MultidimensionalScorer()
        
        assert scorer.base_weights["novelty"] == 0.25
        assert scorer.base_weights["manipulation"] == -0.15
        assert len(scorer.base_weights) == 5
    
    def test_scoring_context(self):
        """Test scoring context calculations."""
        context = ScoringContext(
            game_phase="mid",
            turn_number=25,
            total_agents=10,
            eliminated_agents=3,
            recent_scores=[0.5, 0.6, 0.7],
            problem_complexity=0.8
        )
        
        assert context.elimination_pressure == 0.3
        assert context.game_progress == 0.5
    
    def test_score_contribution(self):
        """Test scoring a contribution."""
        engine = ScoringEngine()
        
        contribution = Message(
            sender_id="agent_1",
            sender_name="Agent 1",
            content="This is an innovative solution that builds on previous ideas",
            message_type="contribution"
        )
        
        context = ScoringContext(
            game_phase="mid",
            turn_number=10,
            total_agents=8,
            eliminated_agents=0,
            recent_scores=[],
            problem_complexity=0.6
        )
        
        metrics = engine.score_contribution(contribution, context)
        
        assert metrics.agent_id == "agent_1"
        assert 0.0 <= metrics.novelty <= 1.0
        assert 0.0 <= metrics.builds_on_others <= 1.0
        assert metrics.weighted_score is not None
    
    def test_manipulation_detection(self):
        """Test manipulation detection in scoring."""
        scorer = MultidimensionalScorer()
        
        # Create repetitive contribution
        contribution = Message(
            sender_id="manipulator",
            content="innovative creative novel brilliant solution",
            message_type="contribution"
        )
        
        context = ScoringContext(
            game_phase="mid",
            turn_number=10,
            total_agents=8,
            eliminated_agents=0,
            recent_scores=[],
            problem_complexity=0.5
        )
        
        # Score multiple similar contributions
        for _ in range(3):
            metrics = scorer.score(contribution, context, [])
            scorer.agent_patterns["manipulator"]["content_lengths"].append(
                len(contribution.content)
            )
        
        # Check manipulation detection
        manipulation_score = scorer._detect_manipulation(
            contribution, [], "manipulator"
        )
        
        assert manipulation_score > 0  # Some manipulation detected


class TestEliminationMechanics:
    """Test elimination mechanics."""
    
    def test_fair_elimination_initialization(self):
        """Test fair elimination setup."""
        eliminator = FairElimination(
            grace_period=5,
            protect_top_percent=0.2
        )
        
        assert eliminator.grace_period == 5
        assert eliminator.protect_top_percent == 0.2
    
    def test_elimination_context(self):
        """Test elimination context."""
        context = EliminationContext(
            turn_number=30,
            total_agents=10,
            active_agents=7,
            elimination_round=3,
            scores={"agent_1": 0.7, "agent_2": 0.3},
            accusations={},
            protections={}
        )
        
        assert context.elimination_rate == 0.3
        assert context.should_eliminate  # Turn 30 is elimination turn
    
    def test_elimination_candidate(self):
        """Test elimination candidate."""
        candidate = EliminationCandidate(
            agent_id="agent_1",
            score=0.4,
            elimination_score=0.6,
            reasons=["Low performance"],
            is_protected=False
        )
        
        assert candidate.can_be_eliminated()
        
        candidate.is_protected = True
        assert not candidate.can_be_eliminated()
    
    def test_elimination_engine(self):
        """Test elimination engine."""
        engine = EliminationEngine(
            enable_protection=True,
            enable_appeals=True
        )
        
        context = EliminationContext(
            turn_number=30,
            total_agents=10,
            active_agents=8,
            elimination_round=3,
            scores={
                "agent_1": 0.8,
                "agent_2": 0.2,
                "agent_3": 0.5
            },
            accusations={},
            protections={}
        )
        
        agent_states = {
            "agent_1": AgentState(
                agent_id="agent_1",
                is_active=True,
                score=0.8
            ),
            "agent_2": AgentState(
                agent_id="agent_2",
                is_active=True,
                score=0.2
            ),
            "agent_3": AgentState(
                agent_id="agent_3",
                is_active=True,
                score=0.5
            )
        }
        
        eliminated = engine.process_elimination_round(context, agent_states)
        
        # Should eliminate low performers
        assert isinstance(eliminated, list)


class TestCoalitionDetection:
    """Test coalition detection."""
    
    def test_collaboration_pattern(self):
        """Test collaboration pattern detection."""
        pattern = CollaborationPattern(
            agent1="agent_1",
            agent2="agent_2",
            interaction_count=10,
            mutual_support_count=5,
            synchronized_actions=3,
            similarity_score=0.8,
            time_correlation=0.7
        )
        
        assert pattern.collaboration_strength > 0.5
        assert pattern.is_suspicious()
    
    def test_coalition_detector(self):
        """Test coalition detector."""
        detector = CoalitionDetector(
            sensitivity=0.6,
            min_coalition_size=2
        )
        
        messages = [
            Message(
                sender_id="agent_1",
                content="I agree with agent_2",
                message_type="contribution"
            ),
            Message(
                sender_id="agent_2",
                content="Building on agent_1's excellent point",
                message_type="contribution"
            )
        ]
        
        coalitions = detector.analyze_interactions(messages, turn_number=5)
        
        assert isinstance(coalitions, list)
    
    def test_manipulation_detector(self):
        """Test manipulation detection."""
        detector = ManipulationDetector(threshold=0.6)
        
        messages = [
            Message(
                sender_id="manipulator",
                content="Vote together to eliminate agent_3",
                message_type="contribution"
            )
        ]
        
        score = detector.analyze_for_manipulation(
            "manipulator",
            messages,
            {}
        )
        
        assert score >= 0.0


class TestReputationSystem:
    """Test reputation system."""
    
    def test_reputation_engine_initialization(self):
        """Test reputation engine setup."""
        engine = ReputationEngine(
            initial_reputation=0.5,
            decay_enabled=True
        )
        
        assert engine.initial_reputation == 0.5
        assert engine.decay_enabled
    
    def test_reputation_event(self):
        """Test reputation event processing."""
        engine = ReputationEngine()
        
        event = ReputationEvent(
            agent_id="agent_1",
            event_type="contribution",
            impact=0.5,
            factor=ReputationFactor.CONTRIBUTION_QUALITY,
            description="High quality contribution",
            timestamp=datetime.utcnow()
        )
        
        new_rep = engine.update_reputation("agent_1", event)
        
        assert 0.0 <= new_rep <= 1.0
        assert len(engine.reputation_history["agent_1"]) == 1
    
    def test_trust_network(self):
        """Test trust network updates."""
        engine = ReputationEngine()
        
        # Positive interaction
        trust = engine.update_trust("agent_1", "agent_2", True)
        
        assert trust.trustor == "agent_1"
        assert trust.trustee == "agent_2"
        assert trust.positive_interactions == 1
        
        # Get trust level
        level = engine.get_trust_level("agent_1", "agent_2")
        assert 0.0 <= level <= 1.0
    
    def test_credibility_calculation(self):
        """Test credibility scoring."""
        engine = ReputationEngine()
        
        # Add some events
        for i in range(3):
            event = ReputationEvent(
                agent_id="agent_1",
                event_type="contribution",
                impact=0.3,
                factor=ReputationFactor.CONTRIBUTION_QUALITY,
                description="Contribution",
                timestamp=datetime.utcnow()
            )
            engine.update_reputation("agent_1", event)
        
        credibility = engine.calculate_credibility("agent_1")
        
        assert credibility.agent_id == "agent_1"
        assert 0.0 <= credibility.overall_score <= 1.0
        assert ReputationFactor.CONTRIBUTION_QUALITY in credibility.factors


class TestGameStrategies:
    """Test game theory strategies."""
    
    def test_tit_for_tat_strategy(self):
        """Test Tit-for-Tat strategy."""
        strategy = TitForTat("agent_1", forgiveness=0.1)
        
        state = GameState(
            turn=5,
            active_agents=["agent_1", "agent_2"],
            eliminated_agents=[],
            scores={"agent_1": 0.5, "agent_2": 0.5},
            recent_actions=[],
            phase="mid"
        )
        
        # First action should cooperate
        action = strategy.decide_action(state, "agent_2")
        assert action == Action.COOPERATE
        
        # Remember opponent defection
        strategy.memory.remember_action("agent_2", Action.DEFECT)
        
        # Should retaliate
        action = strategy.decide_action(state, "agent_2")
        assert action in [Action.COOPERATE, Action.DEFECT]  # May forgive
    
    def test_adaptive_strategy(self):
        """Test adaptive strategy."""
        strategy = AdaptiveStrategy(
            "agent_1",
            learning_rate=0.1,
            exploration_rate=0.1
        )
        
        state = GameState(
            turn=5,
            active_agents=["agent_1", "agent_2"],
            eliminated_agents=[],
            scores={"agent_1": 0.5, "agent_2": 0.5},
            recent_actions=[],
            phase="mid"
        )
        
        # Should return an action
        action = strategy.decide_action(state, "agent_2")
        assert isinstance(action, Action)
        
        # Update with outcome
        strategy.update(action, 0.7, Action.COOPERATE)
        assert len(strategy.memory.payoffs) == 1


class TestLeaderboard:
    """Test leaderboard system."""
    
    def test_elo_rating_system(self):
        """Test Elo rating calculations."""
        elo = EloRating(
            initial_rating=1500,
            k_factor=32
        )
        
        # Update ratings
        new_winner, new_loser = elo.update_ratings("winner", "loser", draw=False)
        
        assert new_winner > 1500  # Winner gains rating
        assert new_loser < 1500  # Loser loses rating
        assert elo.ratings["winner"].wins == 1
        assert elo.ratings["loser"].losses == 1
    
    def test_leaderboard_management(self):
        """Test leaderboard functionality."""
        leaderboard = Leaderboard(RankingAlgorithm.ELO)
        
        # Record a game
        leaderboard.record_game(
            game_id="game_1",
            rankings=["agent_1", "agent_2", "agent_3"],
            scores={"agent_1": 0.8, "agent_2": 0.6, "agent_3": 0.4},
            eliminations=["agent_3"],
            metadata={"duration": 100}
        )
        
        assert len(leaderboard.history) == 1
        
        # Get leaderboard
        current = leaderboard.get_current_leaderboard(top_n=10)
        assert len(current) > 0
        assert current[0][0] == "agent_1"  # Winner should be first
    
    def test_player_statistics(self):
        """Test player stats tracking."""
        leaderboard = Leaderboard()
        
        # Record multiple games
        for i in range(3):
            leaderboard.record_game(
                game_id=f"game_{i}",
                rankings=["agent_1", "agent_2"] if i < 2 else ["agent_2", "agent_1"],
                scores={"agent_1": 0.7, "agent_2": 0.6},
                eliminations=[],
                metadata={}
            )
        
        stats = leaderboard.get_player_stats("agent_1")
        
        assert stats["player_id"] == "agent_1"
        assert stats["games_played"] > 0
        assert stats["wins"] >= 2
        assert len(stats["recent_games"]) > 0


class TestIntegration:
    """Integration tests for game theory components."""
    
    def test_scoring_elimination_integration(self):
        """Test scoring and elimination working together."""
        scoring_engine = ScoringEngine()
        elimination_engine = EliminationEngine()
        
        # Score contributions
        contributions = [
            Message(sender_id=f"agent_{i}", content=f"Content {i}", 
                   message_type="contribution")
            for i in range(5)
        ]
        
        context = ScoringContext(
            game_phase="mid",
            turn_number=30,
            total_agents=5,
            eliminated_agents=0,
            recent_scores=[],
            problem_complexity=0.5
        )
        
        scores = {}
        for contrib in contributions:
            metrics = scoring_engine.score_contribution(contrib, context)
            scores[contrib.sender_id] = metrics.weighted_score
        
        # Use scores for elimination
        elim_context = EliminationContext(
            turn_number=30,
            total_agents=5,
            active_agents=5,
            elimination_round=1,
            scores=scores,
            accusations={},
            protections={}
        )
        
        agent_states = {
            agent_id: AgentState(agent_id=agent_id, is_active=True, score=score)
            for agent_id, score in scores.items()
        }
        
        eliminated = elimination_engine.process_elimination_round(
            elim_context, agent_states
        )
        
        assert isinstance(eliminated, list)
    
    def test_reputation_coalition_integration(self):
        """Test reputation and coalition detection integration."""
        reputation_engine = ReputationEngine()
        coalition_detector = CoalitionDetector()
        
        # Detect coalition
        messages = [
            Message(sender_id="agent_1", content="Supporting agent_2", 
                   message_type="contribution"),
            Message(sender_id="agent_2", content="Agreeing with agent_1",
                   message_type="contribution")
        ]
        
        coalitions = coalition_detector.analyze_interactions(messages, 5)
        
        # Update reputation based on coalition
        if coalitions:
            for coalition in coalitions:
                for member in coalition.members:
                    event = ReputationEvent(
                        agent_id=member,
                        event_type="coalition_detected",
                        impact=-0.3,
                        factor=ReputationFactor.FAIRNESS,
                        description="Part of detected coalition",
                        timestamp=datetime.utcnow()
                    )
                    reputation_engine.update_reputation(member, event)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])