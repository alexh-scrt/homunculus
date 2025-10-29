"""
Comprehensive Unit Tests for Arena Phase 2 Data Models

This test suite validates all data models created in Phase 2:
- AgentState with lifecycle management
- Message with serialization
- ScoringMetrics with calculations
- Accusation with evidence
- ArenaState with game management

Author: Homunculus Team
"""

import pytest
import json
from datetime import datetime, timedelta
import time

# Import all models to test
from src.arena.models import (
    # Agent models
    AgentState, AgentStatus,
    # Message models
    Message, MessageBatch, MessageType, SenderType,
    # Scoring models
    ScoringMetrics, AgentScorecard,
    # Accusation models
    Accusation, Evidence, AccusationType, AccusationOutcome, EvidenceType,
    # Game models  
    ArenaState, GameStatus, TerminationReason
)


class TestAgentState:
    """Test AgentState model functionality."""
    
    def test_agent_creation(self):
        """Test basic agent creation."""
        agent = AgentState(
            agent_id="agent_001",
            character_name="Ada Lovelace",
            character_profile={"personality": "analytical"}
        )
        
        assert agent.agent_id == "agent_001"
        assert agent.character_name == "Ada Lovelace"
        assert agent.status == AgentStatus.ACTIVE
        assert agent.score == 0.0
        assert agent.is_active
        assert not agent.is_eliminated
    
    def test_agent_validation(self):
        """Test agent validation rules."""
        # Missing agent_id
        with pytest.raises(ValueError, match="agent_id is required"):
            AgentState(agent_id="", character_name="Test", character_profile={})
        
        # Missing character_name
        with pytest.raises(ValueError, match="character_name is required"):
            AgentState(agent_id="001", character_name="", character_profile={})
        
        # Negative score
        with pytest.raises(ValueError, match="score cannot be negative"):
            AgentState(
                agent_id="001", 
                character_name="Test",
                character_profile={},
                score=-1
            )
    
    def test_agent_contribution_tracking(self):
        """Test tracking agent contributions."""
        agent = AgentState(agent_id="001", character_name="Ada", character_profile={})
        
        agent.add_contribution("msg_001")
        agent.add_contribution("msg_002")
        
        assert agent.turns_taken == 2
        assert len(agent.contributions) == 2
        assert "msg_001" in agent.contributions
        assert "msg_002" in agent.contributions
    
    def test_agent_scoring(self):
        """Test agent score updates."""
        agent = AgentState(agent_id="001", character_name="Ada", character_profile={})
        
        agent.update_score(10.5)
        assert agent.score == 10.5
        
        agent.update_score(-5.0)
        assert agent.score == 5.5
        
        # Test minimum score limit
        agent.update_score(-10000)
        assert agent.score == -1000  # Should be clamped to minimum
    
    def test_agent_elimination(self):
        """Test agent elimination process."""
        agent = AgentState(agent_id="001", character_name="Ada", character_profile={})
        
        assert agent.is_active
        assert not agent.is_eliminated
        
        agent.eliminate("Poor performance")
        
        assert not agent.is_active
        assert agent.is_eliminated
        assert agent.status == AgentStatus.ELIMINATED
        assert agent.elimination_reason == "Poor performance"
        assert agent.eliminated_at is not None
    
    def test_agent_champion_status(self):
        """Test champion promotion."""
        agent = AgentState(agent_id="001", character_name="Ada", character_profile={})
        
        agent.make_champion()
        
        assert agent.status == AgentStatus.CHAMPION
        assert agent.previous_wins == 1
        
        # Test returning champion
        agent = AgentState(
            agent_id="001",
            character_name="Ada",
            character_profile={},
            is_returning_champion=True,
            previous_wins=2
        )
        assert agent.is_returning_champion
        assert agent.previous_wins == 2
    
    def test_agent_accusation_tracking(self):
        """Test accusation statistics."""
        agent = AgentState(agent_id="001", character_name="Ada", character_profile={})
        
        agent.record_accusation(was_false=False)
        agent.record_accusation(was_false=True)
        agent.record_accusation(was_false=False)
        
        assert agent.accusations_made == 3
        assert agent.false_accusations == 1
        assert agent.accusation_accuracy == 2/3
    
    def test_agent_serialization(self):
        """Test agent serialization and deserialization."""
        agent = AgentState(
            agent_id="001",
            character_name="Ada",
            character_profile={"test": "data"},
            score=15.5,
            turns_taken=3
        )
        agent.add_contribution("msg_001")
        
        # To dict
        data = agent.to_dict()
        assert data["agent_id"] == "001"
        assert data["score"] == 15.5
        assert data["turns_taken"] == 4  # add_contribution increments turns_taken
        
        # From dict
        restored = AgentState.from_dict(data)
        assert restored.agent_id == "001"
        assert restored.score == 15.5
        assert restored.turns_taken == 4  # Was 3 + 1 from add_contribution
        assert "msg_001" in restored.contributions
        
        # To/from JSON
        json_str = agent.to_json()
        restored = AgentState.from_json(json_str)
        assert restored.agent_id == agent.agent_id
        assert restored.score == agent.score


class TestMessage:
    """Test Message model functionality."""
    
    def test_message_creation(self):
        """Test basic message creation."""
        msg = Message(
            sender_id="agent_001",
            sender_name="Ada",
            content="I propose we consider..."
        )
        
        assert msg.sender_id == "agent_001"
        assert msg.sender_name == "Ada"
        assert msg.content == "I propose we consider..."
        assert msg.message_id is not None
        assert msg.timestamp is not None
    
    def test_message_validation(self):
        """Test message validation."""
        # Missing sender_id
        with pytest.raises(ValueError, match="sender_id is required"):
            Message(sender_id="", content="Test")
        
        # Missing content for non-turn-selection
        with pytest.raises(ValueError, match="content is required"):
            Message(sender_id="001", content="", message_type="contribution")
        
        # Invalid confidence score
        with pytest.raises(ValueError, match="confidence_score must be between"):
            Message(sender_id="001", content="Test", confidence_score=1.5)
    
    def test_message_properties(self):
        """Test message property methods."""
        msg = Message(
            sender_id="agent_001",
            sender_name="Ada",
            sender_type="character",
            message_type="contribution",
            content="Test",
            target_agent_id="agent_002"
        )
        
        assert msg.is_contribution
        assert not msg.is_accusation
        assert not msg.is_system_message
        assert msg.is_targeted
    
    def test_message_references(self):
        """Test message reference handling."""
        msg = Message(sender_id="001", content="Test")
        
        msg.add_reference("msg_001")
        msg.add_reference("msg_002")
        msg.add_reference("msg_001")  # Duplicate
        
        assert len(msg.references) == 2
        assert "msg_001" in msg.references
        assert "msg_002" in msg.references
    
    def test_message_metadata(self):
        """Test message metadata handling."""
        msg = Message(sender_id="001", content="Test")
        
        msg.add_metadata("score", 0.8)
        msg.add_metadata("tags", ["important", "novel"])
        
        assert msg.get_metadata("score") == 0.8
        assert msg.get_metadata("tags") == ["important", "novel"]
        assert msg.get_metadata("missing", "default") == "default"
    
    def test_message_serialization(self):
        """Test message serialization."""
        msg = Message(
            sender_id="001",
            sender_name="Ada",
            content="Test message",
            turn_number=5,
            game_id="game_001"
        )
        msg.add_reference("ref_001")
        msg.add_metadata("test", "data")
        
        # To dict
        data = msg.to_dict()
        assert data["sender_id"] == "001"
        assert data["turn_number"] == 5
        assert "ref_001" in data["references"]
        
        # From dict
        restored = Message.from_dict(data)
        assert restored.sender_id == msg.sender_id
        assert restored.turn_number == msg.turn_number
        assert restored.message_id == msg.message_id
    
    def test_message_batch(self):
        """Test MessageBatch functionality."""
        batch = MessageBatch(game_id="game_001")
        
        msg1 = Message(sender_id="001", content="First", turn_number=1, game_id="game_001")
        msg2 = Message(sender_id="002", content="Second", turn_number=2, game_id="game_001")
        msg3 = Message(sender_id="001", content="Third", turn_number=2, game_id="game_001")
        
        batch.add_message(msg1)
        batch.add_message(msg2)
        batch.add_message(msg3)
        
        assert batch.size == 3
        assert batch.turn_range == (1, 2)
        
        # Filter by sender
        sender_msgs = batch.get_messages_by_sender("001")
        assert len(sender_msgs) == 2
        
        # Wrong game ID
        wrong_game = Message(sender_id="003", content="Wrong", game_id="game_002")
        with pytest.raises(ValueError, match="doesn't match batch"):
            batch.add_message(wrong_game)


class TestScoringMetrics:
    """Test scoring system models."""
    
    def test_scoring_metrics_creation(self):
        """Test basic scoring metrics."""
        metrics = ScoringMetrics(
            novelty=0.8,
            builds_on_others=0.6,
            solves_subproblem=0.4,
            radical_idea=0.9,
            manipulation=0.2
        )
        
        assert metrics.novelty == 0.8
        assert metrics.builds_on_others == 0.6
        assert metrics.total_raw_score == 2.9
        assert metrics.average_metric_score == 0.58
    
    def test_scoring_validation(self):
        """Test scoring metric validation."""
        # Invalid range
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            ScoringMetrics(novelty=1.5)
        
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            ScoringMetrics(builds_on_others=-0.1)
    
    def test_weighted_score_calculation(self):
        """Test weighted score calculation."""
        metrics = ScoringMetrics(
            novelty=0.8,
            builds_on_others=0.6,
            solves_subproblem=0.4,
            radical_idea=0.9,
            manipulation=0.2
        )
        
        weights = {
            "novelty": 0.25,
            "builds_on_others": 0.20,
            "solves_subproblem": 0.25,
            "radical_idea": 0.15,
            "manipulation": 0.15
        }
        
        score = metrics.calculate_weighted_score(weights)
        expected = (0.8*0.25 + 0.6*0.20 + 0.4*0.25 + 0.9*0.15 + 0.2*0.15)
        assert abs(score - expected) < 0.001
        assert metrics.weighted_score == score
        
        # Invalid weights
        bad_weights = {"novelty": 0.5, "builds_on_others": 0.6}  # Sum > 1
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            metrics.calculate_weighted_score(bad_weights)
    
    def test_top_metrics(self):
        """Test getting top scoring metrics."""
        metrics = ScoringMetrics(
            novelty=0.9,
            builds_on_others=0.3,
            solves_subproblem=0.7,
            radical_idea=0.8,
            manipulation=0.2
        )
        
        top = metrics.get_top_metrics(2)
        assert len(top) == 2
        assert top[0] == ("novelty", 0.9)
        assert top[1] == ("radical_idea", 0.8)
    
    def test_agent_scorecard(self):
        """Test AgentScorecard functionality."""
        scorecard = AgentScorecard(agent_id="agent_001")
        
        # Add metrics
        m1 = ScoringMetrics(agent_id="agent_001", novelty=0.8, weighted_score=5.0)
        m2 = ScoringMetrics(agent_id="agent_001", novelty=0.6, weighted_score=4.0)
        m3 = ScoringMetrics(agent_id="agent_001", novelty=0.7, weighted_score=4.5)
        
        scorecard.add_metrics(m1)
        scorecard.add_metrics(m2)
        scorecard.add_metrics(m3)
        
        assert scorecard.total_score == 13.5
        assert scorecard.contribution_count == 3
        assert scorecard.average_score == 4.5
        assert scorecard.average_metric("novelty") == 0.7
    
    def test_scorecard_bonuses_penalties(self):
        """Test scorecard bonus and penalty system."""
        scorecard = AgentScorecard(agent_id="agent_001")
        
        scorecard.apply_bonus(5.0, "Champion bonus")
        assert scorecard.bonus_points == 5.0
        assert scorecard.total_score == 5.0
        
        scorecard.apply_penalty(2.0, "False accusation")
        assert scorecard.penalty_points == 2.0
        assert scorecard.total_score == 3.0
        
        # Invalid values
        with pytest.raises(ValueError):
            scorecard.apply_bonus(-1.0)
        with pytest.raises(ValueError):
            scorecard.apply_penalty(-1.0)
    
    def test_scorecard_statistics(self):
        """Test scorecard statistical methods."""
        scorecard = AgentScorecard(agent_id="agent_001")
        
        # Add some metrics with varying scores
        for score in [3.0, 5.0, 4.0, 6.0, 4.5]:
            m = ScoringMetrics(agent_id="agent_001", weighted_score=score)
            scorecard.add_metrics(m)
        
        stats = scorecard.get_statistics()
        assert stats["contribution_count"] == 5
        assert stats["average_score"] == 4.5
        assert stats["min_score"] == 3.0
        assert stats["max_score"] == 6.0
        assert stats["total_score"] == 22.5


class TestAccusation:
    """Test accusation system models."""
    
    def test_accusation_creation(self):
        """Test basic accusation creation."""
        acc = Accusation(
            accuser_id="agent_001",
            accuser_name="Ada",
            accused_id="agent_002",
            accused_name="Bob",
            accusation_type="false_statement",
            claim="Bob made false claims about X"
        )
        
        assert acc.accuser_id == "agent_001"
        assert acc.accused_id == "agent_002"
        assert acc.outcome == "pending"
        assert acc.is_pending
        assert not acc.is_proven
    
    def test_accusation_validation(self):
        """Test accusation validation."""
        # Self-accusation
        with pytest.raises(ValueError, match="Cannot accuse oneself"):
            Accusation(
                accuser_id="agent_001",
                accused_id="agent_001",
                claim="Test"
            )
        
        # Missing claim
        with pytest.raises(ValueError, match="claim is required"):
            Accusation(
                accuser_id="agent_001",
                accused_id="agent_002",
                claim=""
            )
    
    def test_evidence_handling(self):
        """Test evidence management."""
        acc = Accusation(
            accuser_id="001",
            accused_id="002",
            claim="Cheating occurred"
        )
        
        # Add evidence
        ev1 = Evidence(
            evidence_type="message_reference",
            content="Message shows contradiction",
            message_ids=["msg_001", "msg_002"],
            confidence=0.8
        )
        
        ev2 = Evidence(
            evidence_type="pattern_analysis",
            content="Behavior pattern indicates cheating",
            confidence=0.6
        )
        
        acc.add_evidence(ev1)
        acc.add_evidence(ev2)
        
        assert acc.evidence_count == 2
        assert acc.max_evidence_confidence == 0.8
        assert len(acc.referenced_messages) == 2
        
        # Get evidence by type
        msg_evidence = acc.get_evidence_by_type("message_reference")
        assert len(msg_evidence) == 1
        assert msg_evidence[0].confidence == 0.8
    
    def test_accusation_resolution(self):
        """Test accusation resolution process."""
        acc = Accusation(
            accuser_id="001",
            accused_id="002",
            claim="Test claim"
        )
        
        assert acc.is_pending
        
        # Resolve as proven
        acc.resolve("proven", "Evidence is conclusive", 0.95)
        
        assert acc.outcome == "proven"
        assert acc.is_proven
        assert not acc.is_pending
        assert acc.judge_reasoning == "Evidence is conclusive"
        assert acc.confidence_score == 0.95
        assert acc.resolved_at is not None
        
        # Cannot resolve again
        with pytest.raises(ValueError, match="already resolved"):
            acc.resolve("false", "Changed mind", 0.5)
    
    def test_accusation_withdrawal(self):
        """Test accusation withdrawal."""
        acc = Accusation(
            accuser_id="001",
            accused_id="002",
            claim="Test"
        )
        
        acc.withdraw("Insufficient evidence")
        
        assert acc.outcome == "withdrawn"
        assert "Insufficient evidence" in acc.judge_reasoning
        assert acc.resolved_at is not None
    
    def test_evidence_strength_calculation(self):
        """Test evidence strength calculation."""
        acc = Accusation(
            accuser_id="001",
            accused_id="002",
            claim="Test"
        )
        
        # Add varied evidence
        acc.add_evidence(Evidence(
            evidence_type="system_log",
            content="System detected violation",
            confidence=1.0
        ))
        
        acc.add_evidence(Evidence(
            evidence_type="witness_testimony",
            content="Another agent confirms",
            confidence=0.8
        ))
        
        strength = acc.calculate_evidence_strength()
        assert 0.0 <= strength <= 1.0
        assert strength > 0.8  # System log has high weight


class TestArenaState:
    """Test complete game state management."""
    
    def test_arena_creation(self):
        """Test arena state initialization."""
        arena = ArenaState(
            problem_statement="Solve the trolley problem",
            problem_title="Trolley Problem",
            max_turns=30
        )
        
        assert arena.game_id is not None
        assert arena.status == "initializing"
        assert arena.current_turn == 0
        assert arena.max_turns == 30
        assert len(arena.active_agents) == 0
    
    def test_agent_management(self):
        """Test adding and managing agents."""
        arena = ArenaState()
        
        # Add agents
        agent1 = AgentState("001", "Ada", {})
        agent2 = AgentState("002", "Bob", {})
        
        arena.add_agent(agent1)
        arena.add_agent(agent2)
        
        assert arena.agent_count == 2
        assert arena.get_agent("001") == agent1
        assert "001" in arena.get_active_agent_ids()
        
        # Cannot add duplicate
        with pytest.raises(ValueError, match="already exists"):
            arena.add_agent(AgentState("001", "Duplicate", {}))
        
        # Cannot add after game starts
        arena.start_game()
        with pytest.raises(ValueError, match="Cannot add agents after"):
            arena.add_agent(AgentState("003", "Charlie", {}))
    
    def test_game_flow(self):
        """Test game state transitions."""
        arena = ArenaState()
        
        # Add agents
        arena.add_agent(AgentState("001", "Ada", {}))
        arena.add_agent(AgentState("002", "Bob", {}))
        
        # Start game
        arena.start_game()
        assert arena.status == "in_progress"
        assert arena.started_at is not None
        
        # Advance turns
        arena.advance_turn()
        arena.advance_turn()
        assert arena.current_turn == 2
        
        # Eliminate an agent
        success = arena.eliminate_agent("002", "Poor performance")
        assert success
        assert arena.agent_count == 1
        assert arena.elimination_count == 1
        
        # Check termination
        reason = arena.check_termination()
        assert reason == "single_survivor"
        
        # Terminate game
        arena.terminate_game("single_survivor")
        assert arena.is_completed
        assert arena.winner_id == "001"
    
    def test_message_management(self):
        """Test message handling."""
        arena = ArenaState()
        arena.current_turn = 5
        
        msg1 = Message(sender_id="001", content="First", turn_number=0)
        msg2 = Message(sender_id="002", content="Second", turn_number=0)
        
        arena.add_message(msg1)
        arena.add_message(msg2)
        
        assert len(arena.message_history) == 2
        assert msg1.game_id == arena.game_id
        assert msg1.turn_number == 5  # Should be updated
        
        # Get recent messages
        recent = arena.get_recent_messages(1)
        assert len(recent) == 1
        assert recent[0] == msg2
        
        # Get by turn
        turn_msgs = arena.get_messages_by_turn(5)
        assert len(turn_msgs) == 2
    
    def test_scoring_integration(self):
        """Test scoring system integration."""
        arena = ArenaState()
        
        agent = AgentState("001", "Ada", {})
        arena.add_agent(agent)
        
        # Add scoring metrics
        metrics = ScoringMetrics(
            novelty=0.8,
            builds_on_others=0.6,
            agent_id="001"
        )
        
        arena.update_score("001", metrics)
        
        assert "001" in arena.scorecards
        assert arena.scorecards["001"].contribution_count == 1
        assert agent.score > 0  # Agent score should be updated
        
        # Get all scores
        scores = arena.get_agent_scores()
        assert scores["001"] > 0
    
    def test_accusation_integration(self):
        """Test accusation system integration."""
        arena = ArenaState()
        
        agent1 = AgentState("001", "Ada", {})
        agent2 = AgentState("002", "Bob", {})
        arena.add_agent(agent1)
        arena.add_agent(agent2)
        
        acc = Accusation(
            accuser_id="001",
            accused_id="002",
            claim="Cheating detected"
        )
        
        arena.add_accusation(acc)
        
        assert len(arena.accusation_history) == 1
        assert acc.game_id == arena.game_id
        assert agent1.accusations_made == 1
        
        # Get pending accusations
        pending = arena.get_pending_accusations()
        assert len(pending) == 1
        assert pending[0] == acc
    
    def test_game_summary(self):
        """Test game summary generation."""
        arena = ArenaState(
            problem_title="Test Problem",
            max_turns=20
        )
        
        arena.add_agent(AgentState("001", "Ada", {}, score=15.0))
        arena.add_agent(AgentState("002", "Bob", {}, score=10.0))
        arena.start_game()
        
        summary = arena.get_game_summary()
        
        assert summary["problem_title"] == "Test Problem"
        assert summary["status"] == "in_progress"
        assert summary["active_agents"] == 2
        assert len(summary["top_scores"]) == 2
        assert summary["top_scores"][0] == ("Ada", 15.0)
    
    def test_arena_serialization(self):
        """Test complete arena serialization."""
        arena = ArenaState(problem_title="Test")
        
        # Add complex state
        arena.add_agent(AgentState("001", "Ada", {}))
        arena.add_message(Message(sender_id="001", content="Test"))
        arena.add_accusation(Accusation(
            accuser_id="001",
            accused_id="002", 
            claim="Test"
        ))
        
        # To dict
        data = arena.to_dict()
        assert data["game_id"] == arena.game_id
        assert len(data["active_agents"]) == 1
        assert len(data["message_history"]) == 1
        
        # From dict
        restored = ArenaState.from_dict(data)
        assert restored.game_id == arena.game_id
        assert restored.agent_count == 1
        assert len(restored.message_history) == 1
        
        # To/from JSON
        json_str = arena.to_json()
        restored = ArenaState.from_json(json_str)
        assert restored.game_id == arena.game_id


class TestPhase2Integration:
    """Integration tests for all Phase 2 models working together."""
    
    def test_complete_game_scenario(self):
        """Test a complete mini-game scenario."""
        # Initialize game
        arena = ArenaState(
            problem_title="Test Problem",
            problem_statement="Solve this test problem",
            max_turns=10
        )
        
        # Add agents
        ada = AgentState("ada", "Ada Lovelace", {"personality": "analytical"})
        bob = AgentState("bob", "Bob Builder", {"personality": "practical"})
        charlie = AgentState("charlie", "Charlie Brown", {"personality": "creative"})
        
        arena.add_agent(ada)
        arena.add_agent(bob)
        arena.add_agent(charlie)
        
        # Start game
        arena.start_game()
        assert arena.is_active
        
        # Turn 1: Ada contributes
        arena.advance_turn()
        msg1 = Message(
            sender_id="ada",
            sender_name="Ada Lovelace",
            content="I propose we approach this analytically",
            message_type="contribution"
        )
        arena.add_message(msg1)
        ada.add_contribution(msg1.message_id)
        
        # Score Ada's contribution
        metrics1 = ScoringMetrics(
            novelty=0.8,
            builds_on_others=0.0,  # First contribution
            solves_subproblem=0.6,
            agent_id="ada",
            message_id=msg1.message_id
        )
        arena.update_score("ada", metrics1)
        
        # Turn 2: Bob challenges Ada
        arena.advance_turn()
        msg2 = Message(
            sender_id="bob",
            sender_name="Bob Builder",
            content="I disagree with Ada's approach",
            message_type="contribution",
            target_agent_id="ada"
        )
        arena.add_message(msg2)
        bob.add_contribution(msg2.message_id)
        
        # Turn 3: Charlie accuses Bob
        arena.advance_turn()
        acc = Accusation(
            accuser_id="charlie",
            accuser_name="Charlie Brown",
            accused_id="bob",
            accused_name="Bob Builder",
            accusation_type="false_statement",
            claim="Bob's statement contradicts known facts"
        )
        arena.add_accusation(acc)
        
        # Judge evaluates accusation
        acc.resolve("false", "No contradiction found", 0.3)
        charlie.record_accusation(was_false=True)
        
        # Apply penalty for false accusation
        arena.scorecards["charlie"].apply_penalty(5.0, "False accusation")
        
        # Turn 4: Eliminate Charlie
        arena.advance_turn()
        arena.eliminate_agent("charlie", "Lowest score and false accusation")
        
        elimination_msg = Message(
            sender_id="judge",
            sender_type="judge",
            sender_name="The Judge",
            message_type="elimination",
            content="Charlie Brown has been eliminated: Lowest score and false accusation",
            target_agent_id="charlie"
        )
        arena.add_message(elimination_msg)
        
        # Check game state
        assert arena.agent_count == 2
        assert arena.elimination_count == 1
        assert charlie.is_eliminated
        assert ada.eliminations_witnessed == 1
        assert bob.eliminations_witnessed == 1
        
        # Continue to termination
        arena.advance_turn()
        arena.eliminate_agent("bob", "Score below threshold")
        
        # Check termination
        reason = arena.check_termination()
        assert reason == "single_survivor"
        
        arena.terminate_game("single_survivor")
        
        # Verify final state
        assert arena.is_completed
        assert arena.winner_id == "ada"
        assert arena.winner_name == "Ada Lovelace"
        assert ada.status == AgentStatus.CHAMPION
        
        # Verify serialization of complete game
        game_data = arena.to_dict()
        restored = ArenaState.from_dict(game_data)
        assert restored.winner_id == "ada"
        assert len(restored.message_history) == 3
        assert len(restored.accusation_history) == 1
        assert restored.elimination_count == 2
    
    def test_model_relationships(self):
        """Test that all models work together correctly."""
        # Create interconnected data
        agent = AgentState("001", "Test Agent", {})
        
        message = Message(
            sender_id=agent.agent_id,
            sender_name=agent.character_name,
            content="Test contribution"
        )
        
        metrics = ScoringMetrics(
            agent_id=agent.agent_id,
            message_id=message.message_id,
            novelty=0.7
        )
        
        evidence = Evidence(
            evidence_type="message_reference",
            content="Reference to message",
            message_ids=[message.message_id]
        )
        
        accusation = Accusation(
            accuser_id=agent.agent_id,
            accuser_name=agent.character_name,
            accused_id="002",
            accused_name="Other Agent",
            claim="Test accusation"
        )
        accusation.add_evidence(evidence)
        
        # Verify relationships
        assert metrics.agent_id == agent.agent_id
        assert metrics.message_id == message.message_id
        assert message.message_id in accusation.referenced_messages
        
        # Create arena with all components
        arena = ArenaState()
        arena.add_agent(agent)
        arena.add_message(message)
        arena.update_score(agent.agent_id, metrics)
        arena.add_accusation(accusation)
        
        # Verify integration
        assert arena.get_agent(agent.agent_id) == agent
        assert arena.message_history[0] == message
        assert arena.scorecards[agent.agent_id].contribution_count == 1
        assert arena.accusation_history[0] == accusation


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])