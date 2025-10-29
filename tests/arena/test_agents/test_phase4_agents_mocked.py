"""
Comprehensive Unit Tests for Arena Phase 4 Agents (with mocked Kafka dependencies)

This test suite validates all agent implementations created in Phase 4:
- Base agent classes
- Narrator agent
- Judge agent  
- Turn Selector agent
- Character agent wrapper
- Agent communication

Author: Homunculus Team
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from typing import Dict, Any
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Mock kafka dependencies before importing
sys.modules['kafka'] = MagicMock()
sys.modules['kafka.structs'] = MagicMock()

# Import agent components
from arena.agents.base_agent import BaseAgent, LLMAgent, AgentConfig, AgentRole
from arena.agents.narrator_agent import NarratorAgent
from arena.agents.judge_agent import JudgeAgent
from arena.agents.turn_selector_agent import TurnSelectorAgent
from arena.agents.character_agent import CharacterAgent

# Import models
from arena.models import Message, MessageType, AgentState, ArenaState
from arena.models.homunculus_integration import HomunculusCharacterProfile


class TestBaseAgent:
    """Test base agent functionality."""
    
    def test_agent_config_creation(self):
        """Test creating agent configuration."""
        config = AgentConfig(
            agent_id="test_agent",
            agent_name="Test Agent",
            role=AgentRole.OBSERVER,
            llm_config={"model": "gpt-4"},
            kafka_topics=["test_topic"],
            max_retries=5,
            timeout_seconds=60.0
        )
        
        assert config.agent_id == "test_agent"
        assert config.agent_name == "Test Agent"
        assert config.role == AgentRole.OBSERVER
        assert config.max_retries == 5
        assert config.timeout_seconds == 60.0
    
    @pytest.mark.asyncio
    async def test_base_agent_initialization(self):
        """Test base agent initialization."""
        config = AgentConfig(
            agent_id="test",
            agent_name="Test",
            role=AgentRole.OBSERVER
        )
        
        # Create a concrete implementation for testing
        class TestAgent(BaseAgent):
            async def initialize(self):
                pass
            async def process_message(self, message):
                return None
            async def generate_action(self, context):
                return None
            async def update_state(self, state):
                pass
        
        agent = TestAgent(config)
        
        assert agent.agent_id == "test"
        assert agent.agent_name == "Test"
        assert agent.role == AgentRole.OBSERVER
        assert not agent.is_active
        assert agent.messages_processed == 0
        assert agent.errors_encountered == 0
    
    @pytest.mark.asyncio
    async def test_agent_message_callbacks(self):
        """Test agent message callback registration."""
        config = AgentConfig("test", "Test", AgentRole.OBSERVER)
        
        class TestAgent(BaseAgent):
            async def initialize(self):
                pass
            async def process_message(self, message):
                return None
            async def generate_action(self, context):
                return None
            async def update_state(self, state):
                pass
        
        agent = TestAgent(config)
        
        # Register callbacks
        callback1 = Mock()
        callback2 = Mock()
        
        agent.register_message_callback("contribution", callback1)
        agent.register_message_callback("*", callback2)
        
        assert "contribution" in agent.message_callbacks
        assert "*" in agent.message_callbacks
        assert callback1 in agent.message_callbacks["contribution"]
        assert callback2 in agent.message_callbacks["*"]
    
    def test_agent_statistics(self):
        """Test agent statistics tracking."""
        config = AgentConfig("test", "Test", AgentRole.OBSERVER)
        
        class TestAgent(BaseAgent):
            async def initialize(self):
                pass
            async def process_message(self, message):
                return None
            async def generate_action(self, context):
                return None
            async def update_state(self, state):
                pass
        
        agent = TestAgent(config)
        agent.messages_processed = 10
        agent.errors_encountered = 2
        agent.last_action_time = datetime.utcnow()
        
        stats = agent.get_statistics()
        
        assert stats["agent_id"] == "test"
        assert stats["messages_processed"] == 10
        assert stats["errors_encountered"] == 2
        assert stats["last_action_time"] is not None


class TestLLMAgent:
    """Test LLM agent functionality."""
    
    def test_llm_agent_initialization(self):
        """Test LLM agent initialization."""
        config = AgentConfig(
            agent_id="llm_test",
            agent_name="LLM Test",
            role=AgentRole.CHARACTER,
            llm_config={
                "model": "gpt-4",
                "temperature": 0.8,
                "max_tokens": 500
            }
        )
        
        class TestLLMAgent(LLMAgent):
            async def initialize(self):
                pass
            async def process_message(self, message):
                return None
            async def generate_action(self, context):
                return None
            async def update_state(self, state):
                pass
            async def generate_prompt(self, context):
                return "Test prompt"
        
        agent = TestLLMAgent(config)
        
        assert agent.model_name == "gpt-4"
        assert agent.temperature == 0.8
        assert agent.max_tokens == 500
        assert agent.total_tokens_used == 0
    
    def test_conversation_history_management(self):
        """Test conversation history management."""
        config = AgentConfig("test", "Test", AgentRole.CHARACTER)
        
        class TestLLMAgent(LLMAgent):
            async def initialize(self):
                pass
            async def process_message(self, message):
                return None
            async def generate_action(self, context):
                return None
            async def update_state(self, state):
                pass
            async def generate_prompt(self, context):
                return "Test"
        
        agent = TestLLMAgent(config)
        
        # Add to history
        agent.add_to_history("user", "Hello")
        agent.add_to_history("assistant", "Hi there")
        
        assert len(agent.conversation_history) == 2
        assert agent.conversation_history[0]["role"] == "user"
        assert agent.conversation_history[1]["content"] == "Hi there"
        
        # Clear history
        agent.clear_history()
        assert len(agent.conversation_history) == 0
    
    def test_token_statistics(self):
        """Test token usage statistics."""
        config = AgentConfig("test", "Test", AgentRole.CHARACTER)
        
        class TestLLMAgent(LLMAgent):
            async def initialize(self):
                pass
            async def process_message(self, message):
                return None
            async def generate_action(self, context):
                return None
            async def update_state(self, state):
                pass
            async def generate_prompt(self, context):
                return "Test"
        
        agent = TestLLMAgent(config)
        agent.prompt_tokens_used = 100
        agent.completion_tokens_used = 50
        agent.total_tokens_used = 150
        
        stats = agent.get_token_statistics()
        
        assert stats["total_tokens"] == 150
        assert stats["prompt_tokens"] == 100
        assert stats["completion_tokens"] == 50
        assert "estimated_cost" in stats


class TestNarratorAgent:
    """Test Narrator agent functionality."""
    
    @pytest.mark.asyncio
    async def test_narrator_initialization(self):
        """Test Narrator agent initialization."""
        config = AgentConfig(
            agent_id="narrator",
            agent_name="The Narrator",
            role=AgentRole.NARRATOR,
            metadata={"summary_frequency": 3}
        )
        
        narrator = NarratorAgent(config)
        await narrator.initialize()
        
        assert narrator.role == AgentRole.NARRATOR
        assert narrator.summary_frequency == 3
        assert len(narrator.key_moments) == 0
        assert len(narrator.narrative_arc) == 0
    
    @pytest.mark.asyncio
    async def test_narrator_summary_generation(self):
        """Test narrator summary generation."""
        config = AgentConfig("narrator", "Narrator", AgentRole.NARRATOR)
        narrator = NarratorAgent(config)
        
        # Add contributions
        for i in range(3):
            msg = Message(
                sender_id=f"agent_{i}",
                sender_name=f"Agent {i}",
                content=f"Contribution {i}",
                message_type="contribution"
            )
            narrator.recent_contributions.append(msg)
        
        # Check should summarize
        assert narrator._should_summarize()
        
        # Mock LLM call
        with patch.object(narrator, 'call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Test summary of recent discussion"
            
            summary = await narrator._generate_summary()
            
            assert summary.message_type == "narration"
            assert summary.metadata["narration_type"] == "summary"
            assert len(narrator.recent_contributions) == 0  # Cleared after summary
    
    @pytest.mark.asyncio
    async def test_narrator_accusation_handling(self):
        """Test narrator handling of accusations."""
        config = AgentConfig("narrator", "Narrator", AgentRole.NARRATOR)
        narrator = NarratorAgent(config)
        
        accusation = Message(
            sender_id="accuser",
            sender_name="Accuser",
            content="I accuse Agent X of cheating",
            message_type="accusation",
            metadata={"accused_name": "Agent X"}
        )
        
        with patch.object(narrator, 'call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Tension rises as an accusation is made..."
            
            narration = await narrator._narrate_accusation(accusation)
            
            assert narration.message_type == "narration"
            assert narration.metadata["narration_type"] == "accusation"
            assert len(narrator.key_moments) == 1
            assert narrator.key_moments[0]["type"] == "accusation"


class TestJudgeAgent:
    """Test Judge agent functionality."""
    
    @pytest.mark.asyncio
    async def test_judge_initialization(self):
        """Test Judge agent initialization."""
        config = AgentConfig(
            agent_id="judge",
            agent_name="The Judge",
            role=AgentRole.JUDGE,
            metadata={
                "scoring_weights": {
                    "novelty": 0.3,
                    "builds_on_others": 0.3,
                    "solves_subproblem": 0.4
                }
            }
        )
        
        judge = JudgeAgent(config)
        await judge.initialize()
        
        assert judge.role == AgentRole.JUDGE
        assert judge.scoring_weights["novelty"] == 0.3
        assert judge.contributions_scored == 0
        assert judge.accusations_evaluated == 0
    
    @pytest.mark.asyncio
    async def test_judge_scoring(self):
        """Test judge scoring functionality."""
        config = AgentConfig("judge", "Judge", AgentRole.JUDGE)
        judge = JudgeAgent(config)
        
        scoring_request = Message(
            sender_id="system",
            content="Score contribution",
            message_type="scoring_request",
            metadata={
                "message_id": "msg_123",
                "agent_id": "agent_001",
                "turn_number": 5,
                "content": "This is a test contribution"
            }
        )
        
        with patch.object(judge, 'call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = '{"novelty": 0.8, "builds_on_others": 0.6}'
            
            with patch.object(judge, '_generate_scores', new_callable=AsyncMock) as mock_scores:
                mock_scores.return_value = {
                    "novelty": 0.8,
                    "builds_on_others": 0.6,
                    "solves_subproblem": 0.7,
                    "radical_idea": 0.3,
                    "manipulation": 0.1
                }
                
                score_msg = await judge._score_contribution(scoring_request)
                
                assert score_msg.message_type == "scoring"
                assert "metrics" in score_msg.metadata
                assert score_msg.metadata["agent_id"] == "agent_001"
                assert judge.contributions_scored == 1
    
    @pytest.mark.asyncio
    async def test_judge_accusation_evaluation(self):
        """Test judge accusation evaluation."""
        config = AgentConfig("judge", "Judge", AgentRole.JUDGE)
        judge = JudgeAgent(config)
        
        judge_request = Message(
            sender_id="system",
            content="Evaluate accusation",
            message_type="judge_request",
            metadata={"accusation_id": "acc_123"}
        )
        
        with patch.object(judge, '_generate_verdict', new_callable=AsyncMock) as mock_verdict:
            mock_verdict.return_value = {
                "outcome": "proven",
                "confidence": 0.85,
                "reasoning": "Evidence clearly shows manipulation",
                "penalties": {"score_penalty": 50}
            }
            
            verdict = await judge._evaluate_accusation(judge_request)
            
            assert verdict.message_type == "verdict"
            assert verdict.metadata["outcome"] == "proven"
            assert verdict.metadata["confidence"] == 0.85
            assert judge.accusations_evaluated == 1


class TestTurnSelectorAgent:
    """Test Turn Selector agent functionality."""
    
    @pytest.mark.asyncio
    async def test_turn_selector_initialization(self):
        """Test Turn Selector initialization."""
        config = AgentConfig(
            agent_id="turn_selector",
            agent_name="Turn Selector",
            role=AgentRole.TURN_SELECTOR,
            metadata={
                "strategy_weights": {
                    "fairness": 0.5,
                    "merit": 0.5
                }
            }
        )
        
        selector = TurnSelectorAgent(config)
        await selector.initialize()
        
        assert selector.role == AgentRole.TURN_SELECTOR
        assert selector.strategy_weights["fairness"] == 0.5
        assert selector.epsilon == 0.1
        assert selector.temperature == 1.0
    
    def test_turn_selector_fairness_calculation(self):
        """Test fairness score calculation."""
        config = AgentConfig("selector", "Selector", AgentRole.TURN_SELECTOR)
        selector = TurnSelectorAgent(config)
        
        # Never spoke
        score = selector._calculate_fairness_score("agent_new", 10)
        assert score == 1.0
        
        # Spoke recently
        selector.agent_last_spoke["agent_recent"] = 8
        score = selector._calculate_fairness_score("agent_recent", 10)
        assert score < 1.0
        
        # Spoke long ago (past threshold)
        selector.agent_last_spoke["agent_old"] = 5
        score = selector._calculate_fairness_score("agent_old", 10)
        assert score == 1.0
    
    def test_turn_selector_softmax(self):
        """Test softmax probability calculation."""
        config = AgentConfig("selector", "Selector", AgentRole.TURN_SELECTOR)
        selector = TurnSelectorAgent(config)
        
        scores = [1.0, 2.0, 0.5]
        probs = selector._softmax(scores, temperature=1.0)
        
        assert len(probs) == 3
        assert abs(sum(probs) - 1.0) < 0.001  # Sum to 1
        assert probs[1] > probs[0] > probs[2]  # Ordered by score
    
    @pytest.mark.asyncio
    async def test_turn_selection(self):
        """Test turn selection process."""
        config = AgentConfig("selector", "Selector", AgentRole.TURN_SELECTOR)
        selector = TurnSelectorAgent(config)
        
        request = Message(
            sender_id="system",
            content="Select speaker",
            message_type="turn_request",
            metadata={
                "available_agents": ["agent_1", "agent_2", "agent_3"],
                "current_turn": 5
            }
        )
        
        selection = await selector._select_next_speaker(request)
        
        assert selection.message_type == "turn_selection"
        assert selection.metadata["selected_agent"] in ["agent_1", "agent_2", "agent_3"]
        assert "selection_probabilities" in selection.metadata
        assert len(selector.turn_history) == 1


class TestCharacterAgent:
    """Test Character agent functionality."""
    
    @pytest.mark.asyncio
    async def test_character_initialization(self):
        """Test Character agent initialization."""
        profile = HomunculusCharacterProfile(
            character_name="Ada Lovelace",
            personality_traits=["analytical", "innovative"],
            expertise_areas=["mathematics", "computing"],
            communication_style="formal",
            backstory="Pioneer of computing"
        )
        
        config = AgentConfig(
            agent_id="ada",
            agent_name="Ada",
            role=AgentRole.CHARACTER
        )
        
        character = CharacterAgent(config, profile)
        await character.initialize()
        
        assert character.role == AgentRole.CHARACTER
        assert character.character_profile.character_name == "Ada Lovelace"
        assert "analytical" in character.character_profile.personality_traits
        assert len(character.internal_agents) == 6
    
    @pytest.mark.asyncio
    async def test_character_contribution_generation(self):
        """Test character contribution generation."""
        profile = HomunculusCharacterProfile(
            character_name="Test Character",
            personality_traits=["creative"],
            expertise_areas=["testing"]
        )
        
        config = AgentConfig("test", "Test", AgentRole.CHARACTER)
        character = CharacterAgent(config, profile)
        
        with patch.object(character, 'call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "This is my contribution to the discussion"
            
            with patch.object(character, '_gather_internal_perspectives', new_callable=AsyncMock) as mock_perspectives:
                mock_perspectives.return_value = {
                    "reaper": "Focus on conclusions",
                    "creators_muse": "Be creative"
                }
                
                contribution = await character._generate_contribution()
                
                assert contribution.message_type == "contribution"
                assert contribution.sender_name == "Test Character"
                assert len(character.contribution_history) == 1
    
    def test_character_strategy_update(self):
        """Test character strategy updates."""
        profile = HomunculusCharacterProfile(character_name="Test")
        config = AgentConfig("test", "Test", AgentRole.CHARACTER)
        character = CharacterAgent(config, profile)
        
        # Test phase-based strategy
        character._update_strategy("early")
        assert character.current_strategy == "exploratory"
        
        character._update_strategy("late")
        assert character.current_strategy == "competitive"
        
        character._update_strategy("final")
        assert character.current_strategy == "decisive"
    
    def test_character_internal_agents(self):
        """Test character internal sub-agents."""
        profile = HomunculusCharacterProfile(character_name="Test")
        config = AgentConfig("test", "Test", AgentRole.CHARACTER)
        character = CharacterAgent(config, profile)
        
        # Test each internal agent exists
        assert "reaper" in character.internal_agents
        assert "creators_muse" in character.internal_agents
        assert "conscience" in character.internal_agents
        assert "devil_advocate" in character.internal_agents
        assert "pattern_recognizer" in character.internal_agents
        assert "interface" in character.internal_agents
        
        # Test perspective generation
        context = {"test": "data"}
        perspective = character.internal_agents["reaper"].generate_perspective(context)
        assert isinstance(perspective, str)


class TestAgentIntegration:
    """Integration tests for agent interactions."""
    
    @pytest.mark.asyncio
    async def test_judge_narrator_interaction(self):
        """Test interaction between Judge and Narrator."""
        judge_config = AgentConfig("judge", "Judge", AgentRole.JUDGE)
        narrator_config = AgentConfig("narrator", "Narrator", AgentRole.NARRATOR)
        
        judge = JudgeAgent(judge_config)
        narrator = NarratorAgent(narrator_config)
        
        # Judge scores a contribution
        score_msg = Message(
            sender_id="judge",
            sender_type="judge",
            message_type="scoring",
            content="High quality contribution",
            metadata={
                "agent_id": "test_agent",
                "weighted_score": 0.85
            }
        )
        
        # Narrator processes the scoring
        result = await narrator.process_message(score_msg)
        
        # Narrator might not immediately respond to scoring
        # but would track it for summaries
        assert len(narrator.recent_contributions) == 0  # No contribution tracked
    
    @pytest.mark.asyncio
    async def test_turn_selector_character_interaction(self):
        """Test interaction between Turn Selector and Character."""
        selector_config = AgentConfig("selector", "Selector", AgentRole.TURN_SELECTOR)
        selector = TurnSelectorAgent(selector_config)
        
        profile = HomunculusCharacterProfile(character_name="Ada")
        char_config = AgentConfig("ada", "Ada", AgentRole.CHARACTER)
        character = CharacterAgent(char_config, profile)
        
        # Turn selector selects character
        selection = Message(
            sender_id="selector",
            message_type="turn_selection",
            content="Ada selected",
            metadata={"selected_agent": "ada"}
        )
        
        # Character processes selection
        response = await character.process_message(selection)
        
        # Character should generate contribution when selected
        with patch.object(character, '_generate_contribution', new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = Message(
                sender_id="ada",
                content="My contribution",
                message_type="contribution"
            )
            
            response = await character.process_message(selection)
            assert response is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])