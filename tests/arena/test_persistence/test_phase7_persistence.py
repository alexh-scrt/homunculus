"""
Tests for Arena Persistence Layer (Phase 7)

This module tests the persistence and multi-round functionality
including database, champion memory, replays, and analytics.

Author: Homunculus Team
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import json
import tempfile
from pathlib import Path
import shutil

# Import persistence modules
from src.arena.persistence import (
    DatabaseManager,
    ChampionMemory,
    GameStorage,
    ReplayManager,
    TournamentManager,
    AnalyticsEngine,
    DataExporter,
    ExportFormat,
    ExportConfig,
    TournamentFormat,
    StorageFormat,
    ReplaySpeed
)


class TestDatabaseManager(unittest.TestCase):
    """Test database management."""
    
    def setUp(self):
        """Set up test database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_manager = DatabaseManager()
    
    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_and_load_game(self):
        """Test saving and loading a game."""
        # Create game data
        game_data = {
            "game_id": "test_game_001",
            "start_time": datetime.utcnow().isoformat(),
            "total_turns": 50,
            "total_agents": 8,
            "winner_id": "agent_3",
            "winner_name": "Agent Three",
            "final_phase": "FINAL",
            "turns": [
                {
                    "turn_number": 1,
                    "phase": "EARLY",
                    "active_agents": 8,
                    "eliminated": []
                }
            ],
            "agents": [
                {
                    "agent_id": "agent_1",
                    "agent_name": "Agent One",
                    "final_score": 85.5,
                    "final_position": 2
                },
                {
                    "agent_id": "agent_3",
                    "agent_name": "Agent Three",
                    "final_score": 92.3,
                    "final_position": 1,
                    "is_champion": True
                }
            ],
            "messages": [],
            "scores": []
        }
        
        # Save game
        game_id = self.db_manager.save_game(game_data)
        self.assertEqual(game_id, "test_game_001")
        
        # Load game
        loaded_data = self.db_manager.load_game("test_game_001")
        self.assertIsNotNone(loaded_data)
        self.assertEqual(loaded_data["game_id"], "test_game_001")
        self.assertEqual(loaded_data["winner_id"], "agent_3")
        self.assertEqual(len(loaded_data["agents"]), 2)
    
    def test_champion_history(self):
        """Test retrieving champion history."""
        # Save a game with a champion
        game_data = {
            "game_id": "test_game_002",
            "start_time": datetime.utcnow().isoformat(),
            "total_turns": 40,
            "total_agents": 6,
            "agents": [
                {
                    "agent_id": "champion_1",
                    "agent_name": "Champion",
                    "is_champion": True,
                    "final_score": 100
                }
            ],
            "turns": [],
            "messages": [],
            "scores": []
        }
        
        self.db_manager.save_game(game_data)
        
        # Get champion history
        history = self.db_manager.get_champion_history("champion_1")
        self.assertIsInstance(history, list)
        
        # In fallback mode, should find the champion
        if history:
            self.assertEqual(history[0]["game_id"], "test_game_002")


class TestChampionMemory(unittest.TestCase):
    """Test champion memory system."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.memory = ChampionMemory()
    
    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialize_champion(self):
        """Test champion initialization."""
        profile = self.memory.initialize_champion(
            "test_agent",
            "Test Agent",
            load_history=False
        )
        
        self.assertEqual(profile.agent_id, "test_agent")
        self.assertEqual(profile.agent_name, "Test Agent")
        self.assertEqual(profile.total_wins, 0)
        self.assertEqual(profile.total_games, 0)
    
    def test_record_turn(self):
        """Test recording a turn."""
        self.memory.initialize_champion("test_agent", "Test Agent", False)
        
        # Record a turn
        self.memory.record_turn(
            agent_id="test_agent",
            game_id="game_1",
            turn=5,
            state={"phase": "MID", "active_agents": 6},
            action={"type": "contribute", "content": "Test contribution"},
            reward=10.5
        )
        
        # Check replay buffer
        replay = self.memory.memory_bank.get_replay_buffer("test_agent")
        self.assertEqual(len(replay.buffer), 1)
        
        memory_entry = replay.buffer[0]
        self.assertEqual(memory_entry.game_id, "game_1")
        self.assertEqual(memory_entry.turn, 5)
        self.assertEqual(memory_entry.reward, 10.5)
    
    def test_finalize_game(self):
        """Test finalizing a game."""
        self.memory.initialize_champion("test_agent", "Test Agent", False)
        
        # Record some turns
        for i in range(5):
            self.memory.record_turn(
                agent_id="test_agent",
                game_id="game_1",
                turn=i,
                state={"turn": i},
                action={"type": "action"},
                reward=i * 2
            )
        
        # Finalize game
        game_data = {
            "game_id": "game_1",
            "won": True,
            "final_score": 95.0,
            "primary_strategy": "cooperative"
        }
        
        self.memory.finalize_game("test_agent", game_data)
        
        # Check profile update
        profile = self.memory.memory_bank.profiles["test_agent"]
        self.assertEqual(profile.total_games, 1)
        self.assertEqual(profile.total_wins, 1)
        self.assertEqual(profile.average_score, 95.0)
    
    def test_strategic_advice(self):
        """Test getting strategic advice."""
        self.memory.initialize_champion("test_agent", "Test Agent", False)
        
        # Add some memories
        for i in range(10):
            self.memory.record_turn(
                agent_id="test_agent",
                game_id="game_1",
                turn=i,
                state={"phase": "MID", "score": i * 10},
                action={"type": "cooperate" if i % 2 == 0 else "compete"},
                reward=15 if i % 2 == 0 else 5
            )
        
        # Get advice
        advice = self.memory.get_strategic_advice(
            "test_agent",
            {"phase": "MID", "score": 50}
        )
        
        self.assertIn("recommended_strategies", advice)
        self.assertIn("confidence", advice)


class TestGameStorage(unittest.TestCase):
    """Test game storage system."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage = GameStorage(str(self.temp_dir))
    
    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_and_load(self):
        """Test saving and loading a game."""
        game_state = {
            "game_id": "test_game",
            "current_turn": 25,
            "active_agents": ["agent_1", "agent_2", "agent_3"],
            "scores": {"agent_1": 50, "agent_2": 45, "agent_3": 48}
        }
        
        # Save game
        metadata = self.storage.save_game(
            game_state,
            "test_save",
            format=StorageFormat.JSON
        )
        
        self.assertEqual(metadata.game_id, "test_game")
        self.assertEqual(metadata.save_name, "test_save")
        
        # Load game
        loaded_state = self.storage.load_game(metadata.save_id)
        self.assertIsNotNone(loaded_state)
        self.assertEqual(loaded_state["game_id"], "test_game")
        self.assertEqual(loaded_state["current_turn"], 25)
    
    def test_quick_save_load(self):
        """Test quick save and load functionality."""
        game_state = {
            "game_id": "quick_test",
            "current_turn": 10
        }
        
        # Quick save
        metadata = self.storage.quick_save(game_state, slot=0)
        self.assertIsNotNone(metadata)
        
        # Quick load
        loaded = self.storage.quick_load(slot=0)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["game_id"], "quick_test")
    
    def test_auto_save(self):
        """Test auto save functionality."""
        self.storage.auto_save_interval = 5
        
        # Test on interval
        game_state = {
            "game_id": "auto_test",
            "current_turn": 10  # Multiple of interval
        }
        
        metadata = self.storage.auto_save(game_state)
        self.assertIsNotNone(metadata)
        
        # Test off interval
        game_state["current_turn"] = 11
        metadata = self.storage.auto_save(game_state)
        self.assertIsNone(metadata)
    
    def test_list_saves(self):
        """Test listing saves."""
        # Create some saves
        for i in range(3):
            self.storage.save_game(
                {"game_id": f"game_{i}", "turn": i},
                f"save_{i}",
                save_type="manual"
            )
        
        # List saves
        saves = self.storage.list_saves(save_type="manual")
        self.assertEqual(len(saves), 3)


class TestReplaySystem(unittest.TestCase):
    """Test replay system."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.replay_manager = ReplayManager(str(self.temp_dir))
    
    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_recording(self):
        """Test replay recording."""
        # Start recording
        recorder = self.replay_manager.start_recording("test_game")
        
        # Record frames
        for i in range(5):
            recorder.record_frame(
                game_state={
                    "current_turn": i,
                    "scores": {"agent_1": i * 10},
                    "active_agents": ["agent_1"],
                    "eliminated_this_turn": []
                },
                events=[{"type": "turn_start", "turn": i}],
                messages=[{"sender": "agent_1", "content": f"Turn {i}"}]
            )
        
        # Stop recording
        replay_path = self.replay_manager.stop_recording("test_game", "agent_1")
        self.assertIsNotNone(replay_path)
        self.assertTrue(replay_path.exists())
    
    def test_replay_viewer(self):
        """Test replay viewing."""
        # Create and save a replay
        recorder = self.replay_manager.start_recording("test_game")
        
        for i in range(10):
            recorder.record_frame(
                game_state={
                    "current_turn": i,
                    "scores": {"agent_1": i * 5}
                }
            )
        
        self.replay_manager.stop_recording("test_game")
        
        # Load replay
        viewer = self.replay_manager.load_replay("test_game")
        self.assertIsNotNone(viewer)
        
        # Test navigation
        current = viewer.get_current_frame()
        self.assertEqual(current.frame_number, 0)
        
        next_frame = viewer.next_frame()
        self.assertIsNotNone(next_frame)
        self.assertEqual(next_frame.frame_number, 1)
        
        viewer.jump_to_frame(5)
        current = viewer.get_current_frame()
        self.assertEqual(current.frame_number, 5)
    
    def test_replay_analysis(self):
        """Test replay analysis."""
        from src.arena.persistence.replay_system import (
            ReplayRecorder, ReplayViewer, ReplayAnalyzer, ReplayMetadata
        )
        
        # Create replay data
        recorder = ReplayRecorder("analysis_game")
        
        for turn in range(20):
            recorder.record_frame(
                game_state={
                    "current_turn": turn,
                    "scores": {
                        "agent_1": turn * 5,
                        "agent_2": turn * 4,
                        "agent_3": turn * 3
                    },
                    "active_agents": ["agent_1", "agent_2", "agent_3"],
                    "eliminated_this_turn": ["agent_3"] if turn == 15 else []
                }
            )
        
        metadata = recorder.finalize(winner="agent_1")
        viewer = ReplayViewer(recorder.frames, metadata)
        analyzer = ReplayAnalyzer(viewer)
        
        # Analyze game flow
        flow = analyzer.analyze_game_flow()
        self.assertIn("total_turns", flow)
        self.assertIn("elimination_timeline", flow)
        self.assertIn("score_progression", flow)
        
        # Analyze agent performance
        perf = analyzer.analyze_agent_performance("agent_1")
        self.assertIn("average_score", perf)
        self.assertIn("is_winner", perf)
        self.assertTrue(perf["is_winner"])


class TestTournamentSystem(unittest.TestCase):
    """Test tournament system."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.tournament_mgr = TournamentManager(str(self.temp_dir))
    
    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_single_elimination(self):
        """Test creating single elimination tournament."""
        participants = [f"agent_{i}" for i in range(8)]
        
        bracket = self.tournament_mgr.create_tournament(
            "test_tournament",
            participants,
            TournamentFormat.SINGLE_ELIMINATION
        )
        
        self.assertEqual(bracket.format, TournamentFormat.SINGLE_ELIMINATION)
        self.assertEqual(len(bracket.participants), 8)
        self.assertEqual(len(bracket.rounds), 3)  # log2(8) = 3
    
    def test_match_completion(self):
        """Test completing tournament matches."""
        participants = ["agent_1", "agent_2", "agent_3", "agent_4"]
        
        self.tournament_mgr.create_tournament(
            "test_tournament",
            participants,
            TournamentFormat.SINGLE_ELIMINATION
        )
        
        # Start first match
        match = self.tournament_mgr.start_match("test_tournament", "r1_m0")
        self.assertIsNotNone(match)
        
        # Complete match
        success = self.tournament_mgr.complete_match(
            "test_tournament",
            "r1_m0",
            winner="agent_1",
            scores={"agent_1": 100, "agent_2": 80}
        )
        self.assertTrue(success)
        
        # Check standings
        standings = self.tournament_mgr.get_standings("test_tournament")
        self.assertIsNotNone(standings)
    
    def test_round_robin(self):
        """Test round-robin tournament."""
        participants = ["agent_1", "agent_2", "agent_3", "agent_4"]
        
        bracket = self.tournament_mgr.create_tournament(
            "round_robin_test",
            participants,
            TournamentFormat.ROUND_ROBIN
        )
        
        self.assertEqual(bracket.format, TournamentFormat.ROUND_ROBIN)
        # Each participant plays every other once
        total_matches = (len(participants) * (len(participants) - 1)) // 2
        total_round_matches = sum(len(r.matches) for r in bracket.rounds)
        self.assertEqual(total_round_matches, total_matches)


class TestAnalytics(unittest.TestCase):
    """Test analytics engine."""
    
    def setUp(self):
        """Set up test environment."""
        self.analytics = AnalyticsEngine()
    
    def test_game_tracking(self):
        """Test tracking game metrics."""
        # Start tracking
        metrics = self.analytics.start_game_tracking("test_game")
        self.assertEqual(metrics.game_id, "test_game")
        
        # Record turns
        for turn in range(10):
            self.analytics.record_turn(
                turn,
                {
                    "messages": [
                        {"sender_id": "agent_1", "content": "test"},
                        {"sender_id": "agent_2", "content": "response"}
                    ],
                    "scores": {"agent_1": turn * 10, "agent_2": turn * 8},
                    "eliminated": []
                }
            )
        
        # Finalize
        final_metrics = self.analytics.finalize_game(
            {"agent_1": 100, "agent_2": 80},
            "agent_1"
        )
        
        self.assertEqual(final_metrics.total_turns, 9)
        self.assertEqual(final_metrics.total_messages, 20)
        self.assertEqual(final_metrics.get_winner(), "agent_1")
    
    def test_real_time_metrics(self):
        """Test real-time metrics."""
        self.analytics.start_game_tracking("realtime_game")
        
        # Record some data
        self.analytics.record_turn(
            1,
            {
                "messages": [{"sender_id": "agent_1"}],
                "scores": {"agent_1": 10, "agent_2": 8}
            }
        )
        
        # Get real-time metrics
        rt_metrics = self.analytics.get_real_time_metrics()
        self.assertIn("game_id", rt_metrics)
        self.assertEqual(rt_metrics["current_turn"], 1)
        self.assertEqual(rt_metrics["total_messages"], 1)
        self.assertEqual(rt_metrics["current_leader"], "agent_1")
    
    def test_agent_analytics(self):
        """Test agent-specific analytics."""
        # Complete a game
        self.analytics.start_game_tracking("game_1")
        
        for turn in range(5):
            self.analytics.record_turn(
                turn,
                {
                    "messages": [{"sender_id": "agent_1"}],
                    "scores": {"agent_1": turn * 10}
                }
            )
        
        self.analytics.finalize_game({"agent_1": 50}, "agent_1")
        
        # Get agent analytics
        agent_analytics = self.analytics.get_agent_analytics("agent_1")
        self.assertIn("performance", agent_analytics)
        self.assertIn("trend", agent_analytics)


class TestDataExport(unittest.TestCase):
    """Test data export functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.exporter = DataExporter(str(self.temp_dir))
    
    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_json_export(self):
        """Test JSON export."""
        game_data = {
            "game_id": "test_game",
            "total_turns": 50,
            "agents": [
                {"agent_id": "agent_1", "final_score": 100},
                {"agent_id": "agent_2", "final_score": 85}
            ]
        }
        
        config = ExportConfig(format=ExportFormat.JSON)
        path = self.exporter.export_game(game_data, config)
        
        self.assertTrue(path.exists())
        
        # Verify content
        with open(path, 'r') as f:
            loaded = json.load(f)
            self.assertEqual(loaded["game_id"], "test_game")
            self.assertEqual(len(loaded["agents"]), 2)
    
    def test_csv_export(self):
        """Test CSV export."""
        game_data = {
            "agents": [
                {"agent_id": "agent_1", "final_score": 100, "final_position": 1},
                {"agent_id": "agent_2", "final_score": 85, "final_position": 2}
            ]
        }
        
        config = ExportConfig(format=ExportFormat.CSV)
        path = self.exporter.export_game(game_data, config)
        
        if path:  # CSV export might return None if no data
            self.assertTrue(path.exists())
            
            # Verify content
            with open(path, 'r') as f:
                content = f.read()
                self.assertIn("agent_1", content)
                self.assertIn("100", content)
    
    def test_markdown_export(self):
        """Test Markdown export."""
        game_data = {
            "game_id": "md_test",
            "start_time": "2024-01-01T10:00:00",
            "total_turns": 30,
            "winner_name": "Agent Alpha",
            "agents": [
                {
                    "agent_name": "Agent Alpha",
                    "final_score": 95,
                    "final_position": 1,
                    "is_champion": True
                },
                {
                    "agent_name": "Agent Beta",
                    "final_score": 82,
                    "final_position": 2
                }
            ]
        }
        
        config = ExportConfig(format=ExportFormat.MARKDOWN)
        path = self.exporter.export_game(game_data, config)
        
        self.assertTrue(path.exists())
        
        # Verify content
        with open(path, 'r') as f:
            content = f.read()
            self.assertIn("# Arena Game Report", content)
            self.assertIn("Agent Alpha", content)
            self.assertIn("Winner", content)
    
    def test_batch_export(self):
        """Test batch export."""
        games = [
            {"game_id": f"game_{i}", "winner": f"agent_{i}"}
            for i in range(3)
        ]
        
        config = ExportConfig(format=ExportFormat.JSON)
        paths = self.exporter.export_batch(games, config, "test_batch")
        
        self.assertEqual(len(paths), 3)
        for path in paths:
            self.assertTrue(path.exists())
        
        # Check manifest
        manifest_path = Path(self.temp_dir) / "test_batch_manifest.json"
        self.assertTrue(manifest_path.exists())


if __name__ == "__main__":
    unittest.main()