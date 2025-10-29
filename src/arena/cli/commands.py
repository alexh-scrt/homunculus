"""
CLI Commands Implementation

This module implements all command handlers for the Arena CLI,
including game management, agent operations, tournaments, and analytics.

Author: Homunculus Team
"""

import asyncio
import json
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timedelta
import logging

from src.arena.orchestration import GameOrchestrator
from src.arena.agents import BaseAgent, CharacterAgent
from src.arena.persistence import (
    GameStorage,
    TournamentManager,
    ReplayManager,
    AnalyticsEngine,
    DataExporter,
    ExportConfig,
    ExportFormat,
    TournamentFormat
)
from src.arena.models import ArenaConfig, GameMode
from src.arena.cli.utils import (
    print_table,
    colored_text,
    format_duration,
    confirm_action,
    progress_bar
)

logger = logging.getLogger(__name__)


class GameCommands:
    """
    Handles game-related CLI commands.
    """
    
    def __init__(self):
        """Initialize game commands."""
        self.active_games: Dict[str, GameOrchestrator] = {}
        self.storage = GameStorage()
        self.analytics = AnalyticsEngine()
    
    async def start_game(
        self,
        game_id: str,
        agent_ids: List[str],
        max_turns: int,
        mode: str
    ) -> int:
        """
        Start a new game.
        
        Args:
            game_id: Game identifier
            agent_ids: List of agent IDs
            max_turns: Maximum turns
            mode: Game mode
            
        Returns:
            Exit code
        """
        print(f"Starting game: {game_id}")
        
        # Check if game already exists
        if game_id in self.active_games:
            print(colored_text(f"Game {game_id} already active", "red"))
            return 1
        
        # Create agents
        agents = []
        for agent_id in agent_ids:
            # Load or create agent
            agent = await self._load_agent(agent_id)
            if not agent:
                print(colored_text(f"Failed to load agent: {agent_id}", "red"))
                return 1
            agents.append(agent)
        
        # Create game config
        config = ArenaConfig(
            game_id=game_id,
            max_turns=max_turns,
            elimination_rate=0.2,
            game_mode=GameMode(mode.upper())
        )
        
        # Create orchestrator
        orchestrator = GameOrchestrator(config)
        
        # Initialize game
        try:
            await orchestrator.initialize_game(agents)
            self.active_games[game_id] = orchestrator
            
            # Start analytics tracking
            self.analytics.start_game_tracking(game_id)
            
            print(colored_text(f"Game {game_id} started successfully!", "green"))
            print(f"Players: {', '.join(agent_ids)}")
            print(f"Max turns: {max_turns}")
            print(f"Mode: {mode}")
            
            # Start game loop in background
            asyncio.create_task(self._run_game_loop(game_id))
            
            return 0
            
        except Exception as e:
            logger.error(f"Failed to start game: {e}")
            print(colored_text(f"Error starting game: {e}", "red"))
            return 1
    
    async def _load_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Load or create an agent."""
        # For now, create basic character agents
        # In production, load from database
        return CharacterAgent(
            agent_id=agent_id,
            name=f"Agent {agent_id}",
            character_profile={
                "personality": "competitive",
                "background": "AI agent",
                "goals": ["win the game"]
            }
        )
    
    async def _run_game_loop(self, game_id: str) -> None:
        """Run game loop in background."""
        orchestrator = self.active_games.get(game_id)
        if not orchestrator:
            return
        
        try:
            # Run game
            result = await orchestrator.run_game()
            
            # Finalize analytics
            if result:
                self.analytics.finalize_game(
                    result.get("final_scores", {}),
                    result.get("winner")
                )
            
            print(f"\nGame {game_id} completed!")
            if result.get("winner"):
                print(f"Winner: {result['winner']}")
                
        except Exception as e:
            logger.error(f"Game loop error: {e}")
            print(colored_text(f"Game error: {e}", "red"))
        finally:
            # Remove from active games
            self.active_games.pop(game_id, None)
    
    async def stop_game(self, game_id: str) -> int:
        """Stop a running game."""
        if game_id not in self.active_games:
            print(colored_text(f"Game {game_id} not found", "red"))
            return 1
        
        if not confirm_action(f"Stop game {game_id}?"):
            print("Cancelled")
            return 0
        
        # Stop game
        orchestrator = self.active_games[game_id]
        try:
            await orchestrator.stop_game()
            del self.active_games[game_id]
            print(colored_text(f"Game {game_id} stopped", "yellow"))
            return 0
        except Exception as e:
            print(colored_text(f"Error stopping game: {e}", "red"))
            return 1
    
    async def save_game(self, game_id: str, save_name: Optional[str]) -> int:
        """Save game state."""
        if game_id not in self.active_games:
            print(colored_text(f"Game {game_id} not active", "red"))
            return 1
        
        orchestrator = self.active_games[game_id]
        game_state = orchestrator.get_game_state()
        
        if not save_name:
            save_name = f"{game_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            metadata = self.storage.save_game(game_state, save_name)
            print(colored_text(f"Game saved as: {metadata.save_id}", "green"))
            return 0
        except Exception as e:
            print(colored_text(f"Error saving game: {e}", "red"))
            return 1
    
    async def load_game(self, save_id: str) -> int:
        """Load a saved game."""
        try:
            game_state = self.storage.load_game(save_id)
            if not game_state:
                print(colored_text(f"Save {save_id} not found", "red"))
                return 1
            
            game_id = game_state.get("game_id", save_id)
            
            # Check if already active
            if game_id in self.active_games:
                print(colored_text(f"Game {game_id} already active", "red"))
                return 1
            
            # Create orchestrator and restore state
            config = ArenaConfig(game_id=game_id)
            orchestrator = GameOrchestrator(config)
            orchestrator.restore_state(game_state)
            
            self.active_games[game_id] = orchestrator
            
            print(colored_text(f"Game {game_id} loaded successfully", "green"))
            print(f"Turn: {game_state.get('current_turn', 0)}")
            print(f"Active agents: {len(game_state.get('active_agents', []))}")
            
            return 0
            
        except Exception as e:
            print(colored_text(f"Error loading game: {e}", "red"))
            return 1
    
    async def list_games(self, status: str) -> int:
        """List games."""
        games = []
        
        # Active games
        if status in ["active", "all"]:
            for game_id, orchestrator in self.active_games.items():
                state = orchestrator.get_game_state()
                games.append({
                    "ID": game_id,
                    "Status": "Active",
                    "Turn": state.get("current_turn", 0),
                    "Players": len(state.get("active_agents", [])),
                    "Phase": state.get("phase", "Unknown")
                })
        
        # Saved games
        if status in ["completed", "all"]:
            for save in self.storage.list_saves():
                if save.save_type == "manual":
                    games.append({
                        "ID": save.game_id,
                        "Status": "Saved",
                        "Turn": save.game_turn,
                        "Players": save.active_agents,
                        "Saved": save.created_at.strftime("%Y-%m-%d %H:%M")
                    })
        
        if not games:
            print("No games found")
            return 0
        
        # Print table
        print_table(games)
        return 0
    
    async def watch_game(self, game_id: str, follow: bool) -> int:
        """Watch a game in progress."""
        if game_id not in self.active_games:
            print(colored_text(f"Game {game_id} not found", "red"))
            return 1
        
        orchestrator = self.active_games[game_id]
        
        print(f"Watching game: {game_id}")
        print("Press Ctrl+C to stop watching\n")
        
        try:
            last_turn = -1
            while game_id in self.active_games:
                state = orchestrator.get_game_state()
                current_turn = state.get("current_turn", 0)
                
                # Print updates if turn changed
                if current_turn > last_turn:
                    self._print_game_status(state)
                    last_turn = current_turn
                
                if not follow:
                    break
                
                # Wait a bit before checking again
                await asyncio.sleep(2)
                
        except KeyboardInterrupt:
            print("\nStopped watching")
        
        return 0
    
    def _print_game_status(self, state: Dict[str, Any]) -> None:
        """Print game status."""
        print(f"\n{'='*60}")
        print(f"Turn {state.get('current_turn', 0)} - Phase: {state.get('phase', 'Unknown')}")
        print(f"Active Agents: {len(state.get('active_agents', []))}")
        
        # Show scores
        scores = state.get('scores', {})
        if scores:
            print("\nScores:")
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for agent, score in sorted_scores[:5]:
                print(f"  {agent}: {score:.2f}")
        
        # Show recent eliminations
        eliminated = state.get('eliminated_this_turn', [])
        if eliminated:
            print(f"\nEliminated: {', '.join(eliminated)}")


class AgentCommands:
    """
    Handles agent-related CLI commands.
    """
    
    def __init__(self):
        """Initialize agent commands."""
        self.agents: Dict[str, Dict[str, Any]] = {}
        self._load_agents()
    
    def _load_agents(self) -> None:
        """Load agents from storage."""
        # In production, load from database
        # For now, use some defaults
        self.agents = {
            "alpha": {"name": "Agent Alpha", "type": "character", "wins": 5},
            "beta": {"name": "Agent Beta", "type": "character", "wins": 3},
            "gamma": {"name": "Agent Gamma", "type": "llm", "wins": 7}
        }
    
    async def create_agent(
        self,
        agent_id: str,
        name: str,
        agent_type: str,
        profile_file: Optional[str]
    ) -> int:
        """Create a new agent."""
        if agent_id in self.agents:
            print(colored_text(f"Agent {agent_id} already exists", "red"))
            return 1
        
        # Load profile if provided
        profile = {}
        if profile_file:
            try:
                with open(profile_file, 'r') as f:
                    profile = json.load(f)
            except Exception as e:
                print(colored_text(f"Error loading profile: {e}", "red"))
                return 1
        
        # Create agent
        self.agents[agent_id] = {
            "name": name,
            "type": agent_type,
            "profile": profile,
            "created": datetime.now().isoformat(),
            "wins": 0,
            "games": 0
        }
        
        print(colored_text(f"Agent {agent_id} created successfully!", "green"))
        return 0
    
    async def list_agents(self, agent_type: Optional[str]) -> int:
        """List available agents."""
        agents = []
        
        for agent_id, info in self.agents.items():
            if agent_type and info.get("type") != agent_type:
                continue
            
            agents.append({
                "ID": agent_id,
                "Name": info.get("name", "Unknown"),
                "Type": info.get("type", "unknown"),
                "Wins": info.get("wins", 0),
                "Games": info.get("games", 0),
                "Win Rate": f"{info.get('wins', 0) / max(info.get('games', 1), 1) * 100:.1f}%"
            })
        
        if not agents:
            print("No agents found")
            return 0
        
        print_table(agents)
        return 0
    
    async def show_agent_info(self, agent_id: str) -> int:
        """Show detailed agent information."""
        if agent_id not in self.agents:
            print(colored_text(f"Agent {agent_id} not found", "red"))
            return 1
        
        info = self.agents[agent_id]
        
        print(f"\n{'='*60}")
        print(f"Agent: {info.get('name', agent_id)}")
        print(f"ID: {agent_id}")
        print(f"Type: {info.get('type', 'unknown')}")
        print(f"Created: {info.get('created', 'Unknown')}")
        print(f"\nPerformance:")
        print(f"  Games Played: {info.get('games', 0)}")
        print(f"  Wins: {info.get('wins', 0)}")
        print(f"  Win Rate: {info.get('wins', 0) / max(info.get('games', 1), 1) * 100:.1f}%")
        
        if info.get('profile'):
            print(f"\nProfile:")
            for key, value in info['profile'].items():
                print(f"  {key}: {value}")
        
        print(f"{'='*60}\n")
        return 0
    
    async def show_agent_stats(self, agent_id: str) -> int:
        """Show agent statistics."""
        # In production, query from analytics engine
        print(f"\nStatistics for {agent_id}:")
        print("  Average Score: 85.3")
        print("  Best Score: 112.5")
        print("  Survival Rate: 73.2%")
        print("  Favorite Strategy: Cooperative")
        print("  Nemesis: Agent Beta")
        print("  Best Ally: Agent Delta")
        return 0


class TournamentCommands:
    """
    Handles tournament-related CLI commands.
    """
    
    def __init__(self):
        """Initialize tournament commands."""
        self.tournament_mgr = TournamentManager()
    
    async def create_tournament(
        self,
        tournament_id: str,
        agents: List[str],
        format: str
    ) -> int:
        """Create a new tournament."""
        # Map format string
        format_map = {
            "single-elim": TournamentFormat.SINGLE_ELIMINATION,
            "double-elim": TournamentFormat.DOUBLE_ELIMINATION,
            "round-robin": TournamentFormat.ROUND_ROBIN,
            "swiss": TournamentFormat.SWISS
        }
        
        tournament_format = format_map.get(format)
        if not tournament_format:
            print(colored_text(f"Invalid format: {format}", "red"))
            return 1
        
        try:
            bracket = self.tournament_mgr.create_tournament(
                tournament_id,
                agents,
                tournament_format
            )
            
            print(colored_text(f"Tournament {tournament_id} created!", "green"))
            print(f"Format: {format}")
            print(f"Participants: {len(agents)}")
            print(f"Rounds: {len(bracket.rounds)}")
            
            # Show first round matches
            first_round = bracket.rounds[0] if bracket.rounds else None
            if first_round:
                print(f"\nFirst Round Matches:")
                for match in first_round.matches[:5]:
                    if len(match.participants) >= 2:
                        print(f"  {match.participants[0]} vs {match.participants[1]}")
                
                if len(first_round.matches) > 5:
                    print(f"  ... and {len(first_round.matches) - 5} more matches")
            
            return 0
            
        except Exception as e:
            print(colored_text(f"Error creating tournament: {e}", "red"))
            return 1
    
    async def show_status(self, tournament_id: str) -> int:
        """Show tournament status."""
        standings = self.tournament_mgr.get_standings(tournament_id)
        
        if not standings:
            print(colored_text(f"Tournament {tournament_id} not found", "red"))
            return 1
        
        print(f"\n{'='*60}")
        print(f"Tournament: {tournament_id}")
        print(f"Status: {standings.completed_matches}/{standings.total_matches} matches completed")
        
        if standings.winner:
            print(f"Winner: {colored_text(standings.winner, 'gold')}")
        
        print(f"\nCurrent Standings:")
        for i, (agent, pos) in enumerate(standings.standings[:10], 1):
            if i == 1:
                print(f"  {i}. {colored_text(agent, 'gold')} (Champion)")
            elif i == 2:
                print(f"  {i}. {colored_text(agent, 'silver')}")
            elif i == 3:
                print(f"  {i}. {colored_text(agent, 'bronze')}")
            else:
                print(f"  {i}. {agent}")
        
        print(f"{'='*60}\n")
        return 0
    
    async def show_bracket(self, tournament_id: str) -> int:
        """Show tournament bracket."""
        # In production, generate visual bracket
        print(f"\nTournament Bracket: {tournament_id}")
        print("(Visual bracket would be displayed here)")
        print("Use 'tournament-status' for current standings")
        return 0


class ReplayCommands:
    """
    Handles replay-related CLI commands.
    """
    
    def __init__(self):
        """Initialize replay commands."""
        self.replay_mgr = ReplayManager()
    
    async def list_replays(self, limit: int) -> int:
        """List available replays."""
        replays = self.replay_mgr.list_replays()[:limit]
        
        if not replays:
            print("No replays found")
            return 0
        
        table_data = []
        for replay in replays:
            table_data.append({
                "Game ID": replay['game_id'],
                "Date": replay['created'].strftime("%Y-%m-%d %H:%M"),
                "Turns": replay['turns'],
                "Winner": replay.get('winner', 'N/A'),
                "Size": f"{replay['size'] / 1024:.1f}KB"
            })
        
        print_table(table_data)
        return 0
    
    async def play_replay(self, replay_id: str, speed: float) -> int:
        """Play a game replay."""
        viewer = self.replay_mgr.load_replay(replay_id)
        
        if not viewer:
            print(colored_text(f"Replay {replay_id} not found", "red"))
            return 1
        
        print(f"Playing replay: {replay_id}")
        print(f"Speed: {speed}x")
        print("Controls: [Space] pause/play, [Q] quit, [<] prev, [>] next\n")
        
        # Simple replay loop
        frame_delay = 1.0 / speed if speed > 0 else 0
        
        try:
            while True:
                frame = viewer.get_current_frame()
                if not frame:
                    print("End of replay")
                    break
                
                # Display frame
                print(f"\nFrame {frame.frame_number} / Turn {frame.turn_number}")
                print(f"Active: {len(frame.active_agents)} agents")
                if frame.eliminated:
                    print(f"Eliminated: {', '.join(frame.eliminated)}")
                
                # Auto advance
                if speed > 0:
                    await asyncio.sleep(frame_delay)
                    if not viewer.next_frame():
                        break
                else:
                    # Manual control
                    input("Press Enter for next frame...")
                    if not viewer.next_frame():
                        break
                        
        except KeyboardInterrupt:
            print("\nReplay stopped")
        
        return 0
    
    async def analyze_replay(self, replay_id: str) -> int:
        """Analyze a replay."""
        viewer = self.replay_mgr.load_replay(replay_id)
        
        if not viewer:
            print(colored_text(f"Replay {replay_id} not found", "red"))
            return 1
        
        from src.arena.persistence.replay_system import ReplayAnalyzer
        analyzer = ReplayAnalyzer(viewer)
        
        # Analyze game flow
        analysis = analyzer.analyze_game_flow()
        
        print(f"\n{'='*60}")
        print(f"Game Analysis: {replay_id}")
        print(f"{'='*60}")
        
        print(f"\nGame Summary:")
        print(f"  Total Turns: {analysis['total_turns']}")
        print(f"  Total Frames: {analysis['total_frames']}")
        print(f"  Winner: {analysis.get('winner', 'N/A')}")
        
        print(f"\nElimination Timeline:")
        for event in analysis['elimination_timeline'][:5]:
            print(f"  Turn {event['turn']}: {', '.join(event['eliminated'])}")
        
        print(f"\nKey Moments:")
        key_moments = analyzer.find_key_moments()
        for moment in key_moments[:5]:
            print(f"  Turn {moment['turn']}: {moment['description']}")
        
        print(f"\nStatistics:")
        stats = analyzer.generate_statistics()
        print(f"  Average Turn Duration: {stats['average_turn_duration']:.2f}s")
        print(f"  Total Messages: {stats['total_messages']}")
        print(f"  Elimination Rate: {stats['elimination_rate']:.2f} per turn")
        
        print(f"{'='*60}\n")
        return 0


class StatsCommands:
    """
    Handles statistics and analytics CLI commands.
    """
    
    def __init__(self):
        """Initialize stats commands."""
        self.analytics = AnalyticsEngine()
        self.exporter = DataExporter()
    
    async def show_stats(self, period: str) -> int:
        """Show overall statistics."""
        report = self.analytics.generate_report()
        
        print(f"\n{'='*60}")
        print(f"Arena Statistics ({period})")
        print(f"{'='*60}")
        
        summary = report.get('summary', {})
        global_stats = summary.get('global', {})
        
        print(f"\nGlobal Statistics:")
        print(f"  Total Games: {global_stats.get('total_games', 0)}")
        print(f"  Average Turns: {global_stats.get('average_turns', 0):.1f}")
        print(f"  Average Duration: {global_stats.get('average_duration_seconds', 0):.1f}s")
        print(f"  Unique Agents: {global_stats.get('total_unique_agents', 0)}")
        
        print(f"\nTop Winners:")
        for agent, wins in global_stats.get('most_wins', [])[:5]:
            print(f"  {agent}: {wins} wins")
        
        print(f"\nGame Patterns:")
        for pattern, count in report.get('patterns', {}).items():
            print(f"  {pattern}: {count} occurrences")
        
        print(f"{'='*60}\n")
        return 0
    
    async def show_leaderboard(self, metric: str, limit: int) -> int:
        """Show agent leaderboard."""
        rankings = self.analytics.aggregator.get_agent_rankings(metric)[:limit]
        
        print(f"\n{'='*60}")
        print(f"Agent Leaderboard - {metric.title()}")
        print(f"{'='*60}")
        
        for i, (agent, score) in enumerate(rankings, 1):
            if i == 1:
                print(f"{i:2}. {colored_text(agent, 'gold'):20} {score:.2f} 🥇")
            elif i == 2:
                print(f"{i:2}. {colored_text(agent, 'silver'):20} {score:.2f} 🥈")
            elif i == 3:
                print(f"{i:2}. {colored_text(agent, 'bronze'):20} {score:.2f} 🥉")
            else:
                print(f"{i:2}. {agent:20} {score:.2f}")
        
        print(f"{'='*60}\n")
        return 0
    
    async def export_data(
        self,
        output: str,
        format: str,
        game_ids: Optional[List[str]]
    ) -> int:
        """Export game data."""
        # Map format
        format_map = {
            "json": ExportFormat.JSON,
            "csv": ExportFormat.CSV,
            "excel": ExportFormat.EXCEL,
            "html": ExportFormat.HTML
        }
        
        export_format = format_map.get(format)
        if not export_format:
            print(colored_text(f"Invalid format: {format}", "red"))
            return 1
        
        config = ExportConfig(
            format=export_format,
            include_analytics=True
        )
        
        # For demo, export sample data
        data = {
            "exported_at": datetime.now().isoformat(),
            "games": game_ids or [],
            "statistics": self.analytics.generate_report()
        }
        
        try:
            path = self.exporter.export_game(data, config, output)
            print(colored_text(f"Data exported to: {path}", "green"))
            return 0
        except Exception as e:
            print(colored_text(f"Export failed: {e}", "red"))
            return 1


class ConfigCommands:
    """
    Handles configuration CLI commands.
    """
    
    def __init__(self):
        """Initialize config commands."""
        self.config_file = Path.home() / ".arena" / "config.json"
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        # Default config
        return {
            "data_dir": str(Path.home() / ".arena" / "data"),
            "log_level": "INFO",
            "auto_save": True,
            "auto_save_interval": 10,
            "max_parallel_games": 5,
            "default_max_turns": 100,
            "theme": "default"
        }
    
    def _save_config(self) -> None:
        """Save configuration."""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def load_config(self, config_file: str) -> None:
        """Load configuration from file."""
        try:
            with open(config_file, 'r') as f:
                self.config = json.load(f)
            print(colored_text(f"Config loaded from {config_file}", "green"))
        except Exception as e:
            print(colored_text(f"Error loading config: {e}", "red"))
    
    def show_config(self, key: Optional[str]) -> int:
        """Show configuration."""
        if key:
            value = self.config.get(key)
            if value is None:
                print(colored_text(f"Key not found: {key}", "red"))
                return 1
            print(f"{key}: {value}")
        else:
            print("\nArena Configuration:")
            for k, v in self.config.items():
                print(f"  {k}: {v}")
        return 0
    
    def set_config(self, key: str, value: str) -> int:
        """Set configuration value."""
        # Try to parse value
        try:
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            elif value.isdigit():
                value = int(value)
            elif "." in value and value.replace(".", "").isdigit():
                value = float(value)
        except:
            pass
        
        self.config[key] = value
        self._save_config()
        
        print(colored_text(f"Config updated: {key} = {value}", "green"))
        return 0
    
    def reset_config(self) -> int:
        """Reset configuration to defaults."""
        if not confirm_action("Reset all configuration to defaults?"):
            print("Cancelled")
            return 0
        
        self.config = self._load_config()
        self._save_config()
        
        print(colored_text("Configuration reset to defaults", "green"))
        return 0