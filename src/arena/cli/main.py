"""
Main CLI Entry Point for Arena

This module provides the main command-line interface for Arena,
handling command parsing, execution, and interactive mode.

Author: Homunculus Team
"""

import argparse
import sys
import os
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path
import asyncio
import json

# Add Arena to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.arena.cli.commands import (
    GameCommands,
    AgentCommands,
    TournamentCommands,
    ReplayCommands,
    StatsCommands,
    ConfigCommands
)
from src.arena.cli.utils import (
    setup_logging,
    print_banner,
    print_table,
    confirm_action,
    format_duration,
    colored_text
)

logger = logging.getLogger(__name__)


class ArenaCLI:
    """
    Main Arena CLI application.
    """
    
    def __init__(self):
        """Initialize CLI."""
        self.parser = self._create_parser()
        self.game_commands = GameCommands()
        self.agent_commands = AgentCommands()
        self.tournament_commands = TournamentCommands()
        self.replay_commands = ReplayCommands()
        self.stats_commands = StatsCommands()
        self.config_commands = ConfigCommands()
        
        # Setup logging
        setup_logging()
        
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser."""
        parser = argparse.ArgumentParser(
            prog="arena",
            description="Arena - Competitive AI Agent Training System",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # Add version
        parser.add_argument(
            "--version",
            action="version",
            version="Arena v0.1.0"
        )
        
        # Add verbose flag
        parser.add_argument(
            "-v", "--verbose",
            action="store_true",
            help="Enable verbose output"
        )
        
        # Add config file
        parser.add_argument(
            "-c", "--config",
            type=str,
            help="Configuration file path"
        )
        
        # Create subparsers for commands
        subparsers = parser.add_subparsers(
            title="Commands",
            description="Available commands",
            dest="command",
            help="Command to execute"
        )
        
        # Game commands
        self._add_game_commands(subparsers)
        
        # Agent commands
        self._add_agent_commands(subparsers)
        
        # Tournament commands
        self._add_tournament_commands(subparsers)
        
        # Replay commands
        self._add_replay_commands(subparsers)
        
        # Stats commands
        self._add_stats_commands(subparsers)
        
        # Config commands
        self._add_config_commands(subparsers)
        
        # Interactive mode
        parser_interactive = subparsers.add_parser(
            "interactive",
            help="Enter interactive mode"
        )
        
        return parser
    
    def _add_game_commands(self, subparsers):
        """Add game-related commands."""
        # Start game
        parser_start = subparsers.add_parser(
            "start",
            help="Start a new game"
        )
        parser_start.add_argument(
            "game_id",
            help="Unique game identifier"
        )
        parser_start.add_argument(
            "-a", "--agents",
            type=str,
            nargs="+",
            required=True,
            help="List of agent IDs"
        )
        parser_start.add_argument(
            "-m", "--max-turns",
            type=int,
            default=100,
            help="Maximum number of turns"
        )
        parser_start.add_argument(
            "--mode",
            choices=["competitive", "cooperative", "mixed"],
            default="competitive",
            help="Game mode"
        )
        parser_start.add_argument(
            "-s", "--seed",
            type=str,
            help="Seed question or topic to start the discussion (if not provided, will prompt for input)"
        )
        
        # Stop game
        parser_stop = subparsers.add_parser(
            "stop",
            help="Stop a running game"
        )
        parser_stop.add_argument(
            "game_id",
            help="Game ID to stop"
        )
        
        # Save game
        parser_save = subparsers.add_parser(
            "save",
            help="Save current game state"
        )
        parser_save.add_argument(
            "game_id",
            help="Game ID to save"
        )
        parser_save.add_argument(
            "-n", "--name",
            help="Save name"
        )
        
        # Load game
        parser_load = subparsers.add_parser(
            "load",
            help="Load a saved game"
        )
        parser_load.add_argument(
            "save_id",
            help="Save ID or file path"
        )
        
        # List games
        parser_list = subparsers.add_parser(
            "list",
            help="List games"
        )
        parser_list.add_argument(
            "--status",
            choices=["active", "completed", "all"],
            default="all",
            help="Filter by status"
        )
        
        # Watch game
        parser_watch = subparsers.add_parser(
            "watch",
            help="Watch a game in progress"
        )
        parser_watch.add_argument(
            "game_id",
            help="Game ID to watch"
        )
        parser_watch.add_argument(
            "--follow",
            action="store_true",
            help="Follow game in real-time"
        )
    
    def _add_agent_commands(self, subparsers):
        """Add agent-related commands."""
        # Create agent
        parser_create = subparsers.add_parser(
            "agent-create",
            help="Create a new agent"
        )
        parser_create.add_argument(
            "agent_id",
            help="Unique agent identifier"
        )
        parser_create.add_argument(
            "-n", "--name",
            required=True,
            help="Agent display name"
        )
        parser_create.add_argument(
            "-t", "--type",
            choices=["llm", "character", "narrator", "judge"],
            default="character",
            help="Agent type"
        )
        parser_create.add_argument(
            "--profile",
            type=str,
            help="Character profile JSON file"
        )
        parser_create.add_argument(
            "--research",
            action="store_true",
            help="Automatically research the character using web search (for public figures)"
        )
        
        # List agents
        parser_agents = subparsers.add_parser(
            "agents",
            help="List available agents"
        )
        parser_agents.add_argument(
            "--type",
            help="Filter by agent type"
        )
        
        # Agent info
        parser_info = subparsers.add_parser(
            "agent-info",
            help="Show agent information"
        )
        parser_info.add_argument(
            "agent_id",
            help="Agent ID"
        )
        
        # Agent stats
        parser_stats = subparsers.add_parser(
            "agent-stats",
            help="Show agent statistics"
        )
        parser_stats.add_argument(
            "agent_id",
            help="Agent ID"
        )
    
    def _add_tournament_commands(self, subparsers):
        """Add tournament commands."""
        # Create tournament
        parser_tournament = subparsers.add_parser(
            "tournament",
            help="Create a tournament"
        )
        parser_tournament.add_argument(
            "tournament_id",
            help="Tournament ID"
        )
        parser_tournament.add_argument(
            "-a", "--agents",
            nargs="+",
            required=True,
            help="List of participating agents"
        )
        parser_tournament.add_argument(
            "-f", "--format",
            choices=["single-elim", "double-elim", "round-robin", "swiss"],
            default="single-elim",
            help="Tournament format"
        )
        
        # Tournament status
        parser_tstatus = subparsers.add_parser(
            "tournament-status",
            help="Show tournament status"
        )
        parser_tstatus.add_argument(
            "tournament_id",
            help="Tournament ID"
        )
        
        # Tournament bracket
        parser_bracket = subparsers.add_parser(
            "bracket",
            help="Show tournament bracket"
        )
        parser_bracket.add_argument(
            "tournament_id",
            help="Tournament ID"
        )
    
    def _add_replay_commands(self, subparsers):
        """Add replay commands."""
        # List replays
        parser_replays = subparsers.add_parser(
            "replays",
            help="List available replays"
        )
        parser_replays.add_argument(
            "--limit",
            type=int,
            default=10,
            help="Number of replays to show"
        )
        
        # Play replay
        parser_play = subparsers.add_parser(
            "replay",
            help="Play a game replay"
        )
        parser_play.add_argument(
            "replay_id",
            help="Replay ID or game ID"
        )
        parser_play.add_argument(
            "-s", "--speed",
            type=float,
            default=1.0,
            help="Playback speed"
        )
        
        # Analyze replay
        parser_analyze = subparsers.add_parser(
            "analyze",
            help="Analyze a game replay"
        )
        parser_analyze.add_argument(
            "replay_id",
            help="Replay ID to analyze"
        )
    
    def _add_stats_commands(self, subparsers):
        """Add statistics commands."""
        # Overall stats
        parser_stats = subparsers.add_parser(
            "stats",
            help="Show overall statistics"
        )
        parser_stats.add_argument(
            "--period",
            choices=["day", "week", "month", "all"],
            default="all",
            help="Time period"
        )
        
        # Leaderboard
        parser_leader = subparsers.add_parser(
            "leaderboard",
            help="Show agent leaderboard"
        )
        parser_leader.add_argument(
            "--metric",
            choices=["wins", "score", "performance"],
            default="performance",
            help="Ranking metric"
        )
        parser_leader.add_argument(
            "--limit",
            type=int,
            default=10,
            help="Number of agents to show"
        )
        
        # Export data
        parser_export = subparsers.add_parser(
            "export",
            help="Export game data"
        )
        parser_export.add_argument(
            "output",
            help="Output file path"
        )
        parser_export.add_argument(
            "--format",
            choices=["json", "csv", "excel", "html"],
            default="json",
            help="Export format"
        )
        parser_export.add_argument(
            "--games",
            nargs="+",
            help="Game IDs to export"
        )
    
    def _add_config_commands(self, subparsers):
        """Add configuration commands."""
        # Show config
        parser_config = subparsers.add_parser(
            "config",
            help="Show configuration"
        )
        parser_config.add_argument(
            "key",
            nargs="?",
            help="Configuration key to show"
        )
        
        # Set config
        parser_set = subparsers.add_parser(
            "config-set",
            help="Set configuration value"
        )
        parser_set.add_argument(
            "key",
            help="Configuration key"
        )
        parser_set.add_argument(
            "value",
            help="Configuration value"
        )
        
        # Reset config
        parser_reset = subparsers.add_parser(
            "config-reset",
            help="Reset configuration to defaults"
        )
    
    async def run(self, args: Optional[List[str]] = None) -> int:
        """
        Run the CLI application.
        
        Args:
            args: Command line arguments
            
        Returns:
            Exit code
        """
        # Parse arguments
        parsed_args = self.parser.parse_args(args)
        
        # Set verbose mode
        if parsed_args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Load config if provided
        if parsed_args.config:
            self.config_commands.load_config(parsed_args.config)
        
        # Handle commands
        if not parsed_args.command:
            print_banner()
            self.parser.print_help()
            return 0
        
        try:
            # Game commands
            if parsed_args.command == "start":
                # Handle seed question - prompt user if not provided
                seed_question = parsed_args.seed
                if not seed_question:
                    seed_question = self._prompt_for_seed_question()
                
                return await self.game_commands.start_game(
                    parsed_args.game_id,
                    parsed_args.agents,
                    parsed_args.max_turns,
                    parsed_args.mode,
                    seed_question
                )
            
            elif parsed_args.command == "stop":
                return await self.game_commands.stop_game(parsed_args.game_id)
            
            elif parsed_args.command == "save":
                return await self.game_commands.save_game(
                    parsed_args.game_id,
                    parsed_args.name
                )
            
            elif parsed_args.command == "load":
                return await self.game_commands.load_game(parsed_args.save_id)
            
            elif parsed_args.command == "list":
                return await self.game_commands.list_games(parsed_args.status)
            
            elif parsed_args.command == "watch":
                return await self.game_commands.watch_game(
                    parsed_args.game_id,
                    parsed_args.follow
                )
            
            # Agent commands
            elif parsed_args.command == "agent-create":
                return await self.agent_commands.create_agent(
                    parsed_args.agent_id,
                    parsed_args.name,
                    parsed_args.type,
                    parsed_args.profile,
                    parsed_args.research
                )
            
            elif parsed_args.command == "agents":
                return await self.agent_commands.list_agents(parsed_args.type)
            
            elif parsed_args.command == "agent-info":
                return await self.agent_commands.show_agent_info(parsed_args.agent_id)
            
            elif parsed_args.command == "agent-stats":
                return await self.agent_commands.show_agent_stats(parsed_args.agent_id)
            
            # Tournament commands
            elif parsed_args.command == "tournament":
                return await self.tournament_commands.create_tournament(
                    parsed_args.tournament_id,
                    parsed_args.agents,
                    parsed_args.format
                )
            
            elif parsed_args.command == "tournament-status":
                return await self.tournament_commands.show_status(
                    parsed_args.tournament_id
                )
            
            elif parsed_args.command == "bracket":
                return await self.tournament_commands.show_bracket(
                    parsed_args.tournament_id
                )
            
            # Replay commands
            elif parsed_args.command == "replays":
                return await self.replay_commands.list_replays(parsed_args.limit)
            
            elif parsed_args.command == "replay":
                return await self.replay_commands.play_replay(
                    parsed_args.replay_id,
                    parsed_args.speed
                )
            
            elif parsed_args.command == "analyze":
                return await self.replay_commands.analyze_replay(parsed_args.replay_id)
            
            # Stats commands
            elif parsed_args.command == "stats":
                return await self.stats_commands.show_stats(parsed_args.period)
            
            elif parsed_args.command == "leaderboard":
                return await self.stats_commands.show_leaderboard(
                    parsed_args.metric,
                    parsed_args.limit
                )
            
            elif parsed_args.command == "export":
                return await self.stats_commands.export_data(
                    parsed_args.output,
                    parsed_args.format,
                    parsed_args.games
                )
            
            # Config commands
            elif parsed_args.command == "config":
                return self.config_commands.show_config(parsed_args.key)
            
            elif parsed_args.command == "config-set":
                return self.config_commands.set_config(
                    parsed_args.key,
                    parsed_args.value
                )
            
            elif parsed_args.command == "config-reset":
                return self.config_commands.reset_config()
            
            # Interactive mode
            elif parsed_args.command == "interactive":
                return await self.run_interactive()
            
            else:
                print(f"Unknown command: {parsed_args.command}")
                return 1
                
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            return 130
        except Exception as e:
            logger.error(f"Error executing command: {e}", exc_info=True)
            print(f"Error: {e}")
            return 1
    
    async def run_interactive(self) -> int:
        """
        Run interactive mode.
        
        Returns:
            Exit code
        """
        print_banner()
        print("Welcome to Arena Interactive Mode!")
        print("Type 'help' for available commands or 'quit' to exit.\n")
        
        while True:
            try:
                # Get input
                command = input("arena> ").strip()
                
                if not command:
                    continue
                
                # Check for exit
                if command.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    return 0
                
                # Check for help
                if command.lower() == "help":
                    self.parser.print_help()
                    continue
                
                # Parse and execute command
                args = command.split()
                result = await self.run(args)
                
                if result != 0:
                    print(f"Command failed with code {result}")
                    
            except KeyboardInterrupt:
                print("\nUse 'quit' to exit")
                continue
            except EOFError:
                print("\nGoodbye!")
                return 0
            except Exception as e:
                print(f"Error: {e}")
                continue
    
    def _prompt_for_seed_question(self) -> str:
        """
        Prompt user for a seed question or topic to start the discussion.
        
        Returns:
            Seed question/topic string
        """
        print("\n" + "="*60)
        print("🌱 SEED QUESTION / TOPIC")
        print("="*60)
        print("Please provide a question or topic to start the Arena discussion.")
        print("This will set the context for the agents to debate and explore.")
        print("\nExamples:")
        print("  • What is the future of artificial intelligence?")
        print("  • How should we approach climate change?")
        print("  • Design a perfect city for the year 2050")
        print("  • What makes a good leader?")
        print("  • Should we colonize Mars?")
        print("\n" + "-"*60)
        
        while True:
            try:
                seed_input = input("Enter your question/topic: ").strip()
                
                if not seed_input:
                    print("Please enter a valid question or topic.")
                    continue
                
                # Confirm the input
                print(f"\nYou entered: {colored_text(seed_input, 'cyan')}")
                confirm = input("Is this correct? (y/n): ").strip().lower()
                
                if confirm in ['y', 'yes']:
                    print(f"\n✅ Great! Starting game with topic: {colored_text(seed_input, 'green')}")
                    return seed_input
                else:
                    print("Let's try again...\n")
                    continue
                    
            except KeyboardInterrupt:
                print("\n\nGame cancelled by user.")
                sys.exit(130)
            except EOFError:
                print("\n\nGame cancelled.")
                sys.exit(130)


def main():
    """Main entry point."""
    cli = ArenaCLI()
    
    # Run async main
    try:
        exit_code = asyncio.run(cli.run())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()