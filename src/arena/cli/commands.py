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
from src.arena.orchestration.game_orchestrator import OrchestratorConfig
from src.arena.config import arena_config as arena_system_config
from src.arena.game_theory import ScoringEngine, EliminationEngine
from src.arena.agents import BaseAgent, CharacterAgent, NarratorAgent
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
from src.arena.config.logging_config import setup_arena_logging

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
        mode: str,
        seed_question: str = None
    ) -> int:
        """
        Start a new game.
        
        Args:
            game_id: Game identifier
            agent_ids: List of agent IDs
            max_turns: Maximum turns
            mode: Game mode
            seed_question: Initial question/topic for discussion
            
        Returns:
            Exit code
        """
        print(f"Starting game: {game_id}")
        
        # Set up Arena logging early to avoid verbose CLI logs
        arena_logger = setup_arena_logging(game_id, "logs")
        
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
        
        # Add narrator agent automatically
        narrator = await self._create_narrator_agent(game_id, max_turns)
        agents.append(narrator)
        
        # Add judge agent automatically
        judge = await self._create_judge_agent(game_id)
        agents.append(judge)
        
        # Create game config
        mode_upper = mode.upper()
        if mode_upper not in ["COMPETITIVE", "COOPERATIVE", "MIXED"]:
            print(colored_text(f"Invalid mode: {mode}. Using COMPETITIVE.", "yellow"))
            mode_upper = "COMPETITIVE"
            
        arena_config = ArenaConfig(
            game_id=game_id,
            max_turns=max_turns,
            elimination_rate=0.2,
            game_mode=mode_upper
        )
        
        # Create orchestrator config using Arena configuration
        orchestrator_config = OrchestratorConfig(
            game_id=game_id,
            max_turns=max_turns,
            min_agents=max(len(agents), arena_system_config.min_agents),
            recursion_limit=arena_system_config.recursion_limit,
            checkpoint_frequency=arena_system_config.checkpoint_frequency,
            enable_recovery=arena_system_config.enable_recovery,
            parallel_execution=arena_system_config.parallel_execution,
            timeout_seconds=arena_system_config.timeout_seconds
        )
        
        # Create engines
        scoring_engine = ScoringEngine()
        elimination_engine = EliminationEngine()
        
        # Create orchestrator
        orchestrator = GameOrchestrator(
            config=orchestrator_config,
            agents=agents,
            scoring_engine=scoring_engine,
            elimination_engine=elimination_engine
        )
        
        # Initialize game
        try:
            # Store the arena config for later use
            orchestrator.arena_config = arena_config
            
            # Store seed question in orchestrator for agents to use
            if seed_question:
                orchestrator.seed_question = seed_question
            
            self.active_games[game_id] = orchestrator
            
            # Start analytics tracking
            self.analytics.start_game_tracking(game_id)
            
            print(colored_text(f"Game {game_id} started successfully!", "green"))
            print(f"Players: {', '.join(agent_ids)}")
            print(f"Max turns: {max_turns}")
            print(f"Mode: {mode}")
            if seed_question:
                print(f"Topic: {colored_text(seed_question, 'cyan')}")
            
            # Run game loop and wait for completion
            await self._run_game_loop(game_id)
            
            return 0
            
        except Exception as e:
            logger.error(f"Failed to start game: {e}")
            print(colored_text(f"Error starting game: {e}", "red"))
            return 1
    
    async def _load_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Load or create an agent."""
        from ..agents.base_agent import AgentConfig, AgentRole
        from ..models.homunculus_integration import HomunculusCharacterProfile
        from src.config.character_loader import CharacterLoader
        import yaml
        
        # Create character name from ID
        character_name = agent_id.replace("_", " ").title()
        
        # Create proper AgentConfig using character name
        config = AgentConfig(
            agent_id=agent_id,
            agent_name=character_name,  # Use character name instead of "Agent {id}"
            role=AgentRole.CHARACTER,
            metadata={"source": "arena_cli"}
        )
        
        # Try to load actual character profile from schemas
        try:
            character_loader = CharacterLoader()
            char_config = character_loader.load_character(agent_id)
            
            # Extract character information from loaded config
            character_profile = HomunculusCharacterProfile(
                character_name=char_config.get("character_name", character_name),
                personality_traits=char_config.get("personality_traits", ["competitive", "analytical"]),
                expertise_areas=char_config.get("expertise_areas", ["general"]),
                communication_style=char_config.get("communication_style", "direct"),
                backstory=char_config.get("backstory", f"Arena participant {agent_id}"),
                goals=char_config.get("goals", ["win the game", "demonstrate capability"])
            )
            
            logger.info(f"Loaded character profile for {agent_id}: {character_profile.character_name}")
            
        except Exception as e:
            logger.warning(f"Failed to load character profile for {agent_id}, using defaults: {e}")
            # Fallback to basic character profile
            character_profile = HomunculusCharacterProfile(
                character_name=character_name,
                personality_traits=["competitive", "analytical", "strategic"],
                expertise_areas=["general"],
                communication_style="direct",
                backstory=f"Arena participant {agent_id}",
                goals=["win the game", "demonstrate capability"]
            )
        
        return CharacterAgent(
            config=config,
            character_profile=character_profile
        )
    
    async def _create_narrator_agent(self, game_id: str, max_turns: int) -> NarratorAgent:
        """Create a narrator agent for the game."""
        from ..agents.base_agent import AgentConfig, AgentRole
        
        config = AgentConfig(
            agent_id=f"narrator_{game_id}",
            agent_name="Narrator",
            role=AgentRole.NARRATOR,
            metadata={"game_id": game_id, "max_turns": max_turns}
        )
        
        narrator = NarratorAgent(config)
        # Store max_turns for checking against final turn
        narrator._max_turns = max_turns
        return narrator
    
    async def _create_judge_agent(self, game_id: str):
        """Create a judge agent for the game."""
        from ..agents.base_agent import AgentConfig, AgentRole
        from ..agents.judge_agent import JudgeAgent
        
        config = AgentConfig(
            agent_id=f"judge_{game_id}",
            agent_name="Judge",
            role=AgentRole.JUDGE,
            metadata={"game_id": game_id}
        )
        
        return JudgeAgent(config)
    
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
        # Load actual Homunculus characters from schema files
        import os
        import yaml
        from pathlib import Path
        
        self.agents = {}
        
        # Load character schemas
        schemas_dir = Path("schemas/characters")
        if schemas_dir.exists():
            for schema_file in schemas_dir.glob("*.yaml"):
                try:
                    with open(schema_file, 'r') as f:
                        character_data = yaml.safe_load(f)
                    
                    character_id = character_data.get('character_id', schema_file.stem)
                    character_name = character_data.get('name', character_id.replace('_', ' ').title())
                    
                    self.agents[character_id] = {
                        "name": character_name,
                        "type": "character",
                        "wins": 0,  # Would be loaded from database in production
                        "games": 0,  # Would be loaded from database in production
                        "archetype": character_data.get('archetype', 'unknown'),
                        "profile": {
                            "age": character_data.get('demographics', {}).get('age'),
                            "occupation": character_data.get('demographics', {}).get('occupation'),
                            "personality": character_data.get('initial_agent_states', {}).get('personality', {}).get('big_five', {}),
                            "background": character_data.get('demographics', {}).get('background', '')
                        }
                    }
                except Exception as e:
                    print(f"Warning: Failed to load character schema {schema_file}: {e}")
        
        # If no characters loaded, fall back to placeholders
        if not self.agents:
            print("Warning: No character schemas found, using placeholder agents")
            self.agents = {
                "ada_lovelace": {"name": "Ada Lovelace", "type": "character", "wins": 0, "games": 0},
                "captain_cosmos": {"name": "Captain Cosmos", "type": "character", "wins": 0, "games": 0},
                "zen_master": {"name": "Zen Master", "type": "character", "wins": 0, "games": 0}
            }
    
    async def create_agent(
        self,
        agent_id: str,
        name: str,
        agent_type: str,
        profile_file: Optional[str],
        research: bool = False
    ) -> int:
        """Create a new agent with optional automated research."""
        if agent_id in self.agents:
            print(colored_text(f"Agent {agent_id} already exists", "red"))
            return 1
        
        profile = {}
        
        # Research-based profile creation
        if research and agent_type == "character":
            try:
                print(colored_text(f"🔍 Researching character: {name}", "cyan"))
                print("This may take 30-60 seconds...")
                
                # Import here to avoid circular dependencies
                from ..agents.character_researcher import CharacterResearcher
                
                # Initialize researcher
                researcher = CharacterResearcher()
                
                # Check if this is a public figure
                is_public, confidence = await researcher.is_public_figure(name)
                
                if not is_public or confidence < 0.5:
                    print(colored_text(f"⚠️  '{name}' does not appear to be a well-known public figure.", "yellow"))
                    print(f"Confidence: {confidence:.1f}")
                    
                    # For known patterns, proceed anyway
                    well_known_patterns = ["jobs", "einstein", "shakespeare", "gandhi", "napoleon", 
                                         "mozart", "da vinci", "newton", "tesla", "edison", "bezos", 
                                         "musk", "gates", "obama", "lincoln", "churchill"]
                    
                    should_proceed = any(pattern in name.lower() for pattern in well_known_patterns)
                    
                    if should_proceed:
                        print("Proceeding with research based on name pattern...")
                    else:
                        try:
                            if not confirm_action("Continue with automated research anyway?"):
                                print("Falling back to manual agent creation...")
                                research = False
                            else:
                                print("Proceeding with research...")
                        except (EOFError, KeyboardInterrupt):
                            print("Non-interactive environment detected. Falling back to manual agent creation...")
                            research = False
                
                if research:
                    # Conduct comprehensive research
                    research_result = await researcher.research_character(name)
                    
                    print(f"Research completed! Confidence: {research_result.confidence_score:.2f}")
                    
                    if research_result.confidence_score < 0.3:
                        print(colored_text(f"⚠️  Low research quality. Consider manual creation.", "yellow"))
                        try:
                            if not confirm_action("Use research results anyway?"):
                                print("Falling back to manual agent creation...")
                                research = False
                        except (EOFError, KeyboardInterrupt):
                            print("Using research results in non-interactive environment...")
                            # Proceed with research in non-interactive mode
                    
                    if research:
                        # Generate character profile
                        print("📊 Generating character profile...")
                        character_profile = await researcher.generate_character_profile(research_result)
                        
                        # Save as YAML file
                        yaml_path = await researcher.save_character_profile(character_profile)
                        
                        print(colored_text(f"✅ Character profile saved to: {yaml_path}", "green"))
                        
                        # Display summary
                        self._display_research_summary(research_result, character_profile)
                        
                        # Use generated profile
                        profile = {
                            "research_based": True,
                            "confidence_score": research_result.confidence_score,
                            "yaml_profile_path": str(yaml_path),
                            "research_summary": research_result.to_dict()
                        }
                
            except Exception as e:
                logger.error(f"Research failed for {name}: {e}")
                print(colored_text(f"❌ Research failed: {e}", "red"))
                print("Falling back to manual agent creation...")
                research = False
        
        # Load profile if provided
        if profile_file:
            try:
                with open(profile_file, 'r') as f:
                    additional_profile = json.load(f)
                    profile.update(additional_profile)
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
            "games": 0,
            "research_enabled": research
        }
        
        success_message = f"Agent {agent_id} created successfully!"
        if research and profile.get("research_based"):
            success_message += " (with automated research)"
        
        print(colored_text(success_message, "green"))
        return 0
    
    def _display_research_summary(self, research_result, character_profile: Dict[str, Any]) -> None:
        """Display a summary of the research results."""
        print(f"\n{'='*60}")
        print(colored_text(f"🎭 Character Research Summary: {research_result.name}", "cyan"))
        print(f"{'='*60}")
        
        # Basic info
        print(f"\n📋 Basic Information:")
        demographics = character_profile.get("demographics", {})
        print(f"  Background: {demographics.get('background', 'Unknown')[:80]}...")
        print(f"  Archetype: {character_profile.get('archetype', 'Unknown')}")
        print(f"  Confidence Score: {research_result.confidence_score:.2f}/1.0")
        
        # Personality
        personality = character_profile.get("personality", {})
        traits = personality.get("behavioral_traits", [])
        if traits:
            print(f"\n🧠 Key Personality Traits:")
            for trait in traits[:5]:
                print(f"  • {trait}")
        
        # Expertise
        specialty = character_profile.get("specialty", {})
        expertise = specialty.get("subdomain_knowledge", [])
        if expertise:
            print(f"\n🎯 Areas of Expertise:")
            for area in expertise[:5]:
                print(f"  • {area}")
        
        # Achievements
        achievements = specialty.get("notable_achievements", [])
        if achievements:
            print(f"\n🏆 Notable Achievements:")
            for achievement in achievements[:3]:
                print(f"  • {achievement}")
        
        # Communication style
        comm_style = character_profile.get("communication_style", {})
        print(f"\n💬 Communication Style:")
        print(f"  Style: {comm_style.get('conversation_style', 'Unknown')}")
        print(f"  Formality: {comm_style.get('formality_level', 'Unknown')}")
        
        # Research quality indicators
        print(f"\n📊 Research Quality:")
        print(f"  Biographical Data: {'✅' if research_result.biographical_info else '❌'}")
        print(f"  Personality Traits: {'✅' if research_result.personality_traits.get('traits') else '❌'}")
        print(f"  Achievements Found: {len(research_result.achievements)}")
        print(f"  Expertise Areas: {len(research_result.expertise_areas)}")
        print(f"  Research Queries: {len(research_result.raw_research_data)}")
        
        print(f"{'='*60}\n")
    
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