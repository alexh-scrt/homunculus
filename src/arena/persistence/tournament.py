"""
Tournament System for Arena

This module manages multi-round tournaments, brackets, seasons,
and competitive rankings across multiple games.

Features:
- Single and double elimination brackets
- Round-robin tournaments
- Swiss system tournaments
- Season management
- ELO/Glicko ratings integration
- Tournament history

Author: Homunculus Team
"""

import logging
import json
import random
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import math

logger = logging.getLogger(__name__)


class TournamentFormat(Enum):
    """Tournament format types."""
    SINGLE_ELIMINATION = "single_elimination"
    DOUBLE_ELIMINATION = "double_elimination"
    ROUND_ROBIN = "round_robin"
    SWISS = "swiss"
    LADDER = "ladder"
    LEAGUE = "league"


class MatchStatus(Enum):
    """Match status."""
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class TournamentMatch:
    """Single match in a tournament."""
    match_id: str
    round_number: int
    match_number: int
    participants: List[str]
    status: MatchStatus
    scheduled_time: Optional[datetime] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    winner: Optional[str] = None
    scores: Dict[str, float] = field(default_factory=dict)
    game_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['status'] = self.status.value
        if self.scheduled_time:
            data['scheduled_time'] = self.scheduled_time.isoformat()
        if self.start_time:
            data['start_time'] = self.start_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        return data


@dataclass
class TournamentRound:
    """Single round in a tournament."""
    round_number: int
    round_name: str
    matches: List[TournamentMatch]
    is_complete: bool = False
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def get_winners(self) -> List[str]:
        """Get round winners."""
        winners = []
        for match in self.matches:
            if match.winner:
                winners.append(match.winner)
        return winners
    
    def get_losers(self) -> List[str]:
        """Get round losers."""
        losers = []
        for match in self.matches:
            if match.winner:
                for participant in match.participants:
                    if participant != match.winner:
                        losers.append(participant)
        return losers
    
    def is_ready(self) -> bool:
        """Check if round is ready to start."""
        return all(
            len(match.participants) >= 2
            for match in self.matches
        )


@dataclass
class TournamentBracket:
    """Tournament bracket structure."""
    format: TournamentFormat
    rounds: List[TournamentRound]
    participants: List[str]
    seeds: Optional[Dict[str, int]] = None
    
    # Double elimination specific
    winners_bracket: List[TournamentRound] = field(default_factory=list)
    losers_bracket: List[TournamentRound] = field(default_factory=list)
    
    def get_current_round(self) -> Optional[TournamentRound]:
        """Get current active round."""
        for round in self.rounds:
            if not round.is_complete:
                return round
        return None
    
    def advance_winners(self, round: TournamentRound) -> None:
        """Advance winners to next round."""
        winners = round.get_winners()
        next_round_num = round.round_number + 1
        
        # Find or create next round
        next_round = None
        for r in self.rounds:
            if r.round_number == next_round_num:
                next_round = r
                break
        
        if not next_round:
            return
        
        # Add winners to next round matches
        match_idx = 0
        for i in range(0, len(winners), 2):
            if match_idx < len(next_round.matches):
                match = next_round.matches[match_idx]
                if i < len(winners):
                    match.participants.append(winners[i])
                if i + 1 < len(winners):
                    match.participants.append(winners[i + 1])
                match_idx += 1


@dataclass
class TournamentResults:
    """Tournament results and standings."""
    tournament_id: str
    winner: Optional[str]
    runner_up: Optional[str]
    standings: List[Tuple[str, int]]  # (participant, position)
    total_matches: int
    completed_matches: int
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    def get_top_n(self, n: int) -> List[str]:
        """Get top N finishers."""
        return [p for p, _ in self.standings[:n]]
    
    def get_position(self, participant: str) -> Optional[int]:
        """Get participant's final position."""
        for p, pos in self.standings:
            if p == participant:
                return pos
        return None


class TournamentBuilder:
    """
    Builds tournament brackets based on format.
    """
    
    @staticmethod
    def create_single_elimination(
        participants: List[str],
        seeded: bool = False
    ) -> TournamentBracket:
        """
        Create single elimination bracket.
        
        Args:
            participants: List of participants
            seeded: Whether to use seeding
            
        Returns:
            Tournament bracket
        """
        n = len(participants)
        rounds_needed = math.ceil(math.log2(n))
        total_slots = 2 ** rounds_needed
        
        # Create bracket
        bracket = TournamentBracket(
            format=TournamentFormat.SINGLE_ELIMINATION,
            rounds=[],
            participants=participants.copy()
        )
        
        # Shuffle or seed participants
        if seeded:
            # Simple seeding (1 vs n, 2 vs n-1, etc.)
            ordered = participants.copy()
        else:
            ordered = participants.copy()
            random.shuffle(ordered)
        
        # Add byes if needed
        byes_needed = total_slots - n
        ordered.extend([None] * byes_needed)
        
        # Create first round
        first_round_matches = []
        match_num = 0
        for i in range(0, total_slots, 2):
            p1 = ordered[i]
            p2 = ordered[i + 1] if i + 1 < len(ordered) else None
            
            # Skip if both are byes
            if p1 is None and p2 is None:
                continue
            
            # Create match
            participants = []
            if p1:
                participants.append(p1)
            if p2:
                participants.append(p2)
            
            match = TournamentMatch(
                match_id=f"r1_m{match_num}",
                round_number=1,
                match_number=match_num,
                participants=participants,
                status=MatchStatus.SCHEDULED
            )
            
            # Auto-complete bye matches
            if len(participants) == 1:
                match.winner = participants[0]
                match.status = MatchStatus.COMPLETED
            
            first_round_matches.append(match)
            match_num += 1
        
        first_round = TournamentRound(
            round_number=1,
            round_name="Round 1",
            matches=first_round_matches
        )
        bracket.rounds.append(first_round)
        
        # Create subsequent rounds
        matches_in_round = len(first_round_matches) // 2
        for round_num in range(2, rounds_needed + 1):
            round_matches = []
            
            for match_num in range(matches_in_round):
                match = TournamentMatch(
                    match_id=f"r{round_num}_m{match_num}",
                    round_number=round_num,
                    match_number=match_num,
                    participants=[],  # Will be filled as previous rounds complete
                    status=MatchStatus.SCHEDULED
                )
                round_matches.append(match)
            
            # Name the round
            if matches_in_round == 1:
                round_name = "Final"
            elif matches_in_round == 2:
                round_name = "Semi-Finals"
            elif matches_in_round == 4:
                round_name = "Quarter-Finals"
            else:
                round_name = f"Round {round_num}"
            
            round = TournamentRound(
                round_number=round_num,
                round_name=round_name,
                matches=round_matches
            )
            bracket.rounds.append(round)
            
            matches_in_round = matches_in_round // 2
        
        return bracket
    
    @staticmethod
    def create_round_robin(participants: List[str]) -> TournamentBracket:
        """
        Create round-robin tournament.
        
        Args:
            participants: List of participants
            
        Returns:
            Tournament bracket
        """
        n = len(participants)
        bracket = TournamentBracket(
            format=TournamentFormat.ROUND_ROBIN,
            rounds=[],
            participants=participants.copy()
        )
        
        # Generate all pairings
        pairings = []
        for i in range(n):
            for j in range(i + 1, n):
                pairings.append((participants[i], participants[j]))
        
        # Distribute pairings across rounds
        rounds_needed = n - 1 if n % 2 == 0 else n
        matches_per_round = n // 2
        
        for round_num in range(1, rounds_needed + 1):
            round_matches = []
            used_participants = set()
            
            for p1, p2 in pairings:
                if p1 not in used_participants and p2 not in used_participants:
                    match = TournamentMatch(
                        match_id=f"r{round_num}_m{len(round_matches)}",
                        round_number=round_num,
                        match_number=len(round_matches),
                        participants=[p1, p2],
                        status=MatchStatus.SCHEDULED
                    )
                    round_matches.append(match)
                    used_participants.add(p1)
                    used_participants.add(p2)
                    
                    if len(round_matches) >= matches_per_round:
                        break
            
            # Remove used pairings
            pairings = [(p1, p2) for p1, p2 in pairings
                       if not any(m.participants == [p1, p2] or m.participants == [p2, p1]
                                 for m in round_matches)]
            
            round = TournamentRound(
                round_number=round_num,
                round_name=f"Round {round_num}",
                matches=round_matches
            )
            bracket.rounds.append(round)
        
        return bracket
    
    @staticmethod
    def create_swiss(
        participants: List[str],
        rounds: int = 5
    ) -> TournamentBracket:
        """
        Create Swiss system tournament.
        
        Args:
            participants: List of participants
            rounds: Number of rounds
            
        Returns:
            Tournament bracket
        """
        bracket = TournamentBracket(
            format=TournamentFormat.SWISS,
            rounds=[],
            participants=participants.copy()
        )
        
        # Create empty rounds (pairings determined dynamically)
        for round_num in range(1, rounds + 1):
            round = TournamentRound(
                round_number=round_num,
                round_name=f"Round {round_num}",
                matches=[]
            )
            bracket.rounds.append(round)
        
        return bracket


class TournamentManager:
    """
    Manages tournament execution and state.
    """
    
    def __init__(self, storage_dir: str = "arena_tournaments"):
        """
        Initialize tournament manager.
        
        Args:
            storage_dir: Directory for tournament data
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Active tournaments
        self.active_tournaments: Dict[str, Dict[str, Any]] = {}
        
        # Tournament history
        self.completed_tournaments: List[str] = []
        
        # Load existing tournaments
        self._load_tournaments()
    
    def create_tournament(
        self,
        tournament_id: str,
        participants: List[str],
        format: TournamentFormat,
        **kwargs
    ) -> TournamentBracket:
        """
        Create a new tournament.
        
        Args:
            tournament_id: Tournament ID
            participants: List of participants
            format: Tournament format
            **kwargs: Format-specific options
            
        Returns:
            Tournament bracket
        """
        # Build bracket based on format
        if format == TournamentFormat.SINGLE_ELIMINATION:
            bracket = TournamentBuilder.create_single_elimination(
                participants,
                kwargs.get("seeded", False)
            )
        elif format == TournamentFormat.ROUND_ROBIN:
            bracket = TournamentBuilder.create_round_robin(participants)
        elif format == TournamentFormat.SWISS:
            bracket = TournamentBuilder.create_swiss(
                participants,
                kwargs.get("rounds", 5)
            )
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Store tournament
        self.active_tournaments[tournament_id] = {
            "bracket": bracket,
            "created_at": datetime.utcnow(),
            "status": "active",
            "config": kwargs,
            "match_results": {},
            "participant_stats": {p: {"wins": 0, "losses": 0} for p in participants}
        }
        
        # Save to disk
        self._save_tournament(tournament_id)
        
        logger.info(f"Created tournament {tournament_id} with {len(participants)} participants")
        return bracket
    
    def start_match(
        self,
        tournament_id: str,
        match_id: str
    ) -> Optional[TournamentMatch]:
        """
        Start a tournament match.
        
        Args:
            tournament_id: Tournament ID
            match_id: Match ID
            
        Returns:
            Match object
        """
        tournament = self.active_tournaments.get(tournament_id)
        if not tournament:
            return None
        
        bracket = tournament["bracket"]
        
        # Find match
        for round in bracket.rounds:
            for match in round.matches:
                if match.match_id == match_id:
                    match.status = MatchStatus.IN_PROGRESS
                    match.start_time = datetime.utcnow()
                    self._save_tournament(tournament_id)
                    return match
        
        return None
    
    def complete_match(
        self,
        tournament_id: str,
        match_id: str,
        winner: str,
        scores: Dict[str, float],
        game_id: Optional[str] = None
    ) -> bool:
        """
        Complete a tournament match.
        
        Args:
            tournament_id: Tournament ID
            match_id: Match ID
            winner: Winner participant ID
            scores: Final scores
            game_id: Associated game ID
            
        Returns:
            Success status
        """
        tournament = self.active_tournaments.get(tournament_id)
        if not tournament:
            return False
        
        bracket = tournament["bracket"]
        
        # Find and update match
        match_found = False
        for round in bracket.rounds:
            for match in round.matches:
                if match.match_id == match_id:
                    match.status = MatchStatus.COMPLETED
                    match.end_time = datetime.utcnow()
                    match.winner = winner
                    match.scores = scores
                    match.game_id = game_id
                    match_found = True
                    
                    # Update participant stats
                    for participant in match.participants:
                        if participant in tournament["participant_stats"]:
                            if participant == winner:
                                tournament["participant_stats"][participant]["wins"] += 1
                            else:
                                tournament["participant_stats"][participant]["losses"] += 1
                    
                    break
            if match_found:
                break
        
        if not match_found:
            return False
        
        # Store result
        tournament["match_results"][match_id] = {
            "winner": winner,
            "scores": scores,
            "game_id": game_id,
            "completed_at": datetime.utcnow()
        }
        
        # Check if round is complete
        self._check_round_completion(tournament_id, round)
        
        # Advance winners if needed
        if bracket.format == TournamentFormat.SINGLE_ELIMINATION:
            if round.is_complete:
                bracket.advance_winners(round)
        
        # Check if tournament is complete
        self._check_tournament_completion(tournament_id)
        
        # Save
        self._save_tournament(tournament_id)
        
        return True
    
    def _check_round_completion(
        self,
        tournament_id: str,
        round: TournamentRound
    ) -> None:
        """Check if round is complete."""
        all_complete = all(
            match.status == MatchStatus.COMPLETED
            for match in round.matches
        )
        
        if all_complete and not round.is_complete:
            round.is_complete = True
            round.end_time = datetime.utcnow()
            logger.info(f"Round {round.round_number} complete in tournament {tournament_id}")
    
    def _check_tournament_completion(self, tournament_id: str) -> None:
        """Check if tournament is complete."""
        tournament = self.active_tournaments.get(tournament_id)
        if not tournament:
            return
        
        bracket = tournament["bracket"]
        
        all_complete = all(
            round.is_complete
            for round in bracket.rounds
        )
        
        if all_complete:
            tournament["status"] = "completed"
            tournament["completed_at"] = datetime.utcnow()
            self.completed_tournaments.append(tournament_id)
            logger.info(f"Tournament {tournament_id} completed")
    
    def get_standings(self, tournament_id: str) -> Optional[TournamentResults]:
        """
        Get tournament standings.
        
        Args:
            tournament_id: Tournament ID
            
        Returns:
            Tournament results
        """
        tournament = self.active_tournaments.get(tournament_id)
        if not tournament:
            # Try loading from completed
            tournament = self._load_tournament(tournament_id)
            if not tournament:
                return None
        
        bracket = tournament["bracket"]
        stats = tournament["participant_stats"]
        
        # Calculate standings based on format
        if bracket.format == TournamentFormat.SINGLE_ELIMINATION:
            standings = self._calculate_elimination_standings(bracket)
        elif bracket.format == TournamentFormat.ROUND_ROBIN:
            standings = self._calculate_round_robin_standings(stats)
        elif bracket.format == TournamentFormat.SWISS:
            standings = self._calculate_swiss_standings(bracket, stats)
        else:
            standings = []
        
        # Determine winner and runner-up
        winner = standings[0][0] if standings else None
        runner_up = standings[1][0] if len(standings) > 1 else None
        
        # Count matches
        total_matches = sum(len(r.matches) for r in bracket.rounds)
        completed_matches = sum(
            1 for r in bracket.rounds
            for m in r.matches
            if m.status == MatchStatus.COMPLETED
        )
        
        return TournamentResults(
            tournament_id=tournament_id,
            winner=winner,
            runner_up=runner_up,
            standings=standings,
            total_matches=total_matches,
            completed_matches=completed_matches,
            statistics={
                "format": bracket.format.value,
                "participants": len(bracket.participants),
                "rounds": len(bracket.rounds)
            }
        )
    
    def _calculate_elimination_standings(
        self,
        bracket: TournamentBracket
    ) -> List[Tuple[str, int]]:
        """Calculate standings for elimination tournament."""
        standings = []
        position = 1
        
        # Work backwards from final
        for round in reversed(bracket.rounds):
            round_losers = []
            
            for match in round.matches:
                if match.winner:
                    # Add winner to standings if final round
                    if round.round_number == len(bracket.rounds):
                        standings.append((match.winner, position))
                        position += 1
                    
                    # Add losers
                    for participant in match.participants:
                        if participant != match.winner:
                            round_losers.append(participant)
            
            # Add round losers at same position
            for loser in round_losers:
                standings.append((loser, position))
            
            if round_losers:
                position += len(round_losers)
        
        return standings
    
    def _calculate_round_robin_standings(
        self,
        stats: Dict[str, Dict[str, int]]
    ) -> List[Tuple[str, int]]:
        """Calculate standings for round-robin tournament."""
        # Sort by wins (and win percentage)
        sorted_participants = sorted(
            stats.items(),
            key=lambda x: (x[1]["wins"], -x[1]["losses"]),
            reverse=True
        )
        
        standings = []
        position = 1
        prev_wins = None
        same_position_count = 0
        
        for participant, participant_stats in sorted_participants:
            wins = participant_stats["wins"]
            
            # Handle ties
            if wins == prev_wins:
                same_position_count += 1
            else:
                position += same_position_count
                same_position_count = 1
                prev_wins = wins
            
            standings.append((participant, position))
        
        return standings
    
    def _calculate_swiss_standings(
        self,
        bracket: TournamentBracket,
        stats: Dict[str, Dict[str, int]]
    ) -> List[Tuple[str, int]]:
        """Calculate standings for Swiss tournament."""
        # Similar to round-robin but may include additional tiebreakers
        return self._calculate_round_robin_standings(stats)
    
    def generate_swiss_pairings(
        self,
        tournament_id: str,
        round_number: int
    ) -> List[Tuple[str, str]]:
        """
        Generate pairings for Swiss round.
        
        Args:
            tournament_id: Tournament ID
            round_number: Round number
            
        Returns:
            List of pairings
        """
        tournament = self.active_tournaments.get(tournament_id)
        if not tournament:
            return []
        
        stats = tournament["participant_stats"]
        bracket = tournament["bracket"]
        
        # Get participants sorted by current score
        sorted_participants = sorted(
            stats.items(),
            key=lambda x: x[1]["wins"],
            reverse=True
        )
        
        # Track who has played whom
        played_pairs = set()
        for round in bracket.rounds[:round_number-1]:
            for match in round.matches:
                if len(match.participants) == 2:
                    pair = tuple(sorted(match.participants))
                    played_pairs.add(pair)
        
        # Generate pairings
        pairings = []
        used = set()
        
        for i, (p1, _) in enumerate(sorted_participants):
            if p1 in used:
                continue
            
            # Find best opponent
            for j, (p2, _) in enumerate(sorted_participants[i+1:], i+1):
                if p2 in used:
                    continue
                
                pair = tuple(sorted([p1, p2]))
                if pair not in played_pairs:
                    pairings.append((p1, p2))
                    used.add(p1)
                    used.add(p2)
                    break
        
        return pairings
    
    def _save_tournament(self, tournament_id: str) -> None:
        """Save tournament to disk."""
        tournament = self.active_tournaments.get(tournament_id)
        if not tournament:
            return
        
        file_path = self.storage_dir / f"{tournament_id}.json"
        
        # Prepare data for serialization
        data = {
            "tournament_id": tournament_id,
            "created_at": tournament["created_at"].isoformat(),
            "status": tournament["status"],
            "config": tournament["config"],
            "participant_stats": tournament["participant_stats"],
            "match_results": {}
        }
        
        # Serialize match results
        for match_id, result in tournament["match_results"].items():
            data["match_results"][match_id] = {
                "winner": result["winner"],
                "scores": result["scores"],
                "game_id": result["game_id"],
                "completed_at": result["completed_at"].isoformat()
            }
        
        # Serialize bracket
        bracket = tournament["bracket"]
        data["bracket"] = {
            "format": bracket.format.value,
            "participants": bracket.participants,
            "rounds": []
        }
        
        for round in bracket.rounds:
            round_data = {
                "round_number": round.round_number,
                "round_name": round.round_name,
                "is_complete": round.is_complete,
                "matches": [match.to_dict() for match in round.matches]
            }
            if round.start_time:
                round_data["start_time"] = round.start_time.isoformat()
            if round.end_time:
                round_data["end_time"] = round.end_time.isoformat()
            data["bracket"]["rounds"].append(round_data)
        
        # Save
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_tournament(self, tournament_id: str) -> Optional[Dict[str, Any]]:
        """Load tournament from disk."""
        file_path = self.storage_dir / f"{tournament_id}.json"
        if not file_path.exists():
            return None
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Reconstruct tournament
        tournament = {
            "created_at": datetime.fromisoformat(data["created_at"]),
            "status": data["status"],
            "config": data["config"],
            "participant_stats": data["participant_stats"],
            "match_results": {}
        }
        
        # Reconstruct match results
        for match_id, result in data["match_results"].items():
            tournament["match_results"][match_id] = {
                "winner": result["winner"],
                "scores": result["scores"],
                "game_id": result["game_id"],
                "completed_at": datetime.fromisoformat(result["completed_at"])
            }
        
        # Reconstruct bracket
        bracket_data = data["bracket"]
        bracket = TournamentBracket(
            format=TournamentFormat(bracket_data["format"]),
            rounds=[],
            participants=bracket_data["participants"]
        )
        
        for round_data in bracket_data["rounds"]:
            matches = []
            for match_data in round_data["matches"]:
                match = TournamentMatch(
                    match_id=match_data["match_id"],
                    round_number=match_data["round_number"],
                    match_number=match_data["match_number"],
                    participants=match_data["participants"],
                    status=MatchStatus(match_data["status"]),
                    winner=match_data.get("winner"),
                    scores=match_data.get("scores", {}),
                    game_id=match_data.get("game_id")
                )
                
                if match_data.get("scheduled_time"):
                    match.scheduled_time = datetime.fromisoformat(match_data["scheduled_time"])
                if match_data.get("start_time"):
                    match.start_time = datetime.fromisoformat(match_data["start_time"])
                if match_data.get("end_time"):
                    match.end_time = datetime.fromisoformat(match_data["end_time"])
                
                matches.append(match)
            
            round = TournamentRound(
                round_number=round_data["round_number"],
                round_name=round_data["round_name"],
                matches=matches,
                is_complete=round_data["is_complete"]
            )
            
            if round_data.get("start_time"):
                round.start_time = datetime.fromisoformat(round_data["start_time"])
            if round_data.get("end_time"):
                round.end_time = datetime.fromisoformat(round_data["end_time"])
            
            bracket.rounds.append(round)
        
        tournament["bracket"] = bracket
        return tournament
    
    def _load_tournaments(self) -> None:
        """Load all tournaments from disk."""
        for file_path in self.storage_dir.glob("*.json"):
            tournament_id = file_path.stem
            tournament = self._load_tournament(tournament_id)
            if tournament:
                if tournament["status"] == "completed":
                    self.completed_tournaments.append(tournament_id)
                else:
                    self.active_tournaments[tournament_id] = tournament


class SeasonManager:
    """
    Manages tournament seasons and championships.
    """
    
    def __init__(
        self,
        storage_dir: str = "arena_seasons"
    ):
        """
        Initialize season manager.
        
        Args:
            storage_dir: Directory for season data
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        self.current_season: Optional[Dict[str, Any]] = None
        self.season_history: List[Dict[str, Any]] = []
        
        # Load current season
        self._load_current_season()
    
    def start_season(
        self,
        season_id: str,
        duration_days: int = 90,
        tournament_format: TournamentFormat = TournamentFormat.LEAGUE
    ) -> Dict[str, Any]:
        """
        Start a new season.
        
        Args:
            season_id: Season ID
            duration_days: Season duration
            tournament_format: Tournament format for season
            
        Returns:
            Season data
        """
        self.current_season = {
            "season_id": season_id,
            "start_date": datetime.utcnow(),
            "end_date": datetime.utcnow() + timedelta(days=duration_days),
            "format": tournament_format,
            "tournaments": [],
            "participants": {},
            "leaderboard": {},
            "status": "active"
        }
        
        self._save_season()
        logger.info(f"Started season {season_id}")
        return self.current_season
    
    def add_tournament_to_season(
        self,
        tournament_id: str,
        results: TournamentResults
    ) -> None:
        """
        Add tournament results to season.
        
        Args:
            tournament_id: Tournament ID
            results: Tournament results
        """
        if not self.current_season:
            return
        
        # Add tournament
        self.current_season["tournaments"].append({
            "tournament_id": tournament_id,
            "winner": results.winner,
            "date": datetime.utcnow().isoformat()
        })
        
        # Update participant statistics
        for participant, position in results.standings:
            if participant not in self.current_season["participants"]:
                self.current_season["participants"][participant] = {
                    "tournaments_played": 0,
                    "wins": 0,
                    "top3": 0,
                    "points": 0
                }
            
            stats = self.current_season["participants"][participant]
            stats["tournaments_played"] += 1
            
            if position == 1:
                stats["wins"] += 1
                stats["points"] += 10
            elif position == 2:
                stats["points"] += 7
            elif position == 3:
                stats["points"] += 5
                stats["top3"] += 1
            else:
                stats["points"] += max(0, 10 - position)
            
            if position <= 3:
                stats["top3"] += 1
        
        # Update leaderboard
        self._update_leaderboard()
        
        # Save
        self._save_season()
    
    def _update_leaderboard(self) -> None:
        """Update season leaderboard."""
        if not self.current_season:
            return
        
        leaderboard = []
        for participant, stats in self.current_season["participants"].items():
            leaderboard.append({
                "participant": participant,
                "points": stats["points"],
                "wins": stats["wins"],
                "tournaments": stats["tournaments_played"]
            })
        
        # Sort by points, then wins
        leaderboard.sort(key=lambda x: (x["points"], x["wins"]), reverse=True)
        
        # Add rankings
        for i, entry in enumerate(leaderboard, 1):
            entry["rank"] = i
        
        self.current_season["leaderboard"] = leaderboard
    
    def end_season(self) -> Optional[Dict[str, Any]]:
        """
        End current season.
        
        Returns:
            Season summary
        """
        if not self.current_season:
            return None
        
        self.current_season["status"] = "completed"
        self.current_season["actual_end_date"] = datetime.utcnow().isoformat()
        
        # Determine champion
        if self.current_season["leaderboard"]:
            champion = self.current_season["leaderboard"][0]["participant"]
            self.current_season["champion"] = champion
        
        # Archive season
        self.season_history.append(self.current_season)
        self._save_season_archive(self.current_season)
        
        # Clear current season
        summary = self.current_season.copy()
        self.current_season = None
        self._save_season()
        
        return summary
    
    def get_season_standings(self) -> Optional[List[Dict[str, Any]]]:
        """
        Get current season standings.
        
        Returns:
            Season leaderboard
        """
        if not self.current_season:
            return None
        
        return self.current_season.get("leaderboard", [])
    
    def _save_season(self) -> None:
        """Save current season."""
        if not self.current_season:
            return
        
        file_path = self.storage_dir / "current_season.json"
        with open(file_path, 'w') as f:
            json.dump(self.current_season, f, indent=2, default=str)
    
    def _save_season_archive(self, season: Dict[str, Any]) -> None:
        """Archive completed season."""
        season_id = season["season_id"]
        file_path = self.storage_dir / f"season_{season_id}.json"
        with open(file_path, 'w') as f:
            json.dump(season, f, indent=2, default=str)
    
    def _load_current_season(self) -> None:
        """Load current season."""
        file_path = self.storage_dir / "current_season.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                self.current_season = json.load(f)