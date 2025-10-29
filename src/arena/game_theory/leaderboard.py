"""
Leaderboard and Ranking System for Arena

This module implements comprehensive leaderboard and ranking
systems for tracking agent performance across games.

Features:
- Multiple ranking algorithms (Elo, Glicko, TrueSkill-like)
- Performance metrics tracking
- Historical rankings
- Tournament brackets
- Achievement system

Author: Homunculus Team
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class RankingAlgorithm(Enum):
    """Available ranking algorithms."""
    ELO = "elo"
    GLICKO = "glicko"
    POINTS = "points"
    WEIGHTED = "weighted"


@dataclass
class PlayerRating:
    """Player rating information."""
    player_id: str
    rating: float
    games_played: int
    wins: int
    losses: int
    draws: int
    peak_rating: float
    rating_deviation: float = 350.0  # For Glicko
    volatility: float = 0.06  # For Glicko-2
    last_game: Optional[datetime] = None
    
    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.games_played == 0:
            return 0.0
        return self.wins / self.games_played
    
    @property
    def performance_rating(self) -> float:
        """Calculate performance rating."""
        # Weighted combination of rating and win rate
        return self.rating * 0.7 + (self.win_rate * 1000) * 0.3
    
    def update_peak(self) -> None:
        """Update peak rating if current is higher."""
        if self.rating > self.peak_rating:
            self.peak_rating = self.rating


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for a player."""
    player_id: str
    
    # Basic stats
    total_score: float = 0.0
    average_score: float = 0.0
    score_variance: float = 0.0
    
    # Game outcomes
    eliminations_survived: int = 0
    elimination_position: List[int] = field(default_factory=list)
    champion_count: int = 0
    
    # Contribution metrics
    contribution_quality: float = 0.0
    innovation_score: float = 0.0
    collaboration_score: float = 0.0
    
    # Strategic metrics
    successful_accusations: int = 0
    failed_accusations: int = 0
    times_accused: int = 0
    
    # Temporal metrics
    improvement_rate: float = 0.0
    consistency_score: float = 0.0
    
    def calculate_composite_score(self) -> float:
        """Calculate composite performance score."""
        return (
            self.average_score * 0.3 +
            self.contribution_quality * 0.2 +
            (self.champion_count * 100) * 0.2 +
            self.consistency_score * 0.15 +
            self.innovation_score * 0.15
        )


class EloRating:
    """
    Elo rating system implementation.
    
    Classic chess rating system adapted for multi-player games.
    """
    
    def __init__(
        self,
        initial_rating: float = 1500,
        k_factor: float = 32,
        use_dynamic_k: bool = True
    ):
        """
        Initialize Elo rating system.
        
        Args:
            initial_rating: Starting rating for new players
            k_factor: K-factor for rating changes
            use_dynamic_k: Use dynamic K-factor based on games played
        """
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.use_dynamic_k = use_dynamic_k
        
        self.ratings: Dict[str, PlayerRating] = {}
    
    def get_rating(self, player_id: str) -> float:
        """Get player rating."""
        if player_id not in self.ratings:
            self.initialize_player(player_id)
        return self.ratings[player_id].rating
    
    def initialize_player(self, player_id: str) -> PlayerRating:
        """Initialize new player."""
        rating = PlayerRating(
            player_id=player_id,
            rating=self.initial_rating,
            games_played=0,
            wins=0,
            losses=0,
            draws=0,
            peak_rating=self.initial_rating
        )
        self.ratings[player_id] = rating
        return rating
    
    def update_ratings(
        self,
        winner: str,
        loser: str,
        draw: bool = False
    ) -> Tuple[float, float]:
        """
        Update ratings after a game.
        
        Args:
            winner: Winner's ID
            loser: Loser's ID
            draw: Whether game was a draw
            
        Returns:
            New ratings for (winner, loser)
        """
        # Ensure players exist
        if winner not in self.ratings:
            self.initialize_player(winner)
        if loser not in self.ratings:
            self.initialize_player(loser)
        
        winner_rating = self.ratings[winner]
        loser_rating = self.ratings[loser]
        
        # Calculate expected scores
        winner_expected = self._expected_score(
            winner_rating.rating,
            loser_rating.rating
        )
        loser_expected = 1 - winner_expected
        
        # Actual scores
        if draw:
            winner_actual = 0.5
            loser_actual = 0.5
            winner_rating.draws += 1
            loser_rating.draws += 1
        else:
            winner_actual = 1.0
            loser_actual = 0.0
            winner_rating.wins += 1
            loser_rating.losses += 1
        
        # Get K-factors
        winner_k = self._get_k_factor(winner_rating)
        loser_k = self._get_k_factor(loser_rating)
        
        # Update ratings
        winner_rating.rating += winner_k * (winner_actual - winner_expected)
        loser_rating.rating += loser_k * (loser_actual - loser_expected)
        
        # Update other stats
        winner_rating.games_played += 1
        loser_rating.games_played += 1
        winner_rating.last_game = datetime.utcnow()
        loser_rating.last_game = datetime.utcnow()
        winner_rating.update_peak()
        loser_rating.update_peak()
        
        return winner_rating.rating, loser_rating.rating
    
    def update_multiplayer(
        self,
        rankings: List[str]
    ) -> Dict[str, float]:
        """
        Update ratings for multiplayer game.
        
        Args:
            rankings: List of player IDs in order (first = winner)
            
        Returns:
            New ratings for all players
        """
        new_ratings = {}
        
        # Pairwise updates
        for i, player1 in enumerate(rankings):
            for player2 in rankings[i+1:]:
                # Player1 beat player2
                self.update_ratings(player1, player2, draw=False)
        
        # Get final ratings
        for player in rankings:
            new_ratings[player] = self.get_rating(player)
        
        return new_ratings
    
    def _expected_score(self, rating1: float, rating2: float) -> float:
        """Calculate expected score for player1."""
        return 1.0 / (1.0 + math.pow(10, (rating2 - rating1) / 400))
    
    def _get_k_factor(self, player: PlayerRating) -> float:
        """Get K-factor for player."""
        if not self.use_dynamic_k:
            return self.k_factor
        
        # Dynamic K-factor based on games played and rating
        if player.games_played < 10:
            return 40  # New players
        elif player.games_played < 30:
            return 32  # Intermediate
        elif player.rating < 2000:
            return 24  # Regular players
        else:
            return 16  # High-rated players


class GlickoRating:
    """
    Glicko rating system implementation.
    
    More sophisticated than Elo, includes rating deviation.
    """
    
    def __init__(
        self,
        initial_rating: float = 1500,
        initial_rd: float = 350,
        c_squared: float = 63.2
    ):
        """
        Initialize Glicko rating system.
        
        Args:
            initial_rating: Starting rating
            initial_rd: Starting rating deviation
            c_squared: System constant
        """
        self.initial_rating = initial_rating
        self.initial_rd = initial_rd
        self.c_squared = c_squared
        
        self.ratings: Dict[str, PlayerRating] = {}
    
    def update_ratings(
        self,
        player: str,
        opponents: List[str],
        results: List[float]
    ) -> float:
        """
        Update player rating based on game results.
        
        Args:
            player: Player ID
            opponents: List of opponent IDs
            results: List of results (1=win, 0.5=draw, 0=loss)
            
        Returns:
            New rating
        """
        if player not in self.ratings:
            self.ratings[player] = PlayerRating(
                player_id=player,
                rating=self.initial_rating,
                rating_deviation=self.initial_rd,
                games_played=0,
                wins=0,
                losses=0,
                draws=0,
                peak_rating=self.initial_rating
            )
        
        player_rating = self.ratings[player]
        
        # Ensure opponents exist
        for opp in opponents:
            if opp not in self.ratings:
                self.ratings[opp] = PlayerRating(
                    player_id=opp,
                    rating=self.initial_rating,
                    rating_deviation=self.initial_rd,
                    games_played=0,
                    wins=0,
                    losses=0,
                    draws=0,
                    peak_rating=self.initial_rating
                )
        
        # Calculate new rating
        q = math.log(10) / 400
        g_values = []
        e_values = []
        
        for opp in opponents:
            opp_rating = self.ratings[opp]
            g = self._g_function(opp_rating.rating_deviation)
            e = self._e_function(
                player_rating.rating,
                opp_rating.rating,
                g
            )
            g_values.append(g)
            e_values.append(e)
        
        # Calculate d²
        d_squared = 0
        for g, e in zip(g_values, e_values):
            d_squared += (g * g * e * (1 - e))
        d_squared = 1.0 / (q * q * d_squared)
        
        # Calculate new rating deviation
        new_rd_squared = 1.0 / (
            1.0 / (player_rating.rating_deviation ** 2) + 1.0 / d_squared
        )
        new_rd = math.sqrt(new_rd_squared)
        
        # Calculate rating change
        rating_change = 0
        for opp, result, g, e in zip(opponents, results, g_values, e_values):
            rating_change += g * (result - e)
        rating_change *= q * new_rd_squared
        
        # Update rating
        player_rating.rating += rating_change
        player_rating.rating_deviation = new_rd
        player_rating.games_played += len(opponents)
        player_rating.last_game = datetime.utcnow()
        player_rating.update_peak()
        
        # Update win/loss/draw counts
        for result in results:
            if result == 1.0:
                player_rating.wins += 1
            elif result == 0.0:
                player_rating.losses += 1
            else:
                player_rating.draws += 1
        
        return player_rating.rating
    
    def _g_function(self, rd: float) -> float:
        """G function for Glicko."""
        q = math.log(10) / 400
        return 1.0 / math.sqrt(1 + 3 * q * q * rd * rd / (math.pi * math.pi))
    
    def _e_function(self, r1: float, r2: float, g: float) -> float:
        """E function for Glicko."""
        return 1.0 / (1 + math.pow(10, -g * (r1 - r2) / 400))


class RankingSystem:
    """
    Unified ranking system supporting multiple algorithms.
    """
    
    def __init__(
        self,
        algorithm: RankingAlgorithm = RankingAlgorithm.ELO
    ):
        """
        Initialize ranking system.
        
        Args:
            algorithm: Ranking algorithm to use
        """
        self.algorithm = algorithm
        
        # Initialize specific system
        if algorithm == RankingAlgorithm.ELO:
            self.system = EloRating()
        elif algorithm == RankingAlgorithm.GLICKO:
            self.system = GlickoRating()
        else:
            self.system = None
        
        # Track additional metrics
        self.performance_metrics: Dict[str, PerformanceMetrics] = {}
    
    def process_game_result(
        self,
        rankings: List[str],
        scores: Dict[str, float],
        eliminations: List[str]
    ) -> Dict[str, float]:
        """
        Process game result and update rankings.
        
        Args:
            rankings: Final rankings (first = winner)
            scores: Final scores
            eliminations: Order of eliminations
            
        Returns:
            Updated ratings
        """
        # Update rating system
        if self.algorithm == RankingAlgorithm.ELO:
            new_ratings = self.system.update_multiplayer(rankings)
        elif self.algorithm == RankingAlgorithm.GLICKO:
            # Glicko needs pairwise results
            new_ratings = {}
            for i, player in enumerate(rankings):
                opponents = rankings[:i] + rankings[i+1:]
                results = [0.0] * i + [1.0] * (len(rankings) - i - 1)
                new_rating = self.system.update_ratings(player, opponents, results)
                new_ratings[player] = new_rating
        else:
            # Simple points-based
            new_ratings = scores
        
        # Update performance metrics
        self._update_performance_metrics(rankings, scores, eliminations)
        
        return new_ratings
    
    def _update_performance_metrics(
        self,
        rankings: List[str],
        scores: Dict[str, float],
        eliminations: List[str]
    ) -> None:
        """Update performance metrics for players."""
        for player_id in rankings:
            if player_id not in self.performance_metrics:
                self.performance_metrics[player_id] = PerformanceMetrics(player_id)
            
            metrics = self.performance_metrics[player_id]
            
            # Update basic stats
            player_score = scores.get(player_id, 0)
            metrics.total_score += player_score
            
            # Update position
            position = rankings.index(player_id) + 1
            metrics.elimination_position.append(position)
            
            if position == 1:
                metrics.champion_count += 1
            
            # Calculate averages
            games_played = len(metrics.elimination_position)
            metrics.average_score = metrics.total_score / games_played
            
            # Calculate variance
            if games_played > 1:
                scores_list = [scores.get(player_id, 0)]  # Would need history
                metrics.score_variance = np.var(scores_list)
    
    def get_leaderboard(
        self,
        top_n: Optional[int] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Get current leaderboard.
        
        Args:
            top_n: Number of top players to return
            
        Returns:
            List of (player_id, rating, stats) tuples
        """
        leaderboard = []
        
        if self.system and hasattr(self.system, 'ratings'):
            for player_id, rating_obj in self.system.ratings.items():
                stats = {
                    "games_played": rating_obj.games_played,
                    "win_rate": rating_obj.win_rate,
                    "peak_rating": rating_obj.peak_rating
                }
                
                # Add performance metrics if available
                if player_id in self.performance_metrics:
                    metrics = self.performance_metrics[player_id]
                    stats["champion_count"] = metrics.champion_count
                    stats["average_score"] = metrics.average_score
                    stats["composite_score"] = metrics.calculate_composite_score()
                
                leaderboard.append((player_id, rating_obj.rating, stats))
        
        # Sort by rating
        leaderboard.sort(key=lambda x: x[1], reverse=True)
        
        if top_n:
            return leaderboard[:top_n]
        return leaderboard


class Leaderboard:
    """
    Complete leaderboard management system.
    """
    
    def __init__(
        self,
        ranking_algorithm: RankingAlgorithm = RankingAlgorithm.ELO
    ):
        """
        Initialize leaderboard.
        
        Args:
            ranking_algorithm: Algorithm to use for rankings
        """
        self.ranking_system = RankingSystem(ranking_algorithm)
        
        # Historical tracking
        self.history: List[Dict[str, Any]] = []
        self.season_data: Dict[int, List[Dict]] = defaultdict(list)
        self.current_season = 1
        
        # Achievements
        self.achievements: Dict[str, List[str]] = defaultdict(list)
        
        # Tournament tracking
        self.tournaments: List[Dict[str, Any]] = []
    
    def record_game(
        self,
        game_id: str,
        rankings: List[str],
        scores: Dict[str, float],
        eliminations: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a game result.
        
        Args:
            game_id: Unique game ID
            rankings: Final rankings
            scores: Final scores
            eliminations: Elimination order
            metadata: Additional game metadata
        """
        # Update rankings
        new_ratings = self.ranking_system.process_game_result(
            rankings, scores, eliminations
        )
        
        # Record in history
        record = {
            "game_id": game_id,
            "timestamp": datetime.utcnow(),
            "rankings": rankings,
            "scores": scores,
            "eliminations": eliminations,
            "ratings_after": new_ratings,
            "season": self.current_season,
            "metadata": metadata or {}
        }
        
        self.history.append(record)
        self.season_data[self.current_season].append(record)
        
        # Check for achievements
        self._check_achievements(rankings, scores)
        
        logger.info(f"Recorded game {game_id}, winner: {rankings[0] if rankings else 'none'}")
    
    def _check_achievements(
        self,
        rankings: List[str],
        scores: Dict[str, float]
    ) -> None:
        """Check and award achievements."""
        if not rankings:
            return
        
        winner = rankings[0]
        
        # First win
        if winner not in self.achievements:
            self.achievements[winner].append("first_win")
        
        # Perfect game (no eliminations)
        if len(rankings) == 1:
            self.achievements[winner].append("perfect_game")
        
        # Comeback victory (was in bottom half)
        mid_point = len(rankings) // 2
        if winner in rankings[mid_point:]:
            self.achievements[winner].append("comeback_victory")
        
        # High score
        winner_score = scores.get(winner, 0)
        if winner_score > 0.9:
            self.achievements[winner].append("high_scorer")
    
    def get_current_leaderboard(
        self,
        top_n: Optional[int] = 10
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Get current leaderboard."""
        return self.ranking_system.get_leaderboard(top_n)
    
    def get_player_stats(
        self,
        player_id: str
    ) -> Dict[str, Any]:
        """
        Get comprehensive stats for a player.
        
        Args:
            player_id: Player ID
            
        Returns:
            Player statistics
        """
        stats = {
            "player_id": player_id,
            "current_rating": 0,
            "peak_rating": 0,
            "games_played": 0,
            "wins": 0,
            "win_rate": 0.0,
            "achievements": self.achievements.get(player_id, []),
            "recent_games": [],
            "season_performance": {}
        }
        
        # Get rating info
        if hasattr(self.ranking_system.system, 'ratings'):
            if player_id in self.ranking_system.system.ratings:
                rating_obj = self.ranking_system.system.ratings[player_id]
                stats["current_rating"] = rating_obj.rating
                stats["peak_rating"] = rating_obj.peak_rating
                stats["games_played"] = rating_obj.games_played
                stats["wins"] = rating_obj.wins
                stats["win_rate"] = rating_obj.win_rate
        
        # Get recent games
        for record in reversed(self.history[-10:]):
            if player_id in record["rankings"]:
                position = record["rankings"].index(player_id) + 1
                stats["recent_games"].append({
                    "game_id": record["game_id"],
                    "position": position,
                    "score": record["scores"].get(player_id, 0),
                    "timestamp": record["timestamp"]
                })
        
        # Season performance
        for season, games in self.season_data.items():
            season_wins = sum(1 for g in games 
                            if g["rankings"] and g["rankings"][0] == player_id)
            season_games = sum(1 for g in games 
                             if player_id in g["rankings"])
            
            if season_games > 0:
                stats["season_performance"][season] = {
                    "games": season_games,
                    "wins": season_wins,
                    "win_rate": season_wins / season_games
                }
        
        return stats
    
    def start_new_season(self) -> None:
        """Start a new season."""
        self.current_season += 1
        logger.info(f"Started season {self.current_season}")
    
    def create_tournament(
        self,
        name: str,
        players: List[str],
        format: str = "round_robin"
    ) -> Dict[str, Any]:
        """
        Create a tournament.
        
        Args:
            name: Tournament name
            players: List of player IDs
            format: Tournament format
            
        Returns:
            Tournament information
        """
        tournament = {
            "id": f"tournament_{len(self.tournaments)}",
            "name": name,
            "players": players,
            "format": format,
            "games": [],
            "standings": {},
            "created": datetime.utcnow(),
            "status": "active"
        }
        
        self.tournaments.append(tournament)
        logger.info(f"Created tournament: {name}")
        
        return tournament