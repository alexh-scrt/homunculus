"""
Database Management for Arena

This module provides database schema and operations for persisting
Arena game data using SQLAlchemy with PostgreSQL support.

Features:
- Game and turn records
- Agent performance tracking
- Message history
- Score progression
- Efficient queries for analytics

Author: Homunculus Team
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import json

try:
    from sqlalchemy import (
        create_engine, Column, String, Integer, Float, 
        DateTime, Boolean, JSON, Text, ForeignKey, Index
    )
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import relationship, sessionmaker, Session
    from sqlalchemy.pool import QueuePool
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    # Fallback to simple file storage
    declarative_base = lambda: object

logger = logging.getLogger(__name__)

# Base for SQLAlchemy models
Base = declarative_base() if SQLALCHEMY_AVAILABLE else object


class GameRecord(Base):
    """Record of a complete game."""
    __tablename__ = 'games' if SQLALCHEMY_AVAILABLE else None
    
    if SQLALCHEMY_AVAILABLE:
        game_id = Column(String, primary_key=True)
        start_time = Column(DateTime, nullable=False)
        end_time = Column(DateTime)
        total_turns = Column(Integer, default=0)
        winner_id = Column(String)
        winner_name = Column(String)
        final_phase = Column(String)
        total_agents = Column(Integer)
        configuration = Column(JSON)
        game_metadata = Column(JSON)
        
        # Relationships
        turns = relationship("TurnRecord", back_populates="game", cascade="all, delete-orphan")
        agents = relationship("AgentRecord", back_populates="game", cascade="all, delete-orphan")
        messages = relationship("MessageRecord", back_populates="game", cascade="all, delete-orphan")
        scores = relationship("ScoreRecord", back_populates="game", cascade="all, delete-orphan")
        
        # Indexes for performance
        __table_args__ = (
            Index('idx_game_start_time', start_time),
            Index('idx_game_winner', winner_id),
        )


class TurnRecord(Base):
    """Record of a single turn."""
    __tablename__ = 'turns' if SQLALCHEMY_AVAILABLE else None
    
    if SQLALCHEMY_AVAILABLE:
        id = Column(Integer, primary_key=True, autoincrement=True)
        game_id = Column(String, ForeignKey('games.game_id'), nullable=False)
        turn_number = Column(Integer, nullable=False)
        phase = Column(String)
        active_agents = Column(Integer)
        eliminated_this_turn = Column(JSON)  # List of eliminated agent IDs
        speaker_id = Column(String)
        duration_ms = Column(Integer)
        timestamp = Column(DateTime, default=datetime.utcnow)
        
        # Relationship
        game = relationship("GameRecord", back_populates="turns")
        
        # Index
        __table_args__ = (
            Index('idx_turn_game', game_id, turn_number),
        )


class AgentRecord(Base):
    """Record of an agent's performance."""
    __tablename__ = 'agents' if SQLALCHEMY_AVAILABLE else None
    
    if SQLALCHEMY_AVAILABLE:
        id = Column(Integer, primary_key=True, autoincrement=True)
        game_id = Column(String, ForeignKey('games.game_id'), nullable=False)
        agent_id = Column(String, nullable=False)
        agent_name = Column(String)
        agent_type = Column(String)
        character_profile = Column(JSON)
        final_position = Column(Integer)
        final_score = Column(Float)
        elimination_turn = Column(Integer)
        total_contributions = Column(Integer, default=0)
        total_accusations = Column(Integer, default=0)
        times_accused = Column(Integer, default=0)
        is_champion = Column(Boolean, default=False)
        previous_wins = Column(Integer, default=0)
        
        # Relationship
        game = relationship("GameRecord", back_populates="agents")
        
        # Index
        __table_args__ = (
            Index('idx_agent_game', game_id, agent_id),
            Index('idx_agent_champion', is_champion),
        )


class MessageRecord(Base):
    """Record of a message."""
    __tablename__ = 'messages' if SQLALCHEMY_AVAILABLE else None
    
    if SQLALCHEMY_AVAILABLE:
        id = Column(Integer, primary_key=True, autoincrement=True)
        game_id = Column(String, ForeignKey('games.game_id'), nullable=False)
        message_id = Column(String)
        turn_number = Column(Integer)
        sender_id = Column(String)
        sender_name = Column(String)
        message_type = Column(String)
        content = Column(Text)
        message_metadata = Column(JSON)
        timestamp = Column(DateTime, default=datetime.utcnow)
        
        # Relationship
        game = relationship("GameRecord", back_populates="messages")
        
        # Index
        __table_args__ = (
            Index('idx_message_game_turn', game_id, turn_number),
            Index('idx_message_sender', sender_id),
        )


class ScoreRecord(Base):
    """Record of score progression."""
    __tablename__ = 'scores' if SQLALCHEMY_AVAILABLE else None
    
    if SQLALCHEMY_AVAILABLE:
        id = Column(Integer, primary_key=True, autoincrement=True)
        game_id = Column(String, ForeignKey('games.game_id'), nullable=False)
        agent_id = Column(String, nullable=False)
        turn_number = Column(Integer, nullable=False)
        score = Column(Float, nullable=False)
        score_change = Column(Float)
        novelty = Column(Float)
        builds_on_others = Column(Float)
        solves_subproblem = Column(Float)
        radical_idea = Column(Float)
        manipulation = Column(Float)
        
        # Relationship
        game = relationship("GameRecord", back_populates="scores")
        
        # Index
        __table_args__ = (
            Index('idx_score_game_agent', game_id, agent_id, turn_number),
        )


@dataclass
class DatabaseConfig:
    """Database configuration."""
    connection_string: str = "sqlite:///arena.db"  # Default to SQLite
    pool_size: int = 5
    max_overflow: int = 10
    echo: bool = False
    create_tables: bool = True


class DatabaseManager:
    """
    Manages database operations for Arena.
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        """
        Initialize database manager.
        
        Args:
            config: Database configuration
        """
        self.config = config or DatabaseConfig()
        
        if not SQLALCHEMY_AVAILABLE:
            logger.warning("SQLAlchemy not available, using fallback storage")
            self.engine = None
            self.session_factory = None
            self._init_fallback_storage()
        else:
            # Create engine
            self.engine = create_engine(
                self.config.connection_string,
                poolclass=QueuePool,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                echo=self.config.echo
            )
            
            # Create session factory
            self.session_factory = sessionmaker(bind=self.engine)
            
            # Create tables if needed
            if self.config.create_tables:
                self.create_tables()
    
    def _init_fallback_storage(self):
        """Initialize fallback storage when SQLAlchemy not available."""
        import json
        from pathlib import Path
        
        self.storage_dir = Path("arena_data")
        self.storage_dir.mkdir(exist_ok=True)
        self.games = {}
        self.turns = []
        self.agents = []
        self.messages = []
        self.scores = []
        
        # Load existing data
        games_file = self.storage_dir / "games.json"
        if games_file.exists():
            with open(games_file) as f:
                data = json.load(f)
                self.games = data.get("games", {})
                self.turns = data.get("turns", [])
                self.agents = data.get("agents", [])
                self.messages = data.get("messages", [])
                self.scores = data.get("scores", [])
    
    def create_tables(self):
        """Create database tables."""
        if self.engine:
            Base.metadata.create_all(self.engine)
            logger.info("Database tables created")
    
    def get_session(self) -> Session:
        """Get a new database session."""
        if self.session_factory:
            return self.session_factory()
        return None
    
    def save_game(self, game_data: Dict[str, Any]) -> str:
        """
        Save a complete game to database.
        
        Args:
            game_data: Complete game data
            
        Returns:
            Game ID
        """
        if not SQLALCHEMY_AVAILABLE:
            return self._save_game_fallback(game_data)
        
        session = self.get_session()
        try:
            # Create game record
            game = GameRecord(
                game_id=game_data["game_id"],
                start_time=datetime.fromisoformat(game_data["start_time"]),
                end_time=datetime.fromisoformat(game_data.get("end_time", datetime.utcnow().isoformat())),
                total_turns=game_data["total_turns"],
                winner_id=game_data.get("winner_id"),
                winner_name=game_data.get("winner_name"),
                final_phase=game_data.get("final_phase"),
                total_agents=game_data["total_agents"],
                configuration=game_data.get("configuration", {}),
                game_metadata=game_data.get("metadata", {})
            )
            session.add(game)
            
            # Add turn records
            for turn_data in game_data.get("turns", []):
                turn = TurnRecord(
                    game_id=game_data["game_id"],
                    turn_number=turn_data["turn_number"],
                    phase=turn_data.get("phase"),
                    active_agents=turn_data.get("active_agents"),
                    eliminated_this_turn=turn_data.get("eliminated", []),
                    speaker_id=turn_data.get("speaker_id"),
                    duration_ms=turn_data.get("duration_ms"),
                    timestamp=datetime.fromisoformat(turn_data.get("timestamp", datetime.utcnow().isoformat()))
                )
                session.add(turn)
            
            # Add agent records
            for agent_data in game_data.get("agents", []):
                agent = AgentRecord(
                    game_id=game_data["game_id"],
                    agent_id=agent_data["agent_id"],
                    agent_name=agent_data.get("agent_name"),
                    agent_type=agent_data.get("agent_type"),
                    character_profile=agent_data.get("character_profile"),
                    final_position=agent_data.get("final_position"),
                    final_score=agent_data.get("final_score"),
                    elimination_turn=agent_data.get("elimination_turn"),
                    total_contributions=agent_data.get("total_contributions", 0),
                    total_accusations=agent_data.get("total_accusations", 0),
                    times_accused=agent_data.get("times_accused", 0),
                    is_champion=agent_data.get("is_champion", False),
                    previous_wins=agent_data.get("previous_wins", 0)
                )
                session.add(agent)
            
            # Add message records
            for msg_data in game_data.get("messages", []):
                message = MessageRecord(
                    game_id=game_data["game_id"],
                    message_id=msg_data.get("message_id"),
                    turn_number=msg_data["turn_number"],
                    sender_id=msg_data["sender_id"],
                    sender_name=msg_data.get("sender_name"),
                    message_type=msg_data["message_type"],
                    content=msg_data.get("content"),
                    message_metadata=msg_data.get("metadata", {}),
                    timestamp=datetime.fromisoformat(msg_data.get("timestamp", datetime.utcnow().isoformat()))
                )
                session.add(message)
            
            # Add score records
            for score_data in game_data.get("scores", []):
                score = ScoreRecord(
                    game_id=game_data["game_id"],
                    agent_id=score_data["agent_id"],
                    turn_number=score_data["turn_number"],
                    score=score_data["score"],
                    score_change=score_data.get("score_change"),
                    novelty=score_data.get("novelty"),
                    builds_on_others=score_data.get("builds_on_others"),
                    solves_subproblem=score_data.get("solves_subproblem"),
                    radical_idea=score_data.get("radical_idea"),
                    manipulation=score_data.get("manipulation")
                )
                session.add(score)
            
            session.commit()
            logger.info(f"Game {game_data['game_id']} saved to database")
            return game_data["game_id"]
        
        except Exception as e:
            logger.error(f"Error saving game: {e}")
            session.rollback()
            raise
        finally:
            session.close()
    
    def _save_game_fallback(self, game_data: Dict[str, Any]) -> str:
        """Save game using fallback storage."""
        import json
        
        game_id = game_data["game_id"]
        self.games[game_id] = game_data
        
        # Save to file
        games_file = self.storage_dir / "games.json"
        with open(games_file, 'w') as f:
            json.dump({
                "games": self.games,
                "turns": self.turns,
                "agents": self.agents,
                "messages": self.messages,
                "scores": self.scores
            }, f, indent=2, default=str)
        
        logger.info(f"Game {game_id} saved to fallback storage")
        return game_id
    
    def load_game(self, game_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a game from database.
        
        Args:
            game_id: Game ID to load
            
        Returns:
            Game data or None
        """
        if not SQLALCHEMY_AVAILABLE:
            return self.games.get(game_id)
        
        session = self.get_session()
        try:
            game = session.query(GameRecord).filter_by(game_id=game_id).first()
            if not game:
                return None
            
            # Build game data
            game_data = {
                "game_id": game.game_id,
                "start_time": game.start_time.isoformat(),
                "end_time": game.end_time.isoformat() if game.end_time else None,
                "total_turns": game.total_turns,
                "winner_id": game.winner_id,
                "winner_name": game.winner_name,
                "final_phase": game.final_phase,
                "total_agents": game.total_agents,
                "configuration": game.configuration,
                "metadata": game.game_metadata,
                "turns": [],
                "agents": [],
                "messages": [],
                "scores": []
            }
            
            # Add turns
            for turn in game.turns:
                game_data["turns"].append({
                    "turn_number": turn.turn_number,
                    "phase": turn.phase,
                    "active_agents": turn.active_agents,
                    "eliminated": turn.eliminated_this_turn,
                    "speaker_id": turn.speaker_id,
                    "duration_ms": turn.duration_ms,
                    "timestamp": turn.timestamp.isoformat()
                })
            
            # Add agents
            for agent in game.agents:
                game_data["agents"].append({
                    "agent_id": agent.agent_id,
                    "agent_name": agent.agent_name,
                    "agent_type": agent.agent_type,
                    "character_profile": agent.character_profile,
                    "final_position": agent.final_position,
                    "final_score": agent.final_score,
                    "elimination_turn": agent.elimination_turn,
                    "total_contributions": agent.total_contributions,
                    "total_accusations": agent.total_accusations,
                    "times_accused": agent.times_accused,
                    "is_champion": agent.is_champion,
                    "previous_wins": agent.previous_wins
                })
            
            # Add messages
            for message in game.messages:
                game_data["messages"].append({
                    "message_id": message.message_id,
                    "turn_number": message.turn_number,
                    "sender_id": message.sender_id,
                    "sender_name": message.sender_name,
                    "message_type": message.message_type,
                    "content": message.content,
                    "metadata": message.message_metadata,
                    "timestamp": message.timestamp.isoformat()
                })
            
            # Add scores
            for score in game.scores:
                game_data["scores"].append({
                    "agent_id": score.agent_id,
                    "turn_number": score.turn_number,
                    "score": score.score,
                    "score_change": score.score_change,
                    "novelty": score.novelty,
                    "builds_on_others": score.builds_on_others,
                    "solves_subproblem": score.solves_subproblem,
                    "radical_idea": score.radical_idea,
                    "manipulation": score.manipulation
                })
            
            return game_data
        
        finally:
            session.close()
    
    def get_champion_history(self, agent_id: str) -> List[Dict[str, Any]]:
        """
        Get champion's game history.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            List of games where agent was champion
        """
        if not SQLALCHEMY_AVAILABLE:
            # Fallback implementation
            champion_games = []
            for game_id, game_data in self.games.items():
                for agent in game_data.get("agents", []):
                    if agent["agent_id"] == agent_id and agent.get("is_champion"):
                        champion_games.append(game_data)
            return champion_games
        
        session = self.get_session()
        try:
            champions = session.query(AgentRecord).filter_by(
                agent_id=agent_id,
                is_champion=True
            ).all()
            
            history = []
            for champion in champions:
                game_data = self.load_game(champion.game_id)
                if game_data:
                    history.append(game_data)
            
            return history
        
        finally:
            session.close()
    
    def get_game_statistics(self, game_id: str) -> Dict[str, Any]:
        """
        Get statistics for a game.
        
        Args:
            game_id: Game ID
            
        Returns:
            Game statistics
        """
        game_data = self.load_game(game_id)
        if not game_data:
            return {}
        
        stats = {
            "game_id": game_id,
            "total_turns": game_data["total_turns"],
            "total_agents": game_data["total_agents"],
            "winner": game_data.get("winner_name"),
            "total_messages": len(game_data.get("messages", [])),
            "eliminations_per_turn": {},
            "average_score": 0,
            "highest_score": 0,
            "phase_durations": {}
        }
        
        # Calculate eliminations per turn
        for turn in game_data.get("turns", []):
            eliminated = turn.get("eliminated", [])
            if eliminated:
                stats["eliminations_per_turn"][turn["turn_number"]] = len(eliminated)
        
        # Calculate score statistics
        final_scores = [a["final_score"] for a in game_data.get("agents", []) 
                       if a.get("final_score") is not None]
        if final_scores:
            stats["average_score"] = sum(final_scores) / len(final_scores)
            stats["highest_score"] = max(final_scores)
        
        return stats