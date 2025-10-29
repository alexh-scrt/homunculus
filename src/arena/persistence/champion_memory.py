"""
Champion Memory System for Arena

This module manages persistent memory for champion agents across games,
allowing winners to retain knowledge and improve over time.

Features:
- Champion profile management
- Experience replay buffers
- Strategy evolution tracking
- Cross-game learning
- Memory consolidation

Author: Homunculus Team
"""

import logging
import json
import pickle
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from collections import deque
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ChampionProfile:
    """Profile for a champion agent."""
    agent_id: str
    agent_name: str
    total_wins: int = 0
    total_games: int = 0
    win_rate: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_game: Optional[datetime] = None
    
    # Performance metrics
    average_score: float = 0.0
    highest_score: float = 0.0
    favorite_strategies: List[str] = field(default_factory=list)
    successful_tactics: Dict[str, float] = field(default_factory=dict)
    
    # Learning data
    key_insights: List[str] = field(default_factory=list)
    avoided_mistakes: Set[str] = field(default_factory=set)
    opponent_profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Evolution tracking
    strategy_evolution: List[Dict[str, Any]] = field(default_factory=list)
    skill_progression: Dict[str, List[float]] = field(default_factory=dict)
    
    def update_from_game(self, game_data: Dict[str, Any]) -> None:
        """Update profile from game results."""
        self.total_games += 1
        self.last_game = datetime.utcnow()
        
        if game_data.get("won", False):
            self.total_wins += 1
        
        self.win_rate = self.total_wins / self.total_games if self.total_games > 0 else 0.0
        
        # Update score metrics
        score = game_data.get("final_score", 0.0)
        self.average_score = (
            (self.average_score * (self.total_games - 1) + score) / self.total_games
        )
        self.highest_score = max(self.highest_score, score)
        
        # Track strategies
        strategy = game_data.get("primary_strategy")
        if strategy and strategy not in self.favorite_strategies:
            self.favorite_strategies.append(strategy)
            if len(self.favorite_strategies) > 5:
                self.favorite_strategies.pop(0)
        
        # Update tactics
        for tactic, success_rate in game_data.get("tactics", {}).items():
            current = self.successful_tactics.get(tactic, 0.0)
            self.successful_tactics[tactic] = (current + success_rate) / 2
        
        # Add insights
        insights = game_data.get("key_insights", [])
        for insight in insights:
            if insight not in self.key_insights:
                self.key_insights.append(insight)
                if len(self.key_insights) > 20:
                    self.key_insights.pop(0)
        
        # Track evolution
        evolution_entry = {
            "game": self.total_games,
            "strategy": strategy,
            "score": score,
            "timestamp": self.last_game.isoformat()
        }
        self.strategy_evolution.append(evolution_entry)
        if len(self.strategy_evolution) > 50:
            self.strategy_evolution.pop(0)


@dataclass
class MemoryEntry:
    """Single memory entry for replay."""
    game_id: str
    turn: int
    state: Dict[str, Any]
    action: Dict[str, Any]
    reward: float
    next_state: Optional[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "game_id": self.game_id,
            "turn": self.turn,
            "state": self.state,
            "action": self.action,
            "reward": self.reward,
            "next_state": self.next_state,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class ExperienceReplay:
    """
    Experience replay buffer for champion learning.
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of memories
        """
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        self.priorities: Dict[str, float] = {}
        
    def add(self, memory: MemoryEntry) -> None:
        """Add memory to buffer."""
        self.buffer.append(memory)
        # Calculate priority based on reward
        priority = abs(memory.reward) + 0.01
        memory_id = f"{memory.game_id}_{memory.turn}"
        self.priorities[memory_id] = priority
    
    def sample(self, batch_size: int) -> List[MemoryEntry]:
        """Sample memories from buffer."""
        import random
        
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        # Priority sampling
        if self.priorities:
            # Calculate sampling probabilities
            total_priority = sum(self.priorities.values())
            probs = [
                self.priorities.get(f"{m.game_id}_{m.turn}", 1.0) / total_priority
                for m in self.buffer
            ]
            
            # Sample with probabilities
            indices = random.choices(
                range(len(self.buffer)),
                weights=probs,
                k=batch_size
            )
            return [self.buffer[i] for i in indices]
        else:
            # Uniform sampling
            return random.sample(list(self.buffer), batch_size)
    
    def get_recent(self, n: int = 100) -> List[MemoryEntry]:
        """Get n most recent memories."""
        return list(self.buffer)[-n:]
    
    def get_high_reward(self, n: int = 100) -> List[MemoryEntry]:
        """Get n highest reward memories."""
        sorted_memories = sorted(
            self.buffer,
            key=lambda m: m.reward,
            reverse=True
        )
        return sorted_memories[:n]
    
    def consolidate(self) -> Dict[str, Any]:
        """Consolidate memories into insights."""
        if not self.buffer:
            return {}
        
        # Analyze patterns
        total_reward = sum(m.reward for m in self.buffer)
        avg_reward = total_reward / len(self.buffer)
        
        # Find successful actions
        successful_actions = {}
        for memory in self.buffer:
            if memory.reward > avg_reward:
                action_type = memory.action.get("type", "unknown")
                if action_type not in successful_actions:
                    successful_actions[action_type] = []
                successful_actions[action_type].append(memory.reward)
        
        # Calculate action success rates
        action_stats = {}
        for action_type, rewards in successful_actions.items():
            action_stats[action_type] = {
                "count": len(rewards),
                "avg_reward": sum(rewards) / len(rewards),
                "max_reward": max(rewards)
            }
        
        return {
            "total_experiences": len(self.buffer),
            "average_reward": avg_reward,
            "successful_actions": action_stats,
            "memory_span_games": len(set(m.game_id for m in self.buffer))
        }


class MemoryBank:
    """
    Central memory bank for all champions.
    """
    
    def __init__(self, storage_path: str = "arena_memories"):
        """
        Initialize memory bank.
        
        Args:
            storage_path: Path to storage directory
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # In-memory caches
        self.profiles: Dict[str, ChampionProfile] = {}
        self.replays: Dict[str, ExperienceReplay] = {}
        
        # Load existing memories
        self.load_all()
    
    def get_or_create_profile(self, agent_id: str, agent_name: str) -> ChampionProfile:
        """Get or create champion profile."""
        if agent_id not in self.profiles:
            self.profiles[agent_id] = ChampionProfile(
                agent_id=agent_id,
                agent_name=agent_name
            )
        return self.profiles[agent_id]
    
    def get_replay_buffer(self, agent_id: str) -> ExperienceReplay:
        """Get replay buffer for agent."""
        if agent_id not in self.replays:
            self.replays[agent_id] = ExperienceReplay()
        return self.replays[agent_id]
    
    def record_game(
        self,
        agent_id: str,
        game_data: Dict[str, Any],
        memories: List[MemoryEntry]
    ) -> None:
        """
        Record game results and memories.
        
        Args:
            agent_id: Agent ID
            game_data: Game results data
            memories: List of memory entries
        """
        # Update profile
        profile = self.profiles.get(agent_id)
        if profile:
            profile.update_from_game(game_data)
        
        # Add memories to replay buffer
        replay = self.get_replay_buffer(agent_id)
        for memory in memories:
            replay.add(memory)
        
        # Save to disk
        self.save_agent(agent_id)
    
    def get_champion_insights(self, agent_id: str) -> Dict[str, Any]:
        """
        Get consolidated insights for a champion.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Champion insights
        """
        profile = self.profiles.get(agent_id)
        if not profile:
            return {}
        
        replay = self.get_replay_buffer(agent_id)
        consolidation = replay.consolidate()
        
        return {
            "profile": {
                "total_wins": profile.total_wins,
                "total_games": profile.total_games,
                "win_rate": profile.win_rate,
                "average_score": profile.average_score,
                "highest_score": profile.highest_score
            },
            "strategies": {
                "favorites": profile.favorite_strategies,
                "successful_tactics": profile.successful_tactics,
                "evolution": profile.strategy_evolution[-10:] if profile.strategy_evolution else []
            },
            "learning": {
                "key_insights": profile.key_insights[-10:],
                "avoided_mistakes": list(profile.avoided_mistakes)[-10:],
                "experience_consolidation": consolidation
            }
        }
    
    def save_agent(self, agent_id: str) -> None:
        """Save agent data to disk."""
        # Save profile
        profile = self.profiles.get(agent_id)
        if profile:
            profile_path = self.storage_path / f"{agent_id}_profile.json"
            with open(profile_path, 'w') as f:
                # Convert sets to lists for JSON serialization
                profile_dict = asdict(profile)
                profile_dict['avoided_mistakes'] = list(profile.avoided_mistakes)
                profile_dict['created_at'] = profile.created_at.isoformat()
                if profile.last_game:
                    profile_dict['last_game'] = profile.last_game.isoformat()
                json.dump(profile_dict, f, indent=2)
        
        # Save replay buffer
        replay = self.replays.get(agent_id)
        if replay:
            replay_path = self.storage_path / f"{agent_id}_replay.pkl"
            with open(replay_path, 'wb') as f:
                pickle.dump(replay, f)
    
    def load_agent(self, agent_id: str) -> bool:
        """Load agent data from disk."""
        # Load profile
        profile_path = self.storage_path / f"{agent_id}_profile.json"
        if profile_path.exists():
            with open(profile_path, 'r') as f:
                data = json.load(f)
                # Convert back to proper types
                data['avoided_mistakes'] = set(data.get('avoided_mistakes', []))
                data['created_at'] = datetime.fromisoformat(data['created_at'])
                if data.get('last_game'):
                    data['last_game'] = datetime.fromisoformat(data['last_game'])
                
                self.profiles[agent_id] = ChampionProfile(**data)
        
        # Load replay buffer
        replay_path = self.storage_path / f"{agent_id}_replay.pkl"
        if replay_path.exists():
            with open(replay_path, 'rb') as f:
                self.replays[agent_id] = pickle.load(f)
        
        return agent_id in self.profiles
    
    def load_all(self) -> None:
        """Load all agent memories from disk."""
        # Find all profile files
        profile_files = list(self.storage_path.glob("*_profile.json"))
        
        for profile_file in profile_files:
            agent_id = profile_file.stem.replace("_profile", "")
            self.load_agent(agent_id)
        
        logger.info(f"Loaded {len(self.profiles)} champion profiles")
    
    def save_all(self) -> None:
        """Save all agent memories to disk."""
        for agent_id in self.profiles:
            self.save_agent(agent_id)
        
        logger.info(f"Saved {len(self.profiles)} champion profiles")


class ChampionMemory:
    """
    Main interface for champion memory system.
    """
    
    def __init__(self, memory_bank: Optional[MemoryBank] = None):
        """
        Initialize champion memory.
        
        Args:
            memory_bank: Optional shared memory bank
        """
        self.memory_bank = memory_bank or MemoryBank()
        
    def initialize_champion(
        self,
        agent_id: str,
        agent_name: str,
        load_history: bool = True
    ) -> ChampionProfile:
        """
        Initialize a champion with memory.
        
        Args:
            agent_id: Agent ID
            agent_name: Agent name
            load_history: Whether to load historical data
            
        Returns:
            Champion profile
        """
        if load_history:
            self.memory_bank.load_agent(agent_id)
        
        return self.memory_bank.get_or_create_profile(agent_id, agent_name)
    
    def record_turn(
        self,
        agent_id: str,
        game_id: str,
        turn: int,
        state: Dict[str, Any],
        action: Dict[str, Any],
        reward: float,
        next_state: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a turn for learning.
        
        Args:
            agent_id: Agent ID
            game_id: Game ID
            turn: Turn number
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        memory = MemoryEntry(
            game_id=game_id,
            turn=turn,
            state=state,
            action=action,
            reward=reward,
            next_state=next_state
        )
        
        replay = self.memory_bank.get_replay_buffer(agent_id)
        replay.add(memory)
    
    def finalize_game(
        self,
        agent_id: str,
        game_data: Dict[str, Any]
    ) -> None:
        """
        Finalize game and consolidate memories.
        
        Args:
            agent_id: Agent ID
            game_data: Final game data
        """
        # Get all memories for this game
        replay = self.memory_bank.get_replay_buffer(agent_id)
        game_memories = [
            m for m in replay.buffer
            if m.game_id == game_data.get("game_id")
        ]
        
        # Update profile and save
        self.memory_bank.record_game(agent_id, game_data, game_memories)
    
    def get_champion_knowledge(self, agent_id: str) -> Dict[str, Any]:
        """
        Get champion's accumulated knowledge.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Champion knowledge and insights
        """
        return self.memory_bank.get_champion_insights(agent_id)
    
    def get_strategic_advice(
        self,
        agent_id: str,
        game_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get strategic advice based on memories.
        
        Args:
            agent_id: Agent ID
            game_context: Current game context
            
        Returns:
            Strategic recommendations
        """
        profile = self.memory_bank.profiles.get(agent_id)
        if not profile:
            return {"advice": "No historical data available"}
        
        replay = self.memory_bank.get_replay_buffer(agent_id)
        
        # Analyze similar situations
        similar_memories = []
        for memory in replay.get_recent(500):
            # Simple similarity check (could be more sophisticated)
            state_similarity = self._calculate_similarity(
                memory.state,
                game_context
            )
            if state_similarity > 0.7:
                similar_memories.append(memory)
        
        # Extract successful actions
        successful_actions = {}
        for memory in similar_memories:
            if memory.reward > 0:
                action_type = memory.action.get("type", "unknown")
                if action_type not in successful_actions:
                    successful_actions[action_type] = []
                successful_actions[action_type].append(memory.reward)
        
        # Generate advice
        advice = {
            "recommended_strategies": profile.favorite_strategies[:3],
            "successful_tactics": dict(
                sorted(
                    profile.successful_tactics.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            ),
            "similar_situation_actions": successful_actions,
            "confidence": len(similar_memories) / 100.0 if similar_memories else 0.0
        }
        
        return advice
    
    def _calculate_similarity(
        self,
        state1: Dict[str, Any],
        state2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between two states."""
        # Simple implementation - could be more sophisticated
        common_keys = set(state1.keys()) & set(state2.keys())
        if not common_keys:
            return 0.0
        
        matches = 0
        for key in common_keys:
            if state1[key] == state2[key]:
                matches += 1
            elif isinstance(state1[key], (int, float)) and isinstance(state2[key], (int, float)):
                # Numeric similarity
                diff = abs(state1[key] - state2[key])
                max_val = max(abs(state1[key]), abs(state2[key]), 1.0)
                similarity = 1.0 - (diff / max_val)
                matches += similarity
        
        return matches / len(common_keys)
    
    def merge_champions(
        self,
        champion_ids: List[str],
        new_id: str,
        new_name: str
    ) -> ChampionProfile:
        """
        Merge multiple champions into one.
        
        Args:
            champion_ids: List of champion IDs to merge
            new_id: New champion ID
            new_name: New champion name
            
        Returns:
            Merged champion profile
        """
        # Create new profile
        merged = ChampionProfile(agent_id=new_id, agent_name=new_name)
        
        # Merge profiles
        for cid in champion_ids:
            profile = self.memory_bank.profiles.get(cid)
            if profile:
                merged.total_wins += profile.total_wins
                merged.total_games += profile.total_games
                merged.average_score = (
                    (merged.average_score * merged.total_games +
                     profile.average_score * profile.total_games) /
                    (merged.total_games + profile.total_games)
                    if merged.total_games + profile.total_games > 0 else 0
                )
                merged.highest_score = max(merged.highest_score, profile.highest_score)
                
                # Merge strategies
                for strategy in profile.favorite_strategies:
                    if strategy not in merged.favorite_strategies:
                        merged.favorite_strategies.append(strategy)
                
                # Merge tactics
                for tactic, rate in profile.successful_tactics.items():
                    if tactic in merged.successful_tactics:
                        merged.successful_tactics[tactic] = (
                            merged.successful_tactics[tactic] + rate
                        ) / 2
                    else:
                        merged.successful_tactics[tactic] = rate
                
                # Merge insights
                merged.key_insights.extend(profile.key_insights)
                merged.avoided_mistakes.update(profile.avoided_mistakes)
        
        # Recalculate win rate
        merged.win_rate = (
            merged.total_wins / merged.total_games
            if merged.total_games > 0 else 0.0
        )
        
        # Save merged profile
        self.memory_bank.profiles[new_id] = merged
        
        # Merge replay buffers
        merged_replay = ExperienceReplay(capacity=20000)
        for cid in champion_ids:
            replay = self.memory_bank.replays.get(cid)
            if replay:
                for memory in replay.buffer:
                    merged_replay.add(memory)
        
        self.memory_bank.replays[new_id] = merged_replay
        
        # Save to disk
        self.memory_bank.save_agent(new_id)
        
        return merged