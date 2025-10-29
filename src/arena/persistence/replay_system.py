"""
Replay System for Arena

This module provides functionality for recording, replaying, and analyzing
completed games with frame-by-frame playback.

Features:
- Frame-based replay recording
- Variable speed playback
- Analysis tools
- Replay export/import
- Commentary generation

Author: Homunculus Team
"""

import logging
import json
import pickle
import gzip
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class ReplaySpeed(Enum):
    """Replay playback speeds."""
    SLOW = 0.5
    NORMAL = 1.0
    FAST = 2.0
    VERY_FAST = 4.0
    INSTANT = 0.0


@dataclass
class ReplayFrame:
    """Single frame in a replay."""
    frame_number: int
    turn_number: int
    timestamp: datetime
    game_state: Dict[str, Any]
    events: List[Dict[str, Any]]
    messages: List[Dict[str, Any]]
    scores: Dict[str, float]
    active_agents: List[str]
    eliminated: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReplayFrame':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class ReplayMetadata:
    """Metadata for a replay."""
    replay_id: str
    game_id: str
    created_at: datetime
    total_frames: int
    total_turns: int
    duration: timedelta
    winner: Optional[str]
    final_scores: Dict[str, float]
    participants: List[str]
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['duration'] = self.duration.total_seconds()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReplayMetadata':
        """Create from dictionary."""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['duration'] = timedelta(seconds=data['duration'])
        return cls(**data)


class ReplayRecorder:
    """
    Records game sessions for replay.
    """
    
    def __init__(self, game_id: str):
        """
        Initialize replay recorder.
        
        Args:
            game_id: Game ID
        """
        self.game_id = game_id
        self.frames: List[ReplayFrame] = []
        self.start_time = datetime.utcnow()
        self.current_frame = 0
        self.current_turn = 0
        
    def record_frame(
        self,
        game_state: Dict[str, Any],
        events: Optional[List[Dict[str, Any]]] = None,
        messages: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Record a frame.
        
        Args:
            game_state: Current game state
            events: Events in this frame
            messages: Messages in this frame
        """
        frame = ReplayFrame(
            frame_number=self.current_frame,
            turn_number=self.current_turn,
            timestamp=datetime.utcnow(),
            game_state=game_state.copy(),
            events=events or [],
            messages=messages or [],
            scores=game_state.get("scores", {}),
            active_agents=game_state.get("active_agents", []),
            eliminated=game_state.get("eliminated_this_turn", [])
        )
        
        self.frames.append(frame)
        self.current_frame += 1
        
        # Update turn number if changed
        turn = game_state.get("current_turn", 0)
        if turn != self.current_turn:
            self.current_turn = turn
    
    def finalize(self, winner: Optional[str] = None) -> ReplayMetadata:
        """
        Finalize replay recording.
        
        Args:
            winner: Winner agent ID
            
        Returns:
            Replay metadata
        """
        if not self.frames:
            raise ValueError("No frames recorded")
        
        # Get final scores
        final_frame = self.frames[-1]
        final_scores = final_frame.scores
        
        # Get all participants
        participants = set()
        for frame in self.frames:
            participants.update(frame.active_agents)
        
        # Create metadata
        metadata = ReplayMetadata(
            replay_id=f"{self.game_id}_replay",
            game_id=self.game_id,
            created_at=self.start_time,
            total_frames=len(self.frames),
            total_turns=self.current_turn,
            duration=datetime.utcnow() - self.start_time,
            winner=winner,
            final_scores=final_scores,
            participants=list(participants)
        )
        
        return metadata
    
    def save(self, file_path: Path, compress: bool = True) -> None:
        """
        Save replay to file.
        
        Args:
            file_path: File path
            compress: Whether to compress
        """
        data = {
            "metadata": self.finalize().to_dict(),
            "frames": [frame.to_dict() for frame in self.frames]
        }
        
        if compress:
            with gzip.open(file_path.with_suffix('.replay.gz'), 'wt') as f:
                json.dump(data, f, indent=2)
        else:
            with open(file_path.with_suffix('.replay'), 'w') as f:
                json.dump(data, f, indent=2)


class ReplayViewer:
    """
    Views and controls replay playback.
    """
    
    def __init__(self, frames: List[ReplayFrame], metadata: ReplayMetadata):
        """
        Initialize replay viewer.
        
        Args:
            frames: Replay frames
            metadata: Replay metadata
        """
        self.frames = frames
        self.metadata = metadata
        self.current_frame_index = 0
        self.playback_speed = ReplaySpeed.NORMAL
        self.is_playing = False
        
        # Callbacks
        self.frame_callbacks: List[Callable[[ReplayFrame], None]] = []
        self.event_callbacks: Dict[str, List[Callable]] = {}
    
    @classmethod
    def load(cls, file_path: Path) -> 'ReplayViewer':
        """
        Load replay from file.
        
        Args:
            file_path: File path
            
        Returns:
            Replay viewer
        """
        if file_path.suffix == '.gz':
            with gzip.open(file_path, 'rt') as f:
                data = json.load(f)
        else:
            with open(file_path, 'r') as f:
                data = json.load(f)
        
        metadata = ReplayMetadata.from_dict(data['metadata'])
        frames = [ReplayFrame.from_dict(f) for f in data['frames']]
        
        return cls(frames, metadata)
    
    def get_current_frame(self) -> ReplayFrame:
        """Get current frame."""
        if 0 <= self.current_frame_index < len(self.frames):
            return self.frames[self.current_frame_index]
        return None
    
    def next_frame(self) -> Optional[ReplayFrame]:
        """Move to next frame."""
        if self.current_frame_index < len(self.frames) - 1:
            self.current_frame_index += 1
            frame = self.get_current_frame()
            self._trigger_frame_callbacks(frame)
            return frame
        return None
    
    def previous_frame(self) -> Optional[ReplayFrame]:
        """Move to previous frame."""
        if self.current_frame_index > 0:
            self.current_frame_index -= 1
            frame = self.get_current_frame()
            self._trigger_frame_callbacks(frame)
            return frame
        return None
    
    def jump_to_frame(self, frame_number: int) -> Optional[ReplayFrame]:
        """Jump to specific frame."""
        if 0 <= frame_number < len(self.frames):
            self.current_frame_index = frame_number
            frame = self.get_current_frame()
            self._trigger_frame_callbacks(frame)
            return frame
        return None
    
    def jump_to_turn(self, turn_number: int) -> Optional[ReplayFrame]:
        """Jump to specific turn."""
        for i, frame in enumerate(self.frames):
            if frame.turn_number >= turn_number:
                return self.jump_to_frame(i)
        return None
    
    def set_speed(self, speed: ReplaySpeed) -> None:
        """Set playback speed."""
        self.playback_speed = speed
    
    def play(self) -> None:
        """Start playback."""
        self.is_playing = True
    
    def pause(self) -> None:
        """Pause playback."""
        self.is_playing = False
    
    def reset(self) -> None:
        """Reset to beginning."""
        self.current_frame_index = 0
        self.is_playing = False
    
    def register_frame_callback(self, callback: Callable[[ReplayFrame], None]) -> None:
        """Register frame change callback."""
        self.frame_callbacks.append(callback)
    
    def register_event_callback(self, event_type: str, callback: Callable) -> None:
        """Register event-specific callback."""
        if event_type not in self.event_callbacks:
            self.event_callbacks[event_type] = []
        self.event_callbacks[event_type].append(callback)
    
    def _trigger_frame_callbacks(self, frame: ReplayFrame) -> None:
        """Trigger frame callbacks."""
        for callback in self.frame_callbacks:
            try:
                callback(frame)
            except Exception as e:
                logger.error(f"Frame callback error: {e}")
        
        # Trigger event callbacks
        for event in frame.events:
            event_type = event.get("type")
            if event_type in self.event_callbacks:
                for callback in self.event_callbacks[event_type]:
                    try:
                        callback(event)
                    except Exception as e:
                        logger.error(f"Event callback error: {e}")
    
    def find_frames(self, condition: Callable[[ReplayFrame], bool]) -> List[int]:
        """
        Find frames matching condition.
        
        Args:
            condition: Condition function
            
        Returns:
            List of frame indices
        """
        matching = []
        for i, frame in enumerate(self.frames):
            if condition(frame):
                matching.append(i)
        return matching
    
    def get_turn_summary(self, turn_number: int) -> Dict[str, Any]:
        """
        Get summary for a turn.
        
        Args:
            turn_number: Turn number
            
        Returns:
            Turn summary
        """
        turn_frames = [f for f in self.frames if f.turn_number == turn_number]
        
        if not turn_frames:
            return {}
        
        first_frame = turn_frames[0]
        last_frame = turn_frames[-1]
        
        # Collect all events and messages
        all_events = []
        all_messages = []
        for frame in turn_frames:
            all_events.extend(frame.events)
            all_messages.extend(frame.messages)
        
        return {
            "turn": turn_number,
            "frame_count": len(turn_frames),
            "start_frame": first_frame.frame_number,
            "end_frame": last_frame.frame_number,
            "active_agents": last_frame.active_agents,
            "eliminated": last_frame.eliminated,
            "scores": last_frame.scores,
            "event_count": len(all_events),
            "message_count": len(all_messages),
            "events": all_events,
            "messages": all_messages
        }


class ReplayAnalyzer:
    """
    Analyzes replay data for insights.
    """
    
    def __init__(self, viewer: ReplayViewer):
        """
        Initialize analyzer.
        
        Args:
            viewer: Replay viewer
        """
        self.viewer = viewer
        self.frames = viewer.frames
        self.metadata = viewer.metadata
    
    def analyze_game_flow(self) -> Dict[str, Any]:
        """
        Analyze overall game flow.
        
        Returns:
            Game flow analysis
        """
        # Analyze elimination pattern
        elimination_timeline = []
        for frame in self.frames:
            if frame.eliminated:
                elimination_timeline.append({
                    "turn": frame.turn_number,
                    "frame": frame.frame_number,
                    "eliminated": frame.eliminated
                })
        
        # Analyze score progression
        score_progression = {}
        sample_interval = max(1, len(self.frames) // 20)  # Sample 20 points
        
        for i in range(0, len(self.frames), sample_interval):
            frame = self.frames[i]
            for agent, score in frame.scores.items():
                if agent not in score_progression:
                    score_progression[agent] = []
                score_progression[agent].append({
                    "turn": frame.turn_number,
                    "score": score
                })
        
        # Find critical moments
        critical_moments = []
        
        # Major score changes
        for i in range(1, len(self.frames)):
            prev_scores = self.frames[i-1].scores
            curr_scores = self.frames[i].scores
            
            for agent in curr_scores:
                if agent in prev_scores:
                    change = curr_scores[agent] - prev_scores[agent]
                    if abs(change) > 10:  # Significant change
                        critical_moments.append({
                            "frame": i,
                            "turn": self.frames[i].turn_number,
                            "type": "score_change",
                            "agent": agent,
                            "change": change
                        })
        
        return {
            "total_turns": self.metadata.total_turns,
            "total_frames": self.metadata.total_frames,
            "elimination_timeline": elimination_timeline,
            "score_progression": score_progression,
            "critical_moments": critical_moments[:10],  # Top 10
            "winner": self.metadata.winner,
            "final_scores": self.metadata.final_scores
        }
    
    def analyze_agent_performance(self, agent_id: str) -> Dict[str, Any]:
        """
        Analyze specific agent's performance.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Agent performance analysis
        """
        # Track agent through frames
        active_frames = 0
        score_history = []
        message_count = 0
        event_participation = {}
        
        for frame in self.frames:
            if agent_id in frame.active_agents:
                active_frames += 1
                
                # Track score
                if agent_id in frame.scores:
                    score_history.append(frame.scores[agent_id])
                
                # Count messages
                for msg in frame.messages:
                    if msg.get("sender_id") == agent_id:
                        message_count += 1
                
                # Track events
                for event in frame.events:
                    if agent_id in event.get("participants", []):
                        event_type = event.get("type", "unknown")
                        event_participation[event_type] = event_participation.get(event_type, 0) + 1
        
        # Calculate statistics
        eliminated_turn = None
        for frame in self.frames:
            if agent_id in frame.eliminated:
                eliminated_turn = frame.turn_number
                break
        
        avg_score = sum(score_history) / len(score_history) if score_history else 0
        max_score = max(score_history) if score_history else 0
        final_score = self.metadata.final_scores.get(agent_id, 0)
        
        return {
            "agent_id": agent_id,
            "active_frames": active_frames,
            "survival_rate": active_frames / len(self.frames) if self.frames else 0,
            "eliminated_turn": eliminated_turn,
            "average_score": avg_score,
            "max_score": max_score,
            "final_score": final_score,
            "message_count": message_count,
            "messages_per_turn": message_count / self.metadata.total_turns if self.metadata.total_turns else 0,
            "event_participation": event_participation,
            "is_winner": agent_id == self.metadata.winner
        }
    
    def find_key_moments(self) -> List[Dict[str, Any]]:
        """
        Find key moments in the game.
        
        Returns:
            List of key moments
        """
        key_moments = []
        
        # First elimination
        for frame in self.frames:
            if frame.eliminated:
                key_moments.append({
                    "frame": frame.frame_number,
                    "turn": frame.turn_number,
                    "type": "first_elimination",
                    "description": f"First elimination: {frame.eliminated[0]}"
                })
                break
        
        # Lead changes
        current_leader = None
        for frame in self.frames:
            if frame.scores:
                new_leader = max(frame.scores, key=frame.scores.get)
                if new_leader != current_leader:
                    key_moments.append({
                        "frame": frame.frame_number,
                        "turn": frame.turn_number,
                        "type": "lead_change",
                        "description": f"Lead changed to {new_leader}"
                    })
                    current_leader = new_leader
        
        # Mass eliminations
        for frame in self.frames:
            if len(frame.eliminated) >= 3:
                key_moments.append({
                    "frame": frame.frame_number,
                    "turn": frame.turn_number,
                    "type": "mass_elimination",
                    "description": f"Mass elimination: {len(frame.eliminated)} agents"
                })
        
        # Final turn
        if self.frames:
            final_frame = self.frames[-1]
            key_moments.append({
                "frame": final_frame.frame_number,
                "turn": final_frame.turn_number,
                "type": "game_end",
                "description": f"Game ended. Winner: {self.metadata.winner}"
            })
        
        # Sort by frame number
        key_moments.sort(key=lambda m: m["frame"])
        
        return key_moments
    
    def generate_statistics(self) -> Dict[str, Any]:
        """
        Generate comprehensive game statistics.
        
        Returns:
            Game statistics
        """
        total_messages = sum(len(f.messages) for f in self.frames)
        total_events = sum(len(f.events) for f in self.frames)
        total_eliminations = sum(len(f.eliminated) for f in self.frames)
        
        # Calculate turn durations
        turn_durations = {}
        for i in range(len(self.frames) - 1):
            curr_frame = self.frames[i]
            next_frame = self.frames[i + 1]
            
            if next_frame.turn_number > curr_frame.turn_number:
                duration = (next_frame.timestamp - curr_frame.timestamp).total_seconds()
                turn_durations[curr_frame.turn_number] = duration
        
        avg_turn_duration = sum(turn_durations.values()) / len(turn_durations) if turn_durations else 0
        
        return {
            "game_id": self.metadata.game_id,
            "total_duration": self.metadata.duration.total_seconds(),
            "total_turns": self.metadata.total_turns,
            "total_frames": self.metadata.total_frames,
            "frames_per_turn": self.metadata.total_frames / self.metadata.total_turns if self.metadata.total_turns else 0,
            "total_participants": len(self.metadata.participants),
            "total_messages": total_messages,
            "total_events": total_events,
            "total_eliminations": total_eliminations,
            "elimination_rate": total_eliminations / self.metadata.total_turns if self.metadata.total_turns else 0,
            "average_turn_duration": avg_turn_duration,
            "winner": self.metadata.winner,
            "final_scores": self.metadata.final_scores
        }


class ReplayManager:
    """
    Manages replay recordings and playback.
    """
    
    def __init__(self, storage_dir: str = "arena_replays"):
        """
        Initialize replay manager.
        
        Args:
            storage_dir: Directory for replays
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Active recordings
        self.active_recordings: Dict[str, ReplayRecorder] = {}
    
    def start_recording(self, game_id: str) -> ReplayRecorder:
        """
        Start recording a game.
        
        Args:
            game_id: Game ID
            
        Returns:
            Replay recorder
        """
        recorder = ReplayRecorder(game_id)
        self.active_recordings[game_id] = recorder
        return recorder
    
    def stop_recording(self, game_id: str, winner: Optional[str] = None) -> Optional[Path]:
        """
        Stop recording and save replay.
        
        Args:
            game_id: Game ID
            winner: Winner agent ID
            
        Returns:
            Saved replay path
        """
        recorder = self.active_recordings.get(game_id)
        if not recorder:
            return None
        
        # Save replay
        replay_path = self.storage_dir / f"{game_id}_replay.gz"
        recorder.save(replay_path)
        
        # Clean up
        del self.active_recordings[game_id]
        
        logger.info(f"Replay saved: {replay_path}")
        return replay_path
    
    def load_replay(self, replay_id: str) -> Optional[ReplayViewer]:
        """
        Load a replay.
        
        Args:
            replay_id: Replay ID or path
            
        Returns:
            Replay viewer
        """
        # Try as path first
        replay_path = Path(replay_id)
        if not replay_path.exists():
            # Try in storage directory
            replay_path = self.storage_dir / f"{replay_id}.gz"
            if not replay_path.exists():
                replay_path = self.storage_dir / f"{replay_id}_replay.gz"
        
        if not replay_path.exists():
            logger.error(f"Replay not found: {replay_id}")
            return None
        
        return ReplayViewer.load(replay_path)
    
    def list_replays(self) -> List[Dict[str, Any]]:
        """
        List available replays.
        
        Returns:
            List of replay information
        """
        replays = []
        
        for replay_file in self.storage_dir.glob("*.gz"):
            # Load metadata
            try:
                with gzip.open(replay_file, 'rt') as f:
                    data = json.load(f)
                    metadata = ReplayMetadata.from_dict(data['metadata'])
                    
                    replays.append({
                        "file": replay_file.name,
                        "path": str(replay_file),
                        "replay_id": metadata.replay_id,
                        "game_id": metadata.game_id,
                        "created": metadata.created_at,
                        "turns": metadata.total_turns,
                        "winner": metadata.winner,
                        "size": replay_file.stat().st_size
                    })
            except Exception as e:
                logger.error(f"Error loading replay metadata: {e}")
        
        # Sort by creation time (newest first)
        replays.sort(key=lambda r: r['created'], reverse=True)
        
        return replays
    
    def delete_replay(self, replay_id: str) -> bool:
        """
        Delete a replay.
        
        Args:
            replay_id: Replay ID
            
        Returns:
            Success status
        """
        replay_path = self.storage_dir / f"{replay_id}.gz"
        if not replay_path.exists():
            replay_path = self.storage_dir / f"{replay_id}_replay.gz"
        
        if replay_path.exists():
            replay_path.unlink()
            logger.info(f"Deleted replay: {replay_id}")
            return True
        
        return False
    
    def export_replay(
        self,
        replay_id: str,
        export_path: Path,
        format: str = "json"
    ) -> bool:
        """
        Export replay to different format.
        
        Args:
            replay_id: Replay ID
            export_path: Export path
            format: Export format (json, csv)
            
        Returns:
            Success status
        """
        viewer = self.load_replay(replay_id)
        if not viewer:
            return False
        
        if format == "json":
            # Export as JSON
            data = {
                "metadata": viewer.metadata.to_dict(),
                "frames": [f.to_dict() for f in viewer.frames]
            }
            
            with open(export_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        elif format == "csv":
            # Export as CSV (simplified)
            import csv
            
            with open(export_path, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Header
                writer.writerow([
                    "frame", "turn", "timestamp", "active_agents",
                    "eliminated", "event_count", "message_count"
                ])
                
                # Data
                for frame in viewer.frames:
                    writer.writerow([
                        frame.frame_number,
                        frame.turn_number,
                        frame.timestamp.isoformat(),
                        len(frame.active_agents),
                        len(frame.eliminated),
                        len(frame.events),
                        len(frame.messages)
                    ])
        
        logger.info(f"Exported replay to: {export_path}")
        return True