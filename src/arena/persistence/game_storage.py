"""
Game Storage System for Arena

This module handles saving and loading complete game states,
including compression, versioning, and archival.

Features:
- Game state serialization
- Compression for storage efficiency
- Version compatibility
- Archive management
- Quick save/load functionality

Author: Homunculus Team
"""

import logging
import json
import pickle
import gzip
import shutil
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class StorageFormat(Enum):
    """Storage format options."""
    JSON = "json"
    PICKLE = "pickle"
    COMPRESSED_JSON = "json.gz"
    COMPRESSED_PICKLE = "pkl.gz"


@dataclass
class SaveMetadata:
    """Metadata for a saved game."""
    save_id: str
    game_id: str
    save_name: str
    save_type: str  # "manual", "auto", "checkpoint"
    created_at: datetime
    game_turn: int
    active_agents: int
    file_size: int
    format: StorageFormat
    version: str = "1.0.0"
    description: Optional[str] = None
    tags: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['format'] = self.format.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SaveMetadata':
        """Create from dictionary."""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['format'] = StorageFormat(data['format'])
        return cls(**data)


class SaveGame:
    """
    Handles saving game states.
    """
    
    def __init__(
        self,
        storage_dir: str = "arena_saves",
        default_format: StorageFormat = StorageFormat.COMPRESSED_JSON
    ):
        """
        Initialize save game handler.
        
        Args:
            storage_dir: Directory for saves
            default_format: Default storage format
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.default_format = default_format
        
        # Create subdirectories
        self.manual_dir = self.storage_dir / "manual"
        self.auto_dir = self.storage_dir / "auto"
        self.checkpoint_dir = self.storage_dir / "checkpoint"
        
        for dir in [self.manual_dir, self.auto_dir, self.checkpoint_dir]:
            dir.mkdir(exist_ok=True)
    
    def save(
        self,
        game_state: Dict[str, Any],
        save_name: str,
        save_type: str = "manual",
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        format: Optional[StorageFormat] = None
    ) -> SaveMetadata:
        """
        Save game state.
        
        Args:
            game_state: Complete game state
            save_name: Name for the save
            save_type: Type of save (manual, auto, checkpoint)
            description: Optional description
            tags: Optional tags
            format: Storage format
            
        Returns:
            Save metadata
        """
        format = format or self.default_format
        
        # Generate save ID
        save_id = f"{save_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Determine directory
        if save_type == "manual":
            save_dir = self.manual_dir
        elif save_type == "auto":
            save_dir = self.auto_dir
        else:
            save_dir = self.checkpoint_dir
        
        # Determine file path
        if format == StorageFormat.JSON:
            file_path = save_dir / f"{save_id}.json"
            self._save_json(game_state, file_path)
        elif format == StorageFormat.PICKLE:
            file_path = save_dir / f"{save_id}.pkl"
            self._save_pickle(game_state, file_path)
        elif format == StorageFormat.COMPRESSED_JSON:
            file_path = save_dir / f"{save_id}.json.gz"
            self._save_compressed_json(game_state, file_path)
        elif format == StorageFormat.COMPRESSED_PICKLE:
            file_path = save_dir / f"{save_id}.pkl.gz"
            self._save_compressed_pickle(game_state, file_path)
        
        # Create metadata
        metadata = SaveMetadata(
            save_id=save_id,
            game_id=game_state.get("game_id", "unknown"),
            save_name=save_name,
            save_type=save_type,
            created_at=datetime.utcnow(),
            game_turn=game_state.get("current_turn", 0),
            active_agents=len(game_state.get("active_agents", [])),
            file_size=file_path.stat().st_size,
            format=format,
            description=description,
            tags=tags or []
        )
        
        # Save metadata
        metadata_path = file_path.with_suffix('.meta.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        logger.info(f"Game saved: {save_id} ({format.value})")
        return metadata
    
    def _save_json(self, data: Dict[str, Any], path: Path) -> None:
        """Save as JSON."""
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _save_pickle(self, data: Dict[str, Any], path: Path) -> None:
        """Save as pickle."""
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def _save_compressed_json(self, data: Dict[str, Any], path: Path) -> None:
        """Save as compressed JSON."""
        json_str = json.dumps(data, indent=2, default=str)
        with gzip.open(path, 'wt', encoding='utf-8') as f:
            f.write(json_str)
    
    def _save_compressed_pickle(self, data: Dict[str, Any], path: Path) -> None:
        """Save as compressed pickle."""
        with gzip.open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def quick_save(self, game_state: Dict[str, Any], slot: int = 0) -> SaveMetadata:
        """
        Quick save to numbered slot.
        
        Args:
            game_state: Game state
            slot: Quick save slot number
            
        Returns:
            Save metadata
        """
        return self.save(
            game_state,
            f"quicksave_{slot}",
            save_type="manual",
            description=f"Quick save slot {slot}"
        )
    
    def auto_save(self, game_state: Dict[str, Any]) -> SaveMetadata:
        """
        Automatic save.
        
        Args:
            game_state: Game state
            
        Returns:
            Save metadata
        """
        return self.save(
            game_state,
            "autosave",
            save_type="auto",
            description="Automatic save"
        )
    
    def checkpoint(self, game_state: Dict[str, Any], checkpoint_name: str) -> SaveMetadata:
        """
        Create checkpoint save.
        
        Args:
            game_state: Game state
            checkpoint_name: Checkpoint name
            
        Returns:
            Save metadata
        """
        return self.save(
            game_state,
            f"checkpoint_{checkpoint_name}",
            save_type="checkpoint",
            description=f"Checkpoint: {checkpoint_name}"
        )


class LoadGame:
    """
    Handles loading game states.
    """
    
    def __init__(self, storage_dir: str = "arena_saves"):
        """
        Initialize load game handler.
        
        Args:
            storage_dir: Directory for saves
        """
        self.storage_dir = Path(storage_dir)
        
        # Subdirectories
        self.manual_dir = self.storage_dir / "manual"
        self.auto_dir = self.storage_dir / "auto"
        self.checkpoint_dir = self.storage_dir / "checkpoint"
    
    def load(self, save_id: str) -> Optional[Dict[str, Any]]:
        """
        Load game by save ID.
        
        Args:
            save_id: Save ID
            
        Returns:
            Game state or None
        """
        # Search for save file
        save_file = self._find_save_file(save_id)
        if not save_file:
            logger.error(f"Save not found: {save_id}")
            return None
        
        # Load metadata
        metadata = self._load_metadata(save_file)
        if not metadata:
            logger.error(f"Metadata not found for: {save_id}")
            return None
        
        # Load based on format
        if metadata.format == StorageFormat.JSON:
            return self._load_json(save_file)
        elif metadata.format == StorageFormat.PICKLE:
            return self._load_pickle(save_file)
        elif metadata.format == StorageFormat.COMPRESSED_JSON:
            return self._load_compressed_json(save_file)
        elif metadata.format == StorageFormat.COMPRESSED_PICKLE:
            return self._load_compressed_pickle(save_file)
        
        return None
    
    def _find_save_file(self, save_id: str) -> Optional[Path]:
        """Find save file by ID."""
        # Search all directories
        for dir in [self.manual_dir, self.auto_dir, self.checkpoint_dir]:
            if not dir.exists():
                continue
            
            for file in dir.iterdir():
                if file.stem.startswith(save_id) and not file.name.endswith('.meta.json'):
                    return file
        
        return None
    
    def _load_metadata(self, save_file: Path) -> Optional[SaveMetadata]:
        """Load save metadata."""
        metadata_file = save_file.parent / f"{save_file.stem}.meta.json"
        if not metadata_file.exists():
            # Try to infer metadata from filename
            return None
        
        with open(metadata_file, 'r') as f:
            data = json.load(f)
            return SaveMetadata.from_dict(data)
    
    def _load_json(self, path: Path) -> Dict[str, Any]:
        """Load JSON file."""
        with open(path, 'r') as f:
            return json.load(f)
    
    def _load_pickle(self, path: Path) -> Dict[str, Any]:
        """Load pickle file."""
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def _load_compressed_json(self, path: Path) -> Dict[str, Any]:
        """Load compressed JSON."""
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_compressed_pickle(self, path: Path) -> Dict[str, Any]:
        """Load compressed pickle."""
        with gzip.open(path, 'rb') as f:
            return pickle.load(f)
    
    def quick_load(self, slot: int = 0) -> Optional[Dict[str, Any]]:
        """
        Quick load from numbered slot.
        
        Args:
            slot: Quick save slot number
            
        Returns:
            Game state or None
        """
        return self.load(f"quicksave_{slot}")
    
    def load_latest_auto_save(self) -> Optional[Dict[str, Any]]:
        """
        Load latest auto save.
        
        Returns:
            Game state or None
        """
        if not self.auto_dir.exists():
            return None
        
        # Find latest auto save
        auto_saves = list(self.auto_dir.glob("autosave_*.json*")) + \
                    list(self.auto_dir.glob("autosave_*.pkl*"))
        
        if not auto_saves:
            return None
        
        # Sort by modification time
        latest = max(auto_saves, key=lambda f: f.stat().st_mtime)
        save_id = latest.stem.split('.')[0]  # Remove extensions
        
        return self.load(save_id)
    
    def list_saves(self, save_type: Optional[str] = None) -> List[SaveMetadata]:
        """
        List available saves.
        
        Args:
            save_type: Filter by save type
            
        Returns:
            List of save metadata
        """
        saves = []
        
        # Determine directories to search
        if save_type == "manual":
            dirs = [self.manual_dir]
        elif save_type == "auto":
            dirs = [self.auto_dir]
        elif save_type == "checkpoint":
            dirs = [self.checkpoint_dir]
        else:
            dirs = [self.manual_dir, self.auto_dir, self.checkpoint_dir]
        
        # Search directories
        for dir in dirs:
            if not dir.exists():
                continue
            
            for meta_file in dir.glob("*.meta.json"):
                with open(meta_file, 'r') as f:
                    data = json.load(f)
                    saves.append(SaveMetadata.from_dict(data))
        
        # Sort by creation time (newest first)
        saves.sort(key=lambda s: s.created_at, reverse=True)
        
        return saves


class GameArchive:
    """
    Manages game archives for long-term storage.
    """
    
    def __init__(
        self,
        archive_dir: str = "arena_archives",
        compression_level: int = 6
    ):
        """
        Initialize game archive.
        
        Args:
            archive_dir: Directory for archives
            compression_level: Compression level (1-9)
        """
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(exist_ok=True)
        self.compression_level = compression_level
    
    def archive_game(
        self,
        game_id: str,
        saves_dir: str = "arena_saves"
    ) -> Path:
        """
        Archive all saves for a game.
        
        Args:
            game_id: Game ID
            saves_dir: Directory containing saves
            
        Returns:
            Archive file path
        """
        saves_path = Path(saves_dir)
        archive_name = f"{game_id}_{datetime.utcnow().strftime('%Y%m%d')}.tar.gz"
        archive_path = self.archive_dir / archive_name
        
        # Find all saves for this game
        game_saves = []
        for save_dir in [saves_path / "manual", saves_path / "auto", saves_path / "checkpoint"]:
            if not save_dir.exists():
                continue
            
            for meta_file in save_dir.glob("*.meta.json"):
                with open(meta_file, 'r') as f:
                    data = json.load(f)
                    if data.get("game_id") == game_id:
                        # Get the actual save file
                        save_file = meta_file.parent / meta_file.stem.replace('.meta', '')
                        if save_file.exists():
                            game_saves.append(save_file)
                            game_saves.append(meta_file)
        
        if not game_saves:
            logger.warning(f"No saves found for game: {game_id}")
            return None
        
        # Create tar.gz archive
        import tarfile
        with tarfile.open(archive_path, 'w:gz', compresslevel=self.compression_level) as tar:
            for save_file in game_saves:
                arcname = f"{game_id}/{save_file.parent.name}/{save_file.name}"
                tar.add(save_file, arcname=arcname)
        
        logger.info(f"Archived {len(game_saves)} files for game {game_id}")
        return archive_path
    
    def extract_archive(self, archive_path: Path, target_dir: Optional[str] = None) -> List[Path]:
        """
        Extract game archive.
        
        Args:
            archive_path: Path to archive
            target_dir: Target directory
            
        Returns:
            List of extracted files
        """
        import tarfile
        
        target = Path(target_dir) if target_dir else self.archive_dir / "extracted"
        target.mkdir(exist_ok=True, parents=True)
        
        extracted_files = []
        
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(target)
            for member in tar.getmembers():
                if member.isfile():
                    extracted_files.append(target / member.name)
        
        logger.info(f"Extracted {len(extracted_files)} files from archive")
        return extracted_files
    
    def list_archives(self) -> List[Dict[str, Any]]:
        """
        List available archives.
        
        Returns:
            List of archive information
        """
        archives = []
        
        for archive_file in self.archive_dir.glob("*.tar.gz"):
            info = {
                "filename": archive_file.name,
                "path": str(archive_file),
                "size": archive_file.stat().st_size,
                "created": datetime.fromtimestamp(archive_file.stat().st_ctime),
                "game_id": archive_file.stem.split('_')[0]
            }
            archives.append(info)
        
        # Sort by creation time (newest first)
        archives.sort(key=lambda a: a['created'], reverse=True)
        
        return archives
    
    def cleanup_old_saves(
        self,
        saves_dir: str = "arena_saves",
        days_to_keep: int = 30
    ) -> int:
        """
        Clean up old saves.
        
        Args:
            saves_dir: Directory containing saves
            days_to_keep: Number of days to keep saves
            
        Returns:
            Number of files deleted
        """
        saves_path = Path(saves_dir)
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        deleted = 0
        
        for save_dir in [saves_path / "auto", saves_path / "checkpoint"]:
            if not save_dir.exists():
                continue
            
            for file in save_dir.iterdir():
                # Check file age
                file_time = datetime.fromtimestamp(file.stat().st_mtime)
                if file_time < cutoff_date:
                    file.unlink()
                    deleted += 1
        
        logger.info(f"Deleted {deleted} old save files")
        return deleted


class GameStorage:
    """
    Main interface for game storage operations.
    """
    
    def __init__(
        self,
        storage_dir: str = "arena_saves",
        archive_dir: str = "arena_archives"
    ):
        """
        Initialize game storage.
        
        Args:
            storage_dir: Directory for saves
            archive_dir: Directory for archives
        """
        self.saver = SaveGame(storage_dir)
        self.loader = LoadGame(storage_dir)
        self.archiver = GameArchive(archive_dir)
        
        # Auto-save settings
        self.auto_save_enabled = True
        self.auto_save_interval = 10  # turns
        self.max_auto_saves = 5
        
        # Quick save slots
        self.max_quick_saves = 10
    
    def save_game(
        self,
        game_state: Dict[str, Any],
        save_name: str,
        **kwargs
    ) -> SaveMetadata:
        """Save game state."""
        return self.saver.save(game_state, save_name, **kwargs)
    
    def load_game(self, save_id: str) -> Optional[Dict[str, Any]]:
        """Load game state."""
        return self.loader.load(save_id)
    
    def quick_save(self, game_state: Dict[str, Any], slot: int = 0) -> SaveMetadata:
        """Quick save to slot."""
        if slot >= self.max_quick_saves:
            slot = slot % self.max_quick_saves
        return self.saver.quick_save(game_state, slot)
    
    def quick_load(self, slot: int = 0) -> Optional[Dict[str, Any]]:
        """Quick load from slot."""
        if slot >= self.max_quick_saves:
            slot = slot % self.max_quick_saves
        return self.loader.quick_load(slot)
    
    def auto_save(self, game_state: Dict[str, Any]) -> Optional[SaveMetadata]:
        """Perform auto save if enabled."""
        if not self.auto_save_enabled:
            return None
        
        turn = game_state.get("current_turn", 0)
        if turn % self.auto_save_interval == 0:
            # Clean up old auto saves
            self._cleanup_auto_saves()
            return self.saver.auto_save(game_state)
        
        return None
    
    def _cleanup_auto_saves(self) -> None:
        """Clean up excess auto saves."""
        auto_saves = self.loader.list_saves(save_type="auto")
        
        if len(auto_saves) >= self.max_auto_saves:
            # Delete oldest auto saves
            to_delete = auto_saves[self.max_auto_saves-1:]
            for save in to_delete:
                save_file = self.loader._find_save_file(save.save_id)
                if save_file:
                    save_file.unlink()
                    # Also delete metadata
                    meta_file = save_file.parent / f"{save_file.stem}.meta.json"
                    if meta_file.exists():
                        meta_file.unlink()
    
    def archive_game(self, game_id: str) -> Optional[Path]:
        """Archive a completed game."""
        return self.archiver.archive_game(game_id, self.saver.storage_dir)
    
    def list_saves(self, **kwargs) -> List[SaveMetadata]:
        """List available saves."""
        return self.loader.list_saves(**kwargs)
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        total_size = 0
        save_counts = {"manual": 0, "auto": 0, "checkpoint": 0}
        
        for save_type in ["manual", "auto", "checkpoint"]:
            saves = self.loader.list_saves(save_type=save_type)
            save_counts[save_type] = len(saves)
            for save in saves:
                total_size += save.file_size
        
        archives = self.archiver.list_archives()
        archive_size = sum(a['size'] for a in archives)
        
        return {
            "total_saves": sum(save_counts.values()),
            "save_counts": save_counts,
            "total_archives": len(archives),
            "saves_size_bytes": total_size,
            "archives_size_bytes": archive_size,
            "total_size_bytes": total_size + archive_size,
            "total_size_mb": (total_size + archive_size) / (1024 * 1024)
        }