"""Conversation Manager for persistent multi-user conversation history."""

import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import asdict

try:
    from .character_state import CharacterState
    from ..config.settings import get_settings
    from ..config.logging_config import get_logger
except ImportError:
    try:
        # Fallback imports for when running from different contexts
        from character_state import CharacterState
        from config.settings import get_settings
        from config.logging_config import get_logger
    except ImportError:
        # Additional fallback for script execution
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from core.character_state import CharacterState
        from config.settings import get_settings
        from config.logging_config import get_logger


class ConversationManager:
    """
    Manages persistent conversation history for multiple users and characters.
    
    Creates conversation IDs in format: {user_id}-{character_id}
    Stores full conversation history separate from working memory.
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize the conversation manager."""
        self.logger = get_logger(__name__)
        self.settings = get_settings()
        
        # Set up conversation storage directory
        if data_dir:
            self.conversations_dir = data_dir / "conversations"
        else:
            self.conversations_dir = Path(self.settings.data_dir) / "conversations"
        
        self.conversations_dir.mkdir(parents=True, exist_ok=True)
        
        # Conversation index for metadata and quick lookup
        self.index_file = self.conversations_dir / "conversation_index.json"
        self.conversation_index = self._load_conversation_index()
        
        # File locks for thread safety
        self._file_locks: Dict[str, asyncio.Lock] = {}
        
        self.logger.info(f"ConversationManager initialized - storage: {self.conversations_dir}")
    
    def _load_conversation_index(self) -> Dict[str, Any]:
        """Load the conversation index from file."""
        try:
            if self.index_file.exists():
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            else:
                return {
                    "created": datetime.now().isoformat(),
                    "conversations": {},
                    "users": {},
                    "characters": {}
                }
        except Exception as e:
            self.logger.error(f"Failed to load conversation index: {e}")
            return {
                "created": datetime.now().isoformat(),
                "conversations": {},
                "users": {},
                "characters": {}
            }
    
    async def _save_conversation_index(self) -> None:
        """Save the conversation index to file."""
        try:
            self.conversation_index["last_updated"] = datetime.now().isoformat()
            with open(self.index_file, 'w') as f:
                json.dump(self.conversation_index, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save conversation index: {e}")
    
    def _get_conversation_id(self, user_id: str, character_id: str) -> str:
        """Generate conversation ID from user and character IDs."""
        # Sanitize IDs to be filesystem-safe
        safe_user_id = "".join(c for c in user_id if c.isalnum() or c in '-_').lower()
        safe_character_id = "".join(c for c in character_id if c.isalnum() or c in '-_').lower()
        return f"{safe_user_id}-{safe_character_id}"
    
    def _get_conversation_file(self, conversation_id: str) -> Path:
        """Get the file path for a conversation."""
        return self.conversations_dir / f"{conversation_id}.json"
    
    async def _get_file_lock(self, conversation_id: str) -> asyncio.Lock:
        """Get or create a file lock for the conversation."""
        if conversation_id not in self._file_locks:
            self._file_locks[conversation_id] = asyncio.Lock()
        return self._file_locks[conversation_id]
    
    async def conversation_exists(self, user_id: str, character_id: str) -> bool:
        """Check if a conversation exists between user and character."""
        conversation_id = self._get_conversation_id(user_id, character_id)
        conversation_file = self._get_conversation_file(conversation_id)
        return conversation_file.exists()
    
    async def load_conversation_history(self, user_id: str, character_id: str) -> List[Dict[str, Any]]:
        """Load full conversation history for user-character pair."""
        conversation_id = self._get_conversation_id(user_id, character_id)
        conversation_file = self._get_conversation_file(conversation_id)
        
        if not conversation_file.exists():
            self.logger.info(f"No existing conversation found for {conversation_id}")
            return []
        
        try:
            async with await self._get_file_lock(conversation_id):
                with open(conversation_file, 'r') as f:
                    conversation_data = json.load(f)
                
                history = conversation_data.get('messages', [])
                self.logger.info(f"Loaded {len(history)} messages for conversation {conversation_id}")
                return history
                
        except Exception as e:
            self.logger.error(f"Failed to load conversation {conversation_id}: {e}")
            return []
    
    async def save_conversation_history(
        self, 
        user_id: str, 
        character_id: str, 
        messages: List[Dict[str, Any]],
        character_state: Optional[CharacterState] = None
    ) -> bool:
        """Save full conversation history for user-character pair."""
        conversation_id = self._get_conversation_id(user_id, character_id)
        conversation_file = self._get_conversation_file(conversation_id)
        
        try:
            async with await self._get_file_lock(conversation_id):
                # Create conversation data structure
                conversation_data = {
                    "conversation_id": conversation_id,
                    "user_id": user_id,
                    "character_id": character_id,
                    "created": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                    "message_count": len(messages),
                    "messages": messages,
                    "metadata": {
                        "user_first_message": messages[0]["timestamp"] if messages else None,
                        "last_message": messages[-1]["timestamp"] if messages else None,
                        "character_name": character_state.name if character_state else None
                    }
                }
                
                # If file exists, preserve creation date
                if conversation_file.exists():
                    with open(conversation_file, 'r') as f:
                        existing_data = json.load(f)
                        conversation_data["created"] = existing_data.get("created", conversation_data["created"])
                
                # Save conversation
                with open(conversation_file, 'w') as f:
                    json.dump(conversation_data, f, indent=2, default=str)
                
                # Update index
                await self._update_conversation_index(conversation_id, user_id, character_id, len(messages))
                
                self.logger.info(f"Saved conversation {conversation_id} with {len(messages)} messages")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to save conversation {conversation_id}: {e}")
            return False
    
    async def _update_conversation_index(self, conversation_id: str, user_id: str, character_id: str, message_count: int) -> None:
        """Update the conversation index with latest metadata."""
        # Update conversation entry
        self.conversation_index["conversations"][conversation_id] = {
            "user_id": user_id,
            "character_id": character_id,
            "message_count": message_count,
            "last_updated": datetime.now().isoformat()
        }
        
        # Update user index
        if user_id not in self.conversation_index["users"]:
            self.conversation_index["users"][user_id] = {"conversations": [], "first_seen": datetime.now().isoformat()}
        
        if conversation_id not in self.conversation_index["users"][user_id]["conversations"]:
            self.conversation_index["users"][user_id]["conversations"].append(conversation_id)
        
        # Update character index
        if character_id not in self.conversation_index["characters"]:
            self.conversation_index["characters"][character_id] = {"conversations": [], "first_seen": datetime.now().isoformat()}
        
        if conversation_id not in self.conversation_index["characters"][character_id]["conversations"]:
            self.conversation_index["characters"][character_id]["conversations"].append(conversation_id)
        
        # Save index
        await self._save_conversation_index()
    
    async def get_user_conversations(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all conversations for a specific user."""
        user_data = self.conversation_index["users"].get(user_id, {})
        conversation_ids = user_data.get("conversations", [])
        
        conversations = []
        for conv_id in conversation_ids:
            conv_metadata = self.conversation_index["conversations"].get(conv_id, {})
            if conv_metadata:
                conversations.append({
                    "conversation_id": conv_id,
                    "character_id": conv_metadata["character_id"],
                    "message_count": conv_metadata["message_count"],
                    "last_updated": conv_metadata["last_updated"]
                })
        
        return conversations
    
    async def get_character_conversations(self, character_id: str) -> List[Dict[str, Any]]:
        """Get all conversations for a specific character."""
        char_data = self.conversation_index["characters"].get(character_id, {})
        conversation_ids = char_data.get("conversations", [])
        
        conversations = []
        for conv_id in conversation_ids:
            conv_metadata = self.conversation_index["conversations"].get(conv_id, {})
            if conv_metadata:
                conversations.append({
                    "conversation_id": conv_id,
                    "user_id": conv_metadata["user_id"],
                    "message_count": conv_metadata["message_count"],
                    "last_updated": conv_metadata["last_updated"]
                })
        
        return conversations
    
    async def add_message_to_conversation(
        self, 
        user_id: str, 
        character_id: str, 
        role: str, 
        message: str
    ) -> bool:
        """Add a single message to the conversation history."""
        # Load existing history
        history = await self.load_conversation_history(user_id, character_id)
        
        # Add new message
        new_message = {
            "role": role,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        history.append(new_message)
        
        # Save updated history
        return await self.save_conversation_history(user_id, character_id, history)
    
    async def get_recent_messages(self, user_id: str, character_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get the most recent N messages from conversation history."""
        history = await self.load_conversation_history(user_id, character_id)
        return history[-limit:] if history else []
    
    async def get_conversation_stats(self) -> Dict[str, Any]:
        """Get overall conversation statistics."""
        total_conversations = len(self.conversation_index["conversations"])
        total_users = len(self.conversation_index["users"])
        total_characters = len(self.conversation_index["characters"])
        
        # Calculate total messages
        total_messages = sum(
            conv["message_count"] for conv in self.conversation_index["conversations"].values()
        )
        
        return {
            "total_conversations": total_conversations,
            "total_users": total_users,
            "total_characters": total_characters,
            "total_messages": total_messages,
            "storage_directory": str(self.conversations_dir),
            "index_last_updated": self.conversation_index.get("last_updated", "unknown")
        }