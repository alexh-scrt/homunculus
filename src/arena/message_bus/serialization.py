"""
Message Serialization Utilities for Arena

This module provides serialization and deserialization utilities
for converting between Arena models and wire formats (JSON, MessagePack, etc.).

Features:
- JSON serialization with custom encoders
- MessagePack support for performance
- Compression options
- Schema validation
- Error recovery

Author: Homunculus Team
"""

import json
import logging
import gzip
import base64
from typing import Any, Dict, Optional, Type, TypeVar, Union, List
from datetime import datetime, date
from enum import Enum
from dataclasses import is_dataclass, asdict
import uuid

from ..models import (
    Message, MessageBatch, AgentState, Accusation, 
    ArenaState, ScoringMetrics, Evidence
)


logger = logging.getLogger(__name__)

T = TypeVar('T')


class SerializationFormat(Enum):
    """Supported serialization formats."""
    JSON = "json"
    JSON_COMPRESSED = "json_compressed"
    MSGPACK = "msgpack"  # Future enhancement
    PROTOBUF = "protobuf"  # Future enhancement


class ArenaJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for Arena models.
    
    Handles special types like datetime, UUID, Enum, and dataclasses.
    """
    
    def default(self, obj: Any) -> Any:
        """
        Convert special types to JSON-serializable format.
        
        Args:
            obj: Object to encode
            
        Returns:
            JSON-serializable representation
        """
        # Handle datetime
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        
        # Handle UUID
        if isinstance(obj, uuid.UUID):
            return str(obj)
        
        # Handle Enum
        if isinstance(obj, Enum):
            return obj.value
        
        # Handle dataclasses
        if is_dataclass(obj) and hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif is_dataclass(obj):
            return asdict(obj)
        
        # Handle sets
        if isinstance(obj, set):
            return list(obj)
        
        # Handle bytes
        if isinstance(obj, bytes):
            return base64.b64encode(obj).decode('utf-8')
        
        # Fallback to default
        return super().default(obj)


class MessageSerializer:
    """
    Serializer for Arena messages and models.
    
    This class provides methods to serialize and deserialize Arena
    models to/from various formats with error handling and validation.
    """
    
    @staticmethod
    def serialize(
        obj: Any,
        format: SerializationFormat = SerializationFormat.JSON,
        validate: bool = True
    ) -> bytes:
        """
        Serialize an object to bytes.
        
        Args:
            obj: Object to serialize
            format: Serialization format to use
            validate: Whether to validate before serializing
            
        Returns:
            Serialized bytes
            
        Raises:
            ValueError: If serialization fails
        """
        try:
            # Validate if requested
            if validate and hasattr(obj, '__post_init__'):
                obj.__post_init__()  # Re-run validation
            
            # Convert to dict if it's a model
            if hasattr(obj, 'to_dict'):
                data = obj.to_dict()
            elif is_dataclass(obj):
                data = asdict(obj)
            else:
                data = obj
            
            # Serialize based on format
            if format == SerializationFormat.JSON:
                json_str = json.dumps(data, cls=ArenaJSONEncoder, ensure_ascii=False)
                return json_str.encode('utf-8')
                
            elif format == SerializationFormat.JSON_COMPRESSED:
                json_str = json.dumps(data, cls=ArenaJSONEncoder, ensure_ascii=False)
                return gzip.compress(json_str.encode('utf-8'))
                
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            logger.error(f"Serialization failed: {e}")
            raise ValueError(f"Failed to serialize object: {e}")
    
    @staticmethod
    def deserialize(
        data: bytes,
        target_type: Type[T],
        format: SerializationFormat = SerializationFormat.JSON,
        validate: bool = True
    ) -> T:
        """
        Deserialize bytes to an object.
        
        Args:
            data: Bytes to deserialize
            target_type: Type to deserialize to
            format: Serialization format used
            validate: Whether to validate after deserializing
            
        Returns:
            Deserialized object
            
        Raises:
            ValueError: If deserialization fails
        """
        try:
            # Decompress if needed
            if format == SerializationFormat.JSON_COMPRESSED:
                data = gzip.decompress(data)
            
            # Parse JSON
            if format in [SerializationFormat.JSON, SerializationFormat.JSON_COMPRESSED]:
                json_str = data.decode('utf-8')
                json_data = json.loads(json_str)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Convert to target type
            if hasattr(target_type, 'from_dict'):
                obj = target_type.from_dict(json_data)
            else:
                obj = target_type(**json_data)
            
            # Validate if requested
            if validate and hasattr(obj, '__post_init__'):
                obj.__post_init__()
            
            return obj
            
        except Exception as e:
            logger.error(f"Deserialization failed: {e}")
            raise ValueError(f"Failed to deserialize to {target_type.__name__}: {e}")
    
    @staticmethod
    def serialize_message(
        message: Message,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Serialize a message to a dictionary.
        
        Args:
            message: Message to serialize
            include_metadata: Whether to include metadata
            
        Returns:
            Dictionary representation
        """
        data = {
            "message_id": message.message_id,
            "timestamp": message.timestamp.isoformat(),
            "sender_id": message.sender_id,
            "sender_name": message.sender_name,
            "sender_type": message.sender_type,
            "message_type": message.message_type,
            "content": message.content,
            "turn_number": message.turn_number,
            "game_id": message.game_id
        }
        
        if message.target_agent_id:
            data["target_agent_id"] = message.target_agent_id
        
        if message.confidence_score:
            data["confidence_score"] = message.confidence_score
        
        if message.references:
            data["references"] = list(message.references)
        
        if include_metadata and message.metadata:
            data["metadata"] = message.metadata
        
        return data
    
    @staticmethod
    def deserialize_message(data: Dict[str, Any]) -> Message:
        """
        Deserialize a dictionary to a Message.
        
        Args:
            data: Dictionary to deserialize
            
        Returns:
            Message object
        """
        # Handle timestamp conversion
        if isinstance(data.get('timestamp'), str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        
        # Handle references as set
        if 'references' in data and isinstance(data['references'], list):
            data['references'] = set(data['references'])
        
        return Message.from_dict(data)
    
    @staticmethod
    def serialize_batch(
        messages: Union[MessageBatch, List[Message]],
        compressed: bool = False
    ) -> bytes:
        """
        Serialize a batch of messages.
        
        Args:
            messages: Messages to serialize
            compressed: Whether to compress the output
            
        Returns:
            Serialized bytes
        """
        if isinstance(messages, MessageBatch):
            messages_list = messages.messages
        else:
            messages_list = messages
        
        data = {
            "batch_size": len(messages_list),
            "messages": [MessageSerializer.serialize_message(m) for m in messages_list]
        }
        
        format = SerializationFormat.JSON_COMPRESSED if compressed else SerializationFormat.JSON
        return MessageSerializer.serialize(data, format=format)
    
    @staticmethod
    def deserialize_batch(
        data: bytes,
        compressed: bool = False
    ) -> MessageBatch:
        """
        Deserialize bytes to a MessageBatch.
        
        Args:
            data: Bytes to deserialize
            compressed: Whether the data is compressed
            
        Returns:
            MessageBatch object
        """
        format = SerializationFormat.JSON_COMPRESSED if compressed else SerializationFormat.JSON
        
        if compressed:
            data = gzip.decompress(data)
        
        json_str = data.decode('utf-8')
        json_data = json.loads(json_str)
        
        # Get game_id from first message or use default
        messages = json_data.get('messages', [])
        game_id = messages[0].get('game_id', 'deserialized_batch') if messages else 'deserialized_batch'
        
        batch = MessageBatch(game_id=game_id)
        
        for msg_data in messages:
            message = MessageSerializer.deserialize_message(msg_data)
            # Ensure message has same game_id as batch
            message.game_id = game_id
            batch.add_message(message)
        
        return batch


class ModelSerializer:
    """
    Generic serializer for all Arena models.
    
    Provides a unified interface for serializing any Arena model.
    """
    
    # Model type registry
    MODEL_TYPES = {
        "Message": Message,
        "MessageBatch": MessageBatch,
        "AgentState": AgentState,
        "Accusation": Accusation,
        "ArenaState": ArenaState,
        "ScoringMetrics": ScoringMetrics,
        "Evidence": Evidence
    }
    
    @classmethod
    def serialize(
        cls,
        obj: Any,
        include_type: bool = True,
        compressed: bool = False
    ) -> bytes:
        """
        Serialize any Arena model to bytes.
        
        Args:
            obj: Model object to serialize
            include_type: Whether to include type information
            compressed: Whether to compress output
            
        Returns:
            Serialized bytes
        """
        data = {}
        
        # Include type information if requested
        if include_type:
            type_name = obj.__class__.__name__
            if type_name not in cls.MODEL_TYPES:
                raise ValueError(f"Unknown model type: {type_name}")
            data["__type__"] = type_name
        
        # Get object data
        if hasattr(obj, 'to_dict'):
            data["__data__"] = obj.to_dict()
        else:
            raise ValueError(f"Object {obj} doesn't support serialization")
        
        # Serialize
        format = SerializationFormat.JSON_COMPRESSED if compressed else SerializationFormat.JSON
        return MessageSerializer.serialize(data, format=format)
    
    @classmethod
    def deserialize(
        cls,
        data: bytes,
        target_type: Optional[Type[T]] = None,
        compressed: bool = False
    ) -> T:
        """
        Deserialize bytes to an Arena model.
        
        Args:
            data: Bytes to deserialize
            target_type: Expected type (auto-detect if None)
            compressed: Whether the data is compressed
            
        Returns:
            Deserialized model object
        """
        # Decompress if needed
        if compressed:
            data = gzip.decompress(data)
        
        # Parse JSON
        json_str = data.decode('utf-8')
        json_data = json.loads(json_str)
        
        # Determine type
        if target_type is None:
            if "__type__" not in json_data:
                raise ValueError("No type information in data and target_type not provided")
            
            type_name = json_data["__type__"]
            if type_name not in cls.MODEL_TYPES:
                raise ValueError(f"Unknown model type: {type_name}")
            
            target_type = cls.MODEL_TYPES[type_name]
        
        # Get object data
        obj_data = json_data.get("__data__", json_data)
        
        # Deserialize
        if hasattr(target_type, 'from_dict'):
            return target_type.from_dict(obj_data)
        else:
            return target_type(**obj_data)


def safe_deserialize(
    data: bytes,
    target_type: Type[T],
    default: Optional[T] = None,
    log_errors: bool = True
) -> Optional[T]:
    """
    Safely deserialize data with error handling.
    
    Args:
        data: Data to deserialize
        target_type: Target type
        default: Default value on error
        log_errors: Whether to log errors
        
    Returns:
        Deserialized object or default
    """
    try:
        return MessageSerializer.deserialize(data, target_type)
    except Exception as e:
        if log_errors:
            logger.error(f"Safe deserialization failed: {e}")
        return default