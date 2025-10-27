"""Message protocol definitions for character agent client-server communication."""

from typing import Dict, Any, Optional, Union, Literal
from pydantic import BaseModel
from datetime import datetime
import json


class BaseMessage(BaseModel):
    """Base message class for all client-server communication."""
    
    message_type: str
    message_id: Optional[str] = None
    timestamp: datetime = datetime.now()
    
    def to_json(self) -> str:
        """Convert message to JSON string."""
        return self.model_dump_json()
    
    @classmethod
    def from_json(cls, json_str: str) -> 'BaseMessage':
        """Create message from JSON string."""
        data = json.loads(json_str)
        message_type = data.get('message_type')
        
        # Map message types to their corresponding classes
        type_mapping = {
            'chat_request': ChatRequest,
            'chat_response': ChatResponse,
            'status_request': StatusRequest,
            'status_response': StatusResponse,
            'save_request': SaveRequest,
            'save_response': SaveResponse,
            'load_request': LoadRequest,
            'load_response': LoadResponse,
            'reset_request': ResetRequest,
            'reset_response': ResetResponse,
            'memory_search_request': MemorySearchRequest,
            'memory_search_response': MemorySearchResponse,
            'error_response': ErrorResponse,
            'init_request': InitRequest,
            'init_response': InitResponse,
            'ping_request': PingRequest,
            'ping_response': PingResponse
        }
        
        message_class = type_mapping.get(message_type, BaseMessage)
        return message_class(**data)


# Client -> Server Messages (Requests)

class InitRequest(BaseMessage):
    """Initialize character agent with specific character profile."""
    
    message_type: Literal['init_request'] = 'init_request'
    character_id: str
    character_config: Optional[Dict[str, Any]] = None
    agent_config: Optional[Dict[str, Any]] = None


class ChatRequest(BaseMessage):
    """Send a chat message to the character agent."""
    
    message_type: Literal['chat_request'] = 'chat_request'
    user_message: str
    context: Optional[Dict[str, Any]] = None


class StatusRequest(BaseMessage):
    """Request current character status and state."""
    
    message_type: Literal['status_request'] = 'status_request'
    include_vitals: bool = True
    include_history: bool = False
    include_performance: bool = False


class SaveRequest(BaseMessage):
    """Request to save character state."""
    
    message_type: Literal['save_request'] = 'save_request'
    filename: Optional[str] = None


class LoadRequest(BaseMessage):
    """Request to load character state from file."""
    
    message_type: Literal['load_request'] = 'load_request'
    filename: str


class ResetRequest(BaseMessage):
    """Request to reset character to initial state."""
    
    message_type: Literal['reset_request'] = 'reset_request'
    preserve_memories: bool = True


class MemorySearchRequest(BaseMessage):
    """Request to search character memories."""
    
    message_type: Literal['memory_search_request'] = 'memory_search_request'
    query: str
    limit: int = 5


class PingRequest(BaseMessage):
    """Ping request to check server connectivity."""
    
    message_type: Literal['ping_request'] = 'ping_request'


# Server -> Client Messages (Responses)

class InitResponse(BaseMessage):
    """Response to character initialization."""
    
    message_type: Literal['init_response'] = 'init_response'
    success: bool
    character_name: Optional[str] = None
    character_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ChatResponse(BaseMessage):
    """Response containing character's reply to chat message."""
    
    message_type: Literal['chat_response'] = 'chat_response'
    success: bool
    response_text: Optional[str] = None
    character_insights: Optional[Dict[str, Any]] = None
    response_metadata: Optional[Dict[str, Any]] = None
    generation_info: Optional[Dict[str, Any]] = None
    orchestration_summary: Optional[Dict[str, Any]] = None
    cognitive_summary: Optional[Dict[str, Any]] = None
    character_state_summary: Optional[Dict[str, Any]] = None
    performance_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class StatusResponse(BaseMessage):
    """Response containing character status information."""
    
    message_type: Literal['status_response'] = 'status_response'
    success: bool
    character_summary: Optional[Dict[str, Any]] = None
    vitals: Optional[Dict[str, Any]] = None
    conversation_history: Optional[list] = None
    performance_stats: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class SaveResponse(BaseMessage):
    """Response to save request."""
    
    message_type: Literal['save_response'] = 'save_response'
    success: bool
    filename: Optional[str] = None
    error: Optional[str] = None


class LoadResponse(BaseMessage):
    """Response to load request."""
    
    message_type: Literal['load_response'] = 'load_response'
    success: bool
    character_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ResetResponse(BaseMessage):
    """Response to reset request."""
    
    message_type: Literal['reset_response'] = 'reset_response'
    success: bool
    character_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class MemorySearchResponse(BaseMessage):
    """Response to memory search request."""
    
    message_type: Literal['memory_search_response'] = 'memory_search_response'
    success: bool
    memories: Optional[list] = None
    count: Optional[int] = None
    error: Optional[str] = None


class PingResponse(BaseMessage):
    """Response to ping request."""
    
    message_type: Literal['ping_response'] = 'ping_response'
    success: bool = True
    server_time: datetime = datetime.now()


class ErrorResponse(BaseMessage):
    """Generic error response."""
    
    message_type: Literal['error_response'] = 'error_response'
    error: str
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


# Helper functions

def create_chat_request(user_message: str, context: Optional[Dict[str, Any]] = None, message_id: Optional[str] = None) -> ChatRequest:
    """Helper to create a chat request message."""
    return ChatRequest(
        user_message=user_message,
        context=context,
        message_id=message_id
    )


def create_init_request(character_id: str, character_config: Optional[Dict[str, Any]] = None, message_id: Optional[str] = None) -> InitRequest:
    """Helper to create an init request message."""
    return InitRequest(
        character_id=character_id,
        character_config=character_config,
        message_id=message_id
    )


def create_error_response(error: str, error_code: Optional[str] = None, message_id: Optional[str] = None) -> ErrorResponse:
    """Helper to create an error response message."""
    return ErrorResponse(
        error=error,
        error_code=error_code,
        message_id=message_id
    )


def create_success_response(message_type: str, data: Dict[str, Any], message_id: Optional[str] = None) -> BaseMessage:
    """Helper to create a success response of any type."""
    response_data = {
        'message_type': message_type,
        'success': True,
        'message_id': message_id,
        **data
    }
    
    return BaseMessage.from_json(json.dumps(response_data))


# Message validation functions

def validate_message(message_data: Dict[str, Any]) -> bool:
    """Validate that a message has required fields."""
    required_fields = ['message_type', 'timestamp']
    return all(field in message_data for field in required_fields)


def get_message_type(message_data: Dict[str, Any]) -> Optional[str]:
    """Extract message type from message data."""
    return message_data.get('message_type')