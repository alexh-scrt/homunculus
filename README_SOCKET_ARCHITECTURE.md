# Socket-Based Character Agent Architecture

This document describes the new socket-based architecture for the Character Agent system that allows for decoupled debugging and development.

## Overview

The character agent system has been decoupled into two main components:

1. **Character Agent Server**: Runs character agents as independent processes with WebSocket capabilities
2. **Chat CLI Client**: Connects to the server over WebSocket to interact with characters

## Benefits

- **Easier Debugging**: Character agent runs in separate process, can be debugged independently
- **Process Isolation**: Character agent crashes don't affect CLI
- **Multiple Clients**: Multiple CLI instances can connect to same character
- **Remote Deployment**: Character agent can run on different machine
- **Development Workflow**: Start server once, restart CLI multiple times during development

## Usage

### 1. Start the Character Agent Server

```bash
# Start server without pre-loading a character
python scripts/run_character_server.py

# Start server with a specific character (e.g., m-playful)
python scripts/run_character_server.py --character m-playful

# Start server on different port/host
python scripts/run_character_server.py --port 9000 --host 0.0.0.0

# List available characters
python scripts/run_character_server.py --list-characters

# Validate a character configuration
python scripts/run_character_server.py --validate-character m-playful
```

### 2. Connect Chat CLI to Server

```bash
# Connect to remote server
python scripts/run_chat.py --remote

# Connect to specific character on remote server
python scripts/run_chat.py --remote --character m-playful

# Connect to server on different URL
python scripts/run_chat.py --remote --server-url ws://localhost:9000

# Disable fallback to local mode
python scripts/run_chat.py --remote --no-fallback

# Use vitals-only mode with remote server
python scripts/run_chat.py --remote --vitals-only --character m-playful
```

### 3. Local Mode (Fallback)

The system still supports local mode for backward compatibility:

```bash
# Run in local mode (default)
python scripts/run_chat.py --character m-playful

# Force local mode (no server connection attempt)
python scripts/run_chat.py --character m-playful --no-vitals
```

## Development Workflow

### For Character Agent Development

1. Start the character agent server with your target character:
   ```bash
   python scripts/run_character_server.py --character m-playful
   ```

2. The server will run in the foreground with logging output, making it easy to debug character agent issues.

3. Connect clients as needed:
   ```bash
   python scripts/run_chat.py --remote --character m-playful
   ```

4. Make changes to character agent code, restart server, clients will automatically reconnect.

### For CLI Development

1. Start a character agent server once:
   ```bash
   python scripts/run_character_server.py --character m-playful
   ```

2. Develop and test CLI changes by restarting the client:
   ```bash
   python scripts/run_chat.py --remote --character m-playful
   ```

3. The server keeps running, so you can quickly iterate on CLI changes.

## Architecture Components

### Character Agent Server (`src/server/character_agent_server.py`)

- **WebSocket Server**: Hosts character agents and serves clients
- **Character Management**: Loads and manages character instances
- **Message Protocol**: JSON-based request/response protocol
- **State Persistence**: Handles character state saving/loading
- **Multi-Client Support**: Multiple clients can connect to same character

### Character Agent Client (`src/client/character_agent_client.py`)

- **WebSocket Client**: Connects to character agent server
- **Protocol Implementation**: Handles message serialization/deserialization
- **Reconnection Logic**: Automatic reconnection on connection loss
- **Same Interface**: Provides same interface as local CharacterAgent

### Message Protocol (`src/protocol/messages.py`)

- **Typed Messages**: Pydantic-based message definitions
- **Request/Response**: Chat, status, save, load, reset, memory search
- **Error Handling**: Structured error responses
- **Extensible**: Easy to add new message types

### Remote Chat Interface (`src/cli/chat_interface_remote.py`)

- **Dual Mode**: Can connect to remote server or fallback to local
- **Same Features**: All original features work with remote character
- **Connection Management**: Handles connection states and errors
- **Vitals Display**: Real-time vitals from remote character

## Configuration

### Server Configuration

The server uses the same configuration as the local system:

- **Environment Variables**: Uses `.env` file for Ollama settings
- **Character Schemas**: Loads from `schemas/characters/` directory
- **Save Directory**: Uses `./data/saves` for character state files

### Client Configuration

- **Server URL**: Default `ws://localhost:8765`, configurable via `--server-url`
- **Timeout**: 30 second timeout for requests
- **Reconnection**: 3 attempts with 1 second delay
- **Fallback**: Can fallback to local mode if server unavailable

## Testing

Run the test suite to verify the socket architecture:

```bash
python test_socket_architecture.py
```

This will:
1. Start a test server with m-playful character
2. Connect a client and test all functionality
3. Verify chat messages, vitals, save/load operations
4. Clean up and report results

## Message Protocol

### Request Types

- `init_request`: Initialize character on server
- `chat_request`: Send chat message to character
- `status_request`: Get character status and vitals
- `save_request`: Save character state
- `load_request`: Load character state
- `reset_request`: Reset character to initial state
- `memory_search_request`: Search character memories
- `ping_request`: Test server connectivity

### Response Types

All requests have corresponding response types with success/error status and relevant data.

## Error Handling

- **Connection Errors**: Automatic reconnection attempts
- **Server Errors**: Structured error responses with details
- **Timeout Handling**: Configurable timeouts for all requests
- **Fallback Mode**: Graceful degradation to local mode

## Performance

- **Concurrent Clients**: Server can handle multiple clients
- **Efficient Protocol**: JSON-based with minimal overhead
- **Connection Pooling**: WebSocket connections reused
- **State Caching**: Client caches character state for performance

## Security

- **Local Network**: Default configuration for localhost only
- **No Authentication**: Current version assumes trusted network
- **Future**: Authentication and encryption can be added

## Future Enhancements

- **Authentication**: User authentication and authorization
- **SSL/TLS**: Encrypted connections for remote deployment
- **Load Balancing**: Multiple character agent servers
- **Persistence**: Database storage for character states
- **Monitoring**: Health checks and metrics collection