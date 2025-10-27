#!/usr/bin/env python3
"""Simple test script to verify WebSocket connection to character agent server."""

import asyncio
import json
import websockets
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from protocol.messages import PingRequest, BaseMessage

async def test_connection():
    """Test basic WebSocket connection to server."""
    uri = "ws://localhost:8765"
    
    try:
        print(f"Connecting to {uri}...")
        
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to server!")
            
            # Send a ping request
            ping_request = PingRequest()
            ping_json = ping_request.to_json()
            print(f"📤 Sending ping: {ping_json}")
            
            # Try to parse it back to check JSON validity
            try:
                parsed = json.loads(ping_json)
                print(f"✅ JSON is valid: {parsed}")
            except Exception as e:
                print(f"❌ JSON is invalid: {e}")
                return False
            
            await websocket.send(ping_json)
            print("📤 Ping sent!")
            
            # Wait for response
            response_json = await websocket.recv()
            print(f"📥 Received response: {response_json}")
            
            # Parse response
            response = BaseMessage.from_json(response_json)
            print(f"📋 Parsed response: {response.message_type}")
            
            if response.message_type == 'ping_response':
                print("✅ Ping successful!")
                return True
            else:
                print(f"❌ Unexpected response type: {response.message_type}")
                return False
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_connection())
    sys.exit(0 if success else 1)