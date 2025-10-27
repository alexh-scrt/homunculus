#!/usr/bin/env python3
"""Test script for the socket-based character agent architecture.

This script tests the decoupled architecture by:
1. Starting a character agent server with m-playful character
2. Connecting a client to the server
3. Sending a test message
4. Verifying the response
"""

import asyncio
import sys
import signal
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from server.character_agent_server import CharacterAgentServer
    from client.character_agent_client import CharacterAgentClient, CharacterAgentClientError
    from config.character_loader import CharacterLoader
    from config.logging_config import setup_logging, get_logger
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("Make sure you're running from the project root and dependencies are installed.")
    sys.exit(1)


class SocketArchitectureTest:
    """Test class for socket-based architecture."""
    
    def __init__(self):
        self.server = None
        self.client = None
        self.server_task = None
        self.logger = None
        
    async def setup(self):
        """Setup test environment."""
        # Setup logging
        setup_logging()
        self.logger = get_logger(__name__)
        
        print("🧪 Setting up socket architecture test...")
        
        # Create server
        self.server = CharacterAgentServer(host="localhost", port=8766)  # Use different port
        
        # Create client
        self.client = CharacterAgentClient(
            server_url="ws://localhost:8766",
            timeout=10.0
        )
        
        print("✅ Test setup complete")
    
    async def start_server(self, character_id: str = "m-playful"):
        """Start the character agent server."""
        print(f"🚀 Starting server with character '{character_id}'...")
        
        # Start server in background
        self.server_task = asyncio.create_task(
            self.server.start_server(character_id=character_id)
        )
        
        # Give server time to start
        await asyncio.sleep(2)
        
        if self.server.is_running:
            print("✅ Server started successfully")
            return True
        else:
            print("❌ Failed to start server")
            return False
    
    async def test_connection(self):
        """Test client connection to server."""
        print("🔌 Testing client connection...")
        
        connected = await self.client.connect()
        if connected:
            print("✅ Client connected successfully")
            
            # Test ping
            ping_success = await self.client.ping_server()
            if ping_success:
                print("✅ Ping test successful")
                return True
            else:
                print("❌ Ping test failed")
                return False
        else:
            print("❌ Client connection failed")
            return False
    
    async def test_character_initialization(self):
        """Test character initialization on server."""
        print("🤖 Testing character initialization...")
        
        try:
            success = await self.client.initialize_character("m-playful")
            if success:
                print(f"✅ Character initialized: {self.client.character_name}")
                return True
            else:
                print("❌ Character initialization failed")
                return False
        except CharacterAgentClientError as e:
            print(f"❌ Character initialization error: {e}")
            return False
    
    async def test_chat_message(self):
        """Test sending a chat message."""
        print("💬 Testing chat message...")
        
        try:
            response = await self.client.process_message(
                user_message="Hello! Tell me a joke.",
                context={"test_mode": True}
            )
            
            if response and response.get('response_text'):
                character_response = response['response_text']
                print(f"✅ Got response: {character_response[:100]}...")
                return True
            else:
                print("❌ No response received")
                return False
                
        except CharacterAgentClientError as e:
            print(f"❌ Chat message error: {e}")
            return False
    
    async def test_character_vitals(self):
        """Test getting character vitals."""
        print("📊 Testing character vitals...")
        
        try:
            vitals = await self.client.get_character_vitals()
            if vitals:
                mood = vitals.get('agent_states', {}).get('mood', {})
                neurochemical = vitals.get('neurochemical_state', {})
                print(f"✅ Got vitals - mood: {mood.get('current_state', 'unknown')}")
                print(f"   Neurochemical levels: {len(neurochemical)} hormones tracked")
                return True
            else:
                print("❌ No vitals received")
                return False
                
        except CharacterAgentClientError as e:
            print(f"❌ Vitals error: {e}")
            return False
    
    async def test_save_load(self):
        """Test save and load functionality."""
        print("💾 Testing save/load functionality...")
        
        try:
            # Save character state
            save_path = await self.client.save_character_state("test_save.json")
            print(f"✅ Character state saved to: {save_path}")
            
            # Load character state
            character_info = await self.client.load_character_state("test_save.json")
            if character_info:
                print(f"✅ Character state loaded: {character_info.get('name', 'unknown')}")
                return True
            else:
                print("❌ Load failed")
                return False
                
        except CharacterAgentClientError as e:
            print(f"❌ Save/load error: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup test environment."""
        print("🧹 Cleaning up test environment...")
        
        # Disconnect client
        if self.client:
            await self.client.disconnect()
            print("✅ Client disconnected")
        
        # Stop server
        if self.server and self.server.is_running:
            await self.server.stop_server()
            print("✅ Server stopped")
        
        # Cancel server task
        if self.server_task and not self.server_task.done():
            self.server_task.cancel()
            try:
                await self.server_task
            except asyncio.CancelledError:
                pass
            print("✅ Server task cancelled")
    
    async def run_all_tests(self):
        """Run all tests."""
        print("🧪 Starting socket architecture tests...")
        print("=" * 60)
        
        tests_passed = 0
        total_tests = 6
        
        try:
            # Setup
            await self.setup()
            
            # Test 1: Start server
            if await self.start_server():
                tests_passed += 1
            
            # Test 2: Client connection
            if await self.test_connection():
                tests_passed += 1
            
            # Test 3: Character initialization
            if await self.test_character_initialization():
                tests_passed += 1
            
            # Test 4: Chat message
            if await self.test_chat_message():
                tests_passed += 1
            
            # Test 5: Character vitals
            if await self.test_character_vitals():
                tests_passed += 1
            
            # Test 6: Save/load
            if await self.test_save_load():
                tests_passed += 1
            
        except Exception as e:
            print(f"❌ Test error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await self.cleanup()
        
        print("=" * 60)
        print(f"🧪 Test Results: {tests_passed}/{total_tests} tests passed")
        
        if tests_passed == total_tests:
            print("🎉 All tests passed! Socket architecture is working correctly.")
            return True
        else:
            print("❌ Some tests failed. Check the output above for details.")
            return False


async def main():
    """Main test function."""
    test = SocketArchitectureTest()
    
    # Setup signal handler for cleanup
    def signal_handler(sig, frame):
        print("\n🛑 Test interrupted, cleaning up...")
        asyncio.create_task(test.cleanup())
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        success = await test.run_all_tests()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)