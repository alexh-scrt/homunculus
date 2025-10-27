#!/usr/bin/env python3
"""Character Agent Server startup script.

This script starts the Character Agent WebSocket server that can host character
agents and serve them to remote clients via WebSocket connections.
"""

import asyncio
import argparse
import signal
import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from server.character_agent_server import CharacterAgentServer
    from config.settings import get_settings
    from config.character_loader import CharacterLoader
    from config.logging_config import setup_logging, get_logger
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running from the project root and dependencies are installed.")
    sys.exit(1)


def setup_argument_parser():
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Character Agent WebSocket Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_character_server.py                    # Start server without pre-loading character
  python scripts/run_character_server.py -c m_playful      # Start server with m_playful character
  python scripts/run_character_server.py --port 9000       # Start on different port
  python scripts/run_character_server.py --host 0.0.0.0    # Listen on all interfaces
        """
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='Host address to bind to (default: localhost)'
    )
    
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=8765,
        help='Port to listen on (default: 8765)'
    )
    
    parser.add_argument(
        '--character', '-c',
        type=str,
        help='Character ID to pre-load on startup'
    )
    
    parser.add_argument(
        '--list-characters', '-l',
        action='store_true',
        help='List available characters and exit'
    )
    
    parser.add_argument(
        '--validate-character',
        type=str,
        help='Validate a specific character configuration and exit'
    )
    
    parser.add_argument(
        '--config-check',
        action='store_true',
        help='Check configuration and exit'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser


def list_characters():
    """List available characters and their details."""
    try:
        loader = CharacterLoader()
        characters = loader.list_available_characters()
        
        if not characters:
            print("No characters found in schemas directory.")
            return
        
        print("\n🤖 Available Characters:")
        print("=" * 60)
        
        for char_id in characters:
            try:
                info = loader.get_character_info(char_id)
                print(f"📋 ID: {char_id}")
                print(f"   Name: {info['name']}")
                print(f"   Type: {info['archetype']}")
                
                # Try to get description
                try:
                    config = loader.load_character(char_id)
                    if 'description' in config:
                        print(f"   Description: {config['description']}")
                except Exception:
                    print("   Description: Not available")
                
                print("-" * 40)
                
            except Exception as e:
                print(f"❌ Error loading {char_id}: {e}")
        
        print(f"\n✅ Found {len(characters)} characters total")
        
    except Exception as e:
        print(f"❌ Error listing characters: {e}")


def validate_character(character_id: str):
    """Validate a specific character configuration."""
    try:
        print(f"🔍 Validating character '{character_id}'...")
        
        loader = CharacterLoader()
        
        # Check if character exists
        available = loader.list_available_characters()
        if character_id not in available:
            print(f"❌ Character '{character_id}' not found.")
            print(f"   Available characters: {', '.join(available)}")
            return False
        
        # Try to load character configuration
        config = loader.load_character(character_id)
        print(f"✅ Character configuration loaded successfully")
        print(f"   Name: {config.get('name', 'Unknown')}")
        print(f"   Archetype: {config.get('archetype', 'Unknown')}")
        
        # Validate required fields
        required_fields = ['name', 'archetype', 'personality']
        missing_fields = [field for field in required_fields if field not in config]
        
        if missing_fields:
            print(f"⚠️  Missing required fields: {', '.join(missing_fields)}")
        else:
            print("✅ All required fields present")
        
        # Check personality structure
        if 'personality' in config:
            personality = config['personality']
            if 'big_five' in personality:
                print("✅ Big Five personality traits defined")
            else:
                print("⚠️  Big Five personality traits not defined")
        
        print(f"✅ Character '{character_id}' validation complete")
        return True
        
    except Exception as e:
        print(f"❌ Error validating character '{character_id}': {e}")
        return False


def check_configuration():
    """Check system configuration."""
    try:
        print("🔧 Checking system configuration...")
        
        # Check settings
        settings = get_settings()
        print(f"✅ Settings loaded successfully")
        print(f"   Ollama URL: {settings.ollama_base_url}")
        print(f"   Model: {settings.ollama_model}")
        
        # Check directories
        from pathlib import Path
        
        save_dir = Path("./data/saves")
        if save_dir.exists():
            print(f"✅ Save directory exists: {save_dir}")
        else:
            print(f"⚠️  Save directory will be created: {save_dir}")
        
        schemas_dir = Path("./schemas/characters")
        if schemas_dir.exists():
            character_count = len(list(schemas_dir.glob("*.yaml")))
            print(f"✅ Characters directory exists with {character_count} characters")
        else:
            print(f"❌ Characters directory not found: {schemas_dir}")
        
        print("✅ Configuration check complete")
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False


async def run_server(args):
    """Run the character agent server."""
    
    # Setup logging
    setup_logging()
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger = get_logger(__name__)
    
    # Create server
    server = CharacterAgentServer(host=args.host, port=args.port)
    
    # Display startup information
    print(f"🚀 Starting Character Agent Server")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   WebSocket URL: ws://{args.host}:{args.port}")
    
    if args.character:
        print(f"   Pre-loading character: {args.character}")
    else:
        print(f"   No character pre-loaded (clients can initialize)")
    
    print()
    
    # Setup signal handlers for graceful shutdown
    shutdown_event = asyncio.Event()
    
    def signal_handler(sig, frame):
        print(f"\n🛑 Received signal {sig}, shutting down gracefully...")
        shutdown_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start server (this will run until interrupted)
        server_task = asyncio.create_task(
            server.start_server(character_id=args.character)
        )
        
        # Wait for shutdown signal
        shutdown_task = asyncio.create_task(shutdown_event.wait())
        
        # Wait for either server to complete or shutdown signal
        done, pending = await asyncio.wait(
            [server_task, shutdown_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel any pending tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Stop the server
        await server.stop_server()
        
        print("✅ Server shutdown complete")
        
    except Exception as e:
        logger.error(f"Server error: {e}")
        print(f"❌ Server error: {e}")
        traceback.print_exc()
        return 1
    
    return 0


async def main():
    """Main entry point."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Handle special modes that don't require starting the server
    if args.list_characters:
        list_characters()
        return 0
    
    if args.validate_character:
        success = validate_character(args.validate_character)
        return 0 if success else 1
    
    if args.config_check:
        success = check_configuration()
        return 0 if success else 1
    
    # Validate settings before starting
    try:
        settings = get_settings()
        print(f"⚙️  Configuration:")
        print(f"   Ollama URL: {settings.ollama_base_url}")
        print(f"   Model: {settings.ollama_model}")
        
        # Check if required services are configured
        if not settings.ollama_base_url:
            print("⚠️  Warning: Ollama URL not configured. Check your .env file.")
        
    except Exception as e:
        print(f"❌ Settings error: {e}")
        print("   Check your .env file and configuration.")
        return 1
    
    # Validate character if specified
    if args.character:
        print(f"🔍 Validating character '{args.character}'...")
        if not validate_character(args.character):
            print(f"❌ Cannot start server with invalid character '{args.character}'")
            return 1
    
    # Run the server
    return await run_server(args)


if __name__ == "__main__":
    # Ensure event loop exists for Windows compatibility
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    exit_code = asyncio.run(main())
    sys.exit(exit_code)