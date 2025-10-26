#!/usr/bin/env python3
"""Main entry point for the character agent chat system."""

import asyncio
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from cli.chat_interface import ChatInterface
    from config.settings import get_settings
    from config.character_loader import CharacterLoader
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running from the project root and dependencies are installed.")
    sys.exit(1)


def setup_argument_parser():
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Interactive chat with character agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_chat.py                    # Interactive character selection
  python scripts/run_chat.py --character ada_lovelace  # Chat with specific character
  python scripts/run_chat.py --debug            # Enable debug mode
  python scripts/run_chat.py --list-characters  # List available characters
        """
    )
    
    parser.add_argument(
        '--character', '-c',
        type=str,
        help='Character ID to chat with (skips selection menu)'
    )
    
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug mode (shows agent internals)'
    )
    
    parser.add_argument(
        '--list-characters', '-l',
        action='store_true',
        help='List available characters and exit'
    )
    
    parser.add_argument(
        '--save-dir',
        type=str,
        default='./data/saves',
        help='Directory for saving character states (default: ./data/saves)'
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
        
        print("\nAvailable Characters:")
        print("=" * 60)
        
        for char_id in characters:
            try:
                info = loader.get_character_info(char_id)
                print(f"ID: {char_id}")
                print(f"Name: {info['name']}")
                print(f"Archetype: {info['archetype']}")
                
                # Try to get description
                try:
                    config = loader.load_character(char_id)
                    if 'description' in config:
                        print(f"Description: {config['description']}")
                except Exception:
                    print("Description: Not available")
                
                print("-" * 40)
                
            except Exception as e:
                print(f"Error loading {char_id}: {e}")
        
    except Exception as e:
        print(f"Error listing characters: {e}")


async def run_with_specific_character(character_id: str, debug_mode: bool = False):
    """Run chat with a specific character.
    
    Args:
        character_id: ID of character to load
        debug_mode: Whether to enable debug mode
    """
    try:
        # Verify character exists
        loader = CharacterLoader()
        available = loader.list_available_characters()
        
        if character_id not in available:
            print(f"Error: Character '{character_id}' not found.")
            print(f"Available characters: {', '.join(available)}")
            return
        
        # Create custom interface for specific character
        interface = ChatInterface()
        interface.debug_mode = debug_mode
        
        # Load the specific character
        character = await interface._load_character(character_id)
        if not character:
            print(f"Failed to load character '{character_id}'")
            return
        
        interface.current_character = character
        
        # Display welcome for specific character
        print(f"\nStarting chat with {character.character_name}")
        if debug_mode:
            print("Debug mode enabled")
        print("Type '/help' for commands or start chatting!\n")
        
        # Start conversation loop
        await interface._conversation_loop()
        
        # Auto-save on exit
        await interface._auto_save()
        
    except KeyboardInterrupt:
        print("\nChat interrupted by user. Goodbye!")
    except Exception as e:
        print(f"Error running chat: {e}")


async def main():
    """Main entry point."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Handle list characters
    if args.list_characters:
        list_characters()
        return
    
    # Validate settings
    try:
        settings = get_settings()
        print(f"Using Ollama at: {settings.ollama_base_url}")
        print(f"Model: {settings.ollama_model}")
        
        # Check if required services are configured
        if not settings.ollama_base_url:
            print("Warning: Ollama URL not configured. Check your .env file.")
        
    except Exception as e:
        print(f"Settings error: {e}")
        print("Check your .env file and configuration.")
        return
    
    # Run chat
    try:
        if args.character:
            # Run with specific character
            await run_with_specific_character(args.character, args.debug)
        else:
            # Run full interactive interface
            interface = ChatInterface()
            interface.debug_mode = args.debug
            
            # Set custom save directory if provided
            if args.save_dir:
                interface.save_dir = Path(args.save_dir)
                interface.save_dir.mkdir(parents=True, exist_ok=True)
            
            await interface.run()
    
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Ensure event loop exists for Windows compatibility
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main())