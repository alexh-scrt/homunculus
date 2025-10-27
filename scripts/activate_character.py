#!/usr/bin/env python3
"""Quick character activation script for testing the dual-pane interface."""

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
    from config.logging_config import setup_logging, tail_command
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running from the project root and dependencies are installed.")
    sys.exit(1)


def setup_argument_parser():
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Quick character activation for dual-pane chat interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/activate_character.py m-playful       # Chat with Marcus (playful)
  python scripts/activate_character.py ada_lovelace   # Chat with Ada Lovelace
  python scripts/activate_character.py f-serious --debug      # Enable debug mode
  python scripts/activate_character.py zen_master --vitals-only   # Vitals monitoring only
        """
    )
    
    parser.add_argument(
        'character_id',
        type=str,
        help='Character ID to activate (e.g., m-playful, ada_lovelace)'
    )
    
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug mode (shows agent internals)'
    )
    
    parser.add_argument(
        '--vitals-only', '-v',
        action='store_true',
        help='Show only vitals monitoring (no chat interface)'
    )
    
    parser.add_argument(
        '--no-vitals',
        action='store_true',
        help='Disable vitals display (use original interface)'
    )
    
    return parser


def list_available_characters():
    """List all available characters."""
    try:
        loader = CharacterLoader()
        characters = loader.list_available_characters()
        
        if not characters:
            print("No characters found in schemas directory.")
            return
        
        print("Available Characters:")
        print("=" * 50)
        
        for char_id in sorted(characters):
            try:
                info = loader.get_character_info(char_id)
                name = info.get('name', 'Unknown')
                archetype = info.get('archetype', 'unknown')
                description = info.get('description', 'No description')[:60]
                
                print(f"ID: {char_id}")
                print(f"Name: {name}")
                print(f"Type: {archetype}")
                print(f"Description: {description}...")
                print("-" * 30)
                
            except Exception as e:
                print(f"Error loading {char_id}: {e}")
    
    except Exception as e:
        print(f"Error listing characters: {e}")


async def main():
    """Main entry point for character activation."""
    # Initialize logging first
    setup_logging()
    
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Show logging info
    tail_command()
    
    # Validate character exists
    try:
        loader = CharacterLoader()
        available_characters = loader.list_available_characters()
        
        if args.character_id not in available_characters:
            print(f"Error: Character '{args.character_id}' not found.")
            print("\nAvailable characters:")
            for char_id in sorted(available_characters):
                try:
                    info = loader.get_character_info(char_id)
                    print(f"  {char_id} - {info.get('name', 'Unknown')}")
                except:
                    print(f"  {char_id}")
            print("\nUse 'python scripts/run_chat.py --list-characters' for detailed list.")
            sys.exit(1)
    
    except Exception as e:
        print(f"Error validating character: {e}")
        sys.exit(1)
    
    # Setup interface
    try:
        # Create enhanced interface
        interface = ChatInterface(
            enable_vitals=not args.no_vitals,
            vitals_only=args.vitals_only
        )
        
        # Set debug mode if requested
        if args.debug:
            interface.debug_mode = True
            print(f"[DEBUG] Debug mode enabled")
        
        # Display activation message
        character_info = loader.get_character_info(args.character_id)
        character_name = character_info.get('name', args.character_id)
        
        if args.vitals_only:
            print(f"Activating vitals monitoring for {character_name}...")
            print("Press Ctrl+C to exit")
        else:
            print(f"Activating chat interface with {character_name}...")
            if not args.no_vitals:
                print("✨ Enhanced dual-pane interface with real-time vitals")
        
        # Run the interface
        await interface.run(character_id=args.character_id)
        
    except KeyboardInterrupt:
        print("\n👋 Character session ended. Goodbye!")
    except Exception as e:
        print(f"Error running character interface: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Check if user just wants to see available characters
    if len(sys.argv) == 2 and sys.argv[1] in ['--list', '-l', 'list']:
        list_available_characters()
    else:
        asyncio.run(main())