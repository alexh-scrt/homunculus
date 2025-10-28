#!/usr/bin/env python3
"""Simple chat interface runner."""

import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cli.simple_chat_interface import SimpleChatInterface


async def main():
    """Run the simple chat interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Character Agent Chat Interface")
    parser.add_argument('--character', '-c', help='Character ID to load directly')
    parser.add_argument('--user-id', '-u', default='anonymous', help='User ID for persistent conversations (default: anonymous)')
    
    args = parser.parse_args()
    
    interface = SimpleChatInterface(user_id=args.user_id)
    await interface.run(character_id=args.character)


if __name__ == "__main__":
    asyncio.run(main())