#!/usr/bin/env python3
"""Simple remote chat interface runner."""

import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cli.simple_remote_chat_interface import SimpleRemoteChatInterface


async def main():
    """Run the simple remote chat interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Remote Character Agent Chat Interface")
    parser.add_argument('--server-url', default='ws://localhost:8765', help='Character agent server URL')
    parser.add_argument('--no-fallback', action='store_true', help='Disable fallback to local mode')
    parser.add_argument('--character', '-c', help='Character ID to load directly')
    parser.add_argument('--user-id', '-u', default='anonymous', help='User ID for persistent conversations (default: anonymous)')
    
    args = parser.parse_args()
    
    interface = SimpleRemoteChatInterface(
        server_url=args.server_url,
        fallback_to_local=not args.no_fallback,
        user_id=args.user_id
    )
    
    await interface.run(character_id=args.character)


if __name__ == "__main__":
    asyncio.run(main())