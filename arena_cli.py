#!/usr/bin/env python
"""
Arena CLI Runner

Main entry point for the Arena command-line interface.

Usage:
    python arena_cli.py [command] [options]
    
Examples:
    python arena_cli.py start game1 -a agent1 agent2 agent3
    python arena_cli.py tournament t1 -a agent1 agent2 agent3 agent4 -f single-elim
    python arena_cli.py stats
    python arena_cli.py interactive

Author: Homunculus Team
"""

import sys
import os
from pathlib import Path

# Add Arena to path
sys.path.insert(0, str(Path(__file__).parent))

from src.arena.cli import main


if __name__ == "__main__":
    main()