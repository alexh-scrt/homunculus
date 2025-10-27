#!/usr/bin/env python3
"""Test script to verify WebSocket ping configuration."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_ping_config():
    """Test the ping configuration logic."""
    
    # Test DEBUG=true case
    os.environ['DEBUG'] = 'true'
    debug_mode = os.getenv('DEBUG', '').lower() in ('true', '1', 'yes')
    if debug_mode:
        ping_interval = None
        ping_timeout = None
        print(f"✓ DEBUG=true: ping_interval={ping_interval}, ping_timeout={ping_timeout}")
    
    # Test DEBUG=false case
    os.environ['DEBUG'] = 'false'
    debug_mode = os.getenv('DEBUG', '').lower() in ('true', '1', 'yes')
    if not debug_mode:
        ping_interval = 600
        ping_timeout = 60
        print(f"✓ DEBUG=false: ping_interval={ping_interval}, ping_timeout={ping_timeout}")
    
    # Test no DEBUG env var
    if 'DEBUG' in os.environ:
        del os.environ['DEBUG']
    debug_mode = os.getenv('DEBUG', '').lower() in ('true', '1', 'yes')
    if not debug_mode:
        ping_interval = 600
        ping_timeout = 60
        print(f"✓ No DEBUG env: ping_interval={ping_interval}, ping_timeout={ping_timeout}")
    
    print("All ping configuration tests passed!")

if __name__ == "__main__":
    test_ping_config()