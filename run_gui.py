#!/usr/bin/env python3
"""
Edge AI Video Analytics - GUI Launcher
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Use optimized v2 version
from gui.main_app_v2 import main

if __name__ == "__main__":
    main()
