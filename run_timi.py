#!/usr/bin/env python3
"""Convenience wrapper to run TiMi system."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run main
from timi.main import main
import asyncio

if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║   TiMi - Trade in Minutes                                  ║
║   Rationality-Driven Agentic Trading System                ║
║                                                            ║
║   [WARNING] Trading involves substantial risk              ║
║   [WARNING] Only trade with capital you can afford to lose ║
║   [WARNING] This is NOT financial advice                   ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
""")

    asyncio.run(main())
