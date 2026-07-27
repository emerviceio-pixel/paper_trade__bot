import sys
import asyncio
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.bybit_bot import main

if __name__ == "__main__":
    print("🚀 Starting Bybit HFT Bot...")
    asyncio.run(main())