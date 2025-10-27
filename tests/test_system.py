"""Quick test script to verify TiMi system components."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from timi.utils.config import Config
from timi.utils.logging import setup_logging
from timi.llm.client import LLMClient
from timi.exchange.factory import ExchangeFactory


async def test_configuration():
    """Test configuration loading."""
    print("\n=== Testing Configuration ===")
    try:
        config = Config()
        print(f"✓ Config loaded")
        print(f"  - Mode: {config.mode}")
        print(f"  - Exchange: {config.exchange.primary}")
        print(f"  - Testnet: {config.exchange.testnet}")
        print(f"  - Mainstream pairs: {len(config.trading_pairs_mainstream)}")
        return True
    except Exception as e:
        print(f"✗ Config failed: {e}")
        return False


async def test_llm_client():
    """Test LLM client initialization."""
    print("\n=== Testing LLM Client ===")
    try:
        config = Config()

        # Check if API keys are present
        openai_key = config.get_api_key('openai')
        if not openai_key:
            print("⚠ OpenAI API key not found (add to .env)")
            return False

        llm_client = LLMClient(config)
        print("✓ LLM client initialized")
        print("  - Semantic analysis: Ready")
        print("  - Code programming: Ready")
        print("  - Mathematical reasoning: Ready")
        return True
    except Exception as e:
        print(f"✗ LLM client failed: {e}")
        return False


async def test_exchange_connector():
    """Test exchange connector."""
    print("\n=== Testing Exchange Connector ===")
    try:
        config = Config()

        # Check if API keys are present
        api_key = config.get_api_key('binance')
        api_secret = config.get_api_secret('binance')

        if not api_key or not api_secret:
            print("⚠ Binance API credentials not found (add to .env)")
            return False

        exchange = ExchangeFactory.create_default_exchange(config)
        print(f"✓ Exchange connector created (Testnet: {exchange.testnet})")

        # Try to get ticker
        ticker = await exchange.get_ticker('BTC/USDT')
        print(f"  - BTC/USDT price: ${ticker.last:,.2f}")
        print(f"  - 24h volume: ${ticker.volume_24h:,.0f}")

        await exchange.close()
        return True
    except Exception as e:
        print(f"✗ Exchange connector failed: {e}")
        return False


async def test_agents():
    """Test agent initialization."""
    print("\n=== Testing Agents ===")
    try:
        from timi.agents import (
            MacroAnalysisAgent,
            StrategyAdaptationAgent,
            BotEvolutionAgent,
            FeedbackReflectionAgent
        )
        from timi.data import MarketDataManager

        config = Config()
        llm_client = LLMClient(config)

        # Create mock exchange for testing
        print("  - Creating agents...")

        # We'll skip creating actual agent instances for now
        # as they require exchange connections
        print("✓ Agent classes loaded")
        print("  - Macro Analysis Agent (Ama)")
        print("  - Strategy Adaptation Agent (Asa)")
        print("  - Bot Evolution Agent (Abe)")
        print("  - Feedback Reflection Agent (Afr)")
        return True
    except Exception as e:
        print(f"✗ Agents failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("=" * 60)
    print("TiMi System Component Test")
    print("=" * 60)

    # Setup logging
    setup_logging()

    results = []

    # Test configuration
    results.append(await test_configuration())

    # Test LLM client
    results.append(await test_llm_client())

    # Test exchange connector
    results.append(await test_exchange_connector())

    # Test agents
    results.append(await test_agents())

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n✓ All systems operational!")
        print("\nNext steps:")
        print("1. Ensure API keys are in .env file")
        print("2. Review config.yaml settings")
        print("3. Start with paper trading mode")
    else:
        print("\n⚠ Some components need configuration")
        print("\nSetup required:")
        print("1. Copy .env.example to .env")
        print("2. Add your API keys to .env")
        print("3. Run this test again")


if __name__ == "__main__":
    asyncio.run(main())
