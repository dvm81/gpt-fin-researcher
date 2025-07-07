#!/usr/bin/env python
"""
Test market data fetching functionality
"""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.nodes.market_data import MarketDataFetcher


def test_market_data():
    """Test market data fetching and technical indicators."""
    
    fetcher = MarketDataFetcher()
    
    # Test tickers
    tickers = ["AAPL", "TSLA", "MSFT"]
    
    for ticker in tickers:
        print(f"\n{'='*60}")
        print(f"📊 Testing {ticker}")
        print(f"{'='*60}")
        
        try:
            # Fetch OHLCV data
            print("\n📈 Fetching price data...")
            market_data = fetcher.fetch_ohlcv(ticker, period="3mo")
            
            print(f"✅ Data points: {len(market_data.data_points)}")
            print(f"📅 Period: {market_data.start_date.date()} to {market_data.end_date.date()}")
            print(f"💰 Total Return: {market_data.total_return}%")
            print(f"📊 Volatility: {market_data.volatility}%")
            print(f"📉 Avg Volume: {market_data.average_volume:,}")
            
            # Latest price
            latest = market_data.data_points[-1]
            print(f"\n💵 Latest Price:")
            print(f"   Date: {latest.timestamp.date()}")
            print(f"   Close: ${latest.close}")
            print(f"   Volume: {latest.volume:,}")
            
            # Calculate indicators
            print("\n📊 Technical Indicators:")
            indicators = fetcher.calculate_indicators(ticker)
            
            print(f"   SMA 20: ${indicators.sma_20}")
            print(f"   SMA 50: ${indicators.sma_50}")
            print(f"   RSI 14: {indicators.rsi_14}")
            print(f"   MACD: {indicators.macd}")
            
            # Signals
            if indicators.rsi_14:
                if indicators.rsi_14 > 70:
                    print("   ⚠️  RSI indicates overbought")
                elif indicators.rsi_14 < 30:
                    print("   ⚠️  RSI indicates oversold")
            
            # Get market context
            print("\n🌍 Market Context:")
            context = fetcher.get_market_context(ticker)
            
            print(f"   vs 52W High: {context.price_vs_52w_high}%")
            print(f"   vs 52W Low: {context.price_vs_52w_low}%")
            print(f"   Days since high: {context.days_since_high}")
            print(f"   vs SPY (1M): {context.vs_spy_1m}%")
            print(f"   Trend: {context.trend_strength} ({context.volatility_regime} volatility)")
            print(f"   Volume trend: {context.volume_trend}")
            
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test market data fetching")
    parser.add_argument("--ticker", help="Specific ticker to test")
    
    args = parser.parse_args()
    
    if args.ticker:
        fetcher = MarketDataFetcher()
        market_data = fetcher.fetch_ohlcv(args.ticker)
        indicators = fetcher.calculate_indicators(args.ticker)
        context = fetcher.get_market_context(args.ticker)
        
        print(f"✅ {args.ticker}: Total return {market_data.total_return}%, RSI {indicators.rsi_14}")
    else:
        test_market_data()