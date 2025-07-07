#!/usr/bin/env python
"""
Test strategy generation functionality
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.nodes.strategy_generator import StrategyGenerator


def test_strategy_generation():
    """Test strategy generation with mock data."""
    
    generator = StrategyGenerator()
    
    # Mock financial factors
    mock_factors = {
        "overall_sentiment": {
            "score": 0.6,
            "confidence": 0.8,
            "reasoning": "Management expresses confidence in Q4 growth despite supply chain challenges"
        },
        "revenue_growth": 12.5,
        "guidance_raised": True,
        "risk_factors": ["Supply chain disruptions", "Competitive pressure"],
        "competitive_advantages": ["Strong brand recognition", "AI technology leadership"],
        "growth_drivers": ["Cloud services expansion", "AI product adoption"]
    }
    
    # Mock market context
    mock_context = {
        "price_vs_52w_high": -15.2,
        "price_vs_52w_low": 34.8,
        "vs_spy_1m": 2.3,
        "trend_strength": 0.7,
        "volatility_regime": "Normal"
    }
    
    # Mock technical indicators
    mock_indicators = {
        "rsi_14": 58.5,
        "sma_20": 485.20,
        "sma_50": 475.80,
        "current_price": 492.15
    }
    
    print("🎯 Testing Strategy Generation")
    print("=" * 50)
    
    try:
        # Generate strategy
        strategy = generator.generate_strategy(
            ticker="AAPL",
            financial_factors=mock_factors,
            market_context=mock_context,
            technical_indicators=mock_indicators
        )
        
        print(f"✅ Generated Strategy: {strategy.name}")
        print(f"📝 Description: {strategy.description}")
        
        print(f"\n📈 Entry Conditions:")
        for i, condition in enumerate(strategy.entry_conditions, 1):
            print(f"   {i}. {condition}")
        
        print(f"\n📉 Exit Conditions:")
        for i, condition in enumerate(strategy.exit_conditions, 1):
            print(f"   {i}. {condition}")
        
        print(f"\n⚖️ Risk Management:")
        print(f"   Stop Loss: {strategy.risk_management['stop_loss']*100}%")
        print(f"   Take Profit: {strategy.risk_management['take_profit']*100}%")
        print(f"   Max Position: {strategy.risk_management['max_position_size']*100}%")
        print(f"   Max Drawdown: {strategy.risk_management['max_drawdown']*100}%")
        
        print(f"\n💰 Capital & Timeline:")
        print(f"   Initial Capital: ${strategy.initial_capital:,.0f}")
        print(f"   Backtest Period: {strategy.start_date} to {strategy.end_date}")
        print(f"   Position Sizing: {strategy.position_sizing}")
        
        # Generate and show Backtrader code preview
        backtrader_code = generator.generate_backtrader_code(strategy)
        
        print(f"\n💻 Generated Backtrader Code Preview:")
        print("-" * 60)
        # Show first 20 lines
        code_lines = backtrader_code.split('\n')[:20]
        for i, line in enumerate(code_lines, 1):
            print(f"{i:2d}: {line}")
        print("    ... (truncated)")
        print("-" * 60)
        
        # Test different market scenarios
        print(f"\n🧪 Testing Different Scenarios:")
        print("-" * 30)
        
        scenarios = [
            ("Bullish", {"overall_sentiment": {"score": 0.8, "confidence": 0.9, "reasoning": "Strong growth outlook"}, "revenue_growth": 20.0}),
            ("Bearish", {"overall_sentiment": {"score": -0.6, "confidence": 0.8, "reasoning": "Declining margins and competition"}, "revenue_growth": -5.0}),
            ("Neutral", {"overall_sentiment": {"score": 0.1, "confidence": 0.7, "reasoning": "Mixed signals from management"}, "revenue_growth": 3.0})
        ]
        
        for scenario_name, scenario_data in scenarios:
            test_factors = {**mock_factors, **scenario_data}
            test_strategy = generator.generate_strategy("TEST", test_factors, mock_context, mock_indicators)
            print(f"   {scenario_name}: {test_strategy.name}")
            print(f"      SL: {test_strategy.risk_management['stop_loss']*100}% | TP: {test_strategy.risk_management['take_profit']*100}%")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_strategy_generation()