#!/usr/bin/env python
"""
Test backtest visualization functionality
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.visualization import BacktestVisualizer
from src.schemas import BacktestResults, TradingStrategy


def create_sample_data():
    """Create sample backtest data for testing."""
    
    # Create sample strategies
    strategies = [
        TradingStrategy(
            strategy_id="test_001",
            name="MA Crossover Strategy",
            description="Simple moving average crossover strategy",
            entry_conditions=["Price > SMA(20)", "RSI < 70"],
            exit_conditions=["Price < SMA(20)", "Stop loss at 5%"],
            position_sizing="fixed",
            risk_management={
                "stop_loss": 0.05,
                "take_profit": 0.15,
                "max_position_size": 0.25
            },
            initial_capital=100000.0,
            start_date="2023-01-01",
            end_date="2023-12-31",
            backtrader_code="# Sample strategy code"
        ),
        TradingStrategy(
            strategy_id="test_002",
            name="RSI Mean Reversion",
            description="RSI-based mean reversion strategy",
            entry_conditions=["RSI < 30", "Price > SMA(50)"],
            exit_conditions=["RSI > 70", "Stop loss at 8%"],
            position_sizing="fixed",
            risk_management={
                "stop_loss": 0.08,
                "take_profit": 0.12,
                "max_position_size": 0.20
            },
            initial_capital=100000.0,
            start_date="2023-01-01",
            end_date="2023-12-31",
            backtrader_code="# Sample strategy code"
        ),
        TradingStrategy(
            strategy_id="test_003",
            name="Momentum Breakout",
            description="Momentum-based breakout strategy",
            entry_conditions=["Price > 52W High", "Volume > Average"],
            exit_conditions=["Price < SMA(10)", "Stop loss at 6%"],
            position_sizing="fixed",
            risk_management={
                "stop_loss": 0.06,
                "take_profit": 0.18,
                "max_position_size": 0.30
            },
            initial_capital=100000.0,
            start_date="2023-01-01",
            end_date="2023-12-31",
            backtrader_code="# Sample strategy code"
        )
    ]
    
    # Create sample backtest results
    results = [
        BacktestResults(
            strategy_id="test_001",
            total_return=15.30,
            annualized_return=15.30,
            sharpe_ratio=1.25,
            max_drawdown=8.50,
            win_rate=62.5,
            total_trades=24,
            winning_trades=15,
            losing_trades=9,
            avg_win=450.75,
            avg_loss=280.25,
            volatility=12.5,
            calmar_ratio=1.80,
            sortino_ratio=1.45,
            execution_time=2.5,
            data_points=252
        ),
        BacktestResults(
            strategy_id="test_002",
            total_return=8.75,
            annualized_return=8.75,
            sharpe_ratio=0.85,
            max_drawdown=12.20,
            win_rate=58.0,
            total_trades=31,
            winning_trades=18,
            losing_trades=13,
            avg_win=320.50,
            avg_loss=195.80,
            volatility=15.2,
            calmar_ratio=0.72,
            sortino_ratio=1.10,
            execution_time=3.1,
            data_points=252
        ),
        BacktestResults(
            strategy_id="test_003",
            total_return=22.10,
            annualized_return=22.10,
            sharpe_ratio=1.65,
            max_drawdown=15.80,
            win_rate=45.0,
            total_trades=20,
            winning_trades=9,
            losing_trades=11,
            avg_win=820.90,
            avg_loss=340.60,
            volatility=18.5,
            calmar_ratio=1.40,
            sortino_ratio=1.85,
            execution_time=2.8,
            data_points=252
        )
    ]
    
    return strategies, results


def test_visualization():
    """Test the visualization components."""
    
    print("🎨 Testing Backtest Visualization")
    print("=" * 50)
    
    # Create sample data
    strategies, results = create_sample_data()
    
    # Initialize visualizer
    visualizer = BacktestVisualizer()
    
    # Test dashboard creation
    print("\n📊 Creating performance dashboard...")
    try:
        dashboard_path = visualizer.create_performance_dashboard(
            results, strategies, "AAPL"
        )
        if dashboard_path:
            print(f"✅ Dashboard created: {dashboard_path}")
        else:
            print("❌ Dashboard creation failed")
    except Exception as e:
        print(f"❌ Error creating dashboard: {e}")
        import traceback
        traceback.print_exc()
    
    # Test individual strategy charts
    print(f"\n📊 Creating individual strategy charts...")
    for result, strategy in zip(results, strategies):
        try:
            chart_path = visualizer.create_individual_strategy_chart(
                result, strategy, "AAPL"
            )
            if chart_path:
                print(f"✅ Individual chart created: {chart_path}")
            else:
                print(f"❌ Individual chart creation failed for {strategy.name}")
        except Exception as e:
            print(f"❌ Error creating chart for {strategy.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✅ Visualization testing completed!")
    print(f"📁 Charts saved to: {visualizer.output_dir}")
    

if __name__ == "__main__":
    test_visualization()