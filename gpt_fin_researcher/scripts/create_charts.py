#!/usr/bin/env python
"""
Create visualization charts for backtest results
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.nodes.backtester import StrategyBacktester
from src.nodes.strategy_generator import StrategyGenerator
from src.visualization import BacktestVisualizer
from src.schemas import TradingStrategy


def create_sample_strategy(ticker: str) -> TradingStrategy:
    """Create a sample strategy for testing."""
    
    return TradingStrategy(
        strategy_id=f"demo_{ticker}",
        name=f"{ticker} Demo Strategy",
        description="Simple moving average crossover strategy for demonstration",
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
        backtrader_code="""
import backtrader as bt

class DemoStrategy(bt.Strategy):
    params = (
        ('sma_period', 20),
        ('rsi_period', 14),
        ('rsi_threshold', 70),
        ('stop_loss', 0.05),
        ('take_profit', 0.15),
    )
    
    def __init__(self):
        self.sma = bt.indicators.SimpleMovingAverage(
            self.datas[0].close, period=self.params.sma_period)
        self.rsi = bt.indicators.RSI(
            self.datas[0].close, period=self.params.rsi_period)
        self.order = None
        self.entry_price = None
        
    def next(self):
        if self.order:
            return
            
        if not self.position:
            if (self.datas[0].close[0] > self.sma[0] and 
                self.rsi[0] < self.params.rsi_threshold):
                self.order = self.buy()
                self.entry_price = self.datas[0].close[0]
        else:
            current_price = self.datas[0].close[0]
            pnl = (current_price - self.entry_price) / self.entry_price
            
            if (pnl <= -self.params.stop_loss or 
                pnl >= self.params.take_profit or
                self.datas[0].close[0] < self.sma[0]):
                self.order = self.sell()
                
    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}')
            else:
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}')
                self.entry_price = None
        self.order = None
        
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')
"""
    )


def main():
    parser = argparse.ArgumentParser(description='Create backtest visualization charts')
    parser.add_argument('--ticker', type=str, required=True, help='Stock ticker symbol')
    parser.add_argument('--period', type=str, default='1y', help='Backtest period (default: 1y)')
    
    args = parser.parse_args()
    
    print(f"📊 Creating Backtest Charts for {args.ticker}")
    print("=" * 50)
    
    # Create sample strategy
    strategy = create_sample_strategy(args.ticker)
    
    # Run backtest
    print(f"🔄 Running backtest for {args.ticker}...")
    backtester = StrategyBacktester(initial_capital=100000)
    
    try:
        result = backtester.backtest_strategy(
            strategy=strategy,
            ticker=args.ticker,
            start_date="2023-01-01",
            end_date="2023-12-31"
        )
        
        print(f"✅ Backtest completed:")
        print(f"   Total Return: {result.total_return:.2f}%")
        print(f"   Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"   Max Drawdown: {result.max_drawdown:.2f}%")
        print(f"   Total Trades: {result.total_trades}")
        
        # Create visualizations
        print(f"\\n🎨 Creating visualizations...")
        visualizer = BacktestVisualizer()
        
        # Create individual strategy chart
        individual_chart = visualizer.create_individual_strategy_chart(
            result, strategy, args.ticker
        )
        print(f"✅ Individual chart: {individual_chart}")
        
        # Create dashboard (with single strategy)
        dashboard_chart = visualizer.create_performance_dashboard(
            [result], [strategy], args.ticker
        )
        print(f"✅ Dashboard chart: {dashboard_chart}")
        
        print(f"\\n📁 Charts saved to: {visualizer.output_dir}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()