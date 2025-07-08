#!/usr/bin/env python
"""
Test backtesting functionality
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.nodes.backtester import StrategyBacktester
from src.schemas import TradingStrategy


def test_backtesting():
    """Test backtesting with a simple strategy."""
    
    # Create a simple test strategy
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    test_strategy = TradingStrategy(
        strategy_id="test_001",
        name="Simple MA Crossover",
        description="Buy when price crosses above SMA20, sell when below",
        entry_conditions=["Price > SMA(20)", "RSI < 70"],
        exit_conditions=["Price < SMA(20)", "Stop loss at 5%"],
        position_sizing="fixed",
        risk_management={
            "stop_loss": 0.05,
            "take_profit": 0.15,
            "max_position_size": 0.25,
            "max_drawdown": 0.20
        },
        initial_capital=100000.0,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        backtrader_code="""
import backtrader as bt

class SimpleMAStrategy(bt.Strategy):
    params = (
        ('sma_period', 20),
        ('stop_loss', 0.05),
        ('take_profit', 0.15),
    )
    
    def __init__(self):
        self.sma = bt.indicators.SimpleMovingAverage(
            self.datas[0].close, period=self.params.sma_period)
        self.order = None
        self.entry_price = None
        
    def next(self):
        if self.order:
            return
            
        if not self.position:
            if self.datas[0].close[0] > self.sma[0]:
                self.order = self.buy()
                self.entry_price = self.datas[0].close[0]
        else:
            # Check stop loss and take profit
            current_price = self.datas[0].close[0]
            pnl = (current_price - self.entry_price) / self.entry_price
            
            if pnl <= -self.params.stop_loss or pnl >= self.params.take_profit:
                self.order = self.sell()
            elif self.datas[0].close[0] < self.sma[0]:
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
    
    print("🧪 Testing Backtesting Module")
    print("=" * 50)
    
    # Initialize backtester
    backtester = StrategyBacktester(initial_capital=100000)
    
    # Test different tickers
    tickers = ["AAPL", "MSFT", "TSLA"]
    
    for ticker in tickers:
        print(f"\n📊 Backtesting {ticker}")
        print("-" * 40)
        
        try:
            # Run backtest
            results = backtester.backtest_strategy(
                strategy=test_strategy,
                ticker=ticker,
                start_date=test_strategy.start_date,
                end_date=test_strategy.end_date
            )
            
            # Display results
            print(f"\n📈 Results for {ticker}:")
            print(f"  Total Return: {results.total_return}%")
            print(f"  Annualized Return: {results.annualized_return}%")
            print(f"  Sharpe Ratio: {results.sharpe_ratio}")
            print(f"  Max Drawdown: {results.max_drawdown}%")
            print(f"  Win Rate: {results.win_rate}%")
            print(f"  Total Trades: {results.total_trades}")
            print(f"  Winning Trades: {results.winning_trades}")
            print(f"  Losing Trades: {results.losing_trades}")
            
            # Performance rating
            if results.total_return > 10 and results.sharpe_ratio > 1:
                print("  ⭐ Excellent Performance!")
            elif results.total_return > 0:
                print("  ✅ Profitable Strategy")
            else:
                print("  ❌ Unprofitable Strategy")
                
            # Risk assessment
            if results.max_drawdown > 20:
                print("  ⚠️  High Risk (Drawdown > 20%)")
            
        except Exception as e:
            print(f"❌ Error backtesting {ticker}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    test_backtesting()