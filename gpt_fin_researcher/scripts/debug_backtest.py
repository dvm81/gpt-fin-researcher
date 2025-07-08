#!/usr/bin/env python
"""Debug backtesting issues"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import backtrader as bt
import yfinance as yf
from datetime import datetime, timedelta

# Simple strategy for testing
class TestStrategy(bt.Strategy):
    def __init__(self):
        self.sma = bt.indicators.SimpleMovingAverage(self.datas[0].close, period=20)
        
    def next(self):
        if self.datas[0].close[0] > self.sma[0] and not self.position:
            self.buy(size=100)
        elif self.datas[0].close[0] < self.sma[0] and self.position:
            self.sell(size=100)

# Test basic functionality
print("Testing basic backtrader functionality...")

end_date = datetime.now()
start_date = end_date - timedelta(days=365)

# Download data
print(f"Downloading AAPL data...")
df = yf.download("AAPL", start=start_date, end=end_date, progress=False)
print(f"Downloaded {len(df)} rows")
print(df.head())

# Create cerebro
cerebro = bt.Cerebro()
cerebro.broker.setcash(100000.0)

# Add data
data = bt.feeds.PandasData(dataname=df)
cerebro.adddata(data)

# Add strategy
cerebro.addstrategy(TestStrategy)

# Run
print(f"\nStarting value: ${cerebro.broker.getvalue():,.2f}")
cerebro.run()
print(f"Final value: ${cerebro.broker.getvalue():,.2f}")

print("\n✅ Basic functionality works!")