"""Backtesting node for GPT-Fin-Researcher.

This module executes trading strategies using Backtrader
and calculates performance metrics.
"""

import os
import sys
from datetime import datetime, timedelta
from io import StringIO
from typing import Any, Dict, List, Optional

import backtrader as bt
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from ..schemas import BacktestResults, TradingStrategy

# Load environment variables
load_dotenv()


class StrategyBacktester:
    """Execute trading strategies and calculate performance metrics."""
    
    def __init__(self, initial_capital: float = 100000.0):
        """Initialize the backtester.
        
        Args:
            initial_capital: Starting capital for backtesting
        """
        self.initial_capital = initial_capital
        self.results = []
    
    def backtest_strategy(self,
                         strategy: TradingStrategy,
                         ticker: str,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> BacktestResults:
        """Backtest a trading strategy.
        
        Args:
            strategy: TradingStrategy object with executable code
            ticker: Stock ticker symbol
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            
        Returns:
            BacktestResults with performance metrics
        """
        # Use strategy dates if not provided
        start_date = start_date or strategy.start_date
        end_date = end_date or strategy.end_date
        
        # Create Cerebro engine
        cerebro = bt.Cerebro()
        
        # Set initial capital
        cerebro.broker.setcash(self.initial_capital)
        
        # Add data feed
        try:
            data = self._get_data_feed(ticker, start_date, end_date)
            cerebro.adddata(data)
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return self._get_error_results(strategy.strategy_id, str(e))
        
        # Execute strategy code to create strategy class
        strategy_class = self._create_strategy_class(strategy.backtrader_code)
        if not strategy_class:
            return self._get_error_results(strategy.strategy_id, "Failed to create strategy class")
        
        # Add strategy
        cerebro.addstrategy(strategy_class)
        
        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
        
        # Set commission
        cerebro.broker.setcommission(commission=0.001)  # 0.1%
        
        # Capture strategy output
        original_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            # Run backtest
            start_time = datetime.now()
            print(f"Starting value: ${cerebro.broker.getvalue():,.2f}")
            results = cerebro.run()
            execution_time = (datetime.now() - start_time).total_seconds()
            print(f"Final value: ${cerebro.broker.getvalue():,.2f}")
            
            # Get strategy instance
            strat = results[0]
            
            # Calculate metrics
            final_value = cerebro.broker.getvalue()
            total_return = ((final_value - self.initial_capital) / self.initial_capital) * 100
            
            # Get analyzer results
            sharpe = strat.analyzers.sharpe.get_analysis()
            drawdown = strat.analyzers.drawdown.get_analysis()
            returns = strat.analyzers.returns.get_analysis()
            trades = strat.analyzers.trades.get_analysis()
            sqn = strat.analyzers.sqn.get_analysis()
            
            # Calculate additional metrics
            sharpe_ratio = sharpe.get('sharperatio', 0)
            max_drawdown = drawdown.get('max', {}).get('drawdown', 0)
            
            # Trade statistics
            total_trades = trades.get('total', {}).get('total', 0)
            won_trades = trades.get('won', {}).get('total', 0)
            lost_trades = trades.get('lost', {}).get('total', 0)
            
            win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
            
            avg_win = trades.get('won', {}).get('pnl', {}).get('average', 0)
            avg_loss = abs(trades.get('lost', {}).get('pnl', {}).get('average', 0))
            
            # Annualized return
            days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
            years = days / 365.0
            annualized_return = ((final_value / self.initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
            
            # Create results
            backtest_results = BacktestResults(
                strategy_id=strategy.strategy_id,
                total_return=round(total_return, 2),
                annualized_return=round(annualized_return, 2),
                sharpe_ratio=round(sharpe_ratio, 2) if sharpe_ratio else 0,
                max_drawdown=round(abs(max_drawdown), 2),
                win_rate=round(win_rate, 2),
                total_trades=total_trades,
                winning_trades=won_trades,
                losing_trades=lost_trades,
                avg_win=round(avg_win, 2),
                avg_loss=round(avg_loss, 2),
                volatility=round(returns.get('rnorm100', 0), 2),
                calmar_ratio=round(annualized_return / abs(max_drawdown), 2) if max_drawdown != 0 else 0,
                sortino_ratio=round(sqn.get('sqn', 0), 2),
                execution_time=round(execution_time, 2),
                data_points=len(data)
            )
            
            # Get trade log
            trade_log = sys.stdout.getvalue()
            
            return backtest_results
            
        except Exception as e:
            print(f"Error during backtest: {e}")
            return self._get_error_results(strategy.strategy_id, str(e))
        
        finally:
            sys.stdout = original_stdout
    
    def _get_data_feed(self, ticker: str, start_date: str, end_date: str) -> bt.feeds.PandasData:
        """Get historical data feed for backtesting.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Backtrader data feed
        """
        # Download data from Yahoo Finance
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
        
        if df.empty:
            raise ValueError(f"No data found for {ticker}")
        
        # Handle multi-index columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Prepare data for Backtrader
        df.index = pd.to_datetime(df.index)
        df['OpenInterest'] = 0  # Required by Backtrader
        
        # Create Pandas data feed
        data = bt.feeds.PandasData(
            dataname=df,
            datetime=None,
            open='Open',
            high='High',
            low='Low',
            close='Close',
            volume='Volume',
            openinterest='OpenInterest'
        )
        
        return data
    
    def _create_strategy_class(self, code: str):
        """Create strategy class from generated code.
        
        Args:
            code: Generated Backtrader strategy code
            
        Returns:
            Strategy class or None if failed
        """
        try:
            # Create a namespace for execution
            namespace = {
                'bt': bt,
                'datetime': datetime
            }
            
            # Execute the code
            exec(code, namespace)
            
            # Find the strategy class
            for name, obj in namespace.items():
                if (isinstance(obj, type) and 
                    issubclass(obj, bt.Strategy) and 
                    obj != bt.Strategy):
                    return obj
            
            print("No strategy class found in generated code")
            return None
            
        except Exception as e:
            print(f"Error creating strategy class: {e}")
            return None
    
    def _get_error_results(self, strategy_id: str, error_msg: str) -> BacktestResults:
        """Return error results when backtest fails."""
        return BacktestResults(
            strategy_id=strategy_id,
            total_return=-99.99,
            annualized_return=-99.99,
            sharpe_ratio=-99.99,
            max_drawdown=99.99,
            win_rate=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            avg_win=0,
            avg_loss=0,
            volatility=0,
            calmar_ratio=-99.99,
            sortino_ratio=-99.99,
            execution_time=0,
            data_points=0
        )
    
    def plot_results(self, cerebro: bt.Cerebro, filename: str = None):
        """Plot backtest results.
        
        Args:
            cerebro: Cerebro instance after running
            filename: Optional filename to save plot
        """
        try:
            cerebro.plot(style='candlestick', volume=False)
            if filename:
                import matplotlib.pyplot as plt
                plt.savefig(filename)
        except Exception as e:
            print(f"Error plotting results: {e}")


def backtest_strategies(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node for backtesting trading strategies.
    
    Args:
        state: Graph state with strategies and market data
        
    Returns:
        Updated state with backtest results
    """
    strategies = state.get("strategies", [])
    market_context = state.get("market_context")
    
    if not strategies:
        return {
            **state,
            "error": "No strategies to backtest"
        }
    
    # Get ticker
    ticker = market_context.ticker if market_context else None
    
    if not ticker:
        # Try to extract from strategy or state
        for strategy in strategies:
            if hasattr(strategy, 'name') and any(t in strategy.name for t in ['AAPL', 'TSLA', 'MSFT', 'NVDA']):
                ticker = next(t for t in ['AAPL', 'TSLA', 'MSFT', 'NVDA'] if t in strategy.name)
                break
    
    if not ticker:
        return {
            **state,
            "error": "No ticker found for backtesting"
        }
    
    # Initialize backtester
    backtester = StrategyBacktester(
        initial_capital=state.get("initial_capital", 100000.0)
    )
    
    backtest_results = []
    
    # Backtest each strategy
    for strategy in strategies:
        try:
            print(f"\n📊 Backtesting: {strategy.name}")
            print(f"   Ticker: {ticker}")
            print(f"   Period: {strategy.start_date} to {strategy.end_date}")
            
            # Run backtest
            results = backtester.backtest_strategy(
                strategy=strategy,
                ticker=ticker,
                start_date=strategy.start_date,
                end_date=strategy.end_date
            )
            
            backtest_results.append(results)
            
            # Print results summary
            print(f"\n📈 Performance Metrics:")
            print(f"   Total Return: {results.total_return}%")
            print(f"   Annualized Return: {results.annualized_return}%")
            print(f"   Sharpe Ratio: {results.sharpe_ratio}")
            print(f"   Max Drawdown: {results.max_drawdown}%")
            print(f"   Win Rate: {results.win_rate}%")
            print(f"   Total Trades: {results.total_trades}")
            
            if results.total_return > 0:
                print("   ✅ Strategy is profitable!")
            else:
                print("   ❌ Strategy is not profitable")
            
        except Exception as e:
            print(f"❌ Error backtesting strategy: {e}")
            backtest_results.append(
                backtester._get_error_results(strategy.strategy_id, str(e))
            )
    
    return {
        **state,
        "backtest_results": backtest_results,
        "current_step": "backtesting_complete"
    }