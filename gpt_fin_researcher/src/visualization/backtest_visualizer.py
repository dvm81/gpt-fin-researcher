"""Backtest Results Visualization for GPT-Fin-Researcher.

This module creates interactive charts and plots to visualize
trading strategy performance and backtest results.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any

import backtrader as bt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from matplotlib.patches import Rectangle

from ..schemas import BacktestResults, TradingStrategy


class BacktestVisualizer:
    """Create visualizations for backtest results."""
    
    def __init__(self, output_dir: str = "./backtest_charts"):
        """Initialize the visualizer.
        
        Args:
            output_dir: Directory to save chart files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def create_performance_dashboard(self, 
                                   results: List[BacktestResults],
                                   strategies: List[TradingStrategy],
                                   ticker: str) -> str:
        """Create a comprehensive performance dashboard.
        
        Args:
            results: List of backtest results
            strategies: List of trading strategies
            ticker: Stock ticker symbol
            
        Returns:
            Path to saved dashboard image
        """
        if not results or not strategies:
            return None
            
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # Main title
        fig.suptitle(f'{ticker} Trading Strategy Performance Dashboard', 
                    fontsize=20, fontweight='bold')
        
        # 1. Performance Summary Table
        ax1 = fig.add_subplot(gs[0, :])
        self._create_performance_table(ax1, results, strategies)
        
        # 2. Returns Comparison
        ax2 = fig.add_subplot(gs[1, 0])
        self._create_returns_comparison(ax2, results, strategies)
        
        # 3. Risk Metrics
        ax3 = fig.add_subplot(gs[1, 1])
        self._create_risk_metrics(ax3, results, strategies)
        
        # 4. Win Rate Analysis
        ax4 = fig.add_subplot(gs[1, 2])
        self._create_win_rate_analysis(ax4, results, strategies)
        
        # 5. Equity Curve (if we have detailed data)
        ax5 = fig.add_subplot(gs[2, :])
        self._create_equity_curve(ax5, results, strategies, ticker)
        
        # 6. Drawdown Analysis
        ax6 = fig.add_subplot(gs[3, 0])
        self._create_drawdown_analysis(ax6, results, strategies)
        
        # 7. Trade Distribution
        ax7 = fig.add_subplot(gs[3, 1])
        self._create_trade_distribution(ax7, results, strategies)
        
        # 8. Performance Ranking
        ax8 = fig.add_subplot(gs[3, 2])
        self._create_performance_ranking(ax8, results, strategies)
        
        # Save dashboard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{ticker}_dashboard_{timestamp}.png"
        filepath = self.output_dir / filename
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _create_performance_table(self, ax, results: List[BacktestResults], strategies: List[TradingStrategy]):
        """Create performance summary table."""
        ax.axis('off')
        
        # Prepare data
        table_data = []
        headers = ['Strategy', 'Total Return', 'Annualized Return', 'Sharpe Ratio', 
                  'Max Drawdown', 'Win Rate', 'Total Trades']
        
        for result, strategy in zip(results, strategies):
            row = [
                strategy.name[:20] + '...' if len(strategy.name) > 20 else strategy.name,
                f"{result.total_return:.2f}%",
                f"{result.annualized_return:.2f}%",
                f"{result.sharpe_ratio:.2f}",
                f"{result.max_drawdown:.2f}%",
                f"{result.win_rate:.2f}%",
                f"{result.total_trades}"
            ]
            table_data.append(row)
        
        # Create table
        table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # Style table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color code performance
        for i, result in enumerate(results):
            row = i + 1
            # Color total return column
            if result.total_return > 0:
                table[(row, 1)].set_facecolor('#E8F5E8')
            else:
                table[(row, 1)].set_facecolor('#FFE8E8')
        
        ax.set_title('Strategy Performance Summary', fontsize=14, fontweight='bold', pad=20)
    
    def _create_returns_comparison(self, ax, results: List[BacktestResults], strategies: List[TradingStrategy]):
        """Create returns comparison chart."""
        strategy_names = [s.name[:15] + '...' if len(s.name) > 15 else s.name for s in strategies]
        total_returns = [r.total_return for r in results]
        annualized_returns = [r.annualized_return for r in results]
        
        x = np.arange(len(strategy_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, total_returns, width, label='Total Return', alpha=0.8)
        bars2 = ax.bar(x + width/2, annualized_returns, width, label='Annualized Return', alpha=0.8)
        
        # Color bars based on performance
        for bar, return_val in zip(bars1, total_returns):
            bar.set_color('#4CAF50' if return_val > 0 else '#F44336')
        
        for bar, return_val in zip(bars2, annualized_returns):
            bar.set_color('#81C784' if return_val > 0 else '#EF5350')
        
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Return (%)')
        ax.set_title('Returns Comparison', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(strategy_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    def _create_risk_metrics(self, ax, results: List[BacktestResults], strategies: List[TradingStrategy]):
        """Create risk metrics visualization."""
        strategy_names = [s.name[:15] + '...' if len(s.name) > 15 else s.name for s in strategies]
        sharpe_ratios = [r.sharpe_ratio for r in results]
        max_drawdowns = [r.max_drawdown for r in results]
        
        # Create scatter plot
        colors = ['#4CAF50' if sr > 1 else '#FFC107' if sr > 0.5 else '#F44336' for sr in sharpe_ratios]
        
        scatter = ax.scatter(max_drawdowns, sharpe_ratios, 
                           c=colors, s=100, alpha=0.7, edgecolors='black')
        
        # Add labels
        for i, name in enumerate(strategy_names):
            ax.annotate(name, (max_drawdowns[i], sharpe_ratios[i]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, ha='left')
        
        ax.set_xlabel('Max Drawdown (%)')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title('Risk-Return Profile', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add reference lines
        ax.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Good Sharpe (1.0)')
        ax.axvline(x=20, color='red', linestyle='--', alpha=0.5, label='High Risk (20%)')
        ax.legend()
    
    def _create_win_rate_analysis(self, ax, results: List[BacktestResults], strategies: List[TradingStrategy]):
        """Create win rate analysis."""
        strategy_names = [s.name[:15] + '...' if len(s.name) > 15 else s.name for s in strategies]
        win_rates = [r.win_rate for r in results]
        total_trades = [r.total_trades for r in results]
        
        # Create bubble chart
        sizes = [max(50, t * 10) for t in total_trades]  # Minimum size 50
        colors = ['#4CAF50' if wr > 50 else '#FFC107' if wr > 40 else '#F44336' for wr in win_rates]
        
        scatter = ax.scatter(range(len(strategy_names)), win_rates, 
                           s=sizes, c=colors, alpha=0.7, edgecolors='black')
        
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Win Rate (%)')
        ax.set_title('Win Rate Analysis\n(Bubble size = # trades)', fontweight='bold')
        ax.set_xticks(range(len(strategy_names)))
        ax.set_xticklabels(strategy_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Add reference line
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% Win Rate')
        ax.legend()
    
    def _create_equity_curve(self, ax, results: List[BacktestResults], strategies: List[TradingStrategy], ticker: str):
        """Create equity curve visualization."""
        # For now, create a simulated equity curve based on the results
        # In a real implementation, you'd capture the actual equity curve from backtrader
        
        ax.set_title(f'{ticker} Strategy Equity Curves', fontweight='bold')
        
        # Generate sample equity curves based on final returns
        days = 252  # Trading days in a year
        
        for i, (result, strategy) in enumerate(zip(results, strategies)):
            # Simulate equity curve
            initial_value = 100000
            final_value = initial_value * (1 + result.total_return / 100)
            
            # Create a curved path to final value
            x = np.linspace(0, days, days)
            
            # Add some volatility based on max drawdown
            volatility = result.max_drawdown / 100 * 0.3
            random_walk = np.random.normal(0, volatility, days)
            
            # Calculate daily returns that reach the final value
            daily_return = (final_value / initial_value) ** (1/days) - 1
            
            equity = [initial_value]
            for j in range(1, days):
                # Add trend + noise
                value = equity[-1] * (1 + daily_return + random_walk[j] * 0.01)
                equity.append(value)
            
            # Normalize to actual final value
            equity = np.array(equity)
            equity = equity * (final_value / equity[-1])
            
            ax.plot(x, equity, label=strategy.name[:20], linewidth=2)
        
        # Add buy & hold benchmark
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")
            if not hist.empty:
                benchmark = hist['Close'].values
                benchmark = (benchmark / benchmark[0]) * 100000
                ax.plot(range(len(benchmark)), benchmark, 
                       label=f'{ticker} Buy & Hold', linestyle='--', alpha=0.7)
        except:
            pass
        
        ax.set_xlabel('Trading Days')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    def _create_drawdown_analysis(self, ax, results: List[BacktestResults], strategies: List[TradingStrategy]):
        """Create drawdown analysis."""
        strategy_names = [s.name[:15] + '...' if len(s.name) > 15 else s.name for s in strategies]
        drawdowns = [r.max_drawdown for r in results]
        
        bars = ax.bar(strategy_names, drawdowns, color='#F44336', alpha=0.7)
        
        # Add warning zones
        ax.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='Moderate Risk (10%)')
        ax.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='High Risk (20%)')
        
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Max Drawdown (%)')
        ax.set_title('Maximum Drawdown Analysis', fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    def _create_trade_distribution(self, ax, results: List[BacktestResults], strategies: List[TradingStrategy]):
        """Create trade distribution analysis."""
        strategy_names = [s.name[:15] + '...' if len(s.name) > 15 else s.name for s in strategies]
        winning_trades = [r.winning_trades for r in results]
        losing_trades = [r.losing_trades for r in results]
        
        x = np.arange(len(strategy_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, winning_trades, width, label='Winning Trades', 
                      color='#4CAF50', alpha=0.8)
        bars2 = ax.bar(x + width/2, losing_trades, width, label='Losing Trades', 
                      color='#F44336', alpha=0.8)
        
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Number of Trades')
        ax.set_title('Trade Distribution', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(strategy_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _create_performance_ranking(self, ax, results: List[BacktestResults], strategies: List[TradingStrategy]):
        """Create performance ranking."""
        # Calculate composite score
        scores = []
        for result in results:
            # Normalize metrics and create composite score
            return_score = max(0, result.total_return / 100)  # Normalize to 0-1+
            sharpe_score = max(0, result.sharpe_ratio / 2)    # Normalize to 0-1+
            drawdown_score = max(0, 1 - result.max_drawdown / 50)  # Invert and normalize
            
            composite_score = (return_score * 0.4 + sharpe_score * 0.4 + drawdown_score * 0.2)
            scores.append(composite_score)
        
        # Sort by score
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        sorted_names = [strategies[i].name[:15] + '...' if len(strategies[i].name) > 15 
                       else strategies[i].name for i in sorted_indices]
        sorted_scores = [scores[i] for i in sorted_indices]
        
        # Create horizontal bar chart
        bars = ax.barh(sorted_names, sorted_scores, color='#2196F3', alpha=0.8)
        
        ax.set_xlabel('Composite Performance Score')
        ax.set_title('Strategy Performance Ranking', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add score labels
        for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
            ax.text(score + 0.01, i, f'{score:.2f}', 
                   va='center', fontsize=8)
    
    def create_individual_strategy_chart(self, 
                                       result: BacktestResults,
                                       strategy: TradingStrategy,
                                       ticker: str) -> str:
        """Create detailed chart for individual strategy.
        
        Args:
            result: Backtest result
            strategy: Trading strategy
            ticker: Stock ticker symbol
            
        Returns:
            Path to saved chart
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{strategy.name} - {ticker} Detailed Analysis', 
                    fontsize=16, fontweight='bold')
        
        # 1. Key metrics display
        ax1 = axes[0, 0]
        self._create_metrics_display(ax1, result, strategy)
        
        # 2. Risk-return scatter
        ax2 = axes[0, 1]
        self._create_risk_return_scatter(ax2, result, strategy)
        
        # 3. Trade analysis
        ax3 = axes[1, 0]
        self._create_trade_analysis(ax3, result, strategy)
        
        # 4. Performance summary
        ax4 = axes[1, 1]
        self._create_performance_summary(ax4, result, strategy)
        
        # Save chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c for c in strategy.name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"{ticker}_{safe_name}_{timestamp}.png"
        filepath = self.output_dir / filename
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _create_metrics_display(self, ax, result: BacktestResults, strategy: TradingStrategy):
        """Create metrics display."""
        ax.axis('off')
        
        metrics = [
            ('Total Return', f"{result.total_return:.2f}%"),
            ('Annualized Return', f"{result.annualized_return:.2f}%"),
            ('Sharpe Ratio', f"{result.sharpe_ratio:.2f}"),
            ('Max Drawdown', f"{result.max_drawdown:.2f}%"),
            ('Win Rate', f"{result.win_rate:.2f}%"),
            ('Total Trades', f"{result.total_trades}"),
            ('Avg Win', f"${result.avg_win:.2f}"),
            ('Avg Loss', f"${result.avg_loss:.2f}"),
        ]
        
        y_pos = 0.9
        for label, value in metrics:
            ax.text(0.1, y_pos, label, fontsize=12, fontweight='bold', transform=ax.transAxes)
            ax.text(0.6, y_pos, value, fontsize=12, transform=ax.transAxes)
            y_pos -= 0.1
        
        ax.set_title('Key Metrics', fontweight='bold')
    
    def _create_risk_return_scatter(self, ax, result: BacktestResults, strategy: TradingStrategy):
        """Create risk-return scatter plot."""
        ax.scatter(result.max_drawdown, result.total_return, 
                  s=200, c='blue', alpha=0.7, edgecolors='black')
        
        ax.set_xlabel('Max Drawdown (%)')
        ax.set_ylabel('Total Return (%)')
        ax.set_title('Risk-Return Profile', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add quadrant labels
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.axvline(x=20, color='gray', linestyle='-', alpha=0.3)
        
        # Add performance rating
        if result.total_return > 0 and result.max_drawdown < 20:
            rating = "Excellent"
            color = "green"
        elif result.total_return > 0:
            rating = "Good"
            color = "orange"
        else:
            rating = "Poor"
            color = "red"
        
        ax.text(0.05, 0.95, f"Rating: {rating}", transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3),
               fontweight='bold')
    
    def _create_trade_analysis(self, ax, result: BacktestResults, strategy: TradingStrategy):
        """Create trade analysis chart."""
        labels = ['Winning Trades', 'Losing Trades']
        sizes = [result.winning_trades, result.losing_trades]
        colors = ['#4CAF50', '#F44336']
        explode = (0.05, 0.05)
        
        if sum(sizes) > 0:
            ax.pie(sizes, labels=labels, colors=colors, explode=explode,
                  autopct='%1.1f%%', shadow=True, startangle=90)
        else:
            ax.text(0.5, 0.5, 'No trades executed', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
        
        ax.set_title('Trade Distribution', fontweight='bold')
    
    def _create_performance_summary(self, ax, result: BacktestResults, strategy: TradingStrategy):
        """Create performance summary."""
        ax.axis('off')
        
        # Performance rating
        if result.total_return > 20 and result.sharpe_ratio > 1:
            rating = "⭐⭐⭐⭐⭐ Excellent"
        elif result.total_return > 10 and result.sharpe_ratio > 0.5:
            rating = "⭐⭐⭐⭐ Good"
        elif result.total_return > 0:
            rating = "⭐⭐⭐ Average"
        elif result.total_return > -10:
            rating = "⭐⭐ Poor"
        else:
            rating = "⭐ Very Poor"
        
        ax.text(0.5, 0.8, "Performance Rating", ha='center', fontsize=14, 
               fontweight='bold', transform=ax.transAxes)
        ax.text(0.5, 0.6, rating, ha='center', fontsize=16, 
               transform=ax.transAxes)
        
        # Strategy summary
        ax.text(0.5, 0.4, "Strategy Summary", ha='center', fontsize=12, 
               fontweight='bold', transform=ax.transAxes)
        
        summary_text = f"Entry: {strategy.entry_conditions[0] if strategy.entry_conditions else 'N/A'}\n"
        summary_text += f"Exit: {strategy.exit_conditions[0] if strategy.exit_conditions else 'N/A'}\n"
        summary_text += f"Risk: {strategy.risk_management.get('stop_loss', 'N/A')}"
        
        ax.text(0.5, 0.2, summary_text, ha='center', fontsize=10, 
               transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.5", 
               facecolor='lightblue', alpha=0.3))