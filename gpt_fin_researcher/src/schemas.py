"""Data schemas for GPT-Fin-Researcher.

Defines structured data models for SEC filings, financial factors,
and trading strategies using Pydantic for validation.
"""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class FilingType(str, Enum):
    """SEC filing types."""
    TEN_K = "10-K"
    TEN_Q = "10-Q"
    EIGHT_K = "8-K"


class SentimentScore(BaseModel):
    """Sentiment analysis score with confidence."""
    score: float = Field(..., ge=-1.0, le=1.0, description="Sentiment score from -1 to 1")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the sentiment")
    reasoning: str = Field(..., description="Explanation for the sentiment score")


class SECFiling(BaseModel):
    """SEC filing document model."""
    ticker: str = Field(..., description="Stock ticker symbol")
    filing_type: FilingType = Field(..., description="Type of SEC filing")
    filing_date: str = Field(..., description="Filing date in YYYY-MM-DD format")
    text: str = Field(..., description="Full text content of the filing")
    source: str = Field(..., description="Data source identifier")
    metadata: Dict = Field(default_factory=dict, description="Additional metadata")


class FinancialFactors(BaseModel):
    """Extracted financial factors from SEC filings."""
    # Core sentiment
    overall_sentiment: SentimentScore = Field(..., description="Overall document sentiment")
    
    # Financial metrics
    revenue_growth: Optional[float] = Field(None, description="YoY revenue growth rate")
    
    # Guidance indicators
    guidance_raised: bool = Field(False, description="Whether guidance was raised")
    margin_expansion: bool = Field(False, description="Whether margins are expanding")
    debt_concerns: bool = Field(False, description="Whether there are debt concerns")
    
    # Key insights
    competitive_advantages: List[str] = Field(default_factory=list, description="Competitive advantages")
    risk_factors: List[str] = Field(default_factory=list, description="Key risk factors mentioned")
    growth_drivers: List[str] = Field(default_factory=list, description="Growth catalysts identified")


class TradingSignal(str, Enum):
    """Trading action signals."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class TradingStrategy(BaseModel):
    """Generated trading strategy model."""
    strategy_id: str = Field(..., description="Unique strategy identifier")
    name: str = Field(..., description="Strategy name")
    description: str = Field(..., description="Strategy description")
    
    # Strategy parameters
    entry_conditions: List[str] = Field(..., description="Conditions for entering positions")
    exit_conditions: List[str] = Field(..., description="Conditions for exiting positions")
    position_sizing: str = Field(..., description="Position sizing method")
    risk_management: Dict = Field(..., description="Risk management parameters")
    
    # Backtest configuration
    initial_capital: float = Field(100000.0, description="Starting capital for backtest")
    start_date: str = Field(..., description="Backtest start date")
    end_date: str = Field(..., description="Backtest end date")
    
    # Generated code
    backtrader_code: str = Field(..., description="Generated Backtrader strategy code")


class BacktestResults(BaseModel):
    """Backtest performance results."""
    strategy_id: str = Field(..., description="Strategy identifier")
    
    # Performance metrics
    total_return: float = Field(..., description="Total return percentage")
    annualized_return: float = Field(..., description="Annualized return percentage")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown percentage")
    win_rate: float = Field(..., description="Percentage of winning trades")
    
    # Trade statistics
    total_trades: int = Field(..., description="Total number of trades")
    winning_trades: int = Field(..., description="Number of winning trades")
    losing_trades: int = Field(..., description="Number of losing trades")
    avg_win: float = Field(..., description="Average winning trade return")
    avg_loss: float = Field(..., description="Average losing trade return")
    
    # Risk metrics
    volatility: float = Field(..., description="Strategy volatility")
    calmar_ratio: float = Field(..., description="Calmar ratio")
    sortino_ratio: float = Field(..., description="Sortino ratio")
    
    # Execution details
    execution_time: float = Field(..., description="Backtest execution time in seconds")
    data_points: int = Field(..., description="Number of data points processed")