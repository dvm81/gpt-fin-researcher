"""Market data schemas for GPT-Fin-Researcher.

Defines data models for price data, technical indicators,
and market analysis using Pydantic validation.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class Interval(str, Enum):
    """Time intervals for market data."""
    ONE_MIN = "1m"
    FIVE_MIN = "5m"
    FIFTEEN_MIN = "15m"
    THIRTY_MIN = "30m"
    ONE_HOUR = "1h"
    ONE_DAY = "1d"
    ONE_WEEK = "1wk"
    ONE_MONTH = "1mo"


class OHLCV(BaseModel):
    """Open-High-Low-Close-Volume data point."""
    timestamp: datetime = Field(..., description="Timestamp of the data point")
    open: float = Field(..., description="Opening price")
    high: float = Field(..., description="Highest price in period")
    low: float = Field(..., description="Lowest price in period")
    close: float = Field(..., description="Closing price")
    volume: int = Field(..., description="Trading volume")
    adjusted_close: Optional[float] = Field(None, description="Adjusted closing price")


class MarketData(BaseModel):
    """Complete market data for a ticker."""
    ticker: str = Field(..., description="Stock ticker symbol")
    interval: Interval = Field(..., description="Data interval")
    start_date: datetime = Field(..., description="Start date of data")
    end_date: datetime = Field(..., description="End date of data")
    data_points: List[OHLCV] = Field(..., description="Price and volume data")
    
    # Summary statistics
    total_return: Optional[float] = Field(None, description="Total return percentage")
    volatility: Optional[float] = Field(None, description="Annualized volatility")
    average_volume: Optional[int] = Field(None, description="Average daily volume")
    

class TechnicalIndicators(BaseModel):
    """Calculated technical indicators for analysis."""
    ticker: str = Field(..., description="Stock ticker symbol")
    timestamp: datetime = Field(..., description="Calculation timestamp")
    
    # Trend indicators
    sma_20: Optional[float] = Field(None, description="20-day Simple Moving Average")
    sma_50: Optional[float] = Field(None, description="50-day Simple Moving Average")
    sma_200: Optional[float] = Field(None, description="200-day Simple Moving Average")
    ema_12: Optional[float] = Field(None, description="12-day Exponential Moving Average")
    ema_26: Optional[float] = Field(None, description="26-day Exponential Moving Average")
    
    # Momentum indicators
    rsi_14: Optional[float] = Field(None, description="14-day Relative Strength Index")
    macd: Optional[float] = Field(None, description="MACD line value")
    macd_signal: Optional[float] = Field(None, description="MACD signal line")
    macd_histogram: Optional[float] = Field(None, description="MACD histogram")
    
    # Volatility indicators
    bb_upper: Optional[float] = Field(None, description="Bollinger Band upper")
    bb_middle: Optional[float] = Field(None, description="Bollinger Band middle")
    bb_lower: Optional[float] = Field(None, description="Bollinger Band lower")
    atr_14: Optional[float] = Field(None, description="14-day Average True Range")
    
    # Volume indicators
    obv: Optional[float] = Field(None, description="On-Balance Volume")
    vwap: Optional[float] = Field(None, description="Volume Weighted Average Price")


class MarketContext(BaseModel):
    """Market context for enhanced analysis."""
    ticker: str = Field(..., description="Stock ticker symbol")
    analysis_date: datetime = Field(..., description="Date of analysis")
    
    # Price action context
    price_vs_52w_high: float = Field(..., description="Current price vs 52-week high %")
    price_vs_52w_low: float = Field(..., description="Current price vs 52-week low %")
    days_since_high: int = Field(..., description="Days since 52-week high")
    days_since_low: int = Field(..., description="Days since 52-week low")
    
    # Relative performance
    vs_spy_1m: Optional[float] = Field(None, description="1-month return vs SPY")
    vs_spy_3m: Optional[float] = Field(None, description="3-month return vs SPY")
    vs_spy_6m: Optional[float] = Field(None, description="6-month return vs SPY")
    vs_sector_etf: Optional[float] = Field(None, description="Return vs sector ETF")
    
    # Market regime
    trend_strength: float = Field(..., ge=-1.0, le=1.0, description="Trend strength (-1 to 1)")
    volatility_regime: str = Field(..., description="Low/Normal/High volatility")
    volume_trend: str = Field(..., description="Increasing/Stable/Decreasing volume")


class MarketSignal(BaseModel):
    """Trading signal derived from market data."""
    ticker: str = Field(..., description="Stock ticker symbol")
    timestamp: datetime = Field(..., description="Signal generation time")
    signal_type: str = Field(..., description="Type of signal (technical/fundamental/mixed)")
    
    # Signal details
    direction: str = Field(..., description="bullish/bearish/neutral")
    strength: float = Field(..., ge=0.0, le=1.0, description="Signal strength (0-1)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence level (0-1)")
    
    # Supporting evidence
    technical_factors: List[str] = Field(..., description="Technical reasons")
    fundamental_factors: List[str] = Field(..., description="Fundamental reasons from SEC data")
    risk_factors: List[str] = Field(..., description="Risk considerations")
    
    # Actionable insights
    entry_price: Optional[float] = Field(None, description="Suggested entry price")
    stop_loss: Optional[float] = Field(None, description="Suggested stop loss")
    target_price: Optional[float] = Field(None, description="Price target")
    time_horizon: Optional[str] = Field(None, description="Expected holding period")