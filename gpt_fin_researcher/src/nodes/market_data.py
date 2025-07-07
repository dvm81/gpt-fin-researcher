"""Market data fetching node for GPT-Fin-Researcher.

This module fetches price data and calculates technical indicators
to complement SEC filing analysis.
"""

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from ..schemas.market_data import (
    Interval, MarketContext, MarketData, OHLCV, TechnicalIndicators
)

# Load environment variables
load_dotenv()


class MarketDataFetcher:
    """Fetches and processes market data from Yahoo Finance."""
    
    def __init__(self):
        """Initialize the market data fetcher."""
        self.cache = {}  # Simple in-memory cache
    
    def fetch_ohlcv(self, 
                   ticker: str, 
                   period: str = "2y",
                   interval: str = "1d") -> MarketData:
        """Fetch OHLCV data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            MarketData object with price history
        """
        cache_key = f"{ticker}_{period}_{interval}"
        
        # Check cache first
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if datetime.now() - cached_time < timedelta(minutes=15):
                return cached_data
        
        try:
            # Fetch data from Yahoo Finance
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)
            
            if df.empty:
                raise ValueError(f"No data found for {ticker}")
            
            # Convert to OHLCV objects
            data_points = []
            for idx, row in df.iterrows():
                ohlcv = OHLCV(
                    timestamp=idx.to_pydatetime(),
                    open=round(row['Open'], 2),
                    high=round(row['High'], 2),
                    low=round(row['Low'], 2),
                    close=round(row['Close'], 2),
                    volume=int(row['Volume'])
                )
                data_points.append(ohlcv)
            
            # Calculate summary statistics
            returns = df['Close'].pct_change().dropna()
            total_return = ((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized
            avg_volume = int(df['Volume'].mean())
            
            market_data = MarketData(
                ticker=ticker.upper(),
                interval=Interval(interval),
                start_date=df.index[0].to_pydatetime(),
                end_date=df.index[-1].to_pydatetime(),
                data_points=data_points,
                total_return=round(total_return, 2),
                volatility=round(volatility, 2),
                average_volume=avg_volume
            )
            
            # Cache the result
            self.cache[cache_key] = (market_data, datetime.now())
            
            return market_data
            
        except Exception as e:
            print(f"Error fetching market data for {ticker}: {e}")
            raise
    
    def calculate_indicators(self, ticker: str, df: pd.DataFrame = None) -> TechnicalIndicators:
        """Calculate technical indicators for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            df: Optional DataFrame with OHLCV data
            
        Returns:
            TechnicalIndicators object
        """
        if df is None:
            # Fetch recent data if not provided
            market_data = self.fetch_ohlcv(ticker, period="6mo")
            df = pd.DataFrame([{
                'timestamp': dp.timestamp,
                'open': dp.open,
                'high': dp.high,
                'low': dp.low,
                'close': dp.close,
                'volume': dp.volume
            } for dp in market_data.data_points])
            df.set_index('timestamp', inplace=True)
        
        # Calculate moving averages
        sma_20 = df['close'].rolling(window=20).mean().iloc[-1]
        sma_50 = df['close'].rolling(window=50).mean().iloc[-1]
        sma_200 = df['close'].rolling(window=200).mean().iloc[-1] if len(df) >= 200 else None
        
        # EMA
        ema_12 = df['close'].ewm(span=12, adjust=False).mean().iloc[-1]
        ema_26 = df['close'].ewm(span=26, adjust=False).mean().iloc[-1]
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi_14 = 100 - (100 / (1 + rs)).iloc[-1]
        
        # MACD
        macd_line = ema_12 - ema_26
        macd_signal = df['close'].ewm(span=9, adjust=False).mean().iloc[-1]
        macd_histogram = macd_line - macd_signal
        
        # Bollinger Bands
        bb_middle = sma_20
        bb_std = df['close'].rolling(window=20).std().iloc[-1]
        bb_upper = bb_middle + (2 * bb_std)
        bb_lower = bb_middle - (2 * bb_std)
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr_14 = tr.rolling(window=14).mean().iloc[-1]
        
        # OBV
        obv = ((df['close'] > df['close'].shift()) * df['volume']).cumsum().iloc[-1]
        
        # VWAP (daily)
        vwap = ((df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / 
                df['volume'].cumsum()).iloc[-1]
        
        return TechnicalIndicators(
            ticker=ticker.upper(),
            timestamp=datetime.now(),
            sma_20=round(sma_20, 2) if not pd.isna(sma_20) else None,
            sma_50=round(sma_50, 2) if not pd.isna(sma_50) else None,
            sma_200=round(sma_200, 2) if sma_200 and not pd.isna(sma_200) else None,
            ema_12=round(ema_12, 2),
            ema_26=round(ema_26, 2),
            rsi_14=round(rsi_14, 2) if not pd.isna(rsi_14) else None,
            macd=round(macd_line, 4),
            macd_signal=round(macd_signal, 4),
            macd_histogram=round(macd_histogram, 4),
            bb_upper=round(bb_upper, 2),
            bb_middle=round(bb_middle, 2),
            bb_lower=round(bb_lower, 2),
            atr_14=round(atr_14, 2) if not pd.isna(atr_14) else None,
            obv=int(obv) if not pd.isna(obv) else None,
            vwap=round(vwap, 2) if not pd.isna(vwap) else None
        )
    
    def get_market_context(self, ticker: str) -> MarketContext:
        """Get comprehensive market context for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            MarketContext object with relative performance metrics
        """
        try:
            # Get 1 year of data for context
            stock_data = self.fetch_ohlcv(ticker, period="1y")
            df = pd.DataFrame([{
                'close': dp.close,
                'high': dp.high,
                'low': dp.low,
                'volume': dp.volume,
                'timestamp': dp.timestamp
            } for dp in stock_data.data_points])
            
            current_price = df['close'].iloc[-1]
            
            # 52-week high/low
            high_52w = df['high'].max()
            low_52w = df['low'].min()
            high_date = df[df['high'] == high_52w]['timestamp'].iloc[0]
            low_date = df[df['low'] == low_52w]['timestamp'].iloc[0]
            
            # Price position
            price_vs_high = ((current_price / high_52w) - 1) * 100
            price_vs_low = ((current_price / low_52w) - 1) * 100
            # Handle timezone-aware timestamps
            now = pd.Timestamp.now(tz=high_date.tz) if hasattr(high_date, 'tz') and high_date.tz else datetime.now()
            days_since_high = (now - high_date).days
            days_since_low = (now - low_date).days
            
            # Get SPY data for relative performance
            spy_data = self.fetch_ohlcv("SPY", period="6mo")
            spy_df = pd.DataFrame([{
                'close': dp.close,
                'timestamp': dp.timestamp
            } for dp in spy_data.data_points])
            
            # Calculate relative returns
            stock_1m = self._calculate_return(df, 30)
            spy_1m = self._calculate_return(spy_df, 30)
            vs_spy_1m = stock_1m - spy_1m
            
            stock_3m = self._calculate_return(df, 90)
            spy_3m = self._calculate_return(spy_df, 90)
            vs_spy_3m = stock_3m - spy_3m
            
            # Trend strength (based on moving averages)
            indicators = self.calculate_indicators(ticker, df)
            if indicators.sma_50 and indicators.sma_200:
                if current_price > indicators.sma_50 > indicators.sma_200:
                    trend_strength = 0.8
                elif current_price > indicators.sma_50:
                    trend_strength = 0.5
                elif current_price < indicators.sma_50 < indicators.sma_200:
                    trend_strength = -0.8
                else:
                    trend_strength = -0.5
            else:
                trend_strength = 0.0
            
            # Volatility regime
            recent_vol = df['close'].pct_change().tail(20).std() * np.sqrt(252)
            if recent_vol < 0.15:
                vol_regime = "Low"
            elif recent_vol < 0.25:
                vol_regime = "Normal"
            else:
                vol_regime = "High"
            
            # Volume trend
            recent_avg_vol = df['volume'].tail(20).mean()
            older_avg_vol = df['volume'].iloc[-60:-20].mean()
            if recent_avg_vol > older_avg_vol * 1.2:
                volume_trend = "Increasing"
            elif recent_avg_vol < older_avg_vol * 0.8:
                volume_trend = "Decreasing"
            else:
                volume_trend = "Stable"
            
            return MarketContext(
                ticker=ticker.upper(),
                analysis_date=datetime.now(),
                price_vs_52w_high=round(price_vs_high, 2),
                price_vs_52w_low=round(price_vs_low, 2),
                days_since_high=days_since_high,
                days_since_low=days_since_low,
                vs_spy_1m=round(vs_spy_1m, 2) if vs_spy_1m else None,
                vs_spy_3m=round(vs_spy_3m, 2) if vs_spy_3m else None,
                vs_spy_6m=None,  # TODO: Implement
                vs_sector_etf=None,  # TODO: Implement sector comparison
                trend_strength=round(trend_strength, 2),
                volatility_regime=vol_regime,
                volume_trend=volume_trend
            )
            
        except Exception as e:
            print(f"Error getting market context for {ticker}: {e}")
            raise
    
    def _calculate_return(self, df: pd.DataFrame, days: int) -> float:
        """Calculate return over specified days."""
        if len(df) < days:
            return None
        current = df['close'].iloc[-1]
        past = df['close'].iloc[-days]
        return ((current / past) - 1) * 100


def fetch_market_data(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node for fetching market data.
    
    Args:
        state: Graph state containing ticker information
        
    Returns:
        Updated state with market data
    """
    # Extract ticker from tasks
    ticker = None
    tasks = state.get("tasks", [])
    for task in tasks:
        # Extract ticker from task string
        parts = task.split()
        for part in parts:
            if part.isupper() and 2 <= len(part) <= 5:
                ticker = part
                break
    
    if not ticker:
        return {
            **state,
            "error": "No ticker found for market data fetch"
        }
    
    fetcher = MarketDataFetcher()
    
    try:
        # Fetch comprehensive market data
        market_data = fetcher.fetch_ohlcv(ticker)
        indicators = fetcher.calculate_indicators(ticker)
        context = fetcher.get_market_context(ticker)
        
        print(f"✅ Fetched market data for {ticker}")
        print(f"   Period: {market_data.start_date.date()} to {market_data.end_date.date()}")
        print(f"   Total return: {market_data.total_return}%")
        print(f"   Current RSI: {indicators.rsi_14}")
        print(f"   vs SPY (1M): {context.vs_spy_1m}%")
        
        return {
            **state,
            "market_data": market_data,
            "technical_indicators": indicators,
            "market_context": context,
            "current_step": "market_data_fetch"
        }
        
    except Exception as e:
        print(f"❌ Error fetching market data: {e}")
        return {
            **state,
            "error": f"Market data fetch failed: {str(e)}"
        }