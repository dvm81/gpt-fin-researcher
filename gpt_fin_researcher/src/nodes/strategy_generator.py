"""Strategy generation node for GPT-Fin-Researcher.

This module generates trading strategies based on financial factors
and market data analysis.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from ..schemas import TradingStrategy

# Load environment variables
load_dotenv()


class StrategyGenerator:
    """Generate trading strategies using AI based on comprehensive analysis."""
    
    def __init__(self,
                 model: str = "gpt-4o-mini",
                 temperature: float = 0.3,
                 max_tokens: int = 3000):
        """Initialize the strategy generator.
        
        Args:
            model: OpenAI model to use
            temperature: Sampling temperature (higher = more creative)
            max_tokens: Maximum tokens in response
        """
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def generate_strategy(self,
                         ticker: str,
                         financial_factors: Dict,
                         market_context: Dict,
                         technical_indicators: Dict) -> TradingStrategy:
        """Generate a trading strategy based on comprehensive analysis.
        
        Args:
            ticker: Stock ticker symbol
            financial_factors: Extracted financial insights
            market_context: Current market position
            technical_indicators: Technical analysis metrics
            
        Returns:
            TradingStrategy object with executable code
        """
        prompt = self._build_strategy_prompt(
            ticker, financial_factors, market_context, technical_indicators
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )
            
            strategy_data = json.loads(response.choices[0].message.content)
            
            # Generate strategy ID
            strategy_data["strategy_id"] = f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            return TradingStrategy(**strategy_data)
            
        except Exception as e:
            print(f"Error generating strategy: {e}")
            return self._get_fallback_strategy(ticker)
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for strategy generation."""
        return """You are an expert quantitative trading strategist. Generate trading strategies based on fundamental and technical analysis.

Your strategies should be:
1. Data-driven and based on the provided analysis
2. Include specific entry/exit conditions
3. Have clear risk management rules
4. Be implementable in Backtrader

Return a JSON object with:
- name: Strategy name (descriptive)
- description: 1-2 sentence description
- entry_conditions: List of specific conditions for entering positions
- exit_conditions: List of specific conditions for exiting positions
- position_sizing: Method for determining position size (e.g., "fixed", "kelly", "risk_parity")
- risk_management: Dictionary with stop_loss, take_profit, max_position_size, max_drawdown
- initial_capital: Starting capital (default 100000)
- start_date: Backtest start date (YYYY-MM-DD)
- end_date: Backtest end date (YYYY-MM-DD)
- backtrader_code: Complete Backtrader strategy class implementation

Example risk_management:
{
    "stop_loss": 0.05,  // 5% stop loss
    "take_profit": 0.15,  // 15% take profit
    "max_position_size": 0.25,  // Max 25% of portfolio
    "max_drawdown": 0.20  // Exit if 20% drawdown
}"""
    
    def _build_strategy_prompt(self,
                              ticker: str,
                              financial_factors: Dict,
                              market_context: Dict,
                              technical_indicators: Dict) -> str:
        """Build the strategy generation prompt."""
        
        # Extract key insights
        sentiment = financial_factors.get("overall_sentiment", {})
        revenue_growth = financial_factors.get("revenue_growth", "N/A")
        guidance = "Raised" if financial_factors.get("guidance_raised") else "Not raised"
        risks = financial_factors.get("risk_factors", [])[:2]
        
        # Market position
        price_vs_high = market_context.get("price_vs_52w_high", 0)
        vs_spy = market_context.get("vs_spy_1m", 0)
        trend = market_context.get("trend_strength", 0)
        
        # Technical signals
        rsi = technical_indicators.get("rsi_14", 50)
        sma20 = technical_indicators.get("sma_20", 0)
        sma50 = technical_indicators.get("sma_50", 0)
        
        prompt = f"""Generate a trading strategy for {ticker} based on this analysis:

FUNDAMENTAL ANALYSIS:
- Sentiment: {sentiment.get('score', 0)} ({sentiment.get('reasoning', 'N/A')})
- Revenue Growth: {revenue_growth}%
- Guidance: {guidance}
- Key Risks: {', '.join(risks) if risks else 'None identified'}

MARKET POSITION:
- Price vs 52W High: {price_vs_high}%
- vs SPY (1M): {vs_spy}%
- Trend Strength: {trend}
- Current Price: ${technical_indicators.get('current_price', 0)}

TECHNICAL INDICATORS:
- RSI(14): {rsi}
- SMA(20): ${sma20}
- SMA(50): ${sma50}
- Price > SMA20: {technical_indicators.get('current_price', 0) > sma20 if sma20 else False}

Generate a strategy that:
1. Leverages the fundamental strengths/weaknesses
2. Considers current market position
3. Uses technical indicators for timing
4. Implements appropriate risk management

The backtrader code should be a complete, executable strategy class."""
        
        return prompt
    
    def generate_backtrader_code(self, strategy: TradingStrategy) -> str:
        """Generate executable Backtrader code from strategy."""
        
        code = f"""import backtrader as bt
import datetime

class {strategy.name.replace(' ', '')}(bt.Strategy):
    \"\"\"
    {strategy.description}
    
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    \"\"\"
    
    params = (
        ('stop_loss', {strategy.risk_management.get('stop_loss', 0.05)}),
        ('take_profit', {strategy.risk_management.get('take_profit', 0.15)}),
        ('max_position_pct', {strategy.risk_management.get('max_position_size', 0.25)}),
    )
    
    def __init__(self):
        # Technical indicators
        self.sma20 = bt.indicators.SimpleMovingAverage(
            self.data.close, period=20)
        self.sma50 = bt.indicators.SimpleMovingAverage(
            self.data.close, period=50)
        self.rsi = bt.indicators.RelativeStrengthIndex(period=14)
        
        # Track position
        self.order = None
        self.entry_price = None
        
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status in [order.Completed]:
            if order.isbuy():
                self.entry_price = order.executed.price
                self.log(f'BUY EXECUTED, Price: {{order.executed.price:.2f}}')
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {{order.executed.price:.2f}}')
                self.entry_price = None
                
        self.order = None
    
    def next(self):
        # Check if we have an order pending
        if self.order:
            return
        
        # Check if we are in the market
        if not self.position:
            # Entry conditions
            if self._check_entry_conditions():
                # Calculate position size
                size = self._calculate_position_size()
                self.order = self.buy(size=size)
                self.log(f'BUY CREATE, Size: {{size}}')
        else:
            # Exit conditions
            if self._check_exit_conditions():
                self.order = self.sell()
                self.log('SELL CREATE')
    
    def _check_entry_conditions(self):
        \"\"\"Check if entry conditions are met.\"\"\"
        conditions = []
        
        # Add specific entry conditions based on strategy
        {self._generate_entry_conditions(strategy)}
        
        return all(conditions) if conditions else False
    
    def _check_exit_conditions(self):
        \"\"\"Check if exit conditions are met.\"\"\"
        if self.entry_price is None:
            return False
            
        current_price = self.data.close[0]
        pnl_pct = (current_price - self.entry_price) / self.entry_price
        
        # Stop loss
        if pnl_pct <= -self.params.stop_loss:
            self.log(f'STOP LOSS TRIGGERED: {{pnl_pct:.2%}}')
            return True
            
        # Take profit
        if pnl_pct >= self.params.take_profit:
            self.log(f'TAKE PROFIT TRIGGERED: {{pnl_pct:.2%}}')
            return True
        
        # Add specific exit conditions
        {self._generate_exit_conditions(strategy)}
        
        return False
    
    def _calculate_position_size(self):
        \"\"\"Calculate position size based on risk management.\"\"\"
        cash = self.broker.get_cash()
        price = self.data.close[0]
        max_position_value = cash * self.params.max_position_pct
        size = int(max_position_value / price)
        return size
    
    def log(self, txt, dt=None):
        \"\"\"Logging function.\"\"\"
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{{dt.isoformat()}} {{txt}}')
"""
        
        return code
    
    def _generate_entry_conditions(self, strategy: TradingStrategy) -> str:
        """Generate entry condition checks."""
        conditions = []
        
        for condition in strategy.entry_conditions:
            if "RSI" in condition and "oversold" in condition.lower():
                conditions.append("conditions.append(self.rsi[0] < 30)")
            elif "SMA" in condition and "above" in condition.lower():
                conditions.append("conditions.append(self.data.close[0] > self.sma20[0])")
            elif "trend" in condition.lower():
                conditions.append("conditions.append(self.sma20[0] > self.sma50[0])")
        
        return "\n        ".join(conditions) if conditions else "conditions.append(True)"
    
    def _generate_exit_conditions(self, strategy: TradingStrategy) -> str:
        """Generate exit condition checks."""
        conditions = []
        
        for condition in strategy.exit_conditions:
            if "RSI" in condition and "overbought" in condition.lower():
                conditions.append("if self.rsi[0] > 70:\n            return True")
            elif "SMA" in condition and "below" in condition.lower():
                conditions.append("if self.data.close[0] < self.sma20[0]:\n            return True")
        
        return "\n        ".join(conditions) if conditions else ""
    
    def _get_fallback_strategy(self, ticker: str) -> TradingStrategy:
        """Return a conservative fallback strategy."""
        return TradingStrategy(
            strategy_id=f"{ticker}_fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name="Conservative Buy and Hold",
            description="Simple buy and hold strategy with basic risk management",
            entry_conditions=["Price above 50-day SMA", "RSI below 70"],
            exit_conditions=["Stop loss at 10%", "Take profit at 20%"],
            position_sizing="fixed",
            risk_management={
                "stop_loss": 0.10,
                "take_profit": 0.20,
                "max_position_size": 0.25,
                "max_drawdown": 0.25
            },
            initial_capital=100000.0,
            start_date=(datetime.now().replace(year=datetime.now().year-1)).strftime('%Y-%m-%d'),
            end_date=datetime.now().strftime('%Y-%m-%d'),
            backtrader_code="# Fallback strategy code"
        )


def generate_trading_strategy(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node for generating trading strategies.
    
    Args:
        state: Graph state with financial factors and market data
        
    Returns:
        Updated state with trading strategy
    """
    # Extract analysis results
    factors = state.get("factors", [])
    market_context = state.get("market_context")
    technical_indicators = state.get("technical_indicators")
    
    if not factors or not market_context:
        return {
            **state,
            "error": "Missing required analysis data for strategy generation"
        }
    
    # Get ticker
    ticker = None
    if market_context:
        ticker = market_context.ticker
    
    if not ticker:
        return {
            **state,
            "error": "No ticker found for strategy generation"
        }
    
    # Initialize generator
    generator = StrategyGenerator(
        model=state.get("strategy_model", "gpt-4o-mini"),
        temperature=state.get("strategy_temperature", 0.3)
    )
    
    strategies = []
    
    # Generate strategies for each financial factor analysis
    for factor in factors[:1]:  # Start with one strategy
        try:
            # Convert Pydantic models to dicts
            factor_dict = {
                "overall_sentiment": {
                    "score": factor.overall_sentiment.score,
                    "confidence": factor.overall_sentiment.confidence,
                    "reasoning": factor.overall_sentiment.reasoning
                },
                "revenue_growth": factor.revenue_growth,
                "guidance_raised": factor.guidance_raised,
                "risk_factors": factor.risk_factors,
                "competitive_advantages": factor.competitive_advantages,
                "growth_drivers": factor.growth_drivers
            }
            
            context_dict = {
                "price_vs_52w_high": market_context.price_vs_52w_high,
                "price_vs_52w_low": market_context.price_vs_52w_low,
                "vs_spy_1m": market_context.vs_spy_1m,
                "trend_strength": market_context.trend_strength,
                "volatility_regime": market_context.volatility_regime
            }
            
            indicators_dict = {
                "rsi_14": technical_indicators.rsi_14,
                "sma_20": technical_indicators.sma_20,
                "sma_50": technical_indicators.sma_50,
                "current_price": technical_indicators.sma_20  # Approximate
            }
            
            # Generate strategy
            strategy = generator.generate_strategy(
                ticker=ticker,
                financial_factors=factor_dict,
                market_context=context_dict,
                technical_indicators=indicators_dict
            )
            
            # Generate executable code
            strategy.backtrader_code = generator.generate_backtrader_code(strategy)
            
            strategies.append(strategy)
            
            print(f"✅ Generated strategy: {strategy.name}")
            print(f"   Entry: {', '.join(strategy.entry_conditions[:2])}")
            print(f"   Exit: {', '.join(strategy.exit_conditions[:2])}")
            print(f"   Risk: SL={strategy.risk_management['stop_loss']*100}%, TP={strategy.risk_management['take_profit']*100}%")
            
        except Exception as e:
            print(f"❌ Error generating strategy: {e}")
    
    return {
        **state,
        "strategies": strategies,
        "current_step": "strategy_generation"
    }