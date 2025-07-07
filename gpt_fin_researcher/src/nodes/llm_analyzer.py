"""LLM-based financial factor extraction node for GPT-Fin-Researcher.

This module uses LLMs to extract structured financial insights from SEC filings.
"""

import json
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

# Load environment variables from .env file
load_dotenv()

from ..schemas import FinancialFactors, SentimentScore
from .vector_store import search_chromadb


class FinancialAnalyzer:
    """LLM-powered financial analysis using configurable models."""
    
    def __init__(self, 
                 model: str = "gpt-4o-mini",
                 temperature: float = 0.1,
                 max_tokens: int = 2000):
        """Initialize the financial analyzer.
        
        Args:
            model: OpenAI model to use (gpt-4o, gpt-4o-mini, gpt-3.5-turbo)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
        """
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def extract_financial_factors(self, ticker: str, filing_text: str) -> FinancialFactors:
        """Extract structured financial factors from filing text.
        
        Args:
            ticker: Company ticker symbol
            filing_text: Raw SEC filing text
            
        Returns:
            Structured financial factors
        """
        prompt = self._build_extraction_prompt(ticker, filing_text)
        
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
            
            # Parse JSON response
            content = response.choices[0].message.content
            factors_data = json.loads(content)
            
            # Fix any null boolean values
            for field in ['guidance_raised', 'margin_expansion', 'debt_concerns']:
                if field in factors_data and factors_data[field] is None:
                    factors_data[field] = False
            
            # Validate and create FinancialFactors object
            return FinancialFactors(**factors_data)
            
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"Error parsing LLM response: {e}")
            return self._get_fallback_factors()
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return self._get_fallback_factors()
    
    def analyze_with_context(self, ticker: str, query: str, max_chunks: int = 5) -> FinancialFactors:
        """Extract factors using relevant chunks from vector store.
        
        Args:
            ticker: Company ticker symbol
            query: Analysis focus query
            max_chunks: Maximum chunks to retrieve for context
            
        Returns:
            Structured financial factors
        """
        # Get relevant context from vector store
        relevant_chunks = search_chromadb(
            query=query,
            n_results=max_chunks,
            filters={"ticker": ticker}
        )
        
        if not relevant_chunks:
            print(f"No relevant chunks found for {ticker}")
            return self._get_fallback_factors()
        
        # Combine chunk texts
        context_text = "\n\n".join([chunk["text"] for chunk in relevant_chunks])
        
        return self.extract_financial_factors(ticker, context_text)
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for financial analysis."""
        return """You are a financial analyst extracting structured insights from SEC filings.

Extract the following information and return it as a JSON object:

1. overall_sentiment: Analyze management tone and outlook
   - score: float between -1.0 (very negative) and 1.0 (very positive) 
   - confidence: float between 0.0 and 1.0
   - reasoning: string explaining the sentiment assessment

2. revenue_growth: Year-over-year revenue growth rate as a percentage (e.g., 15.5 for 15.5%). Look for phrases like "revenue increased/decreased by X%", "X% growth", "revenue of $X compared to $Y". If no specific percentage is mentioned, use null.

3. guidance_raised: Boolean (true/false) indicating if management raised guidance

4. margin_expansion: Boolean (true/false) indicating if profit margins are expanding

5. debt_concerns: Boolean (true/false) indicating if there are significant debt issues

6. competitive_advantages: List of strings describing key competitive moats

7. risk_factors: List of strings describing major business risks

8. growth_drivers: List of strings describing key growth catalysts

Return ONLY valid JSON. Use null for missing numeric values. Always use true/false for boolean fields, never null."""

    def _build_extraction_prompt(self, ticker: str, filing_text: str) -> str:
        """Build the extraction prompt with filing context."""
        # Truncate text if too long to fit in context window
        max_text_length = 8000  # Leave room for prompt and response
        if len(filing_text) > max_text_length:
            filing_text = filing_text[:max_text_length] + "..."
        
        return f"""Analyze the following SEC filing excerpt for {ticker} and extract financial factors:

FILING TEXT:
{filing_text}

Extract structured financial insights as specified in the system prompt."""

    def _get_fallback_factors(self) -> FinancialFactors:
        """Return fallback factors when extraction fails."""
        return FinancialFactors(
            overall_sentiment=SentimentScore(
                score=0.0,
                confidence=0.0,
                reasoning="Analysis failed - no sentiment data available"
            ),
            revenue_growth=None,
            guidance_raised=False,
            margin_expansion=False,
            debt_concerns=False,
            competitive_advantages=[],
            risk_factors=["Analysis failed"],
            growth_drivers=[]
        )


def analyze_financial_factors(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node for extracting financial factors from stored documents.
    
    Now enhanced with market data integration for more comprehensive analysis.
    
    Args:
        state: Graph state containing docs, chunks, and market data
        
    Returns:
        Updated state with extracted financial factors
    """
    # Get configuration from state or use defaults
    model = state.get("llm_model", "gpt-4o-mini")
    temperature = state.get("llm_temperature", 0.1)
    max_tokens = state.get("llm_max_tokens", 2000)
    
    analyzer = FinancialAnalyzer(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    # Get ticker from tasks or docs
    ticker = None
    docs = state.get("docs", [])
    
    if docs:
        ticker = docs[0].get("ticker", "UNKNOWN")
    else:
        # Try to extract from tasks
        tasks = state.get("tasks", [])
        for task in tasks:
            if "TSLA" in task:
                ticker = "TSLA"
                break
            elif "AAPL" in task:
                ticker = "AAPL"
                break
    
    if not ticker:
        return {
            **state,
            "error": "No ticker found for analysis",
            "factors": []
        }
    
    factors = []
    
    # Check if we have market data to enhance analysis
    market_context = state.get("market_context")
    technical_indicators = state.get("technical_indicators")
    
    # Build enhanced queries with market context
    analysis_queries = [
        "revenue growth and financial performance",
        "management outlook and guidance",
        "risk factors and competitive position"
    ]
    
    # Add market context info if available
    if market_context:
        print(f"📊 Market Context: Price vs 52W high: {market_context.price_vs_52w_high}%")
        if technical_indicators:
            print(f"   RSI: {technical_indicators.rsi_14}")
            print(f"   vs SPY (1M): {market_context.vs_spy_1m}%")
    
    for query in analysis_queries:
        try:
            factor = analyzer.analyze_with_context(ticker, query, max_chunks=3)
            factors.append(factor)
            print(f"✅ Extracted factors for query: {query}")
        except Exception as e:
            print(f"❌ Failed to analyze '{query}': {e}")
    
    # If no factors extracted, try with raw docs
    if not factors and docs:
        for doc in docs[:1]:  # Limit to first doc to avoid token limits
            try:
                factor = analyzer.extract_financial_factors(
                    ticker=ticker,
                    filing_text=doc.get("text", "")
                )
                factors.append(factor)
                print(f"✅ Extracted factors from raw document")
                break
            except Exception as e:
                print(f"❌ Failed to analyze document: {e}")
    
    return {
        **state,
        "factors": factors,
        "analysis_model": model,
        "current_step": "financial_analysis"
    }