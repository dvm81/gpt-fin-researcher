#!/usr/bin/env python
"""
Test the LLM financial analysis functionality
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from src.nodes.llm_analyzer import FinancialAnalyzer
from src.nodes.vector_store import get_collection_info


def test_llm_analysis():
    """Test LLM analysis with existing vector store data."""
    
    # Check if we have data in vector store
    info = get_collection_info()
    
    if "error" in info:
        print(f"❌ {info['error']}")
        print("Run main.py first to populate the vector store")
        return
    
    print("📊 Vector Store Info:")
    print(f"   Total chunks: {info['total_chunks']}")
    print(f"   Companies: {', '.join(info['tickers'])}")
    print()
    
    # Initialize analyzer with gpt-4o-mini (cheaper for testing)
    analyzer = FinancialAnalyzer(
        model="gpt-4o-mini",
        temperature=0.1
    )
    
    # Test analysis for each ticker
    for ticker in info['tickers']:
        if ticker == "Unknown":
            continue
            
        print(f"🔍 Analyzing {ticker}...")
        print("=" * 40)
        
        try:
            # Analyze with different queries
            queries = [
                "revenue financial performance earnings quarterly results year over year growth",
                "management guidance outlook forecast future expectations", 
                "risk factors competitive advantages market position"
            ]
            
            for query in queries:
                print(f"\n📝 Query: {query}")
                factors = analyzer.analyze_with_context(ticker, query, max_chunks=2)
                
                print(f"   Sentiment: {factors.overall_sentiment.score:.2f} ({factors.overall_sentiment.reasoning})")
                if factors.revenue_growth is not None:
                    print(f"   Revenue Growth: {factors.revenue_growth}%")
                else:
                    print(f"   Revenue Growth: Not found in this excerpt")
                print(f"   Guidance Raised: {factors.guidance_raised}")
                print(f"   Margin Expansion: {factors.margin_expansion}")
                print(f"   Debt Concerns: {factors.debt_concerns}")
                
                if factors.competitive_advantages:
                    print(f"   Advantages: {', '.join(factors.competitive_advantages[:2])}")
                
                if factors.risk_factors:
                    print(f"   Top Risk: {factors.risk_factors[0]}")
                
                print()
                
        except Exception as e:
            print(f"❌ Error analyzing {ticker}: {e}")
            print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test LLM financial analysis")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")
    
    args = parser.parse_args()
    
    print(f"🤖 Testing LLM analysis with {args.model}")
    print("Make sure OPENAI_API_KEY is set in your environment")
    print()
    
    test_llm_analysis()