#!/usr/bin/env python
"""
GPT-Fin-Researcher - Main Entry Point
=====================================

Simple, clean interface to run the financial research pipeline.
"""

import argparse
import json
from pprint import pprint

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.graph import app


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="GPT Financial Researcher")
    parser.add_argument("--ticker", default="TSLA", help="Stock ticker (default: TSLA)")
    parser.add_argument("--filing", default="10-K", choices=["10-K", "10-Q"], 
                       help="Filing type (default: 10-K)")
    parser.add_argument("--output", help="Save results to JSON file")
    parser.add_argument("--inspect", action="store_true", help="Show detailed inspection")
    
    args = parser.parse_args()
    
    print(f"🔍 Analyzing {args.filing} for {args.ticker}")
    print("=" * 50)
    
    # Run the analysis
    result = app.invoke({
        "tasks": [f"Investigate {args.ticker} {args.filing}"]
    })
    
    if args.inspect:
        # Show detailed results
        if "docs" in result and result["docs"]:
            doc = result["docs"][0]
            print(f"\n📊 Filing: {doc.get('ticker')} {doc.get('filing_type')}")
            print(f"📅 Date: {doc.get('filing_date')}")
            print(f"📄 Text length: {len(doc.get('text', '')):,} chars")
            print(f"📈 Word count: {len(doc.get('text', '').split()):,} words")
            
            if "chunks" in result:
                print(f"🔍 Chunks created: {result.get('chunk_count', 0)}")
        
        # Show LLM analysis results
        if "factors" in result and result["factors"]:
            print(f"\n💡 Financial Analysis:")
            print("-" * 40)
            for i, factor in enumerate(result["factors"]):
                print(f"\nAnalysis #{i+1}:")
                print(f"  Sentiment: {factor.overall_sentiment.score:.2f}")
                print(f"  Reasoning: {factor.overall_sentiment.reasoning}")
                if factor.revenue_growth is not None:
                    print(f"  Revenue Growth: {factor.revenue_growth}%")
                print(f"  Guidance Raised: {factor.guidance_raised}")
                print(f"  Margin Expansion: {factor.margin_expansion}")
                print(f"  Debt Concerns: {factor.debt_concerns}")
                
                if factor.competitive_advantages:
                    print(f"  Advantages: {', '.join(factor.competitive_advantages[:2])}")
                if factor.risk_factors:
                    print(f"  Top Risk: {factor.risk_factors[0]}")
                if factor.growth_drivers:
                    print(f"  Growth Driver: {factor.growth_drivers[0]}")
        
        # Show generated strategies
        if "strategies" in result and result["strategies"]:
            print(f"\n🎯 Trading Strategies:")
            print("-" * 40)
            for i, strategy in enumerate(result["strategies"]):
                print(f"\nStrategy #{i+1}: {strategy.name}")
                print(f"  Description: {strategy.description}")
                print(f"  Entry: {', '.join(strategy.entry_conditions[:2])}")
                print(f"  Exit: {', '.join(strategy.exit_conditions[:2])}")
                print(f"  Risk Management:")
                print(f"    Stop Loss: {strategy.risk_management['stop_loss']*100}%")
                print(f"    Take Profit: {strategy.risk_management['take_profit']*100}%")
                print(f"    Max Position: {strategy.risk_management['max_position_size']*100}%")
        
        # Show backtest results
        if "backtest_results" in result and result["backtest_results"]:
            print(f"\n💰 Backtest Results:")
            print("-" * 40)
            for i, backtest in enumerate(result["backtest_results"]):
                print(f"\nBacktest #{i+1}:")
                print(f"  Total Return: {backtest.total_return}%")
                print(f"  Annualized Return: {backtest.annualized_return}%")
                print(f"  Sharpe Ratio: {backtest.sharpe_ratio}")
                print(f"  Max Drawdown: {backtest.max_drawdown}%")
                print(f"  Win Rate: {backtest.win_rate}%")
                print(f"  Total Trades: {backtest.total_trades}")
                
                if backtest.total_return > 0:
                    print("  ✅ Strategy is profitable!")
                else:
                    print("  ❌ Strategy is not profitable")
        
        print(f"\n📄 Filing Preview:")
        print("-" * 40)
        if result.get("docs"):
            preview = result["docs"][0].get("text", "")[:1000]
            print(preview)
            if len(preview) >= 1000:
                print("...")
        print("-" * 40)
    else:
        # Simple summary
        if result.get("docs"):
            doc = result["docs"][0]
            print(f"✅ Successfully fetched {doc.get('filing_type')} for {doc.get('ticker')}")
            print(f"📄 Content: {len(doc.get('text', '')):,} characters")
            print(f"🔍 Chunks: {result.get('chunk_count', 0)}")
            
            # Show brief LLM analysis
            if "factors" in result and result["factors"]:
                factor = result["factors"][0]  # Show first analysis
                print(f"\n💡 Key Insights:")
                print(f"  Sentiment: {factor.overall_sentiment.score:.2f} - {factor.overall_sentiment.reasoning[:100]}...")
                if factor.revenue_growth is not None:
                    print(f"  Revenue Growth: {factor.revenue_growth}%")
                if factor.competitive_advantages:
                    print(f"  Key Advantage: {factor.competitive_advantages[0]}")
                if factor.risk_factors:
                    print(f"  Top Risk: {factor.risk_factors[0]}")
            
            # Show strategy summary
            if "strategies" in result and result["strategies"]:
                strategy = result["strategies"][0]
                print(f"\n🎯 Trading Strategy: {strategy.name}")
                print(f"  Entry: {strategy.entry_conditions[0] if strategy.entry_conditions else 'N/A'}")
                print(f"  Stop Loss: {strategy.risk_management['stop_loss']*100}%")
                
                # Show backtest summary
                if "backtest_results" in result and result["backtest_results"]:
                    backtest = result["backtest_results"][0]
                    print(f"\n💰 Backtest Performance:")
                    print(f"  Return: {backtest.total_return}% | Sharpe: {backtest.sharpe_ratio}")
                    print(f"  Win Rate: {backtest.win_rate}% | Trades: {backtest.total_trades}")
        else:
            print("❌ No filing data retrieved")
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            # Prepare clean output
            output_data = {
                "metadata": {
                    "ticker": args.ticker,
                    "filing_type": args.filing,
                    "timestamp": result.get("docs", [{}])[0].get("filing_date"),
                    "chunks_count": result.get("chunk_count", 0)
                },
                "content": result.get("docs", [{}])[0].get("text", "")[:5000] if result.get("docs") else "",
                "full_result": result
            }
            json.dump(output_data, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")


if __name__ == "__main__":
    main()