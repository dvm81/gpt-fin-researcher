#!/usr/bin/env python
"""
Simple filing inspector
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph import app


def inspect_filing(ticker="TSLA", filing_type="10-K"):
    """Inspect what we fetch from SEC."""
    
    print(f"🔍 Inspecting {filing_type} for {ticker}")
    print("=" * 40)
    
    result = app.invoke({
        "tasks": [f"Investigate {ticker} {filing_type}"]
    })
    
    if result.get("docs"):
        doc = result["docs"][0]
        text = doc.get("text", "")
        
        print(f"✅ Success!")
        print(f"📊 Ticker: {doc.get('ticker')}")
        print(f"📅 Date: {doc.get('filing_date')}")
        print(f"📄 Length: {len(text):,} chars")
        print(f"📈 Words: {len(text.split()):,}")
        print(f"🔍 Chunks: {result.get('chunk_count', 0)}")
        
        # Show sections found
        sections = ["=== BUSINESS ===", "=== RISK_FACTORS ===", "=== MDA ===", "=== FINANCIALS ==="]
        found = [s for s in sections if s in text]
        if found:
            print(f"📋 Sections: {', '.join([s.strip('=').strip() for s in found])}")
        
        print(f"\n📄 Preview (first 1000 chars):")
        print("-" * 40)
        print(text[:1000])
        if len(text) > 1000:
            print("...")
        print("-" * 40)
        
    else:
        print("❌ No data retrieved")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="TSLA")
    parser.add_argument("--type", default="10-K", choices=["10-K", "10-Q"])
    
    args = parser.parse_args()
    inspect_filing(args.ticker, args.type)