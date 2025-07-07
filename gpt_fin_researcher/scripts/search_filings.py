#!/usr/bin/env python
"""
Search stored SEC filings in ChromaDB
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nodes import search_chromadb, get_collection_info


def search_filings(query: str, n_results: int = 5, ticker: str = None):
    """Search for relevant filing chunks."""
    
    print(f"🔍 Searching for: '{query}'")
    if ticker:
        print(f"   Filtering by ticker: {ticker}")
    print("=" * 50)
    
    # Build filters if ticker specified
    filters = {"ticker": ticker} if ticker else None
    
    # Search
    results = search_chromadb(query, n_results=n_results, filters=filters)
    
    if not results:
        print("❌ No results found")
        return
    
    print(f"✅ Found {len(results)} relevant chunks\n")
    
    for i, chunk in enumerate(results):
        print(f"#{i+1} [{chunk['metadata'].get('ticker')} {chunk['metadata'].get('filing_type')}]")
        print(f"📅 Date: {chunk['metadata'].get('filing_date')}")
        if chunk.get('distance') is not None:
            print(f"📊 Relevance: {1 - chunk['distance']:.2%}")
        print(f"📄 Text preview:")
        print("-" * 40)
        preview = chunk['text'][:300]
        print(preview)
        if len(chunk['text']) > 300:
            print("...")
        print("-" * 40)
        print()


def show_collection_info():
    """Show information about stored filings."""
    info = get_collection_info()
    
    if "error" in info:
        print(f"❌ {info['error']}")
        return
    
    print("📊 ChromaDB Collection Info")
    print("=" * 40)
    print(f"📁 Collection: {info['collection_name']}")
    print(f"📄 Total chunks: {info['total_chunks']}")
    print(f"🏢 Companies: {', '.join(info['tickers'])}")
    print(f"📋 Filing types: {', '.join(info['filing_types'])}")
    print(f"💾 Storage: {info['persist_dir']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Search SEC filings")
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("--ticker", help="Filter by ticker")
    parser.add_argument("--limit", type=int, default=5, help="Number of results")
    parser.add_argument("--info", action="store_true", help="Show collection info")
    
    args = parser.parse_args()
    
    if args.info:
        show_collection_info()
    elif args.query:
        search_filings(args.query, n_results=args.limit, ticker=args.ticker)
    else:
        parser.print_help()