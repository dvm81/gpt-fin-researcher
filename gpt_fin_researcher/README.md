# GPT-Fin-Researcher

AI-powered financial analysis that turns SEC filings into trading insights.

## Quick Start

```bash
# Basic usage - analyze Tesla's 10-K
python main.py

# Analyze Apple's 10-Q with detailed output
python main.py --ticker AAPL --filing 10-Q --inspect

# Save results to file
python main.py --ticker MSFT --output results.json

# Quick inspection of any filing
python scripts/filing_inspector.py --ticker NVDA --type 10-K
```

## What It Does

1. **Fetches Real SEC Filings** - Downloads 10-K/10-Q from SEC EDGAR database
2. **Extracts Key Sections** - Business overview, risk factors, management discussion
3. **Creates Embeddings** - Chunks documents for semantic search
4. **Ready for AI Analysis** - Structured data pipeline for trading strategies

## Directory Structure

```
├── main.py              # Simple entry point
├── src/
│   ├── graph.py         # LangGraph pipeline
│   ├── schemas.py       # Data models
│   └── nodes/
│       ├── sec_loader.py # SEC filing fetcher
│       └── embedder.py   # Document chunker
├── scripts/
│   └── filing_inspector.py # Inspection tools
└── data/               # Downloaded filings
```

## Environment Setup

```bash
# Required for SEC compliance
export SEC_API_USER_AGENT="Your Name your-email@example.com"
```

## Example Output

```
🔍 Analyzing 10-K for TSLA
✅ Successfully fetched 10-K for TSLA
📄 Content: 100,000 characters
🔍 Chunks: 25
📋 Sections: BUSINESS, RISK_FACTORS, MDA, FINANCIALS
```

The system extracts real SEC filing content and prepares it for AI-powered financial analysis!