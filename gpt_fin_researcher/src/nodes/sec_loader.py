"""SEC filing loader node for GPT-Fin-Researcher.

This module fetches 10-K and 10-Q filings from the SEC EDGAR database
using the sec-edgar-downloader library.
"""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import shutil

from sec_edgar_downloader import Downloader


def extract_text_from_filing(file_path: str) -> str:
    """Extract readable text from SEC filing HTML/XML.
    
    Handles full SEC submission files that may contain multiple documents.
    Focuses on extracting the main 10-K/10-Q document content.
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # If this is a full submission file, extract just the main document
    if '<TYPE>10-K' in content or '<TYPE>10-Q' in content or '<TYPE>20-F' in content:
        # Find the main document
        doc_start = content.find('<TYPE>10-K') or content.find('<TYPE>10-Q') or content.find('<TYPE>20-F')
        if doc_start != -1:
            # Find the end of this document (next <TYPE> or end of file)
            doc_end = content.find('<TYPE>', doc_start + 10)
            if doc_end == -1:
                content = content[doc_start:]
            else:
                content = content[doc_start:doc_end]
    
    # Remove script, style elements and XBRL metadata
    content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r'<ix:header>.*?</ix:header>', '', content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r'<div style="display:none">.*?</div>', '', content, flags=re.DOTALL | re.IGNORECASE)
    
    # Extract key sections with improved patterns
    sections = {
        'business': r'ITEM\s+1\.?\s+BUSINESS[^<]*</span>(.*?)(?=ITEM\s+1A\.?\s+RISK\s+FACTORS|ITEM\s+2\.)',
        'risk_factors': r'ITEM\s+1A\.?\s+RISK\s+FACTORS[^<]*</span>(.*?)(?=ITEM\s+1B\.|ITEM\s+2\.)',
        'mda': r'ITEM\s+7\.?\s+MANAGEMENT[^<]*</span>(.*?)(?=ITEM\s+7A\.|ITEM\s+8\.)',
        'financials': r'ITEM\s+8\.?\s+FINANCIAL\s+STATEMENTS[^<]*</span>(.*?)(?=ITEM\s+9\.|\Z)'
    }
    
    extracted_text = []
    
    for section_name, pattern in sections.items():
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            section_text = match.group(1)
            
            # Clean HTML tags but preserve structure
            section_text = re.sub(r'<br[^>]*>', '\n', section_text)
            section_text = re.sub(r'</div>', '\n', section_text)
            section_text = re.sub(r'</p>', '\n\n', section_text)
            section_text = re.sub(r'<[^>]+>', ' ', section_text)
            
            # Decode HTML entities
            import html
            section_text = html.unescape(section_text)
            
            # Clean whitespace
            section_text = re.sub(r'\s+', ' ', section_text)
            section_text = re.sub(r'\n\s*\n', '\n\n', section_text)
            
            # Limit section size but keep more content
            section_text = section_text.strip()[:25000]
            extracted_text.append(f"\n\n=== {section_name.upper()} ===\n{section_text}")
    
    # If no sections found, do a general extraction
    if not extracted_text:
        # Remove all HTML tags
        text = re.sub(r'<[^>]+>', ' ', content)
        
        # Decode HTML entities
        import html
        text = html.unescape(text)
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()[:50000]  # Larger fallback
    
    return '\n'.join(extracted_text)


def sec_loader(state: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch SEC filings based on tasks in state.
    
    Args:
        state: Graph state containing tasks list
        
    Returns:
        Updated state with fetched documents
    """
    tasks = state.get("tasks", [])
    docs = []
    
    # Check for SEC User-Agent
    user_agent = os.environ.get("SEC_API_USER_AGENT")
    if not user_agent:
        print("Warning: SEC_API_USER_AGENT not set. Using default.")
        print("Set it with: export SEC_API_USER_AGENT='Your Name your-email@example.com'")
        user_agent = "Demo User demo@example.com"
    
    # Create download directory
    download_dir = Path("./sec-filings")
    download_dir.mkdir(exist_ok=True)
    
    for task in tasks:
        # Parse task to extract ticker and filing type
        parts = task.split()
        
        ticker = None
        filing_type = "10-K"  # Default to 10-K
        
        for part in parts:
            if part.isupper() and len(part) <= 5 and part not in ["10-K", "10-Q", "8-K", "20-F"]:
                ticker = part
            elif part in ["10-K", "10-Q", "8-K", "20-F"]:
                filing_type = part
        
        if not ticker:
            print(f"No ticker found in task: {task}")
            continue
        
        # Auto-detect filing type for known foreign companies
        foreign_companies = {
            'SPOT': '20-F',  # Spotify (Luxembourg)
            'ASML': '20-F',  # ASML (Netherlands)
            'TSM': '20-F',   # Taiwan Semiconductor
            'NVO': '20-F',   # Novo Nordisk
            'SAP': '20-F',   # SAP (Germany)
        }
        
        if filing_type == "10-K" and ticker in foreign_companies:
            filing_type = foreign_companies[ticker]
            print(f"Auto-detected {filing_type} for foreign company {ticker}")
        
        print(f"Fetching {filing_type} for {ticker}...")
        
        # Initialize downloader for each company
        dl = Downloader(
            company_name=ticker,  # Use ticker as company name
            email_address=user_agent.split()[-1] if '@' in user_agent else "demo@example.com",
            download_folder=str(download_dir)
        )
        
        try:
            # Download latest filing
            result = dl.get(
                filing_type,
                ticker,
                limit=1,  # Get only the latest filing
                download_details=True
            )
            
            # If no filings found and we tried 10-K, try 20-F (foreign companies)
            if result == 0 and filing_type == "10-K":
                print(f"No {filing_type} found for {ticker}, trying 20-F...")
                result = dl.get(
                    "20-F",
                    ticker,
                    limit=1,
                    download_details=True
                )
                if result > 0:
                    filing_type = "20-F"
                    print(f"Found 20-F for {ticker}")
            
            # If still no filings found and we tried 20-F, try 10-K (domestic companies)
            elif result == 0 and filing_type == "20-F":
                print(f"No {filing_type} found for {ticker}, trying 10-K...")
                result = dl.get(
                    "10-K",
                    ticker,
                    limit=1,
                    download_details=True
                )
                if result > 0:
                    filing_type = "10-K"
                    print(f"Found 10-K for {ticker}")
            
            if result == 0:
                print(f"No filings found for {ticker}")
                continue
            
            # Find the downloaded file
            company_dir = download_dir / f"sec-edgar-filings/{ticker}/{filing_type}"
            if company_dir.exists():
                # Get the most recent filing
                filing_dirs = sorted(company_dir.iterdir(), reverse=True)
                if filing_dirs:
                    filing_dir = filing_dirs[0]
                    
                    # Look for the full submission file or primary document
                    filing_file = None
                    for filename in ["full-submission.txt", "primary-document.html", 
                                   f"{ticker.lower()}-{filing_type.lower()}.htm"]:
                        candidate = filing_dir / filename
                        if candidate.exists():
                            filing_file = candidate
                            break
                    
                    # If no standard file found, get the first .txt or .htm file
                    if not filing_file:
                        for ext in ['.txt', '.htm', '.html']:
                            files = list(filing_dir.glob(f"*{ext}"))
                            if files:
                                filing_file = files[0]
                                break
                    
                    if filing_file:
                        text = extract_text_from_filing(str(filing_file))
                        
                        # Extract filing date from directory name
                        filing_date = filing_dir.name
                        
                        # Validate date format
                        try:
                            datetime.strptime(filing_date, "%Y-%m-%d")
                        except ValueError:
                            filing_date = datetime.now().strftime("%Y-%m-%d")
                        
                        docs.append({
                            "ticker": ticker,
                            "filing_type": filing_type,
                            "filing_date": filing_date,
                            "text": text[:100000],  # Increased limit for better context
                            "source": f"SEC EDGAR {filing_type}",
                            "metadata": {
                                "company": ticker,
                                "filing_type": filing_type,
                                "date": filing_date,
                                "task": task,
                                "file_path": str(filing_file)
                            }
                        })
                        
                        print(f"Successfully loaded {filing_type} for {ticker} dated {filing_date}")
                    else:
                        print(f"No readable filing file found for {ticker}")
                        
        except Exception as e:
            print(f"Error fetching {filing_type} for {ticker}: {e}")
            # Fall back to mock data if real fetch fails
            docs.append({
                "ticker": ticker,
                "filing_type": filing_type,
                "filing_date": datetime.now().strftime("%Y-%m-%d"),
                "text": f"Error fetching real filing: {str(e)}. This is placeholder text.",
                "source": f"SEC EDGAR {filing_type} (Error)",
                "metadata": {
                    "company": ticker,
                    "filing_type": filing_type,
                    "error": str(e),
                    "task": task
                }
            })
    
    # Clean up download directory to save space (optional)
    # shutil.rmtree(download_dir, ignore_errors=True)
    
    return {
        **state,
        "docs": state.get("docs", []) + docs
    }