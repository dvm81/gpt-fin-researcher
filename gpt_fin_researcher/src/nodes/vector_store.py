"""Vector store node for GPT-Fin-Researcher using ChromaDB.

This module stores document chunks in ChromaDB for semantic search
and retrieval.
"""

import os
from typing import Any, Dict, List
from pathlib import Path

import chromadb
from chromadb.config import Settings


def store_in_chromadb(state: Dict[str, Any]) -> Dict[str, Any]:
    """Store document chunks in ChromaDB vector database.
    
    Args:
        state: Graph state containing chunks and embeddings
        
    Returns:
        Updated state with vector store information
    """
    chunks = state.get("chunks", [])
    
    if not chunks:
        print("No chunks to store in vector database")
        return state
    
    # Initialize ChromaDB client with persistent storage
    persist_dir = Path("./chroma_db")
    persist_dir.mkdir(exist_ok=True)
    
    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )
    
    # Get or create collection for SEC filings
    collection_name = "sec_filings"
    try:
        collection = client.get_collection(name=collection_name)
        print(f"Using existing collection: {collection_name}")
    except Exception:
        # Create new collection if it doesn't exist
        collection = client.create_collection(
            name=collection_name,
            metadata={"description": "SEC filings for financial analysis"}
        )
        print(f"Created new collection: {collection_name}")
    
    # Prepare data for ChromaDB
    ids = []
    documents = []
    metadatas = []
    
    for chunk in chunks:
        ids.append(chunk["id"])
        documents.append(chunk["text"])
        metadatas.append(chunk["metadata"])
    
    # Add to ChromaDB - let it create embeddings
    try:
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        print(f"✅ Stored {len(chunks)} chunks in ChromaDB")
        
        # Store collection stats
        state["vector_store"] = {
            "type": "chromadb",
            "collection": collection_name,
            "persist_dir": str(persist_dir),
            "chunk_count": len(chunks),
            "total_docs": collection.count()
        }
        
    except Exception as e:
        print(f"Error storing in ChromaDB: {e}")
        state["error"] = f"Vector store error: {str(e)}"
    
    return state


def search_chromadb(query: str, n_results: int = 5, filters: Dict = None) -> List[Dict]:
    """Search ChromaDB for relevant document chunks.
    
    Args:
        query: Search query text
        n_results: Number of results to return
        filters: Optional metadata filters
        
    Returns:
        List of relevant document chunks with metadata
    """
    # Initialize ChromaDB client
    persist_dir = Path("./chroma_db")
    
    if not persist_dir.exists():
        print("No ChromaDB database found")
        return []
    
    client = chromadb.PersistentClient(path=str(persist_dir))
    
    try:
        collection = client.get_collection(name="sec_filings")
        
        # Perform search
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=filters
        )
        
        # Format results
        chunks = []
        for i in range(len(results['ids'][0])):
            chunks.append({
                "id": results['ids'][0][i],
                "text": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i] if 'distances' in results else None
            })
        
        return chunks
        
    except Exception as e:
        print(f"Error searching ChromaDB: {e}")
        return []


def get_collection_info() -> Dict[str, Any]:
    """Get information about the ChromaDB collection."""
    persist_dir = Path("./chroma_db")
    
    if not persist_dir.exists():
        return {"error": "No ChromaDB database found"}
    
    client = chromadb.PersistentClient(path=str(persist_dir))
    
    try:
        collection = client.get_collection(name="sec_filings")
        
        # Get sample documents
        sample = collection.peek(limit=5)
        
        # Get unique tickers
        all_docs = collection.get()
        tickers = set()
        filing_types = set()
        
        for metadata in all_docs.get('metadatas', []):
            if metadata:
                tickers.add(metadata.get('ticker', 'Unknown'))
                filing_types.add(metadata.get('filing_type', 'Unknown'))
        
        return {
            "collection_name": "sec_filings",
            "total_chunks": collection.count(),
            "tickers": list(tickers),
            "filing_types": list(filing_types),
            "sample_chunks": len(sample.get('ids', [])),
            "persist_dir": str(persist_dir)
        }
        
    except Exception as e:
        return {"error": f"Error getting collection info: {str(e)}"}