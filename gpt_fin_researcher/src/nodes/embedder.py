"""Document chunking and embedding node for GPT-Fin-Researcher.

This module chunks SEC filings into smaller segments and creates
embeddings for vector search and retrieval.
"""

import hashlib
from typing import Any, Dict, List

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.embeddings import MockEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding


def chunk_and_embed(state: Dict[str, Any]) -> Dict[str, Any]:
    """Chunk documents and create embeddings for vector search.
    
    Args:
        state: Graph state containing docs
        
    Returns:
        Updated state with document chunks and embeddings
    """
    docs = state.get("docs", [])
    
    # Initialize node parser with reasonable defaults for financial documents
    parser = SentenceSplitter(
        chunk_size=1024,  # Larger chunks for financial context
        chunk_overlap=200,  # Overlap to maintain context
        separator=" ",
        paragraph_separator="\n\n",
    )
    
    # Use mock embeddings for now (replace with OpenAI or other provider)
    embed_model = MockEmbedding(embed_dim=1536)  # Same dimension as OpenAI
    
    all_chunks = []
    all_embeddings = []
    
    for doc in docs:
        # Create LlamaIndex document
        document = Document(
            text=doc.get("text", ""),
            metadata={
                "ticker": doc.get("ticker"),
                "filing_type": doc.get("filing_type"),
                "filing_date": doc.get("filing_date"),
                "source": doc.get("source"),
            }
        )
        
        # Parse into chunks
        chunks = parser.get_nodes_from_documents([document])
        
        # Process each chunk
        for chunk in chunks:
            # Create unique ID for chunk
            chunk_id = hashlib.md5(
                f"{doc.get('ticker')}_{doc.get('filing_date')}_{chunk.text[:100]}".encode()
            ).hexdigest()
            
            # Get embedding
            embedding = embed_model.get_text_embedding(chunk.text)
            
            # Store chunk data
            chunk_data = {
                "id": chunk_id,
                "text": chunk.text,
                "embedding": embedding,
                "metadata": {
                    **chunk.metadata,
                    "chunk_index": chunks.index(chunk),
                    "total_chunks": len(chunks),
                }
            }
            
            all_chunks.append(chunk_data)
            all_embeddings.append(embedding)
    
    # Update state
    return {
        **state,
        "chunks": all_chunks,
        "embeddings": all_embeddings,
        "embedding_model": "mock",  # Track which model was used
        "chunk_count": len(all_chunks),
    }


def create_embeddings_with_openai(state: Dict[str, Any]) -> Dict[str, Any]:
    """Alternative implementation using OpenAI embeddings.
    
    Requires OPENAI_API_KEY environment variable.
    """
    docs = state.get("docs", [])
    
    # Initialize with OpenAI embeddings
    try:
        embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",
            dimensions=1536,
        )
        model_name = "openai-text-embedding-3-small"
    except Exception as e:
        print(f"Failed to initialize OpenAI embeddings: {e}")
        print("Falling back to mock embeddings")
        embed_model = MockEmbedding(embed_dim=1536)
        model_name = "mock"
    
    # Rest of the logic is the same...
    parser = SentenceSplitter(
        chunk_size=1024,
        chunk_overlap=200,
    )
    
    all_chunks = []
    
    for doc in docs:
        document = Document(
            text=doc.get("text", ""),
            metadata={
                "ticker": doc.get("ticker"),
                "filing_type": doc.get("filing_type"),
                "filing_date": doc.get("filing_date"),
            }
        )
        
        chunks = parser.get_nodes_from_documents([document])
        
        for i, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(
                f"{doc.get('ticker')}_{doc.get('filing_date')}_{i}".encode()
            ).hexdigest()
            
            embedding = embed_model.get_text_embedding(chunk.text)
            
            chunk_data = {
                "id": chunk_id,
                "text": chunk.text,
                "embedding": embedding,
                "metadata": {
                    **chunk.metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }
            }
            
            all_chunks.append(chunk_data)
    
    return {
        **state,
        "chunks": all_chunks,
        "embedding_model": model_name,
        "chunk_count": len(all_chunks),
    }