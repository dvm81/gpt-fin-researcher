"""
Clean LangGraph DAG for GPT-Fin-Researcher
"""

from pprint import pprint
from typing import List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from .nodes import sec_loader, chunk_and_embed, store_in_chromadb, analyze_financial_factors
from .schemas import SECFiling, FinancialFactors, TradingStrategy, BacktestResults


class GraphState(TypedDict, total=False):
    # Input
    tasks: List[str]
    
    # Data flow
    docs: List[dict]  # Raw SEC filings
    chunks: List[dict]  # Document chunks with embeddings
    embeddings: List[List[float]]  # Raw embeddings
    embedding_model: str  # Model used for embeddings
    chunk_count: int  # Total chunks created
    filings: List[SECFiling]  # Structured filings
    factors: List[FinancialFactors]  # Extracted factors
    strategies: List[TradingStrategy]  # Generated strategies
    backtest_results: List[BacktestResults]  # Performance results
    
    # Control flow
    error: Optional[str]
    current_step: str
    
    # Vector store info
    vector_store: Optional[dict]


def planner(state: GraphState) -> GraphState:
    """Generate research tasks."""
    # If tasks are already provided, use them; otherwise use default
    if state.get("tasks"):
        return state
    return {"tasks": ["Investigate TSLA 10-K"]}


# Build the graph
g = StateGraph(GraphState)
g.add_node("planner", planner)
g.add_node("sec_loader", sec_loader)
g.add_node("embedder", chunk_and_embed)
g.add_node("vector_store", store_in_chromadb)
g.add_node("analyzer", analyze_financial_factors)

g.set_entry_point("planner")
g.add_edge("planner", "sec_loader")
g.add_edge("sec_loader", "embedder")
g.add_edge("embedder", "vector_store")
g.add_edge("vector_store", "analyzer")
g.add_edge("analyzer", END)

app = g.compile()


if __name__ == "__main__":
    pprint(app.invoke({}))