"""
MCP server for local document

tools:
get_relevant_chunks(query,libraries,top_k)

only retrieve chunk from local rag
"""
from __future__ import annotations

import logging
import sys
from typing import Any, Optional
from pathlib import Path


from fastmcp import FastMCP

from app.retrieval.retriever import CodeRetriever


logger = logging.getLogger(__name__)
mcp = FastMCP("local-doc-context")
_retriever: Optional[CodeRetriever] = None




def get_retriever():
    """
    lazy-load
    """
    global _retriever
    if _retriever is None:
        logger.info("Initializing CodeContextRetriever...")
        _retriever = CodeRetriever()
        logger.info("CodeContextRetriever initialized.")

    return _retriever

@mcp.tool()
def get_code_context(
    query: str,
    library: Optional[str] = None,
    top_k: int = 5,
    max_tokens: int = 2000,
    code_first: bool = True,
) -> dict[str, Any]:
    """
    Retrieve official documentation context for a coding task.
    Parameters
    ----------
    query:
        User's coding question or coding task.

        Example:
            "How do I use Chroma as a LangChain retriever?"

    library:
        Optional library filter.

        Example:
            "langchain"

    top_k:
        Maximum number of final chunks to return.

    max_tokens:
        Maximum total context budget.

    code_first:
        If True, prefer chunks whose metadata has has_code=True.

    Returns
    -------
    dict
        Structured context for coding agents.
    """

    retriever = get_retriever()


    if library is None:
        raise ValueError("library is required")

    return retriever.get_code_context(
        query=query,
        library=library,
        top_k=top_k,
        max_tokens=max_tokens,
        code_first=code_first,
    )

@mcp.tool()
def list_libraries() -> dict[str, Any]:
    """
    List available local documentation libraries.

    """
    retriever = get_retriever()

    if not hasattr(retriever, "list_libraries"):
        return {
            "libraries": [],
            "count": 0,
            "message": "CodeRetriever.list_libraries() is not implemented yet.",
        }

    libraries = retriever.list_libraries()

    return {
        "libraries": libraries,
        "count": len(libraries),
    }



def main() -> None:
    """
    Start MCP server.

    For local IDE integration, stdio is enough.
    """
    logger.info("Starting local-doc-context MCP server...")
    mcp.run()


if __name__ == "__main__":
    main()
