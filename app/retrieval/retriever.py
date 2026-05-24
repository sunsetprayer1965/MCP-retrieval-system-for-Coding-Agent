"""
retrieve relevant document chunk from vector store and format them for coding agent
later, MCp server will call: 
get_relevant_chunks(query, libraries, top_k)

Example:
    data/processed/vectorstore/langchain/
    data/processed/vectorstore/pandas/
    data/processed/vectorstore/chromadb/

"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

logger = logging.getLogger(__name__)

class CodeRetriever:
    """
    It only return the relevant document chunks
    """

    def __init__(
        self,
        vector_store_root: Path = Path("data/processed/vectorstore"),
        collection_name: str = "docs",
    ) -> None:
        self.vector_store_root = vector_store_root
        self.collection_name = collection_name

        self.api_key = (
            os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("Gemini_API_KEY")
            or ""
        ).strip()
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-2",
            google_api_key=self.api_key,
        )

        # Cache loaded Chroma objects.
        # Avoid reloading the same library vector store every tool call.
        self._store_cache: Dict[str, Chroma] = {}
        self._store_filters: Dict[str, Optional[Dict[str, str]]] = {}
        self._store_collection_names: Dict[str, str] = {}

    def _normalize_library(self, library: str) -> str:
        """
        Normalize library name for folder lookup.

        Example:
            "LangChain" -> "langchain"
        """
        if not library or not library.strip():
            raise ValueError("library is required")

        return library.strip().lower()
    
    def _library_vector_dir(self, library: str) -> Path:
        """
        Get the persisted Chroma directory for one library.
        """
        library_key = self._normalize_library(library)
        return self.vector_store_root / library_key

    def _collection_name_for_dir(self, vector_dir: Path) -> str:
        """
        Read the Chroma collection name from sqlite when a store has one collection.
        """
        sqlite_path = vector_dir / "chroma.sqlite3"

        with sqlite3.connect(sqlite_path) as connection:
            rows = connection.execute("select name from collections").fetchall()

        if len(rows) == 1 and rows[0][0]:
            return str(rows[0][0])

        return self.collection_name

    def _dir_contains_library(self, vector_dir: Path, library_key: str) -> bool:
        """
        Check whether a shared vector store contains chunks for a library.
        """
        sqlite_path = vector_dir / "chroma.sqlite3"
        if not sqlite_path.exists():
            return False

        with sqlite3.connect(sqlite_path) as connection:
            row = connection.execute(
                """
                select 1
                from embedding_metadata
                where key in ('source', 'library')
                  and lower(string_value) = ?
                limit 1
                """,
                (library_key,),
            ).fetchone()

        return row is not None

    def _resolve_vector_store(
        self,
        library_key: str,
    ) -> Tuple[Path, Optional[Dict[str, str]], str]:
        """
        Find either a per-library store or a shared store containing the library.
        """
        vector_dir = self._library_vector_dir(library_key)
        if (vector_dir / "chroma.sqlite3").exists():
            return vector_dir, None, self._collection_name_for_dir(vector_dir)

        for candidate in sorted(self.vector_store_root.iterdir()):
            if candidate.is_dir() and self._dir_contains_library(candidate, library_key):
                return (
                    candidate,
                    {"source": library_key},
                    self._collection_name_for_dir(candidate),
                )

        raise FileNotFoundError(
            f"No Chroma vector store found for library='{library_key}' under "
            f"{self.vector_store_root}"
        )
    

    def _load_vector_store(self, library: str) -> Chroma:
        """
        Load an existing vector store for a given library.

        Important:
            This does not create or ingest anything.
            It only opens an existing Chroma vector store.
        """
        library_key = self._normalize_library(library)

        if library_key in self._store_cache:
            return self._store_cache[library_key]

        vector_dir, metadata_filter, collection_name = self._resolve_vector_store(library_key)

        logger.info(
            "Loading existing Chroma vector store: library=%s, path=%s, collection=%s",
            library_key,
            vector_dir,
            collection_name,
        )

        store = Chroma(
            persist_directory=str(vector_dir),
            embedding_function=self.embeddings,
            collection_name=collection_name,
        )

        self._store_cache[library_key] = store
        self._store_filters[library_key] = metadata_filter
        self._store_collection_names[library_key] = collection_name
        return store

    def _get_tokens(self, metadata: Dict[str, Any], content: str) -> int:
        """
        Read token count from metadata.

        Expected metadata:
            tokens: int

        Fallback:
            estimate by len(content) // 4
        """
        value = metadata.get("tokens", metadata.get("token"))

        if isinstance(value, int):
            return value

        if isinstance(value, float):
            return int(value)

        return max(1, len(content) // 4)

    def retrieve_vector_chunks(
        self,
        query: str,
        library: str,
        top_k: int = 5,
        fetch_k: Optional[int] = None,
        code_first: bool = True,
    ) -> List[Tuple[Document, Optional[float]]]:
        """
        Retrieve relevant chunks from one library-specific Chroma vector store.

        Parameters
        ----------
        query:
            Search query or rewritten topic.

        library:
            Required. The target library vector store.

            Example:
                "langchain"
                "pandas"
                "chromadb"

        top_k:
            Final number of chunks to return.

        fetch_k:
            Number of candidates to fetch before re-ranking.
            Usually larger than top_k.

        code_first:
            If True, prefer chunks where metadata["has_code"] is True.

        Returns
        -------
        List[Tuple[Document, Optional[float]]]
            Retrieved chunks and relevance scores.
        """
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if fetch_k is None:
            fetch_k = max(top_k * 4, 20)

        library_key = self._normalize_library(library)
        store = self._load_vector_store(library_key)
        metadata_filter = self._store_filters.get(library_key)

        logger.info(
            "Searching vector store: library=%s, query=%s, top_k=%s, fetch_k=%s",
            library,
            query,
            top_k,
            fetch_k,
        )

        try:
            docs_with_scores = store.similarity_search_with_relevance_scores(
                query,
                k=fetch_k,
                filter=metadata_filter,
            )
        except Exception as e:
            logger.warning(
                "similarity_search_with_relevance_scores failed: %s. "
                "Fallback to as_retriever().invoke().",
                e,
            )

            search_kwargs: Dict[str, Any] = {"k": fetch_k}
            if metadata_filter:
                search_kwargs["filter"] = metadata_filter

            retriever = store.as_retriever(search_kwargs=search_kwargs)
            docs = retriever.invoke(query)
            docs_with_scores = [(doc, None) for doc in docs]

        if code_first:
            
            docs_with_scores = sorted(
                docs_with_scores,
                key=lambda item: (
                    1 if bool((item[0].metadata or {}).get("has_code", False)) else 0,
                    item[1] if item[1] is not None else 0.0,
                ),
                reverse=True,
            )
        

        return docs_with_scores[:top_k]

    def list_libraries(self) -> List[str]:
        """
        List libraries available in per-library or shared Chroma stores.
        """
        libraries = set()

        if not self.vector_store_root.exists():
            return []

        for candidate in self.vector_store_root.iterdir():
            sqlite_path = candidate / "chroma.sqlite3"
            if not sqlite_path.exists():
                continue

            with sqlite3.connect(sqlite_path) as connection:
                rows = connection.execute(
                    """
                    select distinct lower(string_value)
                    from embedding_metadata
                    where key in ('source', 'library')
                      and string_value is not null
                    """
                ).fetchall()

            if rows:
                libraries.update(str(row[0]) for row in rows if row[0])
            else:
                libraries.add(candidate.name.lower())

        return sorted(libraries)
    

    def _format_chunk(
        self,
        doc: Document,
        score: Optional[float],
        rank: int,
        library: str,
    ) -> Dict[str, Any]:
        """
        Convert one LangChain Document into MCP-friendly JSON.

        Assumed metadata schema:
            library
            version
            source_path
            source_url
            doc_title
            section_title
            section_path
            chunk_index
            tokens
            has_code
            content_hash
        """
        metadata = doc.metadata or {}
        content = doc.page_content or ""

        return {
            "rank": rank,
            "score": score,

            "library": metadata.get("library") or library,
            "version": metadata.get("version"),

            "source_path": metadata.get("source_path"),
            "source_url": metadata.get("source_url"),

            "doc_title": metadata.get("doc_title"),
            "section_title": metadata.get("section_title"),
            "section_path": metadata.get("section_path"),

            "chunk_index": metadata.get("chunk_index"),
            "tokens": self._get_tokens(metadata, content),
            "has_code": bool(metadata.get("has_code", False)),
            "content_hash": metadata.get("content_hash"),

            "content": content,

            # Keep full metadata for debugging.
            "metadata": metadata,
        }

    def _apply_token_budget(
        self,
        chunks: List[Dict[str, Any]],
        max_tokens: int,
    ) -> List[Dict[str, Any]]:
        """
        Keep adding chunks until the total token budget is reached.
        """
        selected: List[Dict[str, Any]] = []
        used_tokens = 0

        for chunk in chunks:
            tokens = int(chunk.get("tokens") or 0)

            if selected and used_tokens + tokens > max_tokens:
                break

            selected.append(chunk)
            used_tokens += tokens

        return selected
    
    def _build_context_preview(
        self,
        chunks: List[Dict[str, Any]],
    ) -> str:
        """
        Build markdown context preview.

        This is useful for debugging and for showing what will be injected
        into the coding agent.
        """
        parts = []

        for chunk in chunks:
            title = " > ".join(
                str(x)
                for x in [
                    chunk.get("doc_title"),
                    chunk.get("section_title"),
                ]
                if x
            ) or "Untitled"

            source = chunk.get("source_url") or chunk.get("source_path") or "unknown source"
            content = chunk.get("content", "")

            parts.append(
                f"## {title}\n\n"
                f"Source: {source}\n\n"
                f"{content}"
            )

        return "\n\n---\n\n".join(parts)

    def _build_context_preview(
        self,
        chunks: List[Dict[str, Any]],
    ) -> str:
        """
        Build markdown context preview.

        This is useful for debugging and for showing what will be injected
        into the coding agent.
        """
        parts = []

        for chunk in chunks:
            title = " > ".join(
                str(x)
                for x in [
                    chunk.get("doc_title"),
                    chunk.get("section_title"),
                ]
                if x
            ) or "Untitled"

            source = chunk.get("source_url") or chunk.get("source_path") or "unknown source"
            content = chunk.get("content", "")

            parts.append(
                f"## {title}\n\n"
                f"Source: {source}\n\n"
                f"{content}"
            )

        return "\n\n---\n\n".join(parts)
    

    def get_code_context(
        self,
        query: str,
        library: str,
        top_k: int = 5,
        max_tokens: int = 2000,
        code_first: bool = True,
    ) -> Dict[str, Any]:
        """
        Main function for MCP.

        It searches only the selected library vector store.
        It returns context chunks, not final generated answers.
        """
        start_time = time.perf_counter()

        raw_results = self.retrieve_vector_chunks(
            query=query,
            library=library,
            top_k=top_k,
            code_first=code_first,
        )
        library_key = self._normalize_library(library)

        formatted_chunks = [
            self._format_chunk(
                doc=doc,
                score=score,
                rank=i,
                library=library,
            )
            for i, (doc, score) in enumerate(raw_results, start=1)
        ]

        budgeted_chunks = self._apply_token_budget(
            chunks=formatted_chunks,
            max_tokens=max_tokens,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        return {
            "query": query,
            "library": library,
            "found": len(budgeted_chunks) > 0,
            "results": budgeted_chunks,
           # "context_preview": self._build_context_preview(budgeted_chunks),
            "transparency": {
                "method": "per_library_chroma_vector_search",
                "searched_library": library,
                "collection_name": self._store_collection_names.get(
                    library_key,
                    self.collection_name,
                ),
                "top_k": top_k,
                "max_tokens": max_tokens,
                "code_first": code_first,
                "returned_chunks": len(budgeted_chunks),
               # "latency_ms": round(latency_ms, 2),
            },
            "instruction_to_model": (
                "Use these retrieved official documentation chunks as evidence. "
                "Do not invent unsupported API names, parameters, imports, or behaviors. "
                "If the context is insufficient, say what is missing."
            ),
        }
