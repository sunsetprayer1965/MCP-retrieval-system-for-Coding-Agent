from pathlib import Path
import os
import time
import logging
import hashlib
from typing import List, Optional, Dict, Any

import chromadb
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


logger = logging.getLogger(__name__)


class RAGEngine:
    """
    RAG Engine with:
    - Gemini LLM
    - Gemini embeddings
    - Chroma vector store
    - BM25 keyword search
    - query rewrite OFF by default
    - rerank OFF by default
    """

    def __init__(
        self,
        docs_dir: str = "data/raw",
        vector_dir: str = "data/processed/vectorstore/rag",
        collection_name: str = "python_docs_full",
    ):
        self.docs_dir = Path(docs_dir)
        self.vector_dir = Path(vector_dir)
        self.collection_name = collection_name

        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=self.api_key,
        )

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-2",
            google_api_key=self.api_key,
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        self.vector_store = None

        self.bm25 = None
        self.corpus: List[str] = []
        self.corpus_metadatas: List[Dict[str, Any]] = []

        self.reranker = None

        self.load_vector_store_if_exists()

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------

    def _has_persisted_vector_store(self) -> bool:
        return (
            self.vector_dir.exists()
            and (self.vector_dir / "chroma.sqlite3").exists()
        )

    def _get_response_text(self, response) -> str:
        if hasattr(response, "text") and response.text:
            return response.text
        if hasattr(response, "content"):
            return response.content
        return str(response)

    def _extract_source_name(self, file: Path) -> str:
        """
        Examples:
            data/raw/pypi_packages/seaborn_readme.md -> seaborn
            data/raw/pypi_packages/pandas_readme.md -> pandas
            data/raw/python_docs/3.11/abc.html -> abc.html
        """
        stem = file.stem

        if stem.endswith("_readme"):
            return stem.replace("_readme", "")

        if file.suffix == ".html":
            return file.name

        return stem

    def _make_chunk_id(self, source_name: str, chunk_index: int, text: str) -> str:
        short_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        safe_source = (
            source_name
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
        )
        return f"{safe_source}::chunk_{chunk_index}::{short_hash}"

    def _build_bm25_from_corpus(self) -> None:
        if not self.corpus:
            self.bm25 = None
            logger.warning("No corpus available for BM25.")
            return

        tokenized_corpus = [doc.split() for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 index built with {len(self.corpus)} documents.")

    # ------------------------------------------------------------------
    # Load existing vector store
    # ------------------------------------------------------------------

    def load_vector_store_if_exists(self) -> bool:
        """
        Load existing Chroma vector store from disk.

        Chroma persists documents, metadata, and embeddings.
        BM25 is only in memory, so after loading Chroma, we rebuild BM25
        from documents stored in Chroma.
        """
        if not self._has_persisted_vector_store():
            logger.info("No persisted vector store found.")
            return False

        logger.info(f"Loading existing vector store from {self.vector_dir}")

        self.vector_store = Chroma(
            persist_directory=str(self.vector_dir),
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
        )

        data = self.vector_store._collection.get()
        self.corpus = data.get("documents", []) or []
        self.corpus_metadatas = data.get("metadatas", []) or []

        logger.info(f"Loaded {len(self.corpus)} documents from Chroma.")

        if self.corpus:
            self._build_bm25_from_corpus()
        else:
            logger.warning("Chroma loaded, but no documents found inside collection.")

        return True

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_docs(
        self,
        files: Optional[List[Path]] = None,
        force_rebuild: bool = False,
        sleep_seconds: float = 1.0,
    ) -> None:
        """
        Ingest documents and create Chroma vector store.

        This version avoids Chroma.from_texts() because that performs
        batch embedding internally and can hit Gemini free-tier limits.
        Instead, it embeds chunks one by one and stores them manually.

        Args:
            files:
                Optional list of files to ingest.
            force_rebuild:
                If False, load existing vector store and skip ingestion.
                If True, delete old collection and rebuild.
            sleep_seconds:
                Delay between embedding calls. Useful for Gemini free tier.
        """
        if not force_rebuild:
            if self.vector_store is not None:
                logger.info("Vector store already loaded. Skipping ingestion.")
                return

            if self.load_vector_store_if_exists():
                logger.info("Existing vector store found. Skipping ingestion.")
                return

        logger.info("Starting document ingestion...")

        self.vector_dir.mkdir(parents=True, exist_ok=True)

        docs: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        if files is None:
            files = sorted(
                list(self.docs_dir.rglob("*.md")) +
                list(self.docs_dir.rglob("*.html"))
            )

        logger.info(f"Found {len(files)} files to process.")

        for file in files:
            file = Path(file)

            try:
                source_name = self._extract_source_name(file)
                source_path = str(file)

                if file.suffix == ".md":
                    content = file.read_text(encoding="utf-8", errors="ignore").strip()

                    if not content:
                        logger.warning(f"Skipping empty markdown file: {file}")
                        continue

                    chunks = self.text_splitter.split_text(content)

                    for chunk_idx, chunk in enumerate(chunks):
                        chunk = chunk.strip()
                        if not chunk:
                            continue

                        docs.append(chunk)
                        metadatas.append({
                            "source": source_name,
                            "source_path": source_path,
                            "type": "markdown",
                            "chunk_index": chunk_idx,
                        })

                elif file.suffix == ".html":
                    html_content = file.read_text(encoding="utf-8", errors="ignore")
                    soup = BeautifulSoup(html_content, "html.parser")
                    text = soup.get_text(separator="\n", strip=True)

                    if not text:
                        logger.warning(f"Skipping empty html file: {file}")
                        continue

                    chunks = self.text_splitter.split_text(text)

                    for chunk_idx, chunk in enumerate(chunks):
                        chunk = chunk.strip()
                        if not chunk:
                            continue

                        docs.append(chunk)
                        metadatas.append({
                            "source": source_name,
                            "source_path": source_path,
                            "type": "html",
                            "chunk_index": chunk_idx,
                        })

            except Exception as e:
                logger.error(f"Error processing {file}: {e}")
                continue

        if not docs:
            logger.warning("No documents were ingested.")
            return

        logger.info(f"Prepared {len(docs)} chunks for embedding.")

        client = chromadb.PersistentClient(path=str(self.vector_dir))

        if force_rebuild:
            try:
                client.delete_collection(self.collection_name)
                logger.info(f"Deleted old collection: {self.collection_name}")
            except Exception:
                logger.info(f"No old collection to delete: {self.collection_name}")

        collection = client.get_or_create_collection(name=self.collection_name)

        success_count = 0
        fail_count = 0

        for i, (text, metadata) in enumerate(zip(docs, metadatas)):
            try:
                embedding_result = self.embeddings.embed_documents([text])

                if not embedding_result:
                    raise ValueError("No embedding returned.")

                embedding = embedding_result[0]
                source_name = metadata.get("source", "unknown")
                chunk_id = self._make_chunk_id(source_name, i, text)

                collection.add(
                    ids=[chunk_id],
                    documents=[text],
                    metadatas=[metadata],
                    embeddings=[embedding],
                )

                success_count += 1
                logger.info(
                    f"Stored chunk #{i}: source={source_name}, "
                    f"success={success_count}, fail={fail_count}"
                )

                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)

            except Exception as e:
                fail_count += 1
                logger.warning(
                    f"Failed to embed/store chunk #{i}, "
                    f"source={metadata.get('source', 'unknown')}, error={e}"
                )
                continue

        logger.info(
            f"Ingestion finished. success={success_count}, fail={fail_count}"
        )

        self.vector_store = Chroma(
            client=client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
        )

        data = self.vector_store._collection.get()
        self.corpus = data.get("documents", []) or []
        self.corpus_metadatas = data.get("metadatas", []) or []
        self._build_bm25_from_corpus()

        logger.info("Document ingestion complete.")

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def _vector_search(self, query_text: str, k: int) -> List[Document]:
        if self.vector_store is None:
            loaded = self.load_vector_store_if_exists()
            if not loaded:
                raise ValueError("No vector store found. Run ingest_docs() first.")

        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(query_text)
        logger.info(f"Vector search retrieved {len(docs)} documents.")
        return docs

    def _bm25_search(self, query_text: str, k: int) -> List[Document]:
        if self.bm25 is None or not self.corpus:
            logger.info("BM25 index unavailable. Skipping BM25 search.")
            return []

        try:
            tokenized_query = query_text.split()
            bm25_scores = self.bm25.get_scores(tokenized_query)

            bm25_indices = sorted(
                range(len(bm25_scores)),
                key=lambda i: bm25_scores[i],
                reverse=True,
            )[:k]

            docs = [
                Document(
                    page_content=self.corpus[i],
                    metadata=self.corpus_metadatas[i],
                )
                for i in bm25_indices
            ]

            logger.info(f"BM25 search retrieved {len(docs)} documents.")
            return docs

        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

    def _deduplicate_docs(self, docs: List[Document]) -> List[Document]:
        seen = set()
        unique_docs = []

        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            content_prefix = doc.page_content[:200]
            key = (source, content_prefix)

            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)

        return unique_docs

    def _rrf_merge(
        self,
        vector_docs: List[Document],
        bm25_docs: List[Document],
        k: int,
        rrf_k: int = 60,
    ) -> List[Document]:
        """
        Reciprocal Rank Fusion.
        This is not reranking with a model. It is a lightweight way to merge
        vector and BM25 results.
        """
        scores = {}
        doc_map = {}

        def doc_key(doc: Document):
            return (
                doc.metadata.get("source", "unknown"),
                doc.page_content[:200],
            )

        for rank, doc in enumerate(vector_docs, 1):
            key = doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            doc_map[key] = doc

        for rank, doc in enumerate(bm25_docs, 1):
            key = doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            doc_map[key] = doc

        sorted_keys = sorted(scores.keys(), key=lambda key: scores[key], reverse=True)
        return [doc_map[key] for key in sorted_keys[:k]]

    def _rewrite_query(self, query_text: str) -> str:
        rewrite_prompt = f"""
Rewrite this query to be more specific for document retrieval.
Only return the rewritten query.

Original query: {query_text}

Rewritten query:
""".strip()

        try:
            response = self.llm.invoke(rewrite_prompt)
            rewritten = self._get_response_text(response).strip()
            logger.info(f"Rewritten query: {rewritten}")
            return rewritten or query_text
        except Exception as e:
            logger.warning(f"Query rewriting failed: {e}. Using original query.")
            return query_text

    def _rerank_docs(self, query_text: str, docs: List[Document], k: int) -> List[Document]:
        if not docs:
            return []

        try:
            if self.reranker is None:
                from sentence_transformers import CrossEncoder
                self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

            pairs = [[query_text, doc.page_content] for doc in docs]
            scores = self.reranker.predict(pairs)

            reranked_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True,
            )[:k]

            reranked_docs = [docs[i] for i in reranked_indices]
            logger.info(f"Reranked to top {len(reranked_docs)} documents.")
            return reranked_docs

        except Exception as e:
            logger.warning(f"Reranking failed: {e}. Using original order.")
            return docs[:k]

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        k: int = 5,
        use_rewrite: bool = False,
        use_rerank: bool = False,
        use_bm25: bool = True,
    ) -> Dict[str, Any]:
        logger.info(f"Processing query: {query_text}")

        retrieval_query = self._rewrite_query(query_text) if use_rewrite else query_text

        vector_docs = self._vector_search(retrieval_query, k=k)
        bm25_docs = self._bm25_search(retrieval_query, k=k) if use_bm25 else []

        if use_bm25:
            final_retrieval_docs = self._rrf_merge(vector_docs, bm25_docs, k=k)
        else:
            final_retrieval_docs = self._deduplicate_docs(vector_docs)[:k]

        logger.info(f"Retrieved {len(final_retrieval_docs)} final documents before optional rerank.")

        if use_rerank:
            final_docs = self._rerank_docs(retrieval_query, final_retrieval_docs, k=k)
        else:
            final_docs = final_retrieval_docs[:k]

        if not final_docs:
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "sources": [],
                "query_original": query_text,
                "query_used_for_retrieval": retrieval_query,
                "num_sources": 0,
            }

        context = "\n\n---\n\n".join([
            f"Source: {doc.metadata.get('source', 'Unknown')}\n"
            f"Type: {doc.metadata.get('type', 'unknown')}\n"
            f"Content: {doc.page_content}"
            for doc in final_docs[:3]
        ])

        prompt = f"""
You are a helpful assistant that answers questions about Python programming and libraries based on documentation.

Context:
{context}

Question:
{query_text}

Instructions:
- Answer based on the context above.
- If the context is insufficient, say so.
- Be clear and concise.
- Include code examples only if the context supports them.
- Cite the source names when useful.

Answer:
""".strip()

        try:
            response = self.llm.invoke(prompt)
            answer = self._get_response_text(response)
            logger.info("Generated answer successfully.")
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            answer = f"Error generating answer: {str(e)}"

        sources = [
            {
                "source": doc.metadata.get("source", "Unknown"),
                "source_path": doc.metadata.get("source_path", ""),
                "type": doc.metadata.get("type", "unknown"),
                "content": (
                    doc.page_content[:300] + "..."
                    if len(doc.page_content) > 300
                    else doc.page_content
                ),
            }
            for doc in final_docs
        ]

        return {
            "answer": answer,
            "sources": sources,
            "query_original": query_text,
            "query_used_for_retrieval": retrieval_query,
            "num_sources": len(sources),
        }