"""
Simplified RAG Engine - Built from scratch with working components
No advanced features - just basic retrieval and generation
"""

import os
import logging
from pathlib import Path
from typing import List, Optional
import chromadb
import time

# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
logger = logging.getLogger(__name__)


class SimpleRAGEngine:
    """Simplified RAG engine with only essential features that work."""
    
    def __init__(self):
        """Initialize the RAG engine with working components only."""
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set")
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
          model="gemini-2.5-flash",
            temperature=0,
            google_api_key=self.api_key,
)
        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2",
        google_api_key=self.api_key,
)
        # Vector store path
        self.vector_dir = Path("data/processed/vectorstore/simple_rag")
        self.vector_store = None
        self._load_existing_vector_store()
        logger.info("SimpleRAGEngine initialized successfully")
    
    def _has_persisted_vector_store(self) -> bool:


        return (
        self.vector_dir.exists()
        and (self.vector_dir / "chroma.sqlite3").exists()
    )

    def _load_existing_vector_store(self) -> bool:
        """Load an existing persisted Chroma vector store if it exists."""
        if not self._has_persisted_vector_store():
            logger.info("No persisted vector store found on disk.")
            return False

        logger.info(f"Loading existing vector store from {self.vector_dir}")
        self.vector_store = Chroma(
            persist_directory=str(self.vector_dir),
            embedding_function=self.embeddings,
            collection_name="python_docs",
        )
        logger.info("Existing vector store loaded successfully.")
        return True

    def ingest_docs(self, docs_dir: Path = Path("data/raw")) -> None:
        """
        Ingest documents from the data directory.
        Simplified version - just reads text files and creates embeddings.
        """
        if self.vector_store is not None:
            logger.info("Vector store already loaded in memory. Skipping ingestion.")
            return  

        if self._load_existing_vector_store():
            logger.info("Found persisted vector store on disk. Skipping ingestion.")
            return
        
        logger.info(f"Ingesting documents from {docs_dir}")
        
        documents = []
        
        # Read Python docs
        python_docs_dir = docs_dir / "python_docs" / "3.11"
        if python_docs_dir.exists():
            for html_file in list(python_docs_dir.glob("*.html"))[:20]:  # Limit to 20 for testing
                try:
                    content = html_file.read_text(encoding='utf-8', errors='ignore')
                    # Simple extraction - just get first 2000 characters
                    content = content[:2000]
                    
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": str(html_file.name),
                            "type": "python_doc"
                        }
                    )
                    documents.append(doc)
                except Exception as e:
                    logger.warning(f"Failed to read {html_file}: {e}")
        
        # Read PyPI packages
        pypi_dir = docs_dir / "pypi_packages"
        if pypi_dir.exists():
            for readme_file in pypi_dir.glob("*_readme.md"):
                try:
                    content = readme_file.read_text(encoding='utf-8', errors='ignore')
                    # Limit content size
                    content = content[:2000]
                    
                    package_name = readme_file.stem.replace('_readme', '')
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": package_name,
                            "type": "pypi_package"
                        }
                    )
                    documents.append(doc)
                except Exception as e:
                    logger.warning(f"Failed to read {readme_file}: {e}")
        
        logger.info(f"Loaded {len(documents)} documents")


        if documents:
            self.vector_dir.mkdir(parents=True, exist_ok=True)

            collection_name = "python_docs"

            logger.info("Creating/loading persistent Chroma client...")
            client = chromadb.PersistentClient(path=str(self.vector_dir))
            collection = client.get_or_create_collection(name=collection_name)

            success_count = 0
            fail_count = 0

            logger.info("Start embedding documents one by one...")

            for i, doc in enumerate(documents):
                text = doc.page_content.strip()

                if not text:
                    logger.warning(f"Skip blank document #{i}: {doc.metadata}")
                    continue

                try:
                    embedding = self.embeddings.embed_documents([text])[0]

                    collection.add(
                        ids=[f"doc-{i}"],
                        documents=[text],
                        metadatas=[doc.metadata],
                        embeddings=[embedding],
                    )

                    success_count += 1
                    logger.info(
                        f"Embedded and stored doc #{i}: "
                        f"source={doc.metadata.get('source', 'unknown')}"
                    )

                    # 免费额度下建议放慢一点
                    time.sleep(1)

                except Exception as e:
                    fail_count += 1
                    logger.warning(
                        f"Failed embedding doc #{i}: "
                        f"source={doc.metadata.get('source', 'unknown')}, error={e}"
                    )

            logger.info(
                f"Finished one-by-one embedding. success={success_count}, fail={fail_count}"
            )

            self.vector_store = Chroma(
                client=client,
                collection_name=collection_name,
                embedding_function=self.embeddings,
            )

            logger.info("Vector store initialized successfully")

        else:
            logger.warning("No documents found to ingest")
            
    
    def query(self, query_text: str, k: int = 3) -> str:
        """
        Query the RAG system with a simple question.
        Returns an answer based on retrieved documents.
        """
        logger.info(f"Processing query: {query_text}")
        
        # Load vector store if not already loaded
        if self.vector_store is None:
            if self.vector_dir.exists():
                logger.info("Loading existing vector store...")
                self.vector_store = Chroma(
                    persist_directory=str(self.vector_dir),
                    embedding_function=self.embeddings,
                    collection_name="python_docs"
                )
            else:
                raise ValueError("No vector store found. Run ingest_docs() first.")
        
        # Retrieve relevant documents
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        retrieved_docs = retriever.invoke(query_text)
        
        logger.info(f"Retrieved {len(retrieved_docs)} documents")
        
        # Create context from retrieved documents
        context = "\n\n".join([
            f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
            for doc in retrieved_docs
        ])
        
        # Create prompt
        prompt = f"""You are a helpful Python documentation assistant. Answer the question based on the provided context.

Context:
{context}

Question: {query_text}

Answer (be concise and helpful):"""
        
        # Get answer from LLM
        response = self.llm.invoke(prompt)
        answer = response.content
        
        logger.info("Generated answer successfully")
        
        # Return answer with sources
        sources = [doc.metadata.get('source', 'unknown') for doc in retrieved_docs]
        result = f"{answer}\n\nSources: {', '.join(sources)}"
        
        return result


# Test the simplified RAG engine
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("Testing Simplified RAG Engine")
    print("=" * 80)
    
    # Initialize
    print("\n[1] Initializing RAG engine...")
    rag = SimpleRAGEngine()
    print("✓ Initialized successfully")
    
    # Ingest documents
    print("\n[2] Ingesting documents...")
    rag.ingest_docs()
    print("✓ Documents ingested")
    
    # Test queries
    test_queries = [
        "What is BeautifulSoup?",
        "How do I use pandas?",
        "What is NumPy?"
    ]
    
    for query in test_queries:
        print(f"\n[QUERY] {query}")
        print("-" * 80)
        answer = rag.query(query)
        print(answer)
        print("-" * 80)
    
    print("\n" + "=" * 80)
    print("✓ All tests completed successfully!")
    print("=" * 80)
