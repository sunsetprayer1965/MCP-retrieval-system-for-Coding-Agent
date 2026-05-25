# Local Documentation Context MCP Server

面向 Coding Agent 的本地化官方文档 MCP 检索系统。  
本项目在已有多源官方文档 RAG 系统之上，增加一个 self-hostable MCP server，使 VS Code Copilot、Claude Code、Cursor、Codex 等 coding agents 可以在代码生成过程中调用本地官方文档索引，获取与当前任务相关的文档片段、代码示例和来源 metadata，从而降低 API 幻觉、参数错误、版本不一致等问题。

## 1. Project Motivation

LLM 在写代码时经常依赖训练时记忆，因此容易出现：

- 使用已经过时的 API；
- 编造不存在的参数；
- 混用不同版本的 import 路径；
- 对快速更新的 LLM / ML / Python 生态包理解滞后；
- 在缺少官方文档证据时生成看似合理但无法运行的代码。

本项目的目标不是让 MCP server 直接生成最终代码，而是让 coding agent 在生成代码前能够先获得本地检索到的官方文档上下文。

核心定位：

> A self-hosted Context7-like local documentation MCP server for coding agents.

与云端文档索引类服务相比，本项目强调：

- self-hostable；
- private docs supported；
- local Chroma / BM25 index；
- local cache；
- no vendor lock-in；
- evidence-grounded context；
- source_url / section_title / metadata；
- 可复用已有 60+ Python / ML / LLM 官方文档索引。

---

## 2. Tech Stack

- **Language**: Python
- **RAG / Retrieval**: LangChain, ChromaDB, BM25, RRF
- **MCP Layer**: FastMCP
- **Backend API**: FastAPI
- **Frontend Demo**: Streamlit
- **Document Ingestion**: YAML-driven crawler, Markdown extraction, HTML cleaning, Firecrawl dynamic rendering
- **Metadata / Incremental Update**: content_hash, retrieved_at, source_url, section_title
- **Deployment**: Docker / Docker Compose
- **Monitoring**: Prometheus, Grafana
- **Embedding**: Gemini embedding or compatible embedding backends

---

## 3. System Architecture

```text
Coding Agent
VS Code Copilot / Claude Code / Cursor / Codex
        |
        | MCP tool call
        v
FastMCP Server
        |
        | get_code_context(query, library, top_k, max_tokens, code_first)
        v
CodeRetriever
        |
        | locate per-library vector store
        v
Local Documentation Index
        |
        | Chroma vector search
        | BM25 keyword search
        | RRF fusion ranking
        v
Context Formatter
        |
        | structured context + metadata + instruction_to_model
        v
Coding Agent generates final code
```

---

## 4. Core MCP Tool

### `get_code_context`

Retrieve official documentation context for a coding task.

The MCP server does **not** generate final code.  
It only returns grounded documentation chunks that the coding agent can use as evidence.

### Input Schema

```json
{
  "query": "How do I use Chroma as a LangChain retriever?",
  "library": "langchain",
  "top_k": 5,
  "max_tokens": 2000,
  "code_first": true
}
```

### Field Explanation

| Field | Type | Meaning |
|---|---|---|
| `query` | `str` | Current coding task or documentation question |
| `library` | `str` | Target library, e.g. `langchain`, `pandas`, `chromadb` |
| `top_k` | `int` | Number of candidate chunks to retrieve |
| `max_tokens` | `int` | Maximum returned context budget |
| `code_first` | `bool` | Prefer chunks containing code examples |

### Output Schema

```json
{
  "query": "How do I use Chroma as a LangChain retriever?",
  "library": "langchain",
  "found": true,
  "results": [
    {
      "rank": 1,
      "library": "langchain",
      "version": "0.3",
      "doc_title": "Chroma",
      "section_title": "Use as Retriever",
      "section_path": "Integrations > Vector Stores > Chroma > Retriever",
      "source_path": "data/raw/langchain/integrations/vectorstores/chroma.md",
      "source_url": "https://python.langchain.com/docs/integrations/vectorstores/chroma/",
      "chunk_index": 3,
      "tokens": 420,
      "has_code": true,
      "content_hash": "a8c91d...",
      "content": "..."
    }
  ],
  "context_preview": "...",
  "transparency": {
    "method": "per_library_chroma_vector_search",
    "searched_library": "langchain",
    "top_k": 5,
    "max_tokens": 2000,
    "code_first": true,
    "returned_chunks": 3,
    "latency_ms": 42.1
  },
  "instruction_to_model": "Use these retrieved official documentation chunks as evidence..."
}
```

---

## 5. Metadata Design

Each chunk is stored with structured metadata:

```json
{
  "library": "langchain",
  "version": "0.3",

  "source_path": "data/raw/langchain/integrations/vectorstores/chroma.md",
  "source_url": "https://python.langchain.com/docs/integrations/vectorstores/chroma/",

  "doc_title": "Chroma",
  "section_title": "Initialization",
  "section_path": "Integrations > Vector Stores > Chroma > Initialization",

  "chunk_index": 3,
  "tokens": 420,
  "has_code": true,
  "content_hash": "..."
}
```

### Why These Fields Matter

- `library`: selects the correct per-library vector store.
- `version`: helps avoid cross-version API confusion.
- `source_path`: local traceability and debugging.
- `source_url`: official evidence source.
- `doc_title`: identifies the document page.
- `section_title`: improves retrieval quality and context clarity.
- `section_path`: preserves hierarchical document structure.
- `chunk_index`: enables adjacent chunk merging.
- `tokens`: enables context budget control.
- `has_code`: helps prioritize code examples.
- `content_hash`: supports deduplication and incremental updates.

---

## 6. Retrieval Logic

The intended retrieval pipeline:

```text
query
  ↓
optional query rewrite to topic
  ↓
select library-specific vector store
  ↓
BM25 keyword search
  ↓
Chroma embedding search
  ↓
RRF fusion ranking
  ↓
optional code_first boost
  ↓
token budget filtering
  ↓
structured context returned to coding agent
```

The current MVP first supports per-library Chroma retrieval.  
BM25 + RRF can be enabled as the next retrieval layer.

---

## 7. Document Ingestion

The existing RAG system supports YAML-driven ingestion for 60+ Python / ML / LLM documentation sources.

Supported ingestion modes:

- Markdown extraction;
- HTML cleaning;
- Firecrawl dynamic rendering;
- metadata injection;
- `content_hash`-based incremental update;
- section-based chunking;
- local vector indexing.

Example metadata generated during ingestion:

```python
metadata = {
    "library": library,
    "version": version,
    "source_path": source_path,
    "source_url": source_url,
    "doc_title": doc_title,
    "section_title": section_title,
    "section_path": section_path,
    "chunk_index": chunk_idx,
    "tokens": estimate_tokens(chunk),
    "has_code": has_code_block(chunk),
    "content_hash": content_hash(chunk),
}
```

---

## 8. Per-Library Vector Store Layout

Each library has its own local vector store:

```text
data/processed/vectorstore/
├── langchain/
│   └── chroma.sqlite3
├── chromadb/
│   └── chroma.sqlite3
├── pandas/
│   └── chroma.sqlite3
├── numpy/
│   └── chroma.sqlite3
└── pytorch/
    └── chroma.sqlite3
```

This follows the same high-level idea as local package-based documentation systems:

```text
library
  ↓
open that library's local documentation index
  ↓
search only inside that library
```

This avoids noisy global search and reduces incorrect cross-library matches.

---

## 9. Run with Docker

### Build Image

```bash
docker build -t local-doc-context-mcp .
```

### Run MCP Server Manually

PowerShell:

```powershell
docker run -i --rm `
  --env-file .env `
  -v "${PWD}:/app" `
  -w /app `
  local-doc-context-mcp `
  python -m app.mcp
```

Linux/macOS:

```bash
docker run -i --rm \
  --env-file .env \
  -v "$PWD:/app" \
  -w /app \
  local-doc-context-mcp \
  python -m app.mcp
```

If the command appears to hang, that is normal.  
The MCP stdio server is waiting for an MCP client to call its tools.

---

## 10. Environment Variables

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
```

The embedding key is required because vector search still needs to embed the user query at retrieval time.

---

## 11. Codex MCP Configuration

Create `.codex/config.toml`:

```toml
[mcp_servers.local-doc-context]
command = "docker"
args = [
  "run",
  "-i",
  "--rm",
  "--env-file", "C:\\Users\\73184\\Desktop\\agent-mcp\\.env",
  "-v", "C:\\Users\\73184\\Desktop\\agent-mcp:/app",
  "-w", "/app",
  "local-doc-context-mcp",
  "python", "-m", "app.mcp"
]
```

The `-i` flag is required because MCP stdio communicates through stdin/stdout.

---

## 12. Example Usage

Ask the coding agent:

```text
Use the local-doc-context MCP server to search the langchain documentation for how to use Chroma as a retriever. Do not answer from memory.
```

Expected tool call:

```json
{
  "query": "Chroma retriever usage in LangChain",
  "library": "langchain",
  "top_k": 5,
  "max_tokens": 2000,
  "code_first": true
}
```

The agent should then use the returned documentation chunks as evidence before generating code.

---

## 13. Monitoring

The existing system supports Docker Compose deployment with:

- Prometheus metrics;
- Grafana dashboards;
- API latency monitoring;
- vector retrieval QPS monitoring;
- embedding call cost tracking;
- FastAPI endpoint observability.

---

## 14. Evaluation

Hybrid retrieval evaluation showed:

- MRR@k improved from approximately 20% baseline to 93.3%;
- Top-5 retrieval accuracy reached 75%.

The improvement mainly comes from combining:

- metadata-aware chunking;
- BM25 keyword recall;
- Chroma semantic retrieval;
- RRF fusion ranking.

---

## 15. Roadmap

Planned improvements:

- Add BM25 + RRF into the MCP retrieval path;
- Add optional query-to-topic rewriting;
- Merge adjacent chunks by `source_path + chunk_index`;
- Add `list_libraries` MCP tool;
- Add `show_context` debug tool;
- Support private internal documentation;
- Add deprecated API detection;
- Add migration assistant for version-specific API changes;
- Add local Qdrant backend option.

---

## 16. Project Positioning

This project turns a multi-source official documentation RAG system into a local documentation context server for coding agents.

It is designed for developers who want:

- local-first documentation search;
- private docs support;
- coding-agent-compatible context retrieval;
- grounded code generation;
- no dependency on a single cloud documentation provider.

