# Context Engineering with Qdrant

Build a biomedical research assistant that answers questions over ~27k [PubMed](https://pubmed.ncbi.nlm.nih.gov/) papers using [Qdrant vector search engine](https://qdrant.tech/) tooling and LLM-based (OpenAI) tool routing.

This workshop is a simplified version of the [PubMed Navigator](https://www.youtube.com/watch?v=3NWTi90i6C4) project, covering the vector search part of it.
The goal is to learn the basics of context engineering with Qdrant: best practices, available features, and choices ensuring scalability.

## What you'll learn

### Context engineering

- Designing LLM (OpenAI) tool definitions with clear routing signals
- Writing routing prompts based on best prompt engineering practices

### Qdrant for Context Engineering

Infrastructure and scalability:

- **[Cloud Inference](https://qdrant.tech/documentation/cloud/inference/#use-external-models) through external providers (OpenAI)**. Offload embedding inference to Qdrant server-side to save on latency.
- **[Efficient batch upload](https://qdrant.tech/documentation/concepts/points/#python-client-optimizations)** with the Python client's `upload_points`.
- **[Conditional uploads/updates](https://qdrant.tech/documentation/concepts/points/#update-mode)** to save on latency while broadening or adapting available dataset.
- **[Scalar quantization](https://qdrant.tech/course/essentials/day-4/what-is-quantization/#scalar-quantization)**. Compress vectors used for retrieval, reducing memory usage ~4x. Quantized vectors stay in RAM for fast search; originals live on disk for rescoring.

Capabilities important in high-precision domains:

- **[Hybrid retrieval](https://qdrant.tech/documentation/concepts/hybrid-queries/#hybrid-and-multi-stage-queries)**. Combine dense semantic search with keyword-based search in one call. Keyword search in Qdrant is implemented through [sparse vectors](https://qdrant.tech/documentation/concepts/vectors/#sparse-vectors). Dense vectors catch meaning; [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) formula-based sparse vectors catch exact matches (gene names, drug names) that dense embeddings might miss.
- **[Multistage retrieval](https://qdrant.tech/documentation/concepts/hybrid-queries/#hybrid-and-multi-stage-queries)**. Use [prefetch functionality](https://qdrant.tech/documentation/concepts/hybrid-queries/#hybrid-and-multi-stage-queries) to retrieve candidates cheaply, then rescore and rerank for precision. In this workshop we'll setup multistage retrieval based on [Matryoshka Representation Learning (MRL)](https://huggingface.co/blog/matryoshka) [feature](https://openai.com/index/new-embedding-models-and-api-updates/) of OpenAI embeddings.

Capabilities for meaningful context engineering:
- **[Recommendation API](https://qdrant.tech/documentation/concepts/explore/#recommendation-api)**. Discover papers based on positive/negative constraints in the user's query. Qdrant computes a target vector close to positives and far from negatives, using vector arithmetic.

## What you'll need

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- A [Qdrant Cloud](https://cloud.qdrant.io/) **Free (Forever) Tier** cluster. The UI will guide you through creating a cluster and obtaining your endpoint URL and API key. No credit card needed. [Video walkthrough](https://www.youtube.com/watch?v=xvWIssi_cjQ).
- An [OpenAI API key](https://platform.openai.com/api-keys). You'll need it for both embedding inference (`text-embedding-3-small`) and LLM-based tool routing (`gpt-4o-mini`).

> Embedding all 26,788 abstracts from `data/pubmed_dataset.json` with `text-embedding-3-small` costs approximately **$0.15**. Search queries use `gpt-4o-mini` for tool routing and summarization costing fractions of a cent per query. Expect around **~$0.17** for running the entire pipeline.

## Setup

1. Install dependencies:

```bash
uv sync
```

2. Copy the environment template and fill in your keys:

```bash
cp .env.example .env
```

- For Qdrant Cloud API key & cluster URL, UI guides you through obtaining them after creating a cluster. [Video walkthrough](https://www.youtube.com/watch?v=xvWIssi_cjQ).
- For OpenAI API key, obtain it [here](https://platform.openai.com/api-keys)

```env
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_COLLECTION_NAME=pubmed_papers

OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4o-mini

PUBMED_JSON_PATH=data/pubmed_dataset.json
```

3. You're all set. The dataset is already included at `data/pubmed_dataset.json`.

### About the dataset

The included dataset contains **26,788 PubMed papers**, each with:

- PMID (unique PubMed identifier)
- Title and abstract
- Authors with affiliations
- [MeSH terms](https://www.nlm.nih.gov/mesh/meshhome.html) (Medical Subject Headings — the controlled vocabulary used by NLM to index articles)
- Journal, publication date, and DOI

## Project structure

```
pubmed-navigator-workshop/
├── data/
│   └── pubmed_dataset.json        # PubMed papers dataset
├── workshop/
│   ├── config.py                  # Environment variables
│   ├── cli.py                     # CLI entry point
│   ├── infrastructure/
│   │   ├── search_engine.py       # Qdrant infrastructure
│   │   └── ingestion.py           # Data loading & ingestion orchestration
│   └── context_engineering/
│       ├── prompts.py             # LLM prompt templates
│       ├── tools.py               # Tool definitions for LLM routing
│       ├── search_engine_query.py # Qdrant tools execution
│       └── context.py             # Orchestration pipeline
├── .env.example                   # Environment template
├── Makefile                       # Commands
└── pyproject.toml
```

## Part 1: Data Ingestion

We'll create a Qdrant [collection](https://qdrant.tech/documentation/concepts/collections/) and populate it with PubMed papers.

The paper abstracts get embedded and indexed in Qdrant with OpenAI's `text-embedding-3-small`. All other fields will be stored as [payload](https://qdrant.tech/documentation/concepts/payload/) metadata alongside the vectors.

### Step 1 — Create the collection

```bash
make create-qdrant-collection
```

This creates a configured collection (its name we provided in .env) with three [named vectors](https://qdrant.tech/documentation/concepts/vectors/#named-vectors) — `Dense`, `Reranker`, and `Lexical`:

| Vector | Type | Dimensions | Role |
|--------|------|-----------|------|
| **Dense** | [dense](https://qdrant.tech/documentation/concepts/vectors/#dense-vectors), `text-embedding-3-small` | 1024 | Semantic retrieval, [scalar quantization](https://qdrant.tech/documentation/guides/quantization/#setting-up-scalar-quantization). Quantized vectors are used for retrieval and stored in RAM, originals on disk. |
| **Reranker** | [dense](https://qdrant.tech/documentation/concepts/vectors/#dense-vectors), `text-embedding-3-small` | 1536 | Reranking only. Stored on disk, no [vector index](https://qdrant.tech/documentation/concepts/indexing/#vector-index) built for it. |
| **Lexical** | [sparse](https://qdrant.tech/documentation/concepts/vectors/#sparse-vectors) | varies | [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) keyword matching. |

All collection configuration lives in `search_engine.py`.

#### Cloud Inference + Matryoshka Representation Learning

Both dense vectors come from `text-embedding-3-small`, which supports [MRL](https://huggingface.co/blog/matryoshka) ([OpenAI implementation](https://openai.com/index/new-embedding-models-and-api-updates/)), a training approach that front-loads all important information into the earliest dimensions of an embedding. Due to that, a single embedding can be truncated to any shorter prefix and still be used meaningfully.

We use two truncation levels from one model:
- **1024 dims → Dense**, fast retrieval with lower memory footprint.
- **1536 dims → Reranker**, higher precision for rescoring retrieved candidates.

> **Experiment:** You can change `OPENAI_RETRIEVER_EMBEDDING_DIMENSION` and `OPENAI_RERANKER_EMBEDDING_DIMENSION` in `search_engine.py` to see how dimensionality affects resource usage and retrieval quality. 1536 is the maximum for `text-embedding-3-small`. You can also switch `OPENAI_EMBEDDING_MODEL` to `text-embedding-3-large` for higher dimensions, but it is significantly more expensive.

With [Cloud Inference](https://qdrant.tech/documentation/cloud/inference/#use-external-models) (`CLOUD_INFERENCE = True` in `search_engine.py`), embedding happens server-side: the client sends raw text, Qdrant calls OpenAI. MRL truncation requests for the same text [are deduplicated into one API call](https://qdrant.tech/documentation/concepts/inference/#reduce-vector-dimensionality-with-matryoshka-models), to save costs and latency.

#### Scalar quantization

The `Dense` vector is quantized using [scalar quantization](https://qdrant.tech/course/essentials/day-4/what-is-quantization/#scalar-quantization) and only quantized lighter vectors are kept in RAM (`always_ram=True`). Each float32 in original vector is compressed to 8 bits, reducing memory ~4x. Original vectors stay on disk for rescoring (`rescore=True` at query time), so accuracy is preserved despite compression.

Scalar quantization is a safe default choice. Other quantization methods are also available, see [our documentation on quantization in Qdrant](https://qdrant.tech/documentation/guides/quantization). It's important to keep quantization in mind early: at scale (much more than 30k papers), it saves significantly on latency and RAM costs.

> **Experiment:** The `quantile` parameter (default: `0.99`) in `search_engine.py` → `create_collection()` controls how aggressively outlier values are clipped during quantization.

#### BM25 sparse vectors

Qdrant supports [sparse vectors](https://qdrant.tech/documentation/concepts/vectors/#sparse-vectors) alongside dense vectors. This lets us use the [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) formula, well-known in information retrieval, through [BM25-based sparse vector inference](https://qdrant.tech/documentation/guides/text-search/#bm25).

In biomedical text, this matters: specific terms like gene names ("TP53") or drug names ("metformin") carry precise meaning that dense embeddings sometimes dilute.

> BM25 usage in Qdrant requires an `avg_len` parameter, the average document length used in the formula. We estimate this by sampling the first 300 abstracts (see `_estimate_avg_abstract_len()` in `search_engine.py`). You can adjust `ESTIMATE_BM25_AVG_LEN_ON_X_DOCS` if needed.

> The `Lexical` vector is configured with `modifier=IDF`, which enables [inverse document frequency](https://en.wikipedia.org/wiki/Tf%E2%80%93idf#Inverse_document_frequency) weighting from BM25 formula. This is set in `search_engine.py` → `create_collection()` via `SparseVectorParams`.

### Step 2 — Ingest the data

```bash
make ingest-data-to-qdrant
```

With Cloud Inference enabled (`CLOUD_INFERENCE = True` in `search_engine.py`), ingestion should take around 5 minutes. You can speed it up by playing with the parameters of `upload_points` in `search_engine.py`.
Without Cloud Inference ingestion will take longer.

**Efficient batch upload.** Papers are streamed via a generator into `upload_points` in batches (default: 32). Lazy batching, auto retries, parallelism, see `batch_size`, `parallel`, and `max_retries` parameters in `search_engine.py` → `upsert_points()`.

**Conditional uploads.** By default, points are overwritten during upsertion if a point with this ID already exists in the collection. Pass `ONLY_NEW=1` to switch to `INSERT_ONLY` mode, skipping existing points and saving on latency. [Read more on Conditional Updates here](https://qdrant.tech/documentation/concepts/points/#update-mode).

> **Note:** `ONLY_NEW=1` only speeds up the Qdrant upsert step. Embeddings are still computed for every paper, so it does not reduce inference costs.

To recreate the collection from scratch before ingestion:

```bash
make ingest-data-to-qdrant RECREATE=1
```

### Result

Open your Qdrant Cluster Dashboard (`Cluster UI`). You should see `pubmed_papers` populated with all papers in the list of all collections. [Qdrant WebUI](https://qdrant.tech/documentation/web-ui/).

## Part 2: Context Engineering

Query the collection using a context engineering pipeline that routes natural language questions to the right Qdrant tool via an LLM agent.

### How the pipeline works

Two phases (see `context.py`):

1. **Tool routing** — the LLM reads the question and calls one of two Qdrant tools:
   - `retrieve_papers_based_on_query` — hybrid search (dense + BM25) fused by reranking.
   - `recommend_papers_based_on_constraints` — based on Recommendation API. For queries with negative constraints ("papers about X but not Y").
2. **Summarization** — the LLM summarizes retrieved papers into key findings.

#### Hybrid search with multistage retrieval

`retrieve_papers_based_on_query` (see `search_engine_query.py`) runs a single Qdrant query that combines hybrid retrieval with multistage reranking via [prefetch](https://qdrant.tech/documentation/concepts/hybrid-queries/#hybrid-and-multi-stage-queries):

1. **Prefetch (hybrid)** — two parallel candidate retrievals:
   - Dense semantic search on the 1024-dim vector (with quantization `oversampling` + `rescore` to keep precision high. [Read more on searching with quantization](https://qdrant.tech/documentation/guides/quantization/#searching-with-quantization))
   - BM25 keyword search on the `Lexical` sparse vector
2. **Rerank (multistage)** — fused candidates are rescored using the full 1536-dim `Reranker` vector.

#### Recommendation with constraints

`recommend_papers_based_on_constraints` (see `search_engine_query.py`) uses Qdrant's [Recommendation API](https://qdrant.tech/documentation/concepts/explore/#recommendation-api) with [`AVERAGE_VECTOR` strategy](https://qdrant.tech/documentation/concepts/explore/#average-vector-strategy). Qdrant finds points close to the average of positive vectors and far from the average of negative vectors.

### Prompt and tool engineering

**Tool definitions** (`tools.py`) — each definition tells the LLM **what** the tool does, **when** to use it and **how** to construct arguments.

See best practices: [Anthropic "Best practices for tool definitions"](https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use#best-practices-for-tool-definitions) | [OpenAI function calling tools](https://developers.openai.com/api/docs/guides/function-calling#defining-functions)

**Tool calling** (`context.py`) with key parameters of the OpenAI Responses API call:
- `instructions`, system prompt with routing rules and few-shot examples. Separated from `input` (the user's question) so the LLM has a clear boundary between instructions and user input.
- `tool_choice="required"`, the LLM must call a tool.
- `parallel_tool_calls=False`, not more than one tool call per response.

> **Experiment:** You can change `OPENAI_MODEL` in `.env` to try different models (e.g., `gpt-4o`) and see how tool routing accuracy and summary quality change.

**Routing prompt with few-shot examples** (`prompts.py`): three examples covering the key scenarios:
1. Coherent query, no negative constraints → retrieval (shows filler removal, term preservation)
2. One positive topic, two independent negative constraints → recommendation
3. Two separable positive constraints, one negative one → recommendation

Each example includes a brief explanation so the LLM learns the reasoning, not just the pattern.

See best practices: [Anthropic "Prompting best practices"](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) | [OpenAI "Prompt engineering"](https://developers.openai.com/api/docs/guides/prompt-engineering#prompt-engineering)

### Run a query

```bash
make context-engineering-qdrant QUESTION="your question here"
```

Control how many papers Qdrant returns (default: 5):

```bash
make context-engineering-qdrant QUESTION="your question here" LIMIT=10
```

### Example queries to try

Watch the trace output to see which tool gets called and how arguments are constructed.

These should trigger `retrieve_papers_based_on_query`:

```bash
make context-engineering-qdrant QUESTION="What are the mechanisms of CRISPR-Cas9 delivery using lipid nanoparticles for in vivo genome editing?"

# Smaller limit — see how the summary changes
make context-engineering-qdrant QUESTION="What is the role of autophagy in neurodegenerative diseases?" LIMIT=3
```

These should trigger `recommend_papers_based_on_constraints`:

```bash
make context-engineering-qdrant QUESTION="Find papers on photodynamic therapy for cancer treatment, but not studies focused on skin cancer."

# One positive, two independent negative constraints (should split into separate negative examples)
make context-engineering-qdrant QUESTION="Papers about TP53 mutations in cancer prognosis, but not animal studies and nothing like generic overviews of tumor suppressors."

# Two separable positives + exclusion (positives should split too)
make context-engineering-qdrant QUESTION="Research on BRCA1 DNA repair mechanisms and PD-L1 immunotherapy response biomarkers, excluding pediatric populations."
```

## What's next

Things to experiment with:

- Toggle `CLOUD_INFERENCE` in `search_engine.py` to compare server-side vs. client-side embedding.
- Try different `batch_size` and `parallel` values in `search_engine.py` → `upsert_points()` to optimize ingestion speed.

and

- Change embedding dimensions (`OPENAI_RETRIEVER_EMBEDDING_DIMENSION`, `OPENAI_RERANKER_EMBEDDING_DIMENSION` in `search_engine.py`) and observe how retrieval quality and resource usage change.
- Tune quantization search parameters (`oversampling`, `rescore`) in `search_engine_query.py` → `retrieve_papers_based_on_query()`.
- Change the `limit` on individual prefetch stages in `search_engine_query.py` to control how many candidates each retrieval method (dense vs. BM25) contributes before fusion and reranking.
- Adjust `ESTIMATE_BM25_AVG_LEN_ON_X_DOCS` or the default `avg_len` in `search_engine.py` to see how BM25 scoring changes with different average document length estimates.

and

- Adjust `LIMIT` to see how the number of retrieved papers affects summarization.
- Modify the prompt in `prompts.py` or tool descriptions in `tools.py` and see how the agent's behavior changes.


Check further resources for context engineering with Agents and Qdrant:
- [Qdrant MCP Server](https://github.com/qdrant/mcp-server-qdrant)
- [Qdrant Skills](https://github.com/qdrant/skills)
