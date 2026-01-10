# Track-A Compliant GraphRAG System for Full-Novel Understanding

This system implements a strict Track-A compliant GraphRAG architecture for understanding full novels, with explicit separation between offline graph construction and online query processing.

## Architecture Overview

### Phase 1: Offline Graph Knowledge Construction
1. Input Ingestion & Chunking
2. Meta LLaMA Encoder - Structured Extraction
3. Temporal Normalization Layer
4. Causal Graph Construction
5. Thematic Coherence Layer (Soft Contradictions)
6. Knowledge Graph Assembly
7. Community Detection (Leiden)
8. Community Summarization
9. Pathway Storage

### Phase 2: Online Query Processing
10. Query Encoding
11. Community Selection
12. Map Step (per community)
13. Soft Contradiction Resolution
14. Reduce Step
15. Meta LLaMA Decoder - Final Answer

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Offline Graph Construction

```bash
python offline_graph_construction.py --novel_path <path_to_novel.txt>
```

### Online Query Processing

```bash
python online_query_processing.py --query "Your question about the novel"
```

## Key Constraints

- **NO retrieval-only RAG**
- **NO answering from raw text**
- **NO answering from vector search alone**
- All answers must pass temporal + thematic consistency checks
- All intermediate representations are explicit and inspectable

