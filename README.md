# Track-A GraphRAG System

Complete Track-A compliant GraphRAG system for full-novel understanding.

## Quick Start

All implementation files are located in the `graphrag/` folder.

### Installation

```bash
cd graphrag
pip install -r requirements.txt
```

### Usage

**Offline Graph Construction:**
```bash
cd graphrag
python offline_graph_construction.py --novel_path <path_to_novel.txt>
```

**Online Query Processing:**
```bash
cd graphrag
python online_query_processing.py --query "Your question about the novel"
```

## Documentation

See `graphrag/ARCHITECTURE.md` for complete architecture documentation.

See `graphrag/README.md` for detailed usage instructions.

## Structure

```
graphrag/
├── schemas.py                          # All data schemas
├── config.py                           # Configuration
├── phase1_1_input_ingestion.py        # Phase 1.1: Input Ingestion
├── phase1_2_meta_llama_encoder.py     # Phase 1.2: Structured Extraction
├── phase1_3_temporal_normalization.py # Phase 1.3: Temporal Normalization
├── phase1_4_causal_graph.py           # Phase 1.4: Causal Graph
├── phase1_5_thematic_coherence.py     # Phase 1.5: Thematic Coherence
├── phase1_6_knowledge_graph_assembly.py # Phase 1.6: Graph Assembly
├── phase1_7_community_detection.py    # Phase 1.7: Community Detection
├── phase1_8_community_summarization.py # Phase 1.8: Community Summarization
├── phase1_9_pathway_storage.py        # Phase 1.9: Pathway Storage
├── phase2_1_query_encoding.py         # Phase 2.1: Query Encoding
├── phase2_2_community_selection.py    # Phase 2.2: Community Selection
├── phase2_3_map_step.py               # Phase 2.3: Map Step
├── phase2_4_soft_contradiction_resolution.py # Phase 2.4: Contradiction Resolution
├── phase2_5_reduce_step.py            # Phase 2.5: Reduce Step
├── phase2_6_meta_llama_decoder.py     # Phase 2.6: Final Answer Decoder
├── offline_graph_construction.py      # Phase 1 Orchestration
├── online_query_processing.py         # Phase 2 Orchestration
├── requirements.txt                    # Dependencies
├── README.md                           # Detailed documentation
└── ARCHITECTURE.md                     # Architecture documentation
```

