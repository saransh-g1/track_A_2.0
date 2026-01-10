# Track-A Compliant GraphRAG System Architecture

This document describes the complete architecture of the Track-A compliant GraphRAG system for full-novel understanding.

## System Overview

The system is strictly separated into two phases:

1. **Phase 1: Offline Graph Construction** - Executed once per novel
2. **Phase 2: Online Query Processing** - Executed per query

**Critical Constraint**: Answers are ONLY generated from the constructed knowledge graph, never from raw text or vector search alone.

## Phase 1: Offline Graph Construction

### 1.1 Input Ingestion & Chunking (`phase1_1_input_ingestion.py`)

**Purpose**: Ingest novel text and create chapter-aware chunks.

**Specifications**:
- Chapter-aware chunking
- Chunk size: 600 tokens
- Overlap: 120 tokens

**Output Schema**: `Chunk {chunk_id, chapter_id, text, token_range}`

**Key Methods**:
- `ingest_novel(novel_text)` - Main ingestion method
- `parse_chapters(novel_text)` - Parse chapters from text
- `chunk_text(text, chunk_size, overlap)` - Create overlapping chunks

### 1.2 Meta LLaMA Encoder - Structured Extraction (`phase1_2_meta_llama_encoder.py`)

**Purpose**: Extract structured information from chunks using Meta LLaMA with STRICT JSON schema.

**Specifications**:
- No free-form text allowed
- Extracts: entities, events, relations, claims, themes, temporal markers, causal links
- Enforces JSON schema validation

**Output Schema**: `StructuredExtraction {entities, events, relations, claims, themes, temporal_markers, causal_links}`

**Key Methods**:
- `extract_structured(chunk_text, chunk_id)` - Main extraction method
- `_generate_response(prompt)` - Generate response from Meta LLaMA
- `_extract_json_from_response(response)` - Parse JSON from response

### 1.3 Temporal Normalization Layer (`phase1_3_temporal_normalization.py`)

**Purpose**: Convert local time mentions into a GLOBAL STORY TIMELINE.

**Specifications**:
- Builds story-time index (t_story)
- Creates happens_before / happens_after edges
- Chapter-aware ordering
- Represented as a TEMPORAL DAG

**Output Schema**: `TemporalEvent {event_id, t_story, chapter_id, precedes, original_chunk_id}`

**Key Methods**:
- `normalize_temporal(extractions, chunks)` - Main normalization method
- `validate_temporal_consistency()` - Check for cycles in DAG

### 1.4 Causal Graph Construction (`phase1_4_causal_graph.py`)

**Purpose**: Build explicit CAUSE → EFFECT chains.

**Rules**:
- Only causal links supported by extracted evidence
- No speculative causality
- Temporal validation (cause must happen before effect)

**Output Schema**: `CausalEdge {cause_event_id, effect_event_id, evidence_chunk_id, confidence, evidence_type}`

**Key Methods**:
- `build_causal_graph(extractions, temporal_timeline)` - Main construction method
- `_resolve_transitive_causality(temporal_timeline)` - Resolve transitive chains
- `get_causal_chain(start_event_id)` - Get causal chain from event

### 1.5 Thematic Coherence Layer (`phase1_5_thematic_coherence.py`)

**Purpose**: Model themes as continuous vectors and track evolution over time.

**Specifications**:
- Theme vector dimension: 32–64 (default: 64)
- Every entity and event has a theme vector
- Track theme evolution over time
- Detect soft contradictions (theme_distance < epsilon)

**Output Schema**: `ThematicState {subject_id, theme_vector, temporal_scope, dominant_themes}`

**Key Methods**:
- `build_thematic_coherence(extractions, temporal_timeline, entities, events)` - Main method
- `get_theme_distance(subject_id1, subject_id2)` - Get theme distance
- `has_soft_contradiction(subject_id1, subject_id2)` - Check for soft contradiction

### 1.6 Knowledge Graph Assembly (`phase1_6_knowledge_graph_assembly.py`)

**Purpose**: Assemble complete knowledge graph from all extracted components.

**Node Types**:
- Character
- Event
- Location
- Theme
- Object

**Edge Types**:
- participates_in
- happens_before
- causes
- affects_theme
- contradicts_soft
- related_to

**Output Schema**: `KnowledgeGraph {nodes, edges}`

**Key Methods**:
- `assemble_graph(extractions, temporal_timeline, causal_edges, thematic_states, entities, events)` - Main assembly method
- `get_neighbors(node_id, edge_type)` - Get neighboring nodes
- `get_graph_stats()` - Get graph statistics

### 1.7 Community Detection (`phase1_7_community_detection.py`)

**Purpose**: Apply hierarchical clustering (Leiden) to detect communities.

**Specifications**:
- Uses Leiden algorithm
- Hierarchical levels: 0 (novel), 1 (arcs), 2 (subplots), 3 (scenes)
- Different resolutions for different levels

**Output Schema**: Dictionary mapping level to `{node_id: community_id}`

**Key Methods**:
- `detect_communities(graph, levels)` - Main detection method
- `get_hierarchical_communities()` - Get complete hierarchy
- `_knowledge_graph_to_igraph(graph)` - Convert to igraph format

### 1.8 Community Summarization (`phase1_8_community_summarization.py`)

**Purpose**: Summarize each community using Meta LLaMA.

**Specifications**:
- Preserve temporal span
- Preserve dominant themes
- Preserve causal structure

**Output Schema**: `CommunitySummary {community_id, level, summary_text, temporal_span, dominant_themes, node_ids, causal_structure_summary}`

**Key Methods**:
- `summarize_community(community_id, level, node_ids, graph, temporal_timeline, thematic_states)` - Summarize single community
- `summarize_all_communities(communities, graph, temporal_timeline, thematic_states)` - Summarize all communities

### 1.9 Pathway Storage (`phase1_9_pathway_storage.py`)

**Purpose**: Store embeddings and metadata in Pathway.

**Specifications**:
- Embeddings: 768-dim
- Metadata MUST include: chapter_id, time_range, themes, community_id
- Stores: Entities, Events, Community summaries

**Output Schema**: `PathwayEntity`, `PathwayEvent`, `PathwayCommunitySummary`

**Key Methods**:
- `store_entity(entity, graph_node, thematic_state, temporal_timeline, community_mapping)` - Store entity
- `store_event(event, graph_node, temporal_event, thematic_state, community_mapping)` - Store event
- `store_community_summary(community_summary)` - Store community summary
- `save_to_disk()` - Save to disk
- `load_from_disk()` - Load from disk

## Phase 2: Online Query Processing

### 2.1 Query Encoding (`phase2_1_query_encoding.py`)

**Purpose**: Encode query using Meta LLaMA embedder.

**Output**: `query_embedding ∈ ℝ^768`

**Key Methods**:
- `encode_query(query)` - Encode query to 768-dim vector

### 2.2 Community Selection (`phase2_2_community_selection.py`)

**Purpose**: Select relevant communities.

**Criteria**:
- Vector similarity (Pathway)
- Temporal filters
- Thematic alignment

**Critical**: DO NOT retrieve raw text.

**Key Methods**:
- `select_communities(query_embedding, community_summaries, temporal_filter, thematic_filter, top_k)` - Select communities

### 2.3 Map Step (`phase2_3_map_step.py`)

**Purpose**: Generate partial answers for each selected community.

**Output Schema**: `PartialAnswer {community_id, text, relevance_score, temporal_coverage, thematic_alignment_score, supporting_event_ids, supporting_entity_ids}`

**Key Methods**:
- `generate_partial_answer(query, community_summary, initial_relevance_score)` - Generate partial answer
- `map_all_communities(query, selected_communities, community_summaries)` - Map all communities

### 2.4 Soft Contradiction Resolution (`phase2_4_soft_contradiction_resolution.py`)

**Purpose**: Resolve conflicts between partial answers.

**Rules**:
- Prefer temporal consistency
- Prefer thematic continuity
- Reject answers violating story time

**Key Methods**:
- `resolve_contradictions(partial_answers, temporal_timeline)` - Resolve contradictions
- `_has_contradiction(answer1, answer2, temporal_timeline)` - Check for contradiction
- `_should_reject(answer1, answer2, temporal_timeline)` - Determine if answer should be rejected

### 2.5 Reduce Step (`phase2_5_reduce_step.py`)

**Purpose**: Merge top-K partial answers.

**Enforce**:
- Temporal order
- Causal consistency
- Thematic coherence

**Key Methods**:
- `reduce_partial_answers(partial_answers, top_k)` - Reduce answers
- `merge_answers(partial_answers)` - Merge into single text

### 2.6 Meta LLaMA Decoder (`phase2_6_meta_llama_decoder.py`)

**Purpose**: Generate final answer from reduced partial answers.

**Critical**: Generate ONLY from:
- Reduced partial answers
- Graph-verified facts

**Output Schema**: `FinalAnswer {answer_text, citations, temporal_span, communities_used, confidence}`

**Key Methods**:
- `generate_final_answer(query, reduced_answers)` - Generate final answer
- `_extract_citations(reduced_answers)` - Extract citations

## Main Orchestration Scripts

### `offline_graph_construction.py`

Main script for Phase 1 execution.

**Usage**:
```bash
python offline_graph_construction.py --novel_path <path_to_novel.txt>
```

**Process**:
1. Ingests novel
2. Extracts structured information
3. Builds temporal timeline
4. Constructs causal graph
5. Builds thematic coherence
6. Assembles knowledge graph
7. Detects communities
8. Summarizes communities
9. Stores in Pathway

### `online_query_processing.py`

Main script for Phase 2 execution.

**Usage**:
```bash
python online_query_processing.py --query "Your question about the novel"
```

**Optional Flags**:
- `--temporal_filter START END` - Filter by temporal range
- `--thematic_filter THEME_ID1 THEME_ID2 ...` - Filter by themes

**Process**:
1. Encodes query
2. Selects communities
3. Maps to partial answers
4. Resolves contradictions
5. Reduces answers
6. Generates final answer

## Data Schemas

All schemas are defined in `schemas.py` using Pydantic for validation.

**Key Schemas**:
- `Chunk` - Text chunks
- `StructuredExtraction` - Extracted information
- `TemporalEvent` - Temporal events
- `CausalEdge` - Causal relationships
- `ThematicState` - Thematic states
- `KnowledgeGraph` - Complete graph
- `CommunitySummary` - Community summaries
- `PathwayEntity`, `PathwayEvent`, `PathwayCommunitySummary` - Pathway storage
- `PartialAnswer` - Partial answers
- `FinalAnswer` - Final answers

## Configuration

All configuration is in `config.py`.

**Key Parameters**:
- `CHUNK_SIZE_TOKENS = 600`
- `CHUNK_OVERLAP_TOKENS = 120`
- `THEME_VECTOR_DIMENSION = 64`
- `SOFT_CONTRADICTION_EPSILON = 0.3`
- `TOP_K_COMMUNITIES = 5`
- `TOP_K_PARTIAL_ANSWERS = 3`
- `EMBEDDING_DIMENSION = 768`

## Hard Constraints (Non-Negotiable)

1. **Separate OFFLINE GRAPH CONSTRUCTION from ONLINE QUERY PROCESSING**
2. **No retrieval-only RAG**
3. **No answering from raw text**
4. **No answering from vector search alone**
5. **All answers must pass temporal + thematic consistency checks**
6. **All intermediate representations must be explicit and inspectable**

## File Structure

```
kbs-website/
├── schemas.py                          # All data schemas
├── config.py                           # Configuration
├── phase1_1_input_ingestion.py        # Phase 1.1
├── phase1_2_meta_llama_encoder.py     # Phase 1.2
├── phase1_3_temporal_normalization.py # Phase 1.3
├── phase1_4_causal_graph.py           # Phase 1.4
├── phase1_5_thematic_coherence.py     # Phase 1.5
├── phase1_6_knowledge_graph_assembly.py # Phase 1.6
├── phase1_7_community_detection.py    # Phase 1.7
├── phase1_8_community_summarization.py # Phase 1.8
├── phase1_9_pathway_storage.py        # Phase 1.9
├── phase2_1_query_encoding.py         # Phase 2.1
├── phase2_2_community_selection.py    # Phase 2.2
├── phase2_3_map_step.py               # Phase 2.3
├── phase2_4_soft_contradiction_resolution.py # Phase 2.4
├── phase2_5_reduce_step.py            # Phase 2.5
├── phase2_6_meta_llama_decoder.py     # Phase 2.6
├── offline_graph_construction.py      # Phase 1 orchestration
├── online_query_processing.py         # Phase 2 orchestration
├── requirements.txt                    # Dependencies
├── README.md                           # User guide
└── ARCHITECTURE.md                     # This file
```

## Implementation Notes

1. **Modular Structure**: One file per layer as specified
2. **Explicit Schemas**: All data structures are explicitly defined
3. **No Hidden State**: All intermediate representations are inspectable
4. **Documented Assumptions**: All assumptions are documented in code
5. **Error Handling**: Each layer handles errors gracefully
6. **Validation**: Temporal and causal consistency checks throughout

## Next Steps

1. Run offline graph construction on a novel
2. Test query processing with sample queries
3. Fine-tune parameters in `config.py`
4. Optimize Meta LLaMA model usage
5. Add more sophisticated temporal parsing
6. Enhance thematic coherence detection

