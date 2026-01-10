"""
Configuration for Track-A GraphRAG System.
All assumptions and parameters are documented here.
"""

# Model Configuration
# Using Llama-3-8B-Instruct instead - more stable and works without rope_scaling issues
META_LLAMA_MODEL_NAME = "meta-llama/Llama-3-8B-Instruct"  # Llama 3 8B Instruct (stable)
EMBEDDING_DIMENSION = 4096  # Llama 3 uses 4096-dim embeddings
EMAX_CONTEXT_LENGTH = 8192  # Llama 3 supports 8K context

# Chunking Configuration (Phase 1.1)
CHUNK_SIZE_TOKENS = 600
CHUNK_OVERLAP_TOKENS = 120

# Thematic Coherence Configuration (Phase 1.5)
THEME_VECTOR_DIMENSION = 64  # Between 32-64 as specified
SOFT_CONTRADICTION_EPSILON = 0.3  # Distance threshold for soft contradictions

# Community Detection Configuration (Phase 1.7)
LEIDEN_RESOLUTION = 1.0  # Default resolution for Leiden algorithm
COMMUNITY_LEVELS = [0, 1, 2, 3]  # 0: novel, 1: arcs, 2: subplots, 3: scenes

# Query Processing Configuration (Phase 2)
TOP_K_COMMUNITIES = 5  # Number of communities to select
TOP_K_PARTIAL_ANSWERS = 3  # Number of partial answers to reduce
MIN_RELEVANCE_SCORE = 50.0  # Minimum relevance score for partial answers

# Paths
GRAPH_STORAGE_PATH = "./graph_storage"
PATHWAY_STORAGE_PATH = "./pathway_storage"

# Extraction Prompt Templates
STRUCTURED_EXTRACTION_PROMPT_TEMPLATE = """You are a structured information extractor for novel analysis.

Extract information from the following text chunk following this STRICT JSON schema:

{{
  "entities": [
    {{
      "entity_id": "e_<unique_id>",
      "entity_type": "character|location|object",
      "name": "...",
      "description": "...",
      "attributes": {{}}
    }}
  ],
  "events": [
    {{
      "event_id": "ev_<unique_id>",
      "event_type": "...",
      "description": "...",
      "participants": ["e_..."],
      "location": "e_..."
    }}
  ],
  "relations": [
    {{
      "relation_id": "r_<unique_id>",
      "source_entity_id": "e_...",
      "target_entity_id": "e_...",
      "relation_type": "...",
      "strength": 0.0-1.0
    }}
  ],
  "claims": [
    {{
      "claim_id": "c_<unique_id>",
      "subject": "e_...",
      "predicate": "...",
      "object": "...",
      "certainty": 0.0-1.0
    }}
  ],
  "themes": [
    {{
      "theme_id": "th_<unique_id>",
      "theme_name": "...",
      "description": "...",
      "intensity": 0.0-1.0
    }}
  ],
  "temporal_markers": [
    {{
      "marker_id": "tm_<unique_id>",
      "marker_type": "absolute|relative|sequence",
      "text": "...",
      "reference_event_id": "ev_...",
      "time_value": "..."
    }}
  ],
  "causal_links": [
    {{
      "causal_link_id": "cl_<unique_id>",
      "cause_event_id": "ev_...",
      "effect_event_id": "ev_...",
      "evidence_type": "explicit|implicit|inferred",
      "confidence": 0.0-1.0
    }}
  ]
}}

Text chunk:
{chunk_text}

Return ONLY valid JSON, no additional text."""

