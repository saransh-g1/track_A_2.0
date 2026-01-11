"""
Configuration for Track-A GraphRAG System.
All assumptions and parameters are documented here.
"""

import os
from pathlib import Path

# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'

# Create .env file if it doesn't exist with default model path
if not env_path.exists():
    print(f"[Config] .env file not found at: {env_path}")
    print("[Config] Creating .env file with default model path...")
    default_model_path = "/home/rs/24CS91R03/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
    env_content = f"""# Model Path Configuration
# Set the direct path to your LLaMA model snapshot directory
MODEL_PATH={default_model_path}
"""
    try:
        with open(env_path, 'w') as f:
            f.write(env_content)
        print(f"[Config] Created .env file at: {env_path}")
        print(f"[Config] Default MODEL_PATH: {default_model_path}")
        print("[Config] You can edit .env to change the model path if needed.")
    except Exception as e:
        print(f"[Config] ERROR: Could not create .env file: {e}")
        print(f"[Config] Please manually create .env file at: {env_path}")
        print(f"[Config] With content: MODEL_PATH={default_model_path}")

# Load .env file
try:
    from dotenv import load_dotenv
    # Load .env file from the same directory as this config file
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
        print(f"[Config] Loaded .env file from: {env_path}")
    else:
        print(f"[Config] WARNING: .env file still not found at: {env_path}")
except ImportError:
    # If python-dotenv is not installed, try to load manually
    if env_path.exists():
        print(f"[Config] Loading .env file manually from: {env_path}")
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
                    print(f"[Config] Loaded: {key.strip()}")
    else:
        print(f"[Config] WARNING: .env file not found at: {env_path}")

# Model Configuration
# Load model path from .env file, fallback to default if not set
MODEL_PATH = os.getenv("MODEL_PATH", None)  # Direct path to model snapshot directory

# Debug: Print MODEL_PATH status
if MODEL_PATH:
    print(f"[Config] MODEL_PATH loaded from .env: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"[Config] WARNING: MODEL_PATH does not exist: {MODEL_PATH}")
else:
    print("[Config] WARNING: MODEL_PATH not found in .env file. Will use HuggingFace repo (may download).")
    print(f"[Config] .env file location: {env_path}")
    print(f"[Config] .env file exists: {env_path.exists() if 'env_path' in locals() else 'N/A'}")

META_LLAMA_MODEL_NAME = MODEL_PATH if MODEL_PATH else "meta-llama/Meta-Llama-3-8B-Instruct"  # Use direct path or HuggingFace repo name
EMBEDDING_DIMENSION = 4096  # Llama 3 uses 4096-dim embeddings
EMAX_CONTEXT_LENGTH = 8192  # Llama 3 supports 8K context
MAX_EXTRACTION_TOKENS = 768  # Max tokens for structured extraction (increased from 256 to prevent JSON truncation)
EXTRACTION_BATCH_SIZE = 2  # Batch size for parallel chunk processing (2-4 recommended)

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

# Extraction Prompt Templates (OPTIMIZED for Track-A: claims, temporal markers, entities only)
STRUCTURED_EXTRACTION_PROMPT_TEMPLATE = """You are a structured information extractor for novel analysis.

Extract ONLY the following from the text chunk (STRICT JSON schema):

{{
  "entities": [
    {{
      "entity_id": "e_<unique_id>",
      "entity_type": "character",
      "name": "character name",
      "description": "brief description of the character",
      "attributes": {{"role": "protagonist/antagonist/supporting", "status": "alive/deceased"}}
    }}
  ],
  "claims": [
    {{
      "claim_id": "c_<unique_id>",
      "subject": "e_<entity_id>",
      "predicate": "verb/action",
      "object": "what happened",
      "certainty": 0.9
    }}
  ],
  "temporal_markers": [
    {{
      "marker_id": "tm_<unique_id>",
      "marker_type": "absolute|relative|sequence",
      "text": "exact temporal phrase from text",
      "reference_event_id": "e_<event> or empty",
      "time_value": "extracted time value"
    }}
  ]
}}

Focus on:
- CHARACTERS only (no locations/objects)
- VERIFIABLE CLAIMS about actions and states
- EXPLICIT temporal references in the text

Text chunk:
{chunk_text}

Return ONLY valid JSON, no additional text."""

