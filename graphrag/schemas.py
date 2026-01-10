"""
Data schemas for Track-A GraphRAG System.
All intermediate representations are explicit and inspectable.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum


# ====================================================================
# PHASE 1.1: INPUT INGESTION & CHUNKING
# ====================================================================

class Chunk(BaseModel):
    """Chunk schema for chapter-aware text chunks."""
    chunk_id: int
    chapter_id: int
    text: str
    token_range: Tuple[int, int]  # [start, end]


# ====================================================================
# PHASE 1.2: META LLAMA ENCODER - STRUCTURED EXTRACTION
# ====================================================================

class Entity(BaseModel):
    """Entity extracted from chunk."""
    entity_id: str
    entity_type: str  # "character", "location", "object"
    name: str
    description: Optional[str] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)


class Event(BaseModel):
    """Event extracted from chunk."""
    event_id: str
    event_type: str
    description: str
    participants: List[str]  # entity_ids
    location: Optional[str] = None  # entity_id or location name


class Relation(BaseModel):
    """Relation between entities."""
    relation_id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    strength: Optional[float] = None  # 0-1


class Claim(BaseModel):
    """Verifiable fact extracted from chunk."""
    claim_id: str
    subject: str  # entity_id
    predicate: str
    object: str
    certainty: float = Field(default=1.0, ge=0.0, le=1.0)


class Theme(BaseModel):
    """Theme extracted from chunk."""
    theme_id: str
    theme_name: str
    description: str
    intensity: float = Field(default=0.5, ge=0.0, le=1.0)


class TemporalMarker(BaseModel):
    """Temporal marker extracted from chunk."""
    marker_id: str
    marker_type: str  # "absolute", "relative", "sequence"
    text: str
    reference_event_id: Optional[str] = None
    time_value: Optional[str] = None  # parsed time representation


class CausalLink(BaseModel):
    """Causal link extracted from chunk."""
    causal_link_id: str
    cause_event_id: str
    effect_event_id: str
    evidence_type: str  # "explicit", "implicit", "inferred"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class StructuredExtraction(BaseModel):
    """Complete structured extraction from a single chunk."""
    chunk_id: int
    entities: List[Entity] = Field(default_factory=list)
    events: List[Event] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)
    claims: List[Claim] = Field(default_factory=list)
    themes: List[Theme] = Field(default_factory=list)
    temporal_markers: List[TemporalMarker] = Field(default_factory=list)
    causal_links: List[CausalLink] = Field(default_factory=list)


# ====================================================================
# PHASE 1.3: TEMPORAL NORMALIZATION LAYER
# ====================================================================

class TemporalEvent(BaseModel):
    """Temporal event in global story timeline."""
    event_id: str
    t_story: int  # Global story time index
    chapter_id: int
    precedes: List[str] = Field(default_factory=list)  # event_ids that come after
    original_chunk_id: int


# ====================================================================
# PHASE 1.4: CAUSAL GRAPH CONSTRUCTION
# ====================================================================

class CausalEdge(BaseModel):
    """Explicit causal edge in the graph."""
    cause_event_id: str
    effect_event_id: str
    evidence_chunk_id: int
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_type: str  # "explicit", "implicit"


# ====================================================================
# PHASE 1.5: THEMATIC COHERENCE LAYER
# ====================================================================

class ThematicState(BaseModel):
    """Thematic state of a subject at a point in time."""
    subject_id: str  # entity_id or event_id
    theme_vector: List[float]  # 32-64 dimensions
    temporal_scope: Tuple[int, int]  # [start_t_story, end_t_story]
    dominant_themes: List[str] = Field(default_factory=list)  # theme_ids


# ====================================================================
# PHASE 1.6: KNOWLEDGE GRAPH ASSEMBLY
# ====================================================================

class NodeType(str, Enum):
    """Node types in the knowledge graph."""
    CHARACTER = "character"
    EVENT = "event"
    LOCATION = "location"
    THEME = "theme"
    OBJECT = "object"


class EdgeType(str, Enum):
    """Edge types in the knowledge graph."""
    PARTICIPATES_IN = "participates_in"
    HAPPENS_BEFORE = "happens_before"
    CAUSES = "causes"
    AFFECTS_THEME = "affects_theme"
    CONTRADICTS_SOFT = "contradicts_soft"
    RELATED_TO = "related_to"


class KnowledgeGraphNode(BaseModel):
    """Node in the knowledge graph."""
    node_id: str
    node_type: NodeType
    properties: Dict[str, Any] = Field(default_factory=dict)
    theme_vector: Optional[List[float]] = None
    temporal_range: Optional[Tuple[int, int]] = None  # [start_t_story, end_t_story]


class KnowledgeGraphEdge(BaseModel):
    """Edge in the knowledge graph."""
    edge_id: str
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    properties: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeGraph(BaseModel):
    """Complete knowledge graph."""
    nodes: Dict[str, KnowledgeGraphNode] = Field(default_factory=dict)
    edges: List[KnowledgeGraphEdge] = Field(default_factory=list)


# ====================================================================
# PHASE 1.7 & 1.8: COMMUNITY DETECTION & SUMMARIZATION
# ====================================================================

class CommunitySummary(BaseModel):
    """Summary of a detected community."""
    community_id: str
    level: int  # 0: entire novel, 1: major arcs, 2: subplots, 3: scene-level
    summary_text: str
    temporal_span: Tuple[int, int]  # [start_t_story, end_t_story]
    dominant_themes: List[str] = Field(default_factory=list)  # theme_ids
    node_ids: List[str] = Field(default_factory=list)  # nodes in this community
    causal_structure_summary: Optional[str] = None


# ====================================================================
# PHASE 1.9: PATHWAY STORAGE
# ====================================================================

class PathwayEntity(BaseModel):
    """Entity stored in Pathway."""
    entity_id: str
    embedding: List[float]  # 768-dim
    metadata: Dict[str, Any]  # Must include: chapter_id, time_range, themes, community_id
    text: Optional[str] = None


class PathwayEvent(BaseModel):
    """Event stored in Pathway."""
    event_id: str
    embedding: List[float]  # 768-dim
    metadata: Dict[str, Any]
    text: Optional[str] = None


class PathwayCommunitySummary(BaseModel):
    """Community summary stored in Pathway."""
    community_id: str
    embedding: List[float]  # 768-dim
    metadata: Dict[str, Any]
    summary_text: str


# ====================================================================
# PHASE 2: ONLINE QUERY PROCESSING
# ====================================================================

class PartialAnswer(BaseModel):
    """Partial answer from a single community."""
    community_id: str
    text: str
    relevance_score: float = Field(default=0.0, ge=0.0, le=100.0)
    temporal_coverage: Tuple[int, int]  # [start_t_story, end_t_story]
    thematic_alignment_score: float = Field(default=0.0, ge=0.0, le=1.0)
    supporting_event_ids: List[str] = Field(default_factory=list)
    supporting_entity_ids: List[str] = Field(default_factory=list)


class FinalAnswer(BaseModel):
    """Final answer after map-reduce."""
    answer_text: str
    citations: List[Dict[str, str]] = Field(default_factory=list)  # [{type: "event", id: "..."}, ...]
    temporal_span: Tuple[int, int]
    communities_used: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

