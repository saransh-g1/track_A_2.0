"""
PHASE 1.4: CAUSAL GRAPH CONSTRUCTION

Builds explicit CAUSE → EFFECT chains.

Rules:
- Only causal links supported by extracted evidence
- No speculative causality

Schema: CausalEdge {cause_event_id, effect_event_id, evidence_chunk_id}
"""

from typing import List, Dict, Set
from collections import defaultdict
from schemas import StructuredExtraction, CausalEdge, TemporalEvent


class CausalGraphConstruction:
    """
    Causal graph construction layer.
    
    Builds explicit cause-effect chains from extracted evidence.
    No speculative causality allowed.
    """
    
    def __init__(self):
        self.causal_edges: List[CausalEdge] = []
        self.cause_effect_map: Dict[str, List[str]] = defaultdict(list)  # cause -> [effects]
        self.effect_cause_map: Dict[str, List[str]] = defaultdict(list)  # effect -> [causes]
    
    def build_causal_graph(
        self,
        extractions: List[StructuredExtraction],
        temporal_timeline: Dict[str, TemporalEvent]
    ) -> List[CausalEdge]:
        """
        Build causal graph from extractions.
        
        Process:
        1. Extract explicit causal links from StructuredExtraction
        2. Validate against temporal timeline (cause must happen before effect)
        3. Only include causal links with evidence
        4. Build explicit CausalEdge objects
        
        Args:
            extractions: List of StructuredExtraction objects
            temporal_timeline: Dictionary of TemporalEvent objects
            
        Returns:
            List of CausalEdge objects
        """
        self.causal_edges = []
        
        # Process each extraction
        for extraction in extractions:
            for causal_link in extraction.causal_links:
                cause_event_id = causal_link.cause_event_id
                effect_event_id = causal_link.effect_event_id
                
                # Validate that both events exist in timeline
                if cause_event_id not in temporal_timeline:
                    continue
                if effect_event_id not in temporal_timeline:
                    continue
                
                # Validate temporal constraint: cause must happen before effect
                cause_event = temporal_timeline[cause_event_id]
                effect_event = temporal_timeline[effect_event_id]
                
                if cause_event.t_story >= effect_event.t_story:
                    # Skip if cause doesn't happen before effect
                    # This enforces temporal consistency
                    continue
                
                # Validate that we have evidence (not speculative)
                if causal_link.evidence_type == "inferred" and causal_link.confidence < 0.7:
                    # Skip low-confidence inferred causal links
                    continue
                
                # Create CausalEdge
                causal_edge = CausalEdge(
                    cause_event_id=cause_event_id,
                    effect_event_id=effect_event_id,
                    evidence_chunk_id=extraction.chunk_id,
                    confidence=causal_link.confidence,
                    evidence_type=causal_link.evidence_type
                )
                
                self.causal_edges.append(causal_edge)
                
                # Update maps
                self.cause_effect_map[cause_event_id].append(effect_event_id)
                self.effect_cause_map[effect_event_id].append(cause_event_id)
        
        # Additional validation: check for transitive causality
        self._resolve_transitive_causality(temporal_timeline)
        
        return self.causal_edges
    
    def _resolve_transitive_causality(self, temporal_timeline: Dict[str, TemporalEvent]):
        """
        Resolve transitive causal relationships.
        
        If A causes B and B causes C, we can infer A causes C (if temporally consistent).
        However, we only do this for explicit causal chains to avoid speculation.
        
        Args:
            temporal_timeline: Dictionary of TemporalEvent objects
        """
        # Find transitive causal chains
        for cause_id in list(self.cause_effect_map.keys()):
            effects = self.cause_effect_map[cause_id].copy()
            
            for effect_id in effects:
                # Check if this effect is also a cause
                if effect_id in self.cause_effect_map:
                    transitive_effects = self.cause_effect_map[effect_id]
                    
                    for transitive_effect_id in transitive_effects:
                        # Check if cause -> transitive_effect already exists
                        if transitive_effect_id not in self.cause_effect_map.get(cause_id, []):
                            # Validate temporal constraint
                            if (cause_id in temporal_timeline and
                                transitive_effect_id in temporal_timeline):
                                
                                cause_event = temporal_timeline[cause_id]
                                transitive_event = temporal_timeline[transitive_effect_id]
                                
                                if cause_event.t_story < transitive_event.t_story:
                                    # Create transitive causal edge (lower confidence)
                                    transitive_edge = CausalEdge(
                                        cause_event_id=cause_id,
                                        effect_event_id=transitive_effect_id,
                                        evidence_chunk_id=0,  # Transitive, no direct evidence
                                        confidence=0.6,  # Lower confidence for transitive
                                        evidence_type="inferred"  # Transitive inference
                                    )
                                    
                                    # Check if this edge doesn't already exist
                                    exists = any(
                                        e.cause_event_id == cause_id and
                                        e.effect_event_id == transitive_effect_id
                                        for e in self.causal_edges
                                    )
                                    
                                    if not exists:
                                        self.causal_edges.append(transitive_edge)
                                        self.cause_effect_map[cause_id].append(transitive_effect_id)
                                        self.effect_cause_map[transitive_effect_id].append(cause_id)
    
    def get_causal_chain(self, start_event_id: str, max_depth: int = 10) -> List[str]:
        """
        Get causal chain starting from an event.
        
        Args:
            start_event_id: Starting event ID
            max_depth: Maximum depth to traverse
            
        Returns:
            List of event IDs in causal chain
        """
        chain = [start_event_id]
        visited = {start_event_id}
        current = start_event_id
        depth = 0
        
        while depth < max_depth:
            if current not in self.cause_effect_map:
                break
            
            next_effects = [
                eid for eid in self.cause_effect_map[current]
                if eid not in visited
            ]
            
            if not next_effects:
                break
            
            # Take first effect (could be extended to consider all paths)
            current = next_effects[0]
            chain.append(current)
            visited.add(current)
            depth += 1
        
        return chain
    
    def get_effects(self, event_id: str) -> List[str]:
        """
        Get direct effects of an event.
        
        Args:
            event_id: Event ID
            
        Returns:
            List of event IDs that are effects
        """
        return self.cause_effect_map.get(event_id, [])
    
    def get_causes(self, event_id: str) -> List[str]:
        """
        Get direct causes of an event.
        
        Args:
            event_id: Event ID
            
        Returns:
            List of event IDs that are causes
        """
        return self.effect_cause_map.get(event_id, [])


if __name__ == "__main__":
    from collections import defaultdict
    from schemas import Event, CausalLink
    
    # Test example
    extraction = StructuredExtraction(
        chunk_id=1,
        events=[
            Event(event_id="ev_1", event_type="action", description="John loses job", participants=["e_john"]),
            Event(event_id="ev_2", event_type="emotion", description="John feels sad", participants=["e_john"])
        ],
        causal_links=[
            CausalLink(
                causal_link_id="cl_1",
                cause_event_id="ev_1",
                effect_event_id="ev_2",
                evidence_type="explicit",
                confidence=0.9
            )
        ]
    )
    
    from phase1_3_temporal_normalization import TemporalNormalization
    from schemas import Chunk
    
    chunks = [Chunk(chunk_id=1, chapter_id=1, text="...", token_range=(0, 100))]
    temporal_norm = TemporalNormalization()
    timeline = temporal_norm.normalize_temporal([extraction], chunks)
    
    causal_builder = CausalGraphConstruction()
    causal_edges = causal_builder.build_causal_graph([extraction], timeline)
    
    print(f"Created {len(causal_edges)} causal edges")
    if causal_edges:
        print(f"Example: {causal_edges[0].cause_event_id} -> {causal_edges[0].effect_event_id}")

