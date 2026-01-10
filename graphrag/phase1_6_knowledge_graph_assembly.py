"""
PHASE 1.6: KNOWLEDGE GRAPH ASSEMBLY

Assembles complete knowledge graph from all extracted components.

Node types:
- Character
- Event
- Location
- Theme
- Object

Edge types:
- participates_in
- happens_before
- causes
- affects_theme
- contradicts_soft
"""

from typing import List, Dict, Set
from collections import defaultdict
from schemas import (
    KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge,
    NodeType, EdgeType, StructuredExtraction, Entity, Event,
    Theme, Relation, CausalEdge, TemporalEvent, ThematicState
)


class KnowledgeGraphAssembly:
    """
    Knowledge graph assembly layer.
    
    Combines all extracted information into a unified knowledge graph.
    Graph is queryable independently of vectors.
    """
    
    def __init__(self):
        self.graph = KnowledgeGraph()
    
    def assemble_graph(
        self,
        extractions: List[StructuredExtraction],
        temporal_timeline: Dict[str, TemporalEvent],
        causal_edges: List[CausalEdge],
        thematic_states: Dict[str, ThematicState],
        entities: Dict[str, Entity],
        events: Dict[str, Event]
    ) -> KnowledgeGraph:
        """
        Assemble complete knowledge graph.
        
        Process:
        1. Create nodes for all entities, events, themes, locations, objects
        2. Create edges: participates_in, happens_before, causes, affects_theme, contradicts_soft
        3. Attach theme vectors and temporal ranges to nodes
        
        Args:
            extractions: List of StructuredExtraction objects
            temporal_timeline: Dictionary of TemporalEvent objects
            causal_edges: List of CausalEdge objects
            thematic_states: Dictionary of ThematicState objects
            entities: Dictionary of Entity objects
            events: Dictionary of Event objects
            
        Returns:
            Complete KnowledgeGraph object
        """
        self.graph = KnowledgeGraph()
        
        # Step 1: Create nodes
        self._create_nodes(extractions, temporal_timeline, thematic_states, entities, events)
        
        # Step 2: Create edges
        self._create_edges(extractions, temporal_timeline, causal_edges, thematic_states)
        
        return self.graph
    
    def _create_nodes(
        self,
        extractions: List[StructuredExtraction],
        temporal_timeline: Dict[str, TemporalEvent],
        thematic_states: Dict[str, ThematicState],
        entities: Dict[str, Entity],
        events: Dict[str, Event]
    ):
        """Create all nodes in the knowledge graph."""
        
        # Collect all themes and locations
        all_themes: Dict[str, Theme] = {}
        all_locations: Set[str] = set()
        
        for extraction in extractions:
            for theme in extraction.themes:
                all_themes[theme.theme_id] = theme
            
            for event in extraction.events:
                if event.location:
                    all_locations.add(event.location)
        
        # Create entity nodes
        for entity_id, entity in entities.items():
            node_type = NodeType.CHARACTER if entity.entity_type == "character" else NodeType.OBJECT
            
            # Get theme vector and temporal range
            theme_vector = None
            temporal_range = None
            if entity_id in thematic_states:
                state = thematic_states[entity_id]
                theme_vector = state.theme_vector
                temporal_range = state.temporal_scope
            
            node = KnowledgeGraphNode(
                node_id=entity_id,
                node_type=node_type,
                properties={
                    "name": entity.name,
                    "entity_type": entity.entity_type,
                    "description": entity.description or "",
                    "attributes": entity.attributes
                },
                theme_vector=theme_vector,
                temporal_range=temporal_range
            )
            self.graph.nodes[entity_id] = node
        
        # Create event nodes
        for event_id, event in events.items():
            # Get temporal information
            temporal_range = None
            if event_id in temporal_timeline:
                te = temporal_timeline[event_id]
                temporal_range = (te.t_story, te.t_story)
            
            # Get theme vector
            theme_vector = None
            if event_id in thematic_states:
                state = thematic_states[event_id]
                theme_vector = state.theme_vector
                if temporal_range is None:
                    temporal_range = state.temporal_scope
            
            node = KnowledgeGraphNode(
                node_id=event_id,
                node_type=NodeType.EVENT,
                properties={
                    "event_type": event.event_type,
                    "description": event.description,
                    "participants": event.participants,
                    "location": event.location or ""
                },
                theme_vector=theme_vector,
                temporal_range=temporal_range
            )
            self.graph.nodes[event_id] = node
        
        # Create theme nodes
        for theme_id, theme in all_themes.items():
            # Theme vectors come from thematic states
            theme_vector = None
            if theme_id in thematic_states:
                state = thematic_states[theme_id]
                theme_vector = state.theme_vector
            
            node = KnowledgeGraphNode(
                node_id=theme_id,
                node_type=NodeType.THEME,
                properties={
                    "theme_name": theme.theme_name,
                    "description": theme.description,
                    "intensity": theme.intensity
                },
                theme_vector=theme_vector
            )
            self.graph.nodes[theme_id] = node
        
        # Create location nodes (as entities of type location)
        for location_id in all_locations:
            # Check if location is already an entity
            if location_id not in self.graph.nodes:
                node = KnowledgeGraphNode(
                    node_id=location_id,
                    node_type=NodeType.LOCATION,
                    properties={
                        "name": location_id,
                        "entity_type": "location"
                    }
                )
                self.graph.nodes[location_id] = node
    
    def _create_edges(
        self,
        extractions: List[StructuredExtraction],
        temporal_timeline: Dict[str, TemporalEvent],
        causal_edges: List[CausalEdge],
        thematic_states: Dict[str, ThematicState]
    ):
        """Create all edges in the knowledge graph."""
        edge_id_counter = 1
        
        # participates_in edges (entities participate in events)
        for extraction in extractions:
            for event in extraction.events:
                for participant_id in event.participants:
                    if participant_id in self.graph.nodes and event.event_id in self.graph.nodes:
                        edge = KnowledgeGraphEdge(
                            edge_id=f"e_{edge_id_counter}",
                            source_id=participant_id,
                            target_id=event.event_id,
                            edge_type=EdgeType.PARTICIPATES_IN,
                            weight=1.0
                        )
                        self.graph.edges.append(edge)
                        edge_id_counter += 1
        
        # happens_before edges (from temporal timeline)
        for event_id, temporal_event in temporal_timeline.items():
            if event_id in self.graph.nodes:
                for precedes_id in temporal_event.precedes:
                    if precedes_id in self.graph.nodes:
                        edge = KnowledgeGraphEdge(
                            edge_id=f"e_{edge_id_counter}",
                            source_id=event_id,
                            target_id=precedes_id,
                            edge_type=EdgeType.HAPPENS_BEFORE,
                            weight=1.0,
                            properties={
                                "chapter_id": temporal_event.chapter_id,
                                "t_story_diff": temporal_timeline[precedes_id].t_story - temporal_event.t_story
                            }
                        )
                        self.graph.edges.append(edge)
                        edge_id_counter += 1
        
        # causes edges (from causal graph)
        for causal_edge in causal_edges:
            if (causal_edge.cause_event_id in self.graph.nodes and
                causal_edge.effect_event_id in self.graph.nodes):
                edge = KnowledgeGraphEdge(
                    edge_id=f"e_{edge_id_counter}",
                    source_id=causal_edge.cause_event_id,
                    target_id=causal_edge.effect_event_id,
                    edge_type=EdgeType.CAUSES,
                    weight=causal_edge.confidence,
                    properties={
                        "evidence_chunk_id": causal_edge.evidence_chunk_id,
                        "evidence_type": causal_edge.evidence_type
                    }
                )
                self.graph.edges.append(edge)
                edge_id_counter += 1
        
        # affects_theme edges (entities/events associated with themes)
        for extraction in extractions:
            for entity in extraction.entities:
                for theme in extraction.themes:
                    if entity.entity_id in self.graph.nodes and theme.theme_id in self.graph.nodes:
                        edge = KnowledgeGraphEdge(
                            edge_id=f"e_{edge_id_counter}",
                            source_id=entity.entity_id,
                            target_id=theme.theme_id,
                            edge_type=EdgeType.AFFECTS_THEME,
                            weight=theme.intensity,
                            properties={"chunk_id": extraction.chunk_id}
                        )
                        self.graph.edges.append(edge)
                        edge_id_counter += 1
            
            for event in extraction.events:
                for theme in extraction.themes:
                    if event.event_id in self.graph.nodes and theme.theme_id in self.graph.nodes:
                        edge = KnowledgeGraphEdge(
                            edge_id=f"e_{edge_id_counter}",
                            source_id=event.event_id,
                            target_id=theme.theme_id,
                            edge_type=EdgeType.AFFECTS_THEME,
                            weight=theme.intensity,
                            properties={"chunk_id": extraction.chunk_id}
                        )
                        self.graph.edges.append(edge)
                        edge_id_counter += 1
        
        # contradicts_soft edges (from thematic coherence)
        # This would come from soft contradiction detection
        # For now, we'll add these in a separate pass if thematic_states provides this
        
        # related_to edges (from relations)
        for extraction in extractions:
            for relation in extraction.relations:
                if (relation.source_entity_id in self.graph.nodes and
                    relation.target_entity_id in self.graph.nodes):
                    edge = KnowledgeGraphEdge(
                        edge_id=f"e_{edge_id_counter}",
                        source_id=relation.source_entity_id,
                        target_id=relation.target_entity_id,
                        edge_type=EdgeType.RELATED_TO,
                        weight=relation.strength or 0.5,
                        properties={
                            "relation_type": relation.relation_type,
                            "chunk_id": extraction.chunk_id
                        }
                    )
                    self.graph.edges.append(edge)
                    edge_id_counter += 1
    
    def get_neighbors(self, node_id: str, edge_type: EdgeType = None) -> List[str]:
        """
        Get neighboring nodes.
        
        Args:
            node_id: Node ID
            edge_type: Optional edge type filter
            
        Returns:
            List of neighbor node IDs
        """
        neighbors = []
        for edge in self.graph.edges:
            if edge.source_id == node_id:
                if edge_type is None or edge.edge_type == edge_type:
                    neighbors.append(edge.target_id)
            elif edge.target_id == node_id:
                if edge_type is None or edge.edge_type == edge_type:
                    neighbors.append(edge.source_id)
        return neighbors
    
    def get_graph_stats(self) -> Dict[str, int]:
        """
        Get graph statistics.
        
        Returns:
            Dictionary with node counts by type and edge counts
        """
        stats = {
            "total_nodes": len(self.graph.nodes),
            "total_edges": len(self.graph.edges)
        }
        
        node_type_counts = defaultdict(int)
        for node in self.graph.nodes.values():
            node_type_counts[node.node_type.value] += 1
        
        stats.update(node_type_counts)
        
        edge_type_counts = defaultdict(int)
        for edge in self.graph.edges:
            edge_type_counts[edge.edge_type.value] += 1
        
        stats.update(edge_type_counts)
        
        return stats


if __name__ == "__main__":
    # Test example
    from schemas import StructuredExtraction, Entity, Event, Theme, Chunk
    from phase1_3_temporal_normalization import TemporalNormalization
    from phase1_4_causal_graph import CausalGraphConstruction
    from phase1_5_thematic_coherence import ThematicCoherence
    
    entity = Entity(entity_id="e_1", entity_type="character", name="John")
    event = Event(event_id="ev_1", event_type="action", description="John walks", participants=["e_1"])
    theme = Theme(theme_id="th_1", theme_name="journey", description="...", intensity=0.8)
    
    extraction = StructuredExtraction(
        chunk_id=1,
        entities=[entity],
        events=[event],
        themes=[theme]
    )
    
    chunks = [Chunk(chunk_id=1, chapter_id=1, text="...", token_range=(0, 100))]
    entities_dict = {entity.entity_id: entity}
    events_dict = {event.event_id: event}
    
    temporal_norm = TemporalNormalization()
    timeline = temporal_norm.normalize_temporal([extraction], chunks)
    
    causal_builder = CausalGraphConstruction()
    causal_edges = causal_builder.build_causal_graph([extraction], timeline)
    
    coherence = ThematicCoherence()
    thematic_states = coherence.build_thematic_coherence(
        [extraction],
        timeline,
        entities_dict,
        events_dict
    )
    
    assembler = KnowledgeGraphAssembly()
    graph = assembler.assemble_graph(
        [extraction],
        timeline,
        causal_edges,
        thematic_states,
        entities_dict,
        events_dict
    )
    
    stats = assembler.get_graph_stats()
    print(f"Graph stats: {stats}")

