"""
PHASE 1: OFFLINE GRAPH CONSTRUCTION

Main orchestration script for offline graph knowledge construction.

Executes all Phase 1 steps:
1. Input Ingestion & Chunking
2. Meta LLaMA Encoder - Structured Extraction
3. Temporal Normalization Layer
4. Causal Graph Construction
5. Thematic Coherence Layer
6. Knowledge Graph Assembly
7. Community Detection
8. Community Summarization
9. Pathway Storage
"""

import argparse
import json
import os
from typing import Dict, List
from phase1_1_input_ingestion import InputIngestion
from phase1_2_meta_llama_encoder import MetaLlamaEncoder
from phase1_3_temporal_normalization import TemporalNormalization
from phase1_4_causal_graph import CausalGraphConstruction
from phase1_5_thematic_coherence import ThematicCoherence
from phase1_6_knowledge_graph_assembly import KnowledgeGraphAssembly
from phase1_7_community_detection import CommunityDetection
from phase1_8_community_summarization import CommunitySummarization
from phase1_9_pathway_storage import PathwayStorage
from schemas import Chunk, StructuredExtraction, TemporalEvent, CausalEdge, ThematicState, KnowledgeGraph, CommunitySummary
from config import GRAPH_STORAGE_PATH


class OfflineGraphConstruction:
    """
    Main orchestrator for offline graph construction.
    
    Executes all Phase 1 steps in sequence.
    """
    
    def __init__(self):
        self.chunks: List[Chunk] = []
        self.extractions: List[StructuredExtraction] = []
        self.temporal_timeline: Dict[str, TemporalEvent] = {}
        self.causal_edges: List[CausalEdge] = []
        self.thematic_states: Dict[str, ThematicState] = {}
        self.knowledge_graph: KnowledgeGraph = None
        self.communities: Dict[int, Dict[int, List[str]]] = {}
        self.community_summaries: List[CommunitySummary] = []
        
        # Initialize components
        self.ingestion = InputIngestion()
        self.encoder = MetaLlamaEncoder()
        self.temporal_norm = TemporalNormalization()
        self.causal_builder = CausalGraphConstruction()
        self.thematic_coherence = ThematicCoherence()
        self.graph_assembler = KnowledgeGraphAssembly()
        self.community_detector = CommunityDetection()
        self.summarizer = CommunitySummarization()
        self.pathway_storage = PathwayStorage()
    
    def build_graph(self, novel_path: str):
        """
        Build complete knowledge graph from novel.
        
        Args:
            novel_path: Path to novel text file
        """
        print("=" * 80)
        print("PHASE 1: OFFLINE GRAPH CONSTRUCTION")
        print("=" * 80)
        
        # Step 1: Input Ingestion & Chunking
        print("\n[Step 1/9] Input Ingestion & Chunking...")
        self.chunks = self.ingestion.ingest_from_file(novel_path)
        print(f"Created {len(self.chunks)} chunks from {len(set(c.chapter_id for c in self.chunks))} chapters")
        
        # Step 2: Meta LLaMA Encoder - Structured Extraction (OPTIMIZED with batching)
        print("\n[Step 2/9] Meta LLaMA Encoder - Structured Extraction (OPTIMIZED)...")
        print(f"Processing {len(self.chunks)} chunks in batches...")
        
        start_time = time.time()
        self.extractions = self.encoder.extract_batch(self.chunks)
        total_time = time.time() - start_time
        
        print(f"Extracted information from {len(self.extractions)} chunks in {total_time/60:.1f} minutes")
        print(f"Average time per chunk: {total_time/len(self.chunks):.2f} seconds")
        
        # Step 3: Temporal Normalization Layer
        print("\n[Step 3/9] Temporal Normalization Layer...")
        self.temporal_timeline = self.temporal_norm.normalize_temporal(self.extractions, self.chunks)
        print(f"Created temporal timeline with {len(self.temporal_timeline)} events")
        print(f"Temporal DAG valid: {self.temporal_norm.validate_temporal_consistency()}")
        
        # Step 4: Causal Graph Construction
        print("\n[Step 4/9] Causal Graph Construction...")
        self.causal_edges = self.causal_builder.build_causal_graph(
            self.extractions,
            self.temporal_timeline
        )
        print(f"Created {len(self.causal_edges)} causal edges")
        
        # Step 5: Thematic Coherence Layer
        print("\n[Step 5/9] Thematic Coherence Layer...")
        # Collect entities and events
        entities = {}
        events = {}
        for extraction in self.extractions:
            for entity in extraction.entities:
                entities[entity.entity_id] = entity
            for event in extraction.events:
                events[event.event_id] = event
        
        self.thematic_states = self.thematic_coherence.build_thematic_coherence(
            self.extractions,
            self.temporal_timeline,
            entities,
            events
        )
        print(f"Created {len(self.thematic_states)} thematic states")
        print(f"Detected {len(self.thematic_coherence.soft_contradictions)} soft contradictions")
        
        # Step 6: Knowledge Graph Assembly
        print("\n[Step 6/9] Knowledge Graph Assembly...")
        self.knowledge_graph = self.graph_assembler.assemble_graph(
            self.extractions,
            self.temporal_timeline,
            self.causal_edges,
            self.thematic_states,
            entities,
            events
        )
        stats = self.graph_assembler.get_graph_stats()
        print(f"Assembled knowledge graph:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Step 7: Community Detection
        print("\n[Step 7/9] Community Detection...")
        self.communities = self.community_detector.detect_communities(self.knowledge_graph)
        print(f"Detected communities at {len(self.communities)} levels")
        for level, comms in self.communities.items():
            unique_comms = len(set(comms.values()))
            print(f"  Level {level}: {unique_comms} communities")
        
        # Step 8: Community Summarization
        print("\n[Step 8/9] Community Summarization...")
        hierarchical_communities = self.community_detector.get_hierarchical_communities()
        self.community_summaries = self.summarizer.summarize_all_communities(
            hierarchical_communities,
            self.knowledge_graph,
            self.temporal_timeline,
            self.thematic_states
        )
        print(f"Generated {len(self.community_summaries)} community summaries")
        
        # Step 9: Pathway Storage
        print("\n[Step 9/9] Pathway Storage...")
        # Store entities
        community_mapping = {}
        for level, comms in self.communities.items():
            for node_id, comm_id in comms.items():
                community_mapping[node_id] = comm_id
        
        for entity_id, entity in entities.items():
            graph_node = self.knowledge_graph.nodes.get(entity_id)
            thematic_state = self.thematic_states.get(entity_id)
            self.pathway_storage.store_entity(
                entity,
                graph_node,
                thematic_state,
                self.temporal_timeline,
                community_mapping
            )
        
        # Store events
        for event_id, event in events.items():
            temporal_event = self.temporal_timeline.get(event_id)
            graph_node = self.knowledge_graph.nodes.get(event_id)
            thematic_state = self.thematic_states.get(event_id)
            if temporal_event:
                self.pathway_storage.store_event(
                    event,
                    graph_node,
                    temporal_event,
                    thematic_state,
                    community_mapping
                )
        
        # Store community summaries
        for summary in self.community_summaries:
            self.pathway_storage.store_community_summary(summary)
        
        # Save to disk
        self.pathway_storage.save_to_disk()
        print(f"Stored {len(self.pathway_storage.entities)} entities, "
              f"{len(self.pathway_storage.events)} events, "
              f"{len(self.pathway_storage.community_summaries)} community summaries")
        
        # Save graph to disk
        self._save_graph()
        
        print("\n" + "=" * 80)
        print("OFFLINE GRAPH CONSTRUCTION COMPLETE")
        print("=" * 80)
    
    def _save_graph(self):
        """Save knowledge graph to disk."""
        os.makedirs(GRAPH_STORAGE_PATH, exist_ok=True)
        
        # Save graph structure (simplified - would use proper serialization)
        graph_data = {
            "nodes": {
                node_id: {
                    "node_id": node.node_id,
                    "node_type": node.node_type.value,
                    "properties": node.properties
                }
                for node_id, node in self.knowledge_graph.nodes.items()
            },
            "edges": [
                {
                    "edge_id": edge.edge_id,
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "edge_type": edge.edge_type.value,
                    "weight": edge.weight
                }
                for edge in self.knowledge_graph.edges
            ]
        }
        
        graph_path = os.path.join(GRAPH_STORAGE_PATH, "knowledge_graph.json")
        with open(graph_path, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        print(f"Saved knowledge graph to {graph_path}")


def main():
    parser = argparse.ArgumentParser(description="Offline Graph Construction for Track-A GraphRAG")
    parser.add_argument("--novel_path", type=str, required=True, help="Path to novel text file")
    parser.add_argument("--output_dir", type=str, default="./graph_storage", help="Output directory")
    
    args = parser.parse_args()
    
    builder = OfflineGraphConstruction()
    builder.build_graph(args.novel_path)


if __name__ == "__main__":
    main()

