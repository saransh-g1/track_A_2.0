"""
PHASE 2: ONLINE QUERY PROCESSING

Main orchestration script for online query processing.

Executes all Phase 2 steps:
10. Query Encoding
11. Community Selection
12. Map Step (per community)
13. Soft Contradiction Resolution
14. Reduce Step
15. Meta LLaMA Decoder - Final Answer
"""

import argparse
import os
from phase2_1_query_encoding import QueryEncoding
from phase2_2_community_selection import CommunitySelection
from phase2_3_map_step import MapStep
from phase2_4_soft_contradiction_resolution import SoftContradictionResolution
from phase2_5_reduce_step import ReduceStep
from phase2_6_meta_llama_decoder import MetaLlamaDecoder
from phase1_9_pathway_storage import PathwayStorage
from schemas import FinalAnswer
from config import PATHWAY_STORAGE_PATH, GRAPH_STORAGE_PATH


class OnlineQueryProcessing:
    """
    Main orchestrator for online query processing.
    
    Executes all Phase 2 steps in sequence.
    """
    
    def __init__(self):
        # Initialize components
        self.query_encoder = QueryEncoding()
        self.community_selector = CommunitySelection()
        self.map_step = MapStep()
        self.contradiction_resolver = SoftContradictionResolution()
        self.reduce_step = ReduceStep()
        self.decoder = MetaLlamaDecoder()
        self.pathway_storage = PathwayStorage()
        
        # Load Pathway storage
        if os.path.exists(PATHWAY_STORAGE_PATH):
            self.pathway_storage.load_from_disk()
            print(f"Loaded Pathway storage: {len(self.pathway_storage.community_summaries)} summaries")
        else:
            print(f"Warning: Pathway storage not found at {PATHWAY_STORAGE_PATH}")
            print("Please run offline_graph_construction.py first")
    
    def process_query(
        self,
        query: str,
        temporal_filter: tuple = None,
        thematic_filter: list = None
    ) -> FinalAnswer:
        """
        Process a query through all Phase 2 steps.
        
        Args:
            query: User query string
            temporal_filter: Optional (start_t_story, end_t_story) tuple
            thematic_filter: Optional list of theme IDs
            
        Returns:
            FinalAnswer object
        """
        print("=" * 80)
        print("PHASE 2: ONLINE QUERY PROCESSING")
        print("=" * 80)
        print(f"\nQuery: {query}\n")
        
        # Step 10: Query Encoding
        print("[Step 10/15] Query Encoding...")
        query_embedding = self.query_encoder.encode_query(query)
        print(f"Encoded query to {len(query_embedding)}-dim vector")
        
        # Step 11: Community Selection
        print("\n[Step 11/15] Community Selection...")
        community_summaries = self.pathway_storage.community_summaries
        selected_communities = self.community_selector.select_communities(
            query_embedding,
            community_summaries,
            temporal_filter=temporal_filter,
            thematic_filter=thematic_filter
        )
        print(f"Selected {len(selected_communities)} communities")
        
        # Step 12: Map Step (per community)
        print("\n[Step 12/15] Map Step (per community)...")
        partial_answers = self.map_step.map_all_communities(
            query,
            selected_communities,
            community_summaries
        )
        print(f"Generated {len(partial_answers)} partial answers")
        
        # Step 13: Soft Contradiction Resolution
        print("\n[Step 13/15] Soft Contradiction Resolution...")
        # Load temporal timeline if available (would need to load from disk)
        resolved_answers = self.contradiction_resolver.resolve_contradictions(
            partial_answers,
            temporal_timeline=None  # Would load from disk in production
        )
        print(f"Resolved to {len(resolved_answers)} answers")
        
        # Step 14: Reduce Step
        print("\n[Step 14/15] Reduce Step...")
        reduced_answers = self.reduce_step.reduce_partial_answers(resolved_answers)
        print(f"Reduced to {len(reduced_answers)} answers")
        
        # Step 15: Meta LLaMA Decoder - Final Answer
        print("\n[Step 15/15] Meta LLaMA Decoder - Final Answer...")
        final_answer = self.decoder.generate_final_answer(query, reduced_answers)
        print(f"Generated final answer with confidence {final_answer.confidence:.2f}")
        print(f"Citations: {len(final_answer.citations)}")
        
        print("\n" + "=" * 80)
        print("QUERY PROCESSING COMPLETE")
        print("=" * 80)
        
        return final_answer


def main():
    parser = argparse.ArgumentParser(description="Online Query Processing for Track-A GraphRAG")
    parser.add_argument("--query", type=str, required=True, help="User query")
    parser.add_argument("--temporal_filter", type=int, nargs=2, metavar=("START", "END"),
                       help="Temporal filter: start_t_story end_t_story")
    parser.add_argument("--thematic_filter", type=str, nargs="+",
                       help="Thematic filter: list of theme IDs")
    
    args = parser.parse_args()
    
    processor = OnlineQueryProcessing()
    
    temporal_filter = tuple(args.temporal_filter) if args.temporal_filter else None
    thematic_filter = args.thematic_filter if args.thematic_filter else None
    
    final_answer = processor.process_query(
        args.query,
        temporal_filter=temporal_filter,
        thematic_filter=thematic_filter
    )
    
    print("\n" + "=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)
    print(f"\n{final_answer.answer_text}\n")
    
    if final_answer.citations:
        print("Citations:")
        for citation in final_answer.citations[:5]:  # Show first 5
            print(f"  - {citation['type']}: {citation['id']}")
    
    print(f"\nConfidence: {final_answer.confidence:.2f}")
    print(f"Temporal span: {final_answer.temporal_span}")
    print(f"Communities used: {len(final_answer.communities_used)}")


if __name__ == "__main__":
    main()

