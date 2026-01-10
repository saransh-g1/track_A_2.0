"""
PHASE 2.2: COMMUNITY SELECTION

Select relevant communities using:
- Vector similarity (Pathway)
- Temporal filters
- Thematic alignment

DO NOT retrieve raw text.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from schemas import (
    PathwayCommunitySummary, PartialAnswer
)
from config import TOP_K_COMMUNITIES


class CommunitySelection:
    """
    Community selection layer.
    
    Selects relevant communities based on vector similarity, temporal filters, and thematic alignment.
    """
    
    def __init__(self):
        pass
    
    def select_communities(
        self,
        query_embedding: list,
        community_summaries: Dict[str, PathwayCommunitySummary],
        temporal_filter: Optional[Tuple[int, int]] = None,
        thematic_filter: Optional[List[str]] = None,
        top_k: int = None
    ) -> List[Tuple[str, float]]:
        """
        Select relevant communities.
        
        Process:
        1. Compute vector similarity between query and community embeddings
        2. Apply temporal filter if provided
        3. Apply thematic filter if provided
        4. Rank and return top-K communities
        
        Args:
            query_embedding: Query embedding (768-dim)
            community_summaries: Dictionary of PathwayCommunitySummary objects
            temporal_filter: Optional (start_t_story, end_t_story) tuple
            thematic_filter: Optional list of theme IDs
            top_k: Number of communities to return (default: TOP_K_COMMUNITIES)
            
        Returns:
            List of (community_id, relevance_score) tuples, sorted by relevance
        """
        if top_k is None:
            top_k = TOP_K_COMMUNITIES
        
        query_vec = np.array(query_embedding)
        
        scored_communities = []
        
        for community_id, community_summary in community_summaries.items():
            # Compute vector similarity (cosine similarity)
            community_vec = np.array(community_summary.embedding)
            similarity = np.dot(query_vec, community_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(community_vec) + 1e-8
            )
            
            score = float(similarity)
            
            # Apply temporal filter
            if temporal_filter is not None:
                community_time_range = community_summary.metadata.get("time_range", [0, 0])
                if not self._temporal_overlap(
                    temporal_filter,
                    tuple(community_time_range)
                ):
                    # Skip if no temporal overlap
                    continue
            
            # Apply thematic filter
            if thematic_filter is not None:
                community_themes = community_summary.metadata.get("themes", [])
                thematic_match = any(
                    theme_id in community_themes for theme_id in thematic_filter
                )
                if not thematic_match:
                    # Reduce score if no thematic match (but don't skip)
                    score *= 0.5
            
            scored_communities.append((community_id, score))
        
        # Sort by score (descending)
        scored_communities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-K
        return scored_communities[:top_k]
    
    def _temporal_overlap(
        self,
        range1: Tuple[int, int],
        range2: Tuple[int, int]
    ) -> bool:
        """
        Check if two temporal ranges overlap.
        
        Args:
            range1: First temporal range (start, end)
            range2: Second temporal range (start, end)
            
        Returns:
            True if ranges overlap
        """
        return not (range1[1] < range2[0] or range2[1] < range1[0])


if __name__ == "__main__":
    # Test example
    from schemas import PathwayCommunitySummary
    
    community_summary = PathwayCommunitySummary(
        community_id="c_1_0",
        embedding=[0.1] * 768,
        metadata={
            "time_range": [0, 100],
            "themes": ["th_1", "th_2"]
        },
        summary_text="Test community"
    )
    
    selector = CommunitySelection()
    query_embedding = [0.1] * 768
    selected = selector.select_communities(
        query_embedding,
        {"c_1_0": community_summary}
    )
    print(f"Selected {len(selected)} communities")

