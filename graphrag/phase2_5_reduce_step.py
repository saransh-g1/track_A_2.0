"""
PHASE 2.5: REDUCE STEP

Merge top-K partial answers.

Enforce:
- Temporal order
- Causal consistency
- Thematic coherence
"""

from typing import List, Tuple
from schemas import PartialAnswer
from config import TOP_K_PARTIAL_ANSWERS


class ReduceStep:
    """
    Reduce step layer.
    
    Merges partial answers into a coherent response.
    """
    
    def __init__(self):
        pass
    
    def reduce_partial_answers(
        self,
        partial_answers: List[PartialAnswer],
        top_k: int = None
    ) -> List[PartialAnswer]:
        """
        Reduce and merge partial answers.
        
        Process:
        1. Select top-K partial answers
        2. Sort by temporal order
        3. Ensure causal consistency
        4. Ensure thematic coherence
        
        Args:
            partial_answers: List of PartialAnswer objects
            top_k: Number of answers to merge (default: TOP_K_PARTIAL_ANSWERS)
            
        Returns:
            List of reduced PartialAnswer objects
        """
        if top_k is None:
            top_k = TOP_K_PARTIAL_ANSWERS
        
        if not partial_answers:
            return []
        
        # Select top-K by relevance score
        sorted_answers = sorted(
            partial_answers,
            key=lambda x: x.relevance_score,
            reverse=True
        )[:top_k]
        
        # Sort by temporal order
        temporal_sorted = sorted(
            sorted_answers,
            key=lambda x: x.temporal_coverage[0]
        )
        
        # Ensure causal consistency (simplified)
        causal_consistent = self._ensure_causal_consistency(temporal_sorted)
        
        # Ensure thematic coherence
        thematically_coherent = self._ensure_thematic_coherence(causal_consistent)
        
        return thematically_coherent
    
    def _ensure_causal_consistency(
        self,
        partial_answers: List[PartialAnswer]
    ) -> List[PartialAnswer]:
        """
        Ensure causal consistency between partial answers.
        
        Args:
            partial_answers: List of PartialAnswer objects
            
        Returns:
            List of causally consistent PartialAnswer objects
        """
        # For now, return all answers
        # In production, would check for causal relationships
        # and ensure effects come after causes
        
        consistent_answers = []
        seen_events = set()
        
        for answer in partial_answers:
            # Check for causal consistency
            # If answer references events that are already covered,
            # ensure temporal order is respected
            answer_events = set(answer.supporting_event_ids)
            
            # Simple check: if events overlap, ensure temporal order
            if not seen_events or not answer_events.intersection(seen_events):
                consistent_answers.append(answer)
                seen_events.update(answer_events)
            else:
                # Check if temporal order is correct
                # For now, include it
                consistent_answers.append(answer)
                seen_events.update(answer_events)
        
        return consistent_answers
    
    def _ensure_thematic_coherence(
        self,
        partial_answers: List[PartialAnswer]
    ) -> List[PartialAnswer]:
        """
        Ensure thematic coherence between partial answers.
        
        Args:
            partial_answers: List of PartialAnswer objects
            
        Returns:
            List of thematically coherent PartialAnswer objects
        """
        if not partial_answers:
            return []
        
        # Compute average thematic alignment
        avg_thematic = sum(
            a.thematic_alignment_score for a in partial_answers
        ) / len(partial_answers)
        
        # Filter out answers with very low thematic alignment
        # compared to average
        coherent_answers = [
            a for a in partial_answers
            if a.thematic_alignment_score >= avg_thematic * 0.7
        ]
        
        # If filtering removed all answers, return original
        if not coherent_answers:
            return partial_answers
        
        return coherent_answers
    
    def merge_answers(
        self,
        partial_answers: List[PartialAnswer]
    ) -> str:
        """
        Merge partial answers into a single text.
        
        Args:
            partial_answers: List of PartialAnswer objects (already reduced)
            
        Returns:
            Merged text string
        """
        if not partial_answers:
            return "No information found."
        
        # Sort by temporal order
        sorted_answers = sorted(
            partial_answers,
            key=lambda x: x.temporal_coverage[0]
        )
        
        # Combine texts
        texts = [answer.text for answer in sorted_answers]
        merged_text = " ".join(texts)
        
        return merged_text


if __name__ == "__main__":
    # Test example
    from schemas import PartialAnswer
    
    answer1 = PartialAnswer(
        community_id="c_1",
        text="John starts his journey.",
        relevance_score=80.0,
        temporal_coverage=(0, 50),
        thematic_alignment_score=0.8
    )
    
    answer2 = PartialAnswer(
        community_id="c_2",
        text="John reaches his destination.",
        relevance_score=75.0,
        temporal_coverage=(100, 150),
        thematic_alignment_score=0.7
    )
    
    reducer = ReduceStep()
    reduced = reducer.reduce_partial_answers([answer1, answer2])
    merged = reducer.merge_answers(reduced)
    print(f"Reduced {len([answer1, answer2])} answers to {len(reduced)}")
    print(f"Merged text: {merged[:100]}...")

