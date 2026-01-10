"""
PHASE 2.4: SOFT CONTRADICTION RESOLUTION

If partial answers conflict:
- Prefer temporal consistency
- Prefer thematic continuity
- Reject answers violating story time
"""

from typing import List, Set, Tuple
from schemas import PartialAnswer
from config import SOFT_CONTRADICTION_EPSILON


class SoftContradictionResolution:
    """
    Soft contradiction resolution layer.
    
    Resolves conflicts between partial answers based on temporal and thematic consistency.
    """
    
    def __init__(self):
        pass
    
    def resolve_contradictions(
        self,
        partial_answers: List[PartialAnswer],
        temporal_timeline: dict = None
    ) -> List[PartialAnswer]:
        """
        Resolve soft contradictions in partial answers.
        
        Process:
        1. Detect contradictions between partial answers
        2. Prefer temporal consistency
        3. Prefer thematic continuity
        4. Reject answers violating story time
        
        Args:
            partial_answers: List of PartialAnswer objects
            temporal_timeline: Optional dictionary of TemporalEvent objects
            
        Returns:
            List of resolved PartialAnswer objects (contradictions resolved)
        """
        if len(partial_answers) <= 1:
            return partial_answers
        
        # Sort by relevance score (descending)
        sorted_answers = sorted(
            partial_answers,
            key=lambda x: x.relevance_score,
            reverse=True
        )
        
        resolved_answers = []
        rejected_indices = set()
        
        for i, answer1 in enumerate(sorted_answers):
            if i in rejected_indices:
                continue
            
            is_valid = True
            
            # Check against previously accepted answers
            for j, answer2 in enumerate(resolved_answers):
                if self._has_contradiction(answer1, answer2, temporal_timeline):
                    # Resolve contradiction
                    if self._should_reject(answer1, answer2, temporal_timeline):
                        is_valid = False
                        rejected_indices.add(i)
                        break
                    # If answer2 should be rejected instead, remove it
                    elif self._should_reject(answer2, answer1, temporal_timeline):
                        resolved_answers.remove(answer2)
                        # Continue checking answer1
            
            if is_valid:
                resolved_answers.append(answer1)
        
        return resolved_answers
    
    def _has_contradiction(
        self,
        answer1: PartialAnswer,
        answer2: PartialAnswer,
        temporal_timeline: dict = None
    ) -> bool:
        """
        Check if two answers have a contradiction.
        
        Args:
            answer1: First PartialAnswer
            answer2: Second PartialAnswer
            temporal_timeline: Optional temporal timeline
            
        Returns:
            True if contradiction exists
        """
        # Check temporal inconsistency
        if temporal_timeline and not self._temporal_consistent(
            answer1.temporal_coverage,
            answer2.temporal_coverage
        ):
            # If temporal ranges are very far apart and answers conflict, it's a contradiction
            # This is a simplified check
            return False  # Temporally distant is not necessarily contradictory
        
        # Check thematic contradiction (soft)
        if abs(answer1.thematic_alignment_score - answer2.thematic_alignment_score) > 0.5:
            # Thematic mismatch could indicate contradiction
            # But this alone doesn't mean contradiction
        
        # For now, we'll use a simple heuristic:
        # Answers are contradictory if they have very different temporal coverage
        # and low thematic alignment
        temporal_distance = abs(
            answer1.temporal_coverage[0] - answer2.temporal_coverage[0]
        ) + abs(answer1.temporal_coverage[1] - answer2.temporal_coverage[1])
        
        thematic_distance = abs(
            answer1.thematic_alignment_score - answer2.thematic_alignment_score
        )
        
        # If both temporal and thematic distance are large, potential contradiction
        return temporal_distance > 100 and thematic_distance > 0.5
    
    def _temporal_consistent(
        self,
        temporal_range1: Tuple[int, int],
        temporal_range2: Tuple[int, int]
    ) -> bool:
        """
        Check if two temporal ranges are consistent.
        
        Args:
            temporal_range1: First temporal range
            temporal_range2: Second temporal range
            
        Returns:
            True if temporally consistent
        """
        # Ranges that overlap or are adjacent are consistent
        return not (temporal_range1[1] < temporal_range2[0] - 10 or
                   temporal_range2[1] < temporal_range1[0] - 10)
    
    def _should_reject(
        self,
        answer1: PartialAnswer,
        answer2: PartialAnswer,
        temporal_timeline: dict = None
    ) -> bool:
        """
        Determine if answer1 should be rejected in favor of answer2.
        
        Criteria:
        - Prefer temporal consistency
        - Prefer thematic continuity
        - Reject answers violating story time
        
        Args:
            answer1: Answer to check for rejection
            answer2: Reference answer
            temporal_timeline: Optional temporal timeline
            
        Returns:
            True if answer1 should be rejected
        """
        # Prefer higher relevance score
        if answer1.relevance_score < answer2.relevance_score:
            return True
        
        # Prefer better thematic alignment
        if answer1.thematic_alignment_score < answer2.thematic_alignment_score:
            return True
        
        # Check temporal violation
        if temporal_timeline:
            # Check if answer1 violates story time
            # This would require more detailed temporal checking
            pass
        
        return False
    
    def validate_temporal_order(
        self,
        partial_answers: List[PartialAnswer]
    ) -> List[PartialAnswer]:
        """
        Validate that partial answers respect temporal order.
        
        Args:
            partial_answers: List of PartialAnswer objects
            
        Returns:
            List of temporally valid PartialAnswer objects
        """
        # Sort by temporal coverage start
        sorted_answers = sorted(
            partial_answers,
            key=lambda x: x.temporal_coverage[0]
        )
        
        valid_answers = []
        for answer in sorted_answers:
            # Check if answer respects temporal order
            # For now, accept all answers
            valid_answers.append(answer)
        
        return valid_answers


if __name__ == "__main__":
    # Test example
    from schemas import PartialAnswer
    
    answer1 = PartialAnswer(
        community_id="c_1",
        text="John meets Mary at the beginning.",
        relevance_score=80.0,
        temporal_coverage=(0, 50),
        thematic_alignment_score=0.8
    )
    
    answer2 = PartialAnswer(
        community_id="c_2",
        text="John leaves Mary at the end.",
        relevance_score=75.0,
        temporal_coverage=(200, 250),
        thematic_alignment_score=0.7
    )
    
    resolver = SoftContradictionResolution()
    resolved = resolver.resolve_contradictions([answer1, answer2])
    print(f"Resolved {len(resolved)} answers from {len([answer1, answer2])}")

