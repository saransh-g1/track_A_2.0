"""
PHASE 1.3: TEMPORAL NORMALIZATION LAYER

Converts local time mentions into a GLOBAL STORY TIMELINE.

Builds:
- Story-time index (t_story)
- happens_before / happens_after edges
- Chapter-aware ordering

Represented as a TEMPORAL DAG.
"""

from typing import List, Dict, Set
from collections import defaultdict
from schemas import StructuredExtraction, TemporalEvent, Chunk


class TemporalNormalization:
    """
    Temporal normalization layer.
    
    Converts local temporal markers into global story timeline.
    Builds temporal DAG with explicit ordering.
    """
    
    def __init__(self):
        self.global_timeline: Dict[str, TemporalEvent] = {}
        self.t_story_counter = 0
        self.chapter_event_map: Dict[int, List[str]] = defaultdict(list)
    
    def _infer_chapter_order(self, chapter_id: int, prev_chapter_id: int = None) -> int:
        """
        Infer temporal order based on chapter sequence.
        
        Assumptions:
        - Chapters appear in chronological order by default
        - Later chapters happen after earlier chapters
        
        Args:
            chapter_id: Current chapter ID
            prev_chapter_id: Previous chapter ID (if any)
            
        Returns:
            Initial t_story value for this chapter
        """
        if prev_chapter_id is None or chapter_id > prev_chapter_id:
            return self.t_story_counter
        else:
            # If chapter_id is less than previous, maintain relative order
            return self.t_story_counter - 1
    
    def _parse_temporal_marker(self, marker: str) -> Dict[str, any]:
        """
        Parse temporal marker text into structured information.
        
        This is a simplified parser - in production, would use NLP temporal parsing.
        
        Args:
            marker: Temporal marker text
            
        Returns:
            Dictionary with temporal information
        """
        marker_lower = marker.lower()
        
        # Simple keyword-based parsing
        if any(word in marker_lower for word in ['before', 'earlier', 'previously', 'prior']):
            return {'order': 'before'}
        elif any(word in marker_lower for word in ['after', 'later', 'subsequently', 'then']):
            return {'order': 'after'}
        elif any(word in marker_lower for word in ['during', 'while', 'meanwhile', 'at the same time']):
            return {'order': 'concurrent'}
        elif any(word in marker_lower for word in ['first', 'initially', 'beginning']):
            return {'order': 'first'}
        elif any(word in marker_lower for word in ['last', 'finally', 'end']):
            return {'order': 'last'}
        else:
            return {'order': 'sequential'}  # Default sequential
    
    def _resolve_temporal_order(
        self,
        event_id: str,
        temporal_markers: List,
        all_events: Dict[str, Dict],
        chapter_id: int
    ) -> List[str]:
        """
        Resolve which events this event precedes.
        
        Args:
            event_id: Current event ID
            temporal_markers: List of temporal markers referencing this event
            all_events: Dictionary of all events indexed by event_id
            chapter_id: Current chapter ID
            
        Returns:
            List of event_ids that this event precedes
        """
        precedes = []
        
        for marker in temporal_markers:
            if marker.reference_event_id and marker.reference_event_id in all_events:
                parsed = self._parse_temporal_marker(marker.text)
                order = parsed.get('order', 'sequential')
                
                if order == 'before':
                    # This event happens before the reference event
                    # So reference event is in precedes list
                    if marker.reference_event_id not in precedes:
                        precedes.append(marker.reference_event_id)
                elif order == 'after':
                    # This event happens after the reference event
                    # So we need to add this event to the reference event's precedes
                    ref_event_id = marker.reference_event_id
                    if ref_event_id in self.global_timeline:
                        if event_id not in self.global_timeline[ref_event_id].precedes:
                            self.global_timeline[ref_event_id].precedes.append(event_id)
        
        # Also add events from later chapters to precedes (chapter order implies temporal order)
        for other_event_id, other_event in self.global_timeline.items():
            if other_event.chapter_id > chapter_id and other_event_id not in precedes:
                # Events in later chapters happen after this one
                if event_id not in other_event.precedes:
                    # But don't add directly - maintain DAG structure
                    pass
        
        return precedes
    
    def normalize_temporal(
        self,
        extractions: List[StructuredExtraction],
        chunks: List[Chunk]
    ) -> Dict[str, TemporalEvent]:
        """
        Main normalization method.
        
        Process:
        1. Build event map from extractions
        2. For each event, determine its position in global timeline
        3. Resolve temporal relationships
        4. Build temporal DAG
        
        Args:
            extractions: List of StructuredExtraction objects
            chunks: List of Chunk objects (for chapter_id lookup)
            
        Returns:
            Dictionary of TemporalEvent objects indexed by event_id
        """
        # Build chunk_id -> chunk mapping
        chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
        
        # Build event map
        all_events = {}
        for extraction in extractions:
            chunk = chunk_map[extraction.chunk_id]
            for event in extraction.events:
                all_events[event.event_id] = {
                    'event': event,
                    'chapter_id': chunk.chapter_id,
                    'chunk_id': extraction.chunk_id,
                    'temporal_markers': [
                        m for m in extraction.temporal_markers
                        if m.reference_event_id == event.event_id
                    ]
                }
        
        # Process events in chapter order
        sorted_events = sorted(
            all_events.items(),
            key=lambda x: (x[1]['chapter_id'], x[0])
        )
        
        prev_chapter_id = None
        chapter_t_story_base = {}
        
        for event_id, event_info in sorted_events:
            chapter_id = event_info['chapter_id']
            event = event_info['event']
            chunk_id = event_info['chunk_id']
            temporal_markers = event_info['temporal_markers']
            
            # Initialize chapter base time if needed
            if chapter_id not in chapter_t_story_base:
                if prev_chapter_id is not None:
                    chapter_t_story_base[chapter_id] = chapter_t_story_base[prev_chapter_id] + 100
                else:
                    chapter_t_story_base[chapter_id] = 0
                prev_chapter_id = chapter_id
            
            # Assign initial t_story
            t_story = chapter_t_story_base[chapter_id] + len(self.chapter_event_map[chapter_id])
            
            # Resolve temporal order
            precedes = self._resolve_temporal_order(
                event_id,
                temporal_markers,
                all_events,
                chapter_id
            )
            
            # Create TemporalEvent
            temporal_event = TemporalEvent(
                event_id=event_id,
                t_story=t_story,
                chapter_id=chapter_id,
                precedes=precedes,
                original_chunk_id=chunk_id
            )
            
            self.global_timeline[event_id] = temporal_event
            self.chapter_event_map[chapter_id].append(event_id)
            self.t_story_counter = max(self.t_story_counter, t_story + 1)
        
        return self.global_timeline
    
    def get_temporal_dag(self) -> Dict[str, TemporalEvent]:
        """
        Get the complete temporal DAG.
        
        Returns:
            Dictionary of TemporalEvent objects forming a DAG
        """
        return self.global_timeline
    
    def validate_temporal_consistency(self) -> bool:
        """
        Validate that the temporal DAG has no cycles.
        
        Returns:
            True if DAG is valid (no cycles), False otherwise
        """
        # Check for cycles using DFS
        visited = set()
        rec_stack = set()
        
        def has_cycle(event_id: str) -> bool:
            if event_id in rec_stack:
                return True
            if event_id in visited:
                return False
            
            visited.add(event_id)
            rec_stack.add(event_id)
            
            if event_id in self.global_timeline:
                for next_id in self.global_timeline[event_id].precedes:
                    if has_cycle(next_id):
                        return True
            
            rec_stack.remove(event_id)
            return False
        
        for event_id in self.global_timeline:
            if event_id not in visited:
                if has_cycle(event_id):
                    return False
        
        return True


if __name__ == "__main__":
    # Test example
    from schemas import Event, TemporalMarker
    
    extraction1 = StructuredExtraction(
        chunk_id=1,
        events=[
            Event(
                event_id="ev_1",
                event_type="action",
                description="John enters the room",
                participants=["e_john"]
            )
        ],
        temporal_markers=[
            TemporalMarker(
                marker_id="tm_1",
                marker_type="sequence",
                text="First",
                reference_event_id="ev_1"
            )
        ]
    )
    
    extraction2 = StructuredExtraction(
        chunk_id=2,
        events=[
            Event(
                event_id="ev_2",
                event_type="action",
                description="Mary leaves",
                participants=["e_mary"]
            )
        ],
        temporal_markers=[
            TemporalMarker(
                marker_id="tm_2",
                marker_type="relative",
                text="After John enters",
                reference_event_id="ev_2"
            )
        ]
    )
    
    chunks = [
        Chunk(chunk_id=1, chapter_id=1, text="...", token_range=(0, 100)),
        Chunk(chunk_id=2, chapter_id=1, text="...", token_range=(101, 200))
    ]
    
    normalizer = TemporalNormalization()
    timeline = normalizer.normalize_temporal([extraction1, extraction2], chunks)
    
    print(f"Created {len(timeline)} temporal events")
    print(f"Temporal DAG valid: {normalizer.validate_temporal_consistency()}")

