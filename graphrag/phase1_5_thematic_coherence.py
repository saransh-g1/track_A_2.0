"""
PHASE 1.5: THEMATIC COHERENCE LAYER (SOFT CONTRADICTIONS)

Models themes as continuous vectors.

Requirements:
- Theme vector dimension: 32–64
- Every entity and event has a theme vector
- Track theme evolution over time

Defines:
- SOFT contradiction: theme_distance < epsilon
- HARD contradiction: conflicting claims
"""

import numpy as np
from typing import List, Dict, Tuple, Set
from collections import defaultdict
from schemas import (
    StructuredExtraction, ThematicState, Theme, Entity, Event,
    TemporalEvent, KnowledgeGraphNode
)
from config import THEME_VECTOR_DIMENSION, SOFT_CONTRADICTION_EPSILON


class ThematicCoherence:
    """
    Thematic coherence layer for handling soft contradictions.
    
    Models themes as continuous vectors and tracks evolution over time.
    """
    
    def __init__(self, theme_vector_dim: int = None):
        """
        Initialize thematic coherence layer.
        
        Args:
            theme_vector_dim: Dimension of theme vectors (32-64)
        """
        self.theme_vector_dim = theme_vector_dim or THEME_VECTOR_DIMENSION
        self.thematic_states: Dict[str, ThematicState] = {}
        self.theme_vectors: Dict[str, np.ndarray] = {}  # theme_id -> vector
        self.subject_theme_history: Dict[str, List[Tuple[int, np.ndarray]]] = defaultdict(list)
        self.soft_contradictions: List[Tuple[str, str]] = []  # (subject_id1, subject_id2)
    
    def _initialize_theme_vectors(self, themes: List[Theme]) -> Dict[str, np.ndarray]:
        """
        Initialize theme vectors from extracted themes.
        
        Each theme gets a unique vector representation.
        
        Args:
            themes: List of Theme objects
            
        Returns:
            Dictionary mapping theme_id to vector
        """
        theme_vectors = {}
        
        for theme in themes:
            # Create a unique vector for each theme
            # In practice, this could use embeddings from Meta LLaMA
            # For now, use random initialization with theme properties
            np.random.seed(hash(theme.theme_id) % 2**32)
            vector = np.random.randn(self.theme_vector_dim).astype(np.float32)
            
            # Normalize
            vector = vector / (np.linalg.norm(vector) + 1e-8)
            
            # Scale by intensity
            vector = vector * theme.intensity
            
            theme_vectors[theme.theme_id] = vector
        
        return theme_vectors
    
    def _compute_subject_theme_vector(
        self,
        subject_id: str,
        themes: List[Theme],
        theme_vectors: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Compute theme vector for a subject (entity or event).
        
        Combines multiple themes weighted by their intensity.
        
        Args:
            subject_id: Subject ID (entity_id or event_id)
            themes: List of themes associated with this subject
            theme_vectors: Dictionary of theme vectors
            
        Returns:
            Combined theme vector
        """
        if not themes:
            # Return zero vector if no themes
            return np.zeros(self.theme_vector_dim, dtype=np.float32)
        
        # Weighted combination of theme vectors
        combined = np.zeros(self.theme_vector_dim, dtype=np.float32)
        total_weight = 0.0
        
        for theme in themes:
            if theme.theme_id in theme_vectors:
                vector = theme_vectors[theme.theme_id]
                weight = theme.intensity
                combined += vector * weight
                total_weight += weight
        
        # Normalize
        if total_weight > 0:
            combined = combined / total_weight
        
        return combined
    
    def build_thematic_coherence(
        self,
        extractions: List[StructuredExtraction],
        temporal_timeline: Dict[str, TemporalEvent],
        entities: Dict[str, Entity],
        events: Dict[str, Event]
    ) -> Dict[str, ThematicState]:
        """
        Build thematic coherence model.
        
        Process:
        1. Initialize theme vectors from all extracted themes
        2. For each entity and event, compute theme vector
        3. Track theme evolution over time
        4. Detect soft contradictions
        
        Args:
            extractions: List of StructuredExtraction objects
            temporal_timeline: Dictionary of TemporalEvent objects
            entities: Dictionary of Entity objects indexed by entity_id
            events: Dictionary of Event objects indexed by event_id
            
        Returns:
            Dictionary of ThematicState objects indexed by subject_id
        """
        # Collect all themes
        all_themes = []
        subject_themes: Dict[str, List[Theme]] = defaultdict(list)
        
        for extraction in extractions:
            for theme in extraction.themes:
                if theme not in all_themes:
                    all_themes.append(theme)
            
            # Map themes to entities and events
            for entity in extraction.entities:
                # Find themes mentioned in same chunk
                chunk_themes = [t for t in extraction.themes]
                subject_themes[entity.entity_id].extend(chunk_themes)
            
            for event in extraction.events:
                chunk_themes = [t for t in extraction.themes]
                subject_themes[event.event_id].extend(chunk_themes)
        
        # Initialize theme vectors
        self.theme_vectors = self._initialize_theme_vectors(all_themes)
        
        # Process subjects in temporal order
        all_subjects = {}
        
        # Add entities
        for entity_id, entity in entities.items():
            all_subjects[entity_id] = {
                'type': 'entity',
                'themes': subject_themes.get(entity_id, []),
                't_story': 0  # Entities exist throughout, will be updated by events
            }
        
        # Add events with temporal information
        for event_id, temporal_event in temporal_timeline.items():
            if event_id in events:
                all_subjects[event_id] = {
                    'type': 'event',
                    'themes': subject_themes.get(event_id, []),
                    't_story': temporal_event.t_story,
                    'chapter_id': temporal_event.chapter_id
                }
        
        # Sort by temporal order
        sorted_subjects = sorted(
            all_subjects.items(),
            key=lambda x: x[1].get('t_story', 0)
        )
        
        # Build thematic states
        for subject_id, subject_info in sorted_subjects:
            themes = subject_info['themes']
            
            # Compute theme vector
            theme_vector = self._compute_subject_theme_vector(
                subject_id,
                themes,
                self.theme_vectors
            )
            
            # Determine temporal scope
            if subject_info['type'] == 'event':
                t_story = subject_info['t_story']
                temporal_scope = (t_story, t_story)
            else:
                # For entities, find their temporal range from events
                temporal_scope = self._find_entity_temporal_scope(subject_id, temporal_timeline)
            
            # Get dominant themes (top themes by intensity)
            dominant_themes = sorted(
                themes,
                key=lambda t: t.intensity,
                reverse=True
            )[:3]
            dominant_theme_ids = [t.theme_id for t in dominant_themes]
            
            # Create ThematicState
            thematic_state = ThematicState(
                subject_id=subject_id,
                theme_vector=theme_vector.tolist(),
                temporal_scope=temporal_scope,
                dominant_themes=dominant_theme_ids
            )
            
            self.thematic_states[subject_id] = thematic_state
            
            # Track theme history
            t_story = subject_info.get('t_story', temporal_scope[0])
            self.subject_theme_history[subject_id].append((t_story, theme_vector))
        
        # Detect soft contradictions
        self._detect_soft_contradictions()
        
        return self.thematic_states
    
    def _find_entity_temporal_scope(
        self,
        entity_id: str,
        temporal_timeline: Dict[str, TemporalEvent]
    ) -> Tuple[int, int]:
        """
        Find temporal scope of an entity based on its participation in events.
        
        Args:
            entity_id: Entity ID
            temporal_timeline: Dictionary of TemporalEvent objects
            
        Returns:
            Tuple of (start_t_story, end_t_story)
        """
        # This would require mapping entities to events
        # For now, return full range
        if not temporal_timeline:
            return (0, 0)
        
        min_t = min(te.t_story for te in temporal_timeline.values())
        max_t = max(te.t_story for te in temporal_timeline.values())
        
        return (min_t, max_t)
    
    def _detect_soft_contradictions(self):
        """
        Detect soft contradictions based on theme vector distance.
        
        Soft contradiction: theme_distance < epsilon (very similar themes)
        """
        self.soft_contradictions = []
        subject_ids = list(self.thematic_states.keys())
        
        for i, subject_id1 in enumerate(subject_ids):
            state1 = self.thematic_states[subject_id1]
            vector1 = np.array(state1.theme_vector)
            
            for subject_id2 in subject_ids[i+1:]:
                state2 = self.thematic_states[subject_id2]
                vector2 = np.array(state2.theme_vector)
                
                # Compute cosine distance
                distance = 1.0 - np.dot(vector1, vector2) / (
                    np.linalg.norm(vector1) * np.linalg.norm(vector2) + 1e-8
                )
                
                # Soft contradiction: very similar themes (low distance)
                if distance < SOFT_CONTRADICTION_EPSILON:
                    self.soft_contradictions.append((subject_id1, subject_id2))
    
    def get_theme_distance(self, subject_id1: str, subject_id2: str) -> float:
        """
        Get theme distance between two subjects.
        
        Args:
            subject_id1: First subject ID
            subject_id2: Second subject ID
            
        Returns:
            Cosine distance (0 = identical, 1 = orthogonal)
        """
        if subject_id1 not in self.thematic_states or subject_id2 not in self.thematic_states:
            return 1.0
        
        vector1 = np.array(self.thematic_states[subject_id1].theme_vector)
        vector2 = np.array(self.thematic_states[subject_id2].theme_vector)
        
        distance = 1.0 - np.dot(vector1, vector2) / (
            np.linalg.norm(vector1) * np.linalg.norm(vector2) + 1e-8
        )
        
        return float(distance)
    
    def has_soft_contradiction(self, subject_id1: str, subject_id2: str) -> bool:
        """
        Check if two subjects have a soft contradiction.
        
        Args:
            subject_id1: First subject ID
            subject_id2: Second subject ID
            
        Returns:
            True if soft contradiction exists
        """
        distance = self.get_theme_distance(subject_id1, subject_id2)
        return distance < SOFT_CONTRADICTION_EPSILON


if __name__ == "__main__":
    # Test example
    from schemas import Entity, Event, Theme
    
    theme1 = Theme(theme_id="th_1", theme_name="betrayal", description="...", intensity=0.8)
    theme2 = Theme(theme_id="th_2", theme_name="love", description="...", intensity=0.9)
    
    entity = Entity(entity_id="e_1", entity_type="character", name="John")
    event = Event(event_id="ev_1", event_type="action", description="...", participants=["e_1"])
    
    extraction = StructuredExtraction(
        chunk_id=1,
        entities=[entity],
        events=[event],
        themes=[theme1, theme2]
    )
    
    from phase1_3_temporal_normalization import TemporalNormalization
    from schemas import Chunk, TemporalEvent
    
    chunks = [Chunk(chunk_id=1, chapter_id=1, text="...", token_range=(0, 100))]
    temporal_norm = TemporalNormalization()
    timeline = temporal_norm.normalize_temporal([extraction], chunks)
    
    coherence = ThematicCoherence()
    entities_dict = {entity.entity_id: entity}
    events_dict = {event.event_id: event}
    thematic_states = coherence.build_thematic_coherence(
        [extraction],
        timeline,
        entities_dict,
        events_dict
    )
    
    print(f"Created {len(thematic_states)} thematic states")
    print(f"Detected {len(coherence.soft_contradictions)} soft contradictions")

