"""
PHASE 1.9: PATHWAY STORAGE

Store in Pathway:
- Embeddings (768-dim)
- Metadata filters

Objects to store:
- Entities
- Events
- Community summaries

Metadata MUST include:
- chapter_id
- time_range
- themes
- community_id
"""

import numpy as np
import json
import os
from typing import List, Dict, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from schemas import (
    PathwayEntity, PathwayEvent, PathwayCommunitySummary,
    Entity, Event, CommunitySummary, KnowledgeGraphNode,
    ThematicState, TemporalEvent
)
from config import EMBEDDING_DIMENSION, META_LLAMA_MODEL_NAME, PATHWAY_STORAGE_PATH, MODEL_PATH


class PathwayStorage:
    """
    Pathway storage layer.
    
    Stores embeddings and metadata for entities, events, and community summaries.
    Uses Meta LLaMA embedder for generating embeddings.
    """
    
    def __init__(self, storage_path: str = None, model_name: str = None):
        """
        Initialize Pathway storage.
        
        Args:
            storage_path: Path to storage directory
            model_name: Override default model name if needed
        """
        self.storage_path = storage_path or PATHWAY_STORAGE_PATH
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Use direct model path from .env if available
        self.model_path = model_name or MODEL_PATH or META_LLAMA_MODEL_NAME
        
        print(f"Loading Meta LLaMA model for embeddings: {self.model_path}")
        
        # Load tokenizer and model for embeddings with local_files_only=True
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # For embeddings, we'll use the model's embedding layer
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        self.model.eval()
        
        # Storage dictionaries
        self.entities: Dict[str, PathwayEntity] = {}
        self.events: Dict[str, PathwayEvent] = {}
        self.community_summaries: Dict[str, PathwayCommunitySummary] = {}
    
    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate 768-dim embedding using Meta LLaMA.
        
        Args:
            text: Input text
            
        Returns:
            List of 768 float values
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            # Get embeddings from model (using last hidden state mean pooling)
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # Last layer
            
            # Mean pooling
            embeddings = hidden_states.mean(dim=1).squeeze()
            
            # Ensure 768 dimensions (might need projection if model is different)
            if embeddings.shape[0] != EMBEDDING_DIMENSION:
                # Project to 768 dimensions if needed
                if embeddings.shape[0] > EMBEDDING_DIMENSION:
                    embeddings = embeddings[:EMBEDDING_DIMENSION]
                else:
                    # Pad if smaller (shouldn't happen for LLaMA)
                    padding = torch.zeros(EMBEDDING_DIMENSION - embeddings.shape[0])
                    embeddings = torch.cat([embeddings, padding])
        
        # Convert to list and normalize
        embedding_list = embeddings.cpu().numpy().tolist()
        
        # Normalize
        norm = np.linalg.norm(embedding_list)
        if norm > 0:
            embedding_list = (np.array(embedding_list) / norm).tolist()
        
        return embedding_list[:EMBEDDING_DIMENSION]
    
    def store_entity(
        self,
        entity: Entity,
        graph_node: KnowledgeGraphNode,
        thematic_state: Optional[ThematicState],
        temporal_timeline: Dict[str, TemporalEvent],
        community_mapping: Dict[str, int]
    ) -> PathwayEntity:
        """
        Store an entity in Pathway.
        
        Args:
            entity: Entity object
            graph_node: KnowledgeGraphNode for this entity
            thematic_state: ThematicState for this entity (optional)
            temporal_timeline: Dictionary of TemporalEvent objects
            community_mapping: Dictionary mapping node_id to community_id
            
        Returns:
            PathwayEntity object
        """
        # Generate embedding from entity description
        entity_text = f"{entity.name}. {entity.description or ''}"
        embedding = self._generate_embedding(entity_text)
        
        # Build metadata
        metadata = {
            "chapter_id": 0,  # Entities span chapters, use 0 or determine from events
            "time_range": None,
            "themes": [],
            "community_id": community_mapping.get(entity.entity_id, -1)
        }
        
        # Add temporal information if available
        if thematic_state:
            metadata["time_range"] = list(thematic_state.temporal_scope)
            metadata["themes"] = thematic_state.dominant_themes
        
        # Add entity properties
        metadata["entity_type"] = entity.entity_type
        metadata["attributes"] = entity.attributes
        
        pathway_entity = PathwayEntity(
            entity_id=entity.entity_id,
            embedding=embedding,
            metadata=metadata,
            text=entity_text
        )
        
        self.entities[entity.entity_id] = pathway_entity
        return pathway_entity
    
    def store_event(
        self,
        event: Event,
        graph_node: KnowledgeGraphNode,
        temporal_event: TemporalEvent,
        thematic_state: Optional[ThematicState],
        community_mapping: Dict[str, int]
    ) -> PathwayEvent:
        """
        Store an event in Pathway.
        
        Args:
            event: Event object
            graph_node: KnowledgeGraphNode for this event
            temporal_event: TemporalEvent for this event
            thematic_state: ThematicState for this event (optional)
            community_mapping: Dictionary mapping node_id to community_id
            
        Returns:
            PathwayEvent object
        """
        # Generate embedding from event description
        event_text = f"{event.event_type}: {event.description}"
        embedding = self._generate_embedding(event_text)
        
        # Build metadata
        metadata = {
            "chapter_id": temporal_event.chapter_id,
            "time_range": [temporal_event.t_story, temporal_event.t_story],
            "themes": [],
            "community_id": community_mapping.get(event.event_id, -1),
            "t_story": temporal_event.t_story,
            "event_type": event.event_type,
            "participants": event.participants,
            "location": event.location
        }
        
        # Add thematic information
        if thematic_state:
            metadata["themes"] = thematic_state.dominant_themes
        
        pathway_event = PathwayEvent(
            event_id=event.event_id,
            embedding=embedding,
            metadata=metadata,
            text=event_text
        )
        
        self.events[event.event_id] = pathway_event
        return pathway_event
    
    def store_community_summary(
        self,
        community_summary: CommunitySummary
    ) -> PathwayCommunitySummary:
        """
        Store a community summary in Pathway.
        
        Args:
            community_summary: CommunitySummary object
            
        Returns:
            PathwayCommunitySummary object
        """
        # Generate embedding from summary text
        embedding = self._generate_embedding(community_summary.summary_text)
        
        # Build metadata
        metadata = {
            "chapter_id": 0,  # Communities span chapters
            "time_range": list(community_summary.temporal_span),
            "themes": community_summary.dominant_themes,
            "community_id": community_summary.community_id,
            "level": community_summary.level,
            "node_ids": community_summary.node_ids,
            "causal_structure": community_summary.causal_structure_summary
        }
        
        pathway_summary = PathwayCommunitySummary(
            community_id=community_summary.community_id,
            embedding=embedding,
            metadata=metadata,
            summary_text=community_summary.summary_text
        )
        
        self.community_summaries[community_summary.community_id] = pathway_summary
        return pathway_summary
    
    def save_to_disk(self):
        """Save all Pathway objects to disk."""
        entities_path = os.path.join(self.storage_path, "entities.json")
        events_path = os.path.join(self.storage_path, "events.json")
        summaries_path = os.path.join(self.storage_path, "community_summaries.json")
        
        # Convert to JSON-serializable format
        entities_data = {
            eid: {
                "entity_id": pe.entity_id,
                "embedding": pe.embedding,
                "metadata": pe.metadata,
                "text": pe.text
            }
            for eid, pe in self.entities.items()
        }
        
        events_data = {
            eid: {
                "event_id": pe.event_id,
                "embedding": pe.embedding,
                "metadata": pe.metadata,
                "text": pe.text
            }
            for eid, pe in self.events.items()
        }
        
        summaries_data = {
            cid: {
                "community_id": ps.community_id,
                "embedding": ps.embedding,
                "metadata": ps.metadata,
                "summary_text": ps.summary_text
            }
            for cid, ps in self.community_summaries.items()
        }
        
        with open(entities_path, 'w') as f:
            json.dump(entities_data, f)
        
        with open(events_path, 'w') as f:
            json.dump(events_data, f)
        
        with open(summaries_path, 'w') as f:
            json.dump(summaries_data, f)
        
        print(f"Saved Pathway storage to {self.storage_path}")
    
    def load_from_disk(self):
        """Load all Pathway objects from disk."""
        entities_path = os.path.join(self.storage_path, "entities.json")
        events_path = os.path.join(self.storage_path, "events.json")
        summaries_path = os.path.join(self.storage_path, "community_summaries.json")
        
        if os.path.exists(entities_path):
            with open(entities_path, 'r') as f:
                entities_data = json.load(f)
                for eid, data in entities_data.items():
                    self.entities[eid] = PathwayEntity(**data)
        
        if os.path.exists(events_path):
            with open(events_path, 'r') as f:
                events_data = json.load(f)
                for eid, data in events_data.items():
                    self.events[eid] = PathwayEvent(**data)
        
        if os.path.exists(summaries_path):
            with open(summaries_path, 'r') as f:
                summaries_data = json.load(f)
                for cid, data in summaries_data.items():
                    self.community_summaries[cid] = PathwayCommunitySummary(**data)
        
        print(f"Loaded Pathway storage from {self.storage_path}")


if __name__ == "__main__":
    # Test example
    from schemas import Entity, Event, CommunitySummary, Chunk
    
    entity = Entity(entity_id="e_1", entity_type="character", name="John", description="Protagonist")
    storage = PathwayStorage()
    
    print("Pathway storage layer ready")

