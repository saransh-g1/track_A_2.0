"""
PHASE 1.8: COMMUNITY SUMMARIZATION

For each community:
- Summarize using Meta LLaMA
- Preserve temporal span
- Preserve dominant themes
- Preserve causal structure

Schema: CommunitySummary
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple
from schemas import (
    CommunitySummary, KnowledgeGraph, KnowledgeGraphNode,
    TemporalEvent, ThematicState
)
from config import META_LLAMA_MODEL_NAME, MODEL_PATH


class CommunitySummarization:
    """
    Community summarization layer.
    
    Generates summaries for each detected community using Meta LLaMA.
    Preserves temporal, thematic, and causal information.
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize Meta LLaMA model for summarization.
        
        Args:
            model_name: Override default model name if needed
        """
        # Use direct model path from .env if available
        self.model_path = model_name or MODEL_PATH or META_LLAMA_MODEL_NAME
        
        print(f"Loading Meta LLaMA model for summarization: {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        self.model.eval()
    
    def summarize_community(
        self,
        community_id: int,
        level: int,
        node_ids: List[str],
        graph: KnowledgeGraph,
        temporal_timeline: Dict[str, TemporalEvent],
        thematic_states: Dict[str, ThematicState]
    ) -> CommunitySummary:
        """
        Summarize a single community.
        
        Args:
            community_id: Community ID
            level: Community level (0-3)
            node_ids: List of node IDs in this community
            graph: KnowledgeGraph object
            temporal_timeline: Dictionary of TemporalEvent objects
            thematic_states: Dictionary of ThematicState objects
            
        Returns:
            CommunitySummary object
        """
        # Collect information about the community
        nodes = [graph.nodes[node_id] for node_id in node_ids if node_id in graph.nodes]
        
        # Determine temporal span
        temporal_span = self._compute_temporal_span(node_ids, temporal_timeline)
        
        # Determine dominant themes
        dominant_themes = self._compute_dominant_themes(node_ids, thematic_states)
        
        # Extract causal structure
        causal_structure = self._extract_causal_structure(node_ids, graph)
        
        # Generate summary text
        summary_text = self._generate_summary(
            nodes,
            temporal_span,
            dominant_themes,
            causal_structure,
            level
        )
        
        return CommunitySummary(
            community_id=f"c_{level}_{community_id}",
            level=level,
            summary_text=summary_text,
            temporal_span=temporal_span,
            dominant_themes=dominant_themes,
            node_ids=node_ids,
            causal_structure_summary=causal_structure
        )
    
    def _compute_temporal_span(
        self,
        node_ids: List[str],
        temporal_timeline: Dict[str, TemporalEvent]
    ) -> Tuple[int, int]:
        """
        Compute temporal span of a community.
        
        Args:
            node_ids: List of node IDs
            temporal_timeline: Dictionary of TemporalEvent objects
            
        Returns:
            Tuple of (start_t_story, end_t_story)
        """
        t_stories = []
        
        for node_id in node_ids:
            if node_id in temporal_timeline:
                t_stories.append(temporal_timeline[node_id].t_story)
        
        if not t_stories:
            return (0, 0)
        
        return (min(t_stories), max(t_stories))
    
    def _compute_dominant_themes(
        self,
        node_ids: List[str],
        thematic_states: Dict[str, ThematicState]
    ) -> List[str]:
        """
        Compute dominant themes for a community.
        
        Args:
            node_ids: List of node IDs
            thematic_states: Dictionary of ThematicState objects
            
        Returns:
            List of theme IDs (top 3)
        """
        theme_counts = {}
        
        for node_id in node_ids:
            if node_id in thematic_states:
                state = thematic_states[node_id]
                for theme_id in state.dominant_themes:
                    theme_counts[theme_id] = theme_counts.get(theme_id, 0) + 1
        
        # Sort by frequency and return top 3
        sorted_themes = sorted(
            theme_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        return [theme_id for theme_id, _ in sorted_themes]
    
    def _extract_causal_structure(
        self,
        node_ids: List[str],
        graph: KnowledgeGraph
    ) -> str:
        """
        Extract causal structure summary for a community.
        
        Args:
            node_ids: List of node IDs
            graph: KnowledgeGraph object
            
        Returns:
            Text description of causal structure
        """
        # Find causal edges within this community
        causal_edges = []
        for edge in graph.edges:
            if (edge.edge_type.value == "causes" and
                edge.source_id in node_ids and
                edge.target_id in node_ids):
                causal_edges.append(edge)
        
        if not causal_edges:
            return "No explicit causal structure detected."
        
        # Build causal chain description
        descriptions = []
        for edge in causal_edges[:5]:  # Limit to top 5
            source_node = graph.nodes.get(edge.source_id)
            target_node = graph.nodes.get(edge.target_id)
            if source_node and target_node:
                source_desc = source_node.properties.get("description", edge.source_id)
                target_desc = target_node.properties.get("description", edge.target_id)
                descriptions.append(f"{source_desc} causes {target_desc}")
        
        return "; ".join(descriptions)
    
    def _generate_summary(
        self,
        nodes: List[KnowledgeGraphNode],
        temporal_span: Tuple[int, int],
        dominant_themes: List[str],
        causal_structure: str,
        level: int
    ) -> str:
        """
        Generate summary text using Meta LLaMA.
        
        Args:
            nodes: List of nodes in the community
            temporal_span: Temporal span tuple
            dominant_themes: List of dominant theme IDs
            causal_structure: Causal structure description
            level: Community level
            
        Returns:
            Summary text
        """
        # Build context from nodes
        node_descriptions = []
        for node in nodes[:10]:  # Limit to 10 nodes for context
            if node.node_type.value == "character":
                desc = f"Character: {node.properties.get('name', node.node_id)}"
            elif node.node_type.value == "event":
                desc = f"Event: {node.properties.get('description', node.node_id)}"
            else:
                desc = f"{node.node_type.value}: {node.properties.get('name', node.node_id)}"
            node_descriptions.append(desc)
        
        context = "\n".join(node_descriptions)
        
        level_names = {
            0: "entire novel",
            1: "major arc",
            2: "subplot",
            3: "scene-level cluster"
        }
        level_name = level_names.get(level, f"level {level}")
        
        prompt = f"""Summarize the following {level_name} from a novel.

Temporal span: story time {temporal_span[0]} to {temporal_span[1]}

Dominant themes: {', '.join(dominant_themes) if dominant_themes else 'None specified'}

Causal structure: {causal_structure}

Key elements:
{context}

Provide a concise summary that captures the essence of this {level_name}, preserving temporal, thematic, and causal information."""
        
        # Generate summary
        summary = self._generate_response(prompt, max_length=512)
        
        return summary.strip()
    
    def _generate_response(self, prompt: str, max_length: int = 512) -> str:
        """
        Generate response from Meta LLaMA.
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            
        Returns:
            Generated text
        """
        # Use chat template for Llama 3.1 format
        messages = [
            {
                "role": "system",
                "content": "You are a literary analysis assistant. Provide clear, concise summaries of narrative elements."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # Apply chat template (automatically handles Llama 3.1 format)
        if hasattr(self.tokenizer, "apply_chat_template"):
            full_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback to manual format
            full_prompt = f"{prompt}"
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=8192)
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the new tokens (generated part)
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Remove any Llama 3.1 specific tokens if present
        response = response.replace("<|eot_id|>", "").strip()
        
        return response
    
    def summarize_all_communities(
        self,
        communities: Dict[int, Dict[int, List[str]]],
        graph: KnowledgeGraph,
        temporal_timeline: Dict[str, TemporalEvent],
        thematic_states: Dict[str, ThematicState]
    ) -> List[CommunitySummary]:
        """
        Summarize all communities at all levels.
        
        Args:
            communities: Hierarchical community structure
            graph: KnowledgeGraph object
            temporal_timeline: Dictionary of TemporalEvent objects
            thematic_states: Dictionary of ThematicState objects
            
        Returns:
            List of CommunitySummary objects
        """
        summaries = []
        
        for level, level_communities in communities.items():
            for community_id, node_ids in level_communities.items():
                summary = self.summarize_community(
                    community_id,
                    level,
                    node_ids,
                    graph,
                    temporal_timeline,
                    thematic_states
                )
                summaries.append(summary)
        
        return summaries


if __name__ == "__main__":
    # Test example
    from schemas import KnowledgeGraph, KnowledgeGraphNode, NodeType
    
    graph = KnowledgeGraph()
    graph.nodes["e_1"] = KnowledgeGraphNode(
        node_id="e_1",
        node_type=NodeType.CHARACTER,
        properties={"name": "John", "description": "Protagonist"}
    )
    
    summarizer = CommunitySummarization()
    # summary = summarizer.summarize_community(...)
    print("Community summarization layer ready")

