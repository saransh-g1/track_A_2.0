"""
PHASE 2.3: MAP STEP (PER COMMUNITY)

For each selected community:
- Generate a partial answer
- Score relevance (0–100)
- Track temporal coverage
- Track thematic alignment

Schema: PartialAnswer
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple
from schemas import PartialAnswer, PathwayCommunitySummary
from config import META_LLAMA_MODEL_NAME, MIN_RELEVANCE_SCORE, MODEL_PATH


class MapStep:
    """
    Map step layer.
    
    Generates partial answers for each selected community.
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize Meta LLaMA model for map step.
        
        Args:
            model_name: Override default model name if needed
        """
        # Use direct model path from .env if available
        self.model_path = model_name or MODEL_PATH or META_LLAMA_MODEL_NAME
        
        print(f"Loading Meta LLaMA model for map step: {self.model_path}")
        
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
    
    def generate_partial_answer(
        self,
        query: str,
        community_summary: PathwayCommunitySummary,
        initial_relevance_score: float
    ) -> PartialAnswer:
        """
        Generate a partial answer for a community.
        
        Args:
            query: User query
            community_summary: PathwayCommunitySummary object
            initial_relevance_score: Initial relevance score from community selection
            
        Returns:
            PartialAnswer object
        """
        # Generate partial answer text
        answer_text = self._generate_answer_text(query, community_summary)
        
        # Compute relevance score (0-100)
        relevance_score = self._compute_relevance_score(
            query,
            community_summary,
            answer_text,
            initial_relevance_score
        )
        
        # Get temporal coverage
        temporal_coverage = tuple(community_summary.metadata.get("time_range", [0, 0]))
        
        # Compute thematic alignment score
        thematic_alignment_score = self._compute_thematic_alignment(
            query,
            community_summary
        )
        
        # Get supporting event and entity IDs
        supporting_event_ids = []
        supporting_entity_ids = []
        node_ids = community_summary.metadata.get("node_ids", [])
        for node_id in node_ids:
            if node_id.startswith("ev_"):
                supporting_event_ids.append(node_id)
            elif node_id.startswith("e_"):
                supporting_entity_ids.append(node_id)
        
        return PartialAnswer(
            community_id=community_summary.community_id,
            text=answer_text,
            relevance_score=relevance_score,
            temporal_coverage=temporal_coverage,
            thematic_alignment_score=thematic_alignment_score,
            supporting_event_ids=supporting_event_ids,
            supporting_entity_ids=supporting_entity_ids
        )
    
    def _generate_answer_text(
        self,
        query: str,
        community_summary: PathwayCommunitySummary
    ) -> str:
        """
        Generate partial answer text using Meta LLaMA.
        
        Args:
            query: User query
            community_summary: PathwayCommunitySummary object
            
        Returns:
            Generated answer text
        """
        summary_text = community_summary.summary_text
        metadata = community_summary.metadata
        
        # Build context
        context_parts = [f"Community summary: {summary_text}"]
        
        if metadata.get("causal_structure"):
            context_parts.append(f"Causal structure: {metadata['causal_structure']}")
        
        context = "\n".join(context_parts)
        
        prompt = f"""Based on the following community summary from a novel, answer the user's question.

{context}

User question: {query}

Provide a concise answer that draws from the community summary. Focus on facts and events mentioned in the summary."""
        
        answer = self._generate_response(prompt, max_length=512)
        return answer.strip()
    
    def _compute_relevance_score(
        self,
        query: str,
        community_summary: PathwayCommunitySummary,
        answer_text: str,
        initial_score: float
    ) -> float:
        """
        Compute relevance score (0-100).
        
        Args:
            query: User query
            community_summary: PathwayCommunitySummary object
            answer_text: Generated answer text
            initial_score: Initial relevance score from vector similarity
            
        Returns:
            Relevance score (0-100)
        """
        # Base score from vector similarity (0-1 range)
        base_score = initial_score * 100  # Convert to 0-100
        
        # Adjust based on answer quality
        # Simple heuristic: longer answers might be more relevant
        # In production, could use more sophisticated scoring
        if len(answer_text) > 50:
            base_score *= 1.1  # Boost for substantial answers
        
        # Clamp to 0-100
        return min(100.0, max(0.0, base_score))
    
    def _compute_thematic_alignment(
        self,
        query: str,
        community_summary: PathwayCommunitySummary
    ) -> float:
        """
        Compute thematic alignment score (0-1).
        
        Args:
            query: User query
            community_summary: PathwayCommunitySummary object
            
        Returns:
            Thematic alignment score (0-1)
        """
        # Simple heuristic: check if query mentions themes
        # In production, could use more sophisticated matching
        themes = community_summary.metadata.get("themes", [])
        
        if not themes:
            return 0.5  # Neutral if no themes
        
        # For now, return a default score
        # Could be enhanced with theme-query matching
        return 0.7
    
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
                "content": "You are a helpful assistant that answers questions about novels based on community summaries."
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
    
    def map_all_communities(
        self,
        query: str,
        selected_communities: List[Tuple[str, float]],
        community_summaries: Dict[str, PathwayCommunitySummary]
    ) -> List[PartialAnswer]:
        """
        Generate partial answers for all selected communities.
        
        Args:
            query: User query
            selected_communities: List of (community_id, relevance_score) tuples
            community_summaries: Dictionary of PathwayCommunitySummary objects
            
        Returns:
            List of PartialAnswer objects
        """
        partial_answers = []
        
        for community_id, initial_score in selected_communities:
            if community_id not in community_summaries:
                continue
            
            community_summary = community_summaries[community_id]
            partial_answer = self.generate_partial_answer(
                query,
                community_summary,
                initial_score
            )
            
            # Filter by minimum relevance score
            if partial_answer.relevance_score >= MIN_RELEVANCE_SCORE:
                partial_answers.append(partial_answer)
        
        return partial_answers


if __name__ == "__main__":
    # Test example
    from schemas import PathwayCommunitySummary
    
    community_summary = PathwayCommunitySummary(
        community_id="c_1_0",
        embedding=[0.1] * 768,
        metadata={
            "time_range": [0, 100],
            "themes": ["th_1"],
            "node_ids": ["ev_1", "e_1"]
        },
        summary_text="John goes on a journey."
    )
    
    mapper = MapStep()
    query = "What does John do?"
    partial_answer = mapper.generate_partial_answer(query, community_summary, 0.8)
    print(f"Generated partial answer with relevance {partial_answer.relevance_score}")

