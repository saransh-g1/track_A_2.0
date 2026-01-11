"""
PHASE 2.6: META LLAMA DECODER — FINAL ANSWER

Generate final answer ONLY from:
- Reduced partial answers
- Graph-verified facts

Optionally cite:
- event_id
- entity_id
- community_id
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple
from schemas import PartialAnswer, FinalAnswer
from config import META_LLAMA_MODEL_NAME, MODEL_PATH
from phase2_5_reduce_step import ReduceStep


class MetaLlamaDecoder:
    """
    Meta LLaMA decoder for final answer generation.
    
    Generates final answer from reduced partial answers and graph-verified facts.
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize Meta LLaMA model for decoding.
        
        Args:
            model_name: Override default model name if needed
        """
        # Use direct model path from .env if available
        self.model_path = model_name or MODEL_PATH or META_LLAMA_MODEL_NAME
        
        print(f"Loading Meta LLaMA model for final answer decoding: {self.model_path}")
        
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
        self.reduce_step = ReduceStep()
    
    def generate_final_answer(
        self,
        query: str,
        reduced_answers: List[PartialAnswer]
    ) -> FinalAnswer:
        """
        Generate final answer from reduced partial answers.
        
        Process:
        1. Merge reduced partial answers
        2. Generate final answer using Meta LLaMA
        3. Extract citations
        4. Compute confidence
        
        Args:
            query: User query
            reduced_answers: List of reduced PartialAnswer objects
            
        Returns:
            FinalAnswer object
        """
        # Merge reduced answers
        merged_context = self.reduce_step.merge_answers(reduced_answers)
        
        # Generate final answer
        answer_text = self._generate_answer(query, merged_context, reduced_answers)
        
        # Extract citations
        citations = self._extract_citations(reduced_answers)
        
        # Compute temporal span
        temporal_span = self._compute_temporal_span(reduced_answers)
        
        # Get communities used
        communities_used = [answer.community_id for answer in reduced_answers]
        
        # Compute confidence
        confidence = self._compute_confidence(reduced_answers)
        
        return FinalAnswer(
            answer_text=answer_text,
            citations=citations,
            temporal_span=temporal_span,
            communities_used=communities_used,
            confidence=confidence
        )
    
    def _generate_answer(
        self,
        query: str,
        merged_context: str,
        reduced_answers: List[PartialAnswer]
    ) -> str:
        """
        Generate final answer using Meta LLaMA.
        
        Args:
            query: User query
            merged_context: Merged context from partial answers
            reduced_answers: List of reduced PartialAnswer objects
            
        Returns:
            Generated answer text
        """
        # Build prompt
        prompt = f"""Based on the following verified information from a novel knowledge graph, provide a comprehensive answer to the user's question.

Verified information (from graph-verified facts and community summaries):
{merged_context}

User question: {query}

Provide a clear, comprehensive answer that:
1. Draws ONLY from the verified information provided
2. Maintains temporal and causal consistency
3. Preserves thematic coherence
4. Cites specific events or entities when relevant

Answer:"""
        
        answer = self._generate_response(prompt, max_length=1024)
        return answer.strip()
    
    def _extract_citations(self, reduced_answers: List[PartialAnswer]) -> List[Dict[str, str]]:
        """
        Extract citations from reduced answers.
        
        Args:
            reduced_answers: List of reduced PartialAnswer objects
            
        Returns:
            List of citation dictionaries
        """
        citations = []
        seen_citations = set()
        
        for answer in reduced_answers:
            # Add event citations
            for event_id in answer.supporting_event_ids:
                citation_key = f"event:{event_id}"
                if citation_key not in seen_citations:
                    citations.append({
                        "type": "event",
                        "id": event_id
                    })
                    seen_citations.add(citation_key)
            
            # Add entity citations
            for entity_id in answer.supporting_entity_ids:
                citation_key = f"entity:{entity_id}"
                if citation_key not in seen_citations:
                    citations.append({
                        "type": "entity",
                        "id": entity_id
                    })
                    seen_citations.add(citation_key)
            
            # Add community citation
            citation_key = f"community:{answer.community_id}"
            if citation_key not in seen_citations:
                citations.append({
                    "type": "community",
                    "id": answer.community_id
                })
                seen_citations.add(citation_key)
        
        return citations
    
    def _compute_temporal_span(
        self,
        reduced_answers: List[PartialAnswer]
    ) -> Tuple[int, int]:
        """
        Compute temporal span from reduced answers.
        
        Args:
            reduced_answers: List of reduced PartialAnswer objects
            
        Returns:
            Tuple of (start_t_story, end_t_story)
        """
        if not reduced_answers:
            return (0, 0)
        
        min_t = min(answer.temporal_coverage[0] for answer in reduced_answers)
        max_t = max(answer.temporal_coverage[1] for answer in reduced_answers)
        
        return (min_t, max_t)
    
    def _compute_confidence(
        self,
        reduced_answers: List[PartialAnswer]
    ) -> float:
        """
        Compute confidence score for final answer.
        
        Args:
            reduced_answers: List of reduced PartialAnswer objects
            
        Returns:
            Confidence score (0-1)
        """
        if not reduced_answers:
            return 0.0
        
        # Average relevance and thematic alignment
        avg_relevance = sum(a.relevance_score for a in reduced_answers) / len(reduced_answers)
        avg_thematic = sum(a.thematic_alignment_score for a in reduced_answers) / len(reduced_answers)
        
        # Normalize relevance (0-100 -> 0-1)
        normalized_relevance = avg_relevance / 100.0
        
        # Combine with thematic alignment
        confidence = (normalized_relevance + avg_thematic) / 2.0
        
        # Boost if multiple answers agree
        if len(reduced_answers) > 1:
            confidence *= 1.1
        
        # Clamp to 0-1
        return min(1.0, max(0.0, confidence))
    
    def _generate_response(self, prompt: str, max_length: int = 1024) -> str:
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
                "content": "You are a helpful assistant that provides comprehensive answers based on verified information from knowledge graphs."
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


if __name__ == "__main__":
    # Test example
    from schemas import PartialAnswer
    
    reduced_answers = [
        PartialAnswer(
            community_id="c_1",
            text="John starts his journey.",
            relevance_score=80.0,
            temporal_coverage=(0, 50),
            thematic_alignment_score=0.8,
            supporting_event_ids=["ev_1"],
            supporting_entity_ids=["e_1"]
        )
    ]
    
    decoder = MetaLlamaDecoder()
    query = "What does John do?"
    final_answer = decoder.generate_final_answer(query, reduced_answers)
    print(f"Generated final answer with confidence {final_answer.confidence}")
    print(f"Citations: {len(final_answer.citations)}")

