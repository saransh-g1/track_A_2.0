"""
PHASE 1.2: META LLAMA ENCODER - STRUCTURED EXTRACTION (OPTIMIZED)

OPTIMIZED for Track-A performance:
- Batch processing (2-4 chunks at a time)
- Reduced schema: claims, temporal markers, entities (characters only)
- Hard cap: 256 tokens generation, 2048 tokens input
- 5-20× faster than sequential processing

Extracts ONLY:
- Claims (verifiable facts)
- Temporal markers
- Named entities (characters only)

Output schema: StructuredExtraction (empty lists for unused fields)
"""

import json
import re
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from schemas import (
    StructuredExtraction, Entity, Event, Relation, Claim, Theme,
    TemporalMarker, CausalLink
)
from config import META_LLAMA_MODEL_NAME, STRUCTURED_EXTRACTION_PROMPT_TEMPLATE, MAX_EXTRACTION_TOKENS, MODEL_PATH, EXTRACTION_BATCH_SIZE
import torch


class MetaLlamaEncoder:
    """
    Meta LLaMA encoder for structured extraction.
    
    Enforces STRICT JSON schema - no free-form text allowed.
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize Meta LLaMA model.
        
        Args:
            model_name: Override default model name if needed
        """
        import os
        
        # Use direct model path from .env if available, otherwise use model name
        self.model_path = model_name or MODEL_PATH or META_LLAMA_MODEL_NAME
        
        # Verify model path exists if it's a local path
        if MODEL_PATH:
            # MODEL_PATH is set from .env
            if os.path.exists(MODEL_PATH):
                print(f"Loading Meta LLaMA model from local path: {MODEL_PATH}")
                self.model_path = MODEL_PATH
                use_local = True
            else:
                # MODEL_PATH is set but doesn't exist - this is an error
                raise FileNotFoundError(
                    f"Model path from .env file does not exist: {MODEL_PATH}\n"
                    f"Please check your .env file and ensure MODEL_PATH points to a valid directory.\n"
                    f"Current MODEL_PATH: {MODEL_PATH}"
                )
        elif model_name and os.path.exists(model_name):
            # Custom model_name provided and exists
            print(f"Loading Meta LLaMA model from provided path: {model_name}")
            self.model_path = model_name
            use_local = True
        else:
            # Fallback to HuggingFace repo name (will download) - but we should prevent this
            raise ValueError(
                f"MODEL_PATH not set in .env file!\n"
                f"Please create a .env file in {os.path.dirname(__file__)} with:\n"
                f"MODEL_PATH=/path/to/your/model/snapshot/directory\n"
                f"Current fallback would use: {META_LLAMA_MODEL_NAME} (this would download)"
            )
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=use_local,
            trust_remote_code=True
        )
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=use_local,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        self.model.eval()
    
    def _build_prompt(self, chunk_text: str) -> str:
        """
        Build formatted prompt for extraction.
        
        Args:
            chunk_text: Text content of the chunk
            
        Returns:
            Formatted prompt string
        """
        prompt = STRUCTURED_EXTRACTION_PROMPT_TEMPLATE.format(chunk_text=chunk_text)
        
        # Use chat template for Llama 3.1 format
        messages = [
            {
                "role": "system",
                "content": "You are a structured information extractor. You MUST return ONLY valid JSON following the exact schema provided. No explanations, no additional text, only JSON."
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
            # Fallback to manual format for older tokenizers
            full_prompt = f"{prompt}"
        
        return full_prompt
    
    def _generate_response(self, prompt: str, max_new_tokens: int = None) -> str:
        """
        Generate response from Meta LLaMA (single prompt).
        
        Args:
            prompt: Input prompt (full formatted prompt with chat template)
            max_new_tokens: Maximum generation tokens (defaults to MAX_EXTRACTION_TOKENS from config)
            
        Returns:
            Generated text
        """
        if max_new_tokens is None:
            max_new_tokens = MAX_EXTRACTION_TOKENS
        
        # OPTIMIZED: Reduce input size from 8192 to 2048
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,  # Hard cap at 256
                do_sample=False,
                temperature=0.0,  # Deterministic extraction
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
        
        # Decode only the new tokens (generated part)
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Remove any Llama 3.1 specific tokens if present
        response = response.replace("<|eot_id|>", "").strip()
        
        return response
    
    def _generate_batch(self, prompts: list, max_new_tokens: int = None) -> list:
        """
        Generate responses for a batch of prompts (OPTIMIZED).
        
        Args:
            prompts: List of prompt strings
            max_new_tokens: Maximum generation tokens (defaults to MAX_EXTRACTION_TOKENS)
            
        Returns:
            List of generated text responses
        """
        if max_new_tokens is None:
            max_new_tokens = MAX_EXTRACTION_TOKENS
        
        # Tokenize batch with padding
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048  # OPTIMIZED: Reduced from 8192
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,  # Hard cap at 256
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
        
        # Decode each response (skip input tokens)
        responses = []
        input_lengths = inputs['attention_mask'].sum(dim=1).tolist()
        
        for i, output in enumerate(outputs):
            input_len = input_lengths[i]
            generated_tokens = output[input_len:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            response = response.replace("<|eot_id|>", "").strip()
            responses.append(response)
        
        return responses
    
    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """
        Extract JSON from model response.
        
        Handles cases where model adds extra text.
        
        Args:
            response: Model response string
            
        Returns:
            Parsed JSON dictionary
        """
        # Try to find JSON block
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            # If no JSON block found, try parsing entire response
            json_str = response.strip()
        
        # Clean up common issues
        json_str = json_str.strip()
        
        # Remove markdown code blocks if present
        if json_str.startswith('```'):
            json_str = re.sub(r'^```(?:json)?\s*', '', json_str)
            json_str = re.sub(r'\s*```$', '', json_str)
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Response was: {response[:500]}")
            # Return empty structure on error
            # Return empty structure (optimized schema: only entities, claims, temporal_markers)
            return {
                "entities": [],
                "claims": [],
                "temporal_markers": []
            }
    
    def extract_structured(self, chunk_text: str, chunk_id: int) -> StructuredExtraction:
        """
        Extract structured information from a chunk (OPTIMIZED: claims, temporal markers, entities only).
        
        Args:
            chunk_text: Text content of the chunk
            chunk_id: ID of the chunk
            
        Returns:
            StructuredExtraction object (empty lists for unused fields)
        """
        # Build prompt
        full_prompt = self._build_prompt(chunk_text)
        
        # Generate response
        response = self._generate_response(full_prompt)
        
        # Extract and parse JSON
        extracted_json = self._extract_json_from_response(response)
        
        # Convert to structured objects (OPTIMIZED: only extract what we need)
        entities = [
            Entity(**item) for item in extracted_json.get("entities", [])
            if item.get("entity_type") == "character"  # Only characters
        ]
        
        claims = [
            Claim(**item) for item in extracted_json.get("claims", [])
        ]
        
        temporal_markers = [
            TemporalMarker(**item) for item in extracted_json.get("temporal_markers", [])
        ]
        
        # Return StructuredExtraction with empty lists for unused fields (schema compatibility)
        return StructuredExtraction(
            chunk_id=chunk_id,
            entities=entities,
            events=[],  # Not extracted (optimization)
            relations=[],  # Not extracted (optimization)
            claims=claims,
            themes=[],  # Not extracted (optimization)
            temporal_markers=temporal_markers,
            causal_links=[]  # Not extracted (optimization)
        )
    
    def extract_batch(self, chunks: list, batch_size: int = None) -> list:
        """
        Extract structured information from multiple chunks (OPTIMIZED with batching).
        
        Args:
            chunks: List of Chunk objects
            batch_size: Number of chunks to process in parallel (defaults to EXTRACTION_BATCH_SIZE)
            
        Returns:
            List of StructuredExtraction objects
        """
        if batch_size is None:
            batch_size = EXTRACTION_BATCH_SIZE
        
        results = []
        total_chunks = len(chunks)
        
        # Process chunks in batches
        for batch_start in range(0, total_chunks, batch_size):
            batch_end = min(batch_start + batch_size, total_chunks)
            batch_chunks = chunks[batch_start:batch_end]
            
            # Build prompts for this batch
            prompts = [self._build_prompt(chunk.text) for chunk in batch_chunks]
            
            # Generate responses in batch
            responses = self._generate_batch(prompts)
            
            # Parse each response
            for i, (chunk, response) in enumerate(zip(batch_chunks, responses)):
                extracted_json = self._extract_json_from_response(response)
                
                # Convert to structured objects (OPTIMIZED: only extract what we need)
                entities = [
                    Entity(**item) for item in extracted_json.get("entities", [])
                    if item.get("entity_type") == "character"  # Only characters
                ]
                
                claims = [
                    Claim(**item) for item in extracted_json.get("claims", [])
                ]
                
                temporal_markers = [
                    TemporalMarker(**item) for item in extracted_json.get("temporal_markers", [])
                ]
                
                # Create StructuredExtraction with empty lists for unused fields
                extraction = StructuredExtraction(
                    chunk_id=chunk.chunk_id,
                    entities=entities,
                    events=[],
                    relations=[],
                    claims=claims,
                    themes=[],
                    temporal_markers=temporal_markers,
                    causal_links=[]
                )
                
                results.append(extraction)
            
            print(f"  Processed batch: {batch_end}/{total_chunks} chunks")
        
        return results


if __name__ == "__main__":
    # Test example
    test_chunk = """
    John walked into the room. He saw Mary sitting by the window.
    She looked sad. John asked, "What's wrong?" Mary replied, "I lost my job today."
    This caused John to feel concerned. He promised to help her find a new one.
    """
    
    encoder = MetaLlamaEncoder()
    extraction = encoder.extract_structured(test_chunk, chunk_id=1)
    
    print(f"Extracted {len(extraction.entities)} entities")
    print(f"Extracted {len(extraction.events)} events")
    print(f"Extracted {len(extraction.causal_links)} causal links")

