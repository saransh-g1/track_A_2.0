"""
PHASE 1.2: META LLAMA ENCODER - STRUCTURED EXTRACTION

For EACH chunk, calls Meta LLaMA with a STRICT JSON SCHEMA.
No free-form text allowed.

Extracts:
- Entities (characters, locations, objects)
- Events
- Relations
- Claims (verifiable facts)
- Themes
- Temporal markers
- Causal hints

Output schema: StructuredExtraction
"""

import json
import re
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from schemas import (
    StructuredExtraction, Entity, Event, Relation, Claim, Theme,
    TemporalMarker, CausalLink
)
from config import META_LLAMA_MODEL_NAME, STRUCTURED_EXTRACTION_PROMPT_TEMPLATE
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
        self.model_name = model_name or META_LLAMA_MODEL_NAME
        
        print(f"Loading Meta LLaMA model: {self.model_name}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        self.model.eval()
    
    def _generate_response(self, prompt: str, max_length: int = 2048) -> str:
        """
        Generate response from Meta LLaMA.
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            
        Returns:
            Generated text
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                do_sample=False,
                temperature=0.0,  # Deterministic extraction
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove prompt from response
        if prompt in response:
            response = response[len(prompt):].strip()
        
        return response
    
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
            return {
                "entities": [],
                "events": [],
                "relations": [],
                "claims": [],
                "themes": [],
                "temporal_markers": [],
                "causal_links": []
            }
    
    def extract_structured(self, chunk_text: str, chunk_id: int) -> StructuredExtraction:
        """
        Extract structured information from a chunk.
        
        This is the main extraction method that enforces STRICT JSON schema.
        
        Args:
            chunk_text: Text content of the chunk
            chunk_id: ID of the chunk
            
        Returns:
            StructuredExtraction object with all extracted information
        """
        # Format prompt with chunk text
        prompt = STRUCTURED_EXTRACTION_PROMPT_TEMPLATE.format(chunk_text=chunk_text)
        
        # Add system message for structured output
        full_prompt = f"""<s>[INST] <<SYS>>
You are a structured information extractor. You MUST return ONLY valid JSON following the exact schema provided. No explanations, no additional text, only JSON.
<</SYS>>

{prompt} [/INST]"""
        
        # Generate response
        response = self._generate_response(full_prompt)
        
        # Extract and parse JSON
        extracted_json = self._extract_json_from_response(response)
        
        # Convert to structured objects
        entities = [
            Entity(**item) for item in extracted_json.get("entities", [])
        ]
        
        events = [
            Event(**item) for item in extracted_json.get("events", [])
        ]
        
        relations = [
            Relation(**item) for item in extracted_json.get("relations", [])
        ]
        
        claims = [
            Claim(**item) for item in extracted_json.get("claims", [])
        ]
        
        themes = [
            Theme(**item) for item in extracted_json.get("themes", [])
        ]
        
        temporal_markers = [
            TemporalMarker(**item) for item in extracted_json.get("temporal_markers", [])
        ]
        
        causal_links = [
            CausalLink(**item) for item in extracted_json.get("causal_links", [])
        ]
        
        return StructuredExtraction(
            chunk_id=chunk_id,
            entities=entities,
            events=events,
            relations=relations,
            claims=claims,
            themes=themes,
            temporal_markers=temporal_markers,
            causal_links=causal_links
        )
    
    def extract_batch(self, chunks: list) -> list:
        """
        Extract structured information from multiple chunks.
        
        Args:
            chunks: List of Chunk objects
            
        Returns:
            List of StructuredExtraction objects
        """
        results = []
        for chunk in chunks:
            extraction = self.extract_structured(chunk.text, chunk.chunk_id)
            results.append(extraction)
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

