"""
PHASE 2.1: QUERY ENCODING

Encode query using Meta LLaMA embedder.

Output: query_embedding ∈ ℝ^768
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import META_LLAMA_MODEL_NAME, EMBEDDING_DIMENSION


class QueryEncoding:
    """
    Query encoding layer.
    
    Encodes queries using Meta LLaMA embedder.
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize Meta LLaMA model for query encoding.
        
        Args:
            model_name: Override default model name if needed
        """
        self.model_name = model_name or META_LLAMA_MODEL_NAME
        
        print(f"Loading Meta LLaMA model for query encoding: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        self.model.eval()
    
    def encode_query(self, query: str) -> list:
        """
        Encode a query into 768-dim embedding.
        
        Args:
            query: Query string
            
        Returns:
            List of 768 float values (query_embedding ∈ ℝ^768)
        """
        inputs = self.tokenizer(
            query,
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
            
            # Ensure 768 dimensions
            if embeddings.shape[0] != EMBEDDING_DIMENSION:
                if embeddings.shape[0] > EMBEDDING_DIMENSION:
                    embeddings = embeddings[:EMBEDDING_DIMENSION]
                else:
                    padding = torch.zeros(EMBEDDING_DIMENSION - embeddings.shape[0])
                    embeddings = torch.cat([embeddings, padding])
        
        # Convert to list and normalize
        embedding_list = embeddings.cpu().numpy().tolist()
        
        # Normalize
        norm = np.linalg.norm(embedding_list)
        if norm > 0:
            embedding_list = (np.array(embedding_list) / norm).tolist()
        
        return embedding_list[:EMBEDDING_DIMENSION]


if __name__ == "__main__":
    # Test example
    encoder = QueryEncoding()
    query = "What happens to the protagonist?"
    embedding = encoder.encode_query(query)
    print(f"Query encoded to {len(embedding)}-dim vector")

