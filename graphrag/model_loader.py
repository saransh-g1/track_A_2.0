"""
Model Loading Utility with Caching Support

This module provides utilities for loading models with proper caching configuration
to prevent re-downloading models that are already in the local cache.

The caching logic works as follows:
1. When USE_LOCAL_FILES_ONLY=True, models are loaded from ~/.cache/huggingface/hub/ without network checks
2. The Hugging Face Hub library manages caching under ~/.cache/huggingface/hub/
3. Files are stored in a content-addressed format and reused if already downloaded
4. Setting local_files_only=True prevents Transformers from checking for updates online
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Tuple
from config import META_LLAMA_MODEL_NAME, USE_LOCAL_FILES_ONLY


def load_model_with_cache(
    model_name: Optional[str] = None,
    local_files_only: Optional[bool] = None,
    torch_dtype: Optional[torch.dtype] = None,
    device_map: Optional[str] = None,
    trust_remote_code: bool = True
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Load tokenizer and model with proper caching configuration.
    
    This function ensures that models are loaded from local cache when available,
    preventing unnecessary re-downloads. The caching behavior is controlled by
    the USE_LOCAL_FILES_ONLY configuration setting.
    
    Args:
        model_name: Model name or path. If None, uses META_LLAMA_MODEL_NAME from config
        local_files_only: If True, only use local cache files (no network checks).
                         If None, uses USE_LOCAL_FILES_ONLY from config
        torch_dtype: Data type for model weights. If None, auto-selects based on CUDA availability
        device_map: Device mapping strategy. If None, auto-selects based on CUDA availability
        trust_remote_code: Whether to trust remote code in model loading
        
    Returns:
        Tuple of (tokenizer, model)
        
    Raises:
        OSError: If local_files_only=True and model is not found in cache
    """
    # Use config defaults if not specified
    model_name = model_name or META_LLAMA_MODEL_NAME
    local_files_only = local_files_only if local_files_only is not None else USE_LOCAL_FILES_ONLY
    
    # Auto-select dtype and device_map if not specified
    if torch_dtype is None:
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    if device_map is None:
        device_map = "auto" if torch.cuda.is_available() else None
    
    # Print cache status
    cache_status = "local cache only" if local_files_only else "with network checks"
    print(f"Loading model '{model_name}' from {cache_status}")
    
    if local_files_only:
        print("  → Using cached model files (no network access)")
    else:
        print("  → May check for model updates from Hugging Face Hub")
    
    # Load tokenizer with caching configuration
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code
    )
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with caching configuration
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code
    )
    
    model.eval()
    
    print(f"  → Model loaded successfully")
    
    return tokenizer, model


def get_cache_info(model_name: Optional[str] = None) -> dict:
    """
    Get information about the local cache for a model.
    
    Args:
        model_name: Model name. If None, uses META_LLAMA_MODEL_NAME from config
        
    Returns:
        Dictionary with cache information including:
        - cache_dir: Base cache directory
        - model_cache_path: Path to model cache (if exists)
        - exists: Whether model exists in cache
        - cache_size: Size of cache (if available)
        - num_repos: Number of cached repos (if available)
    """
    model_name = model_name or META_LLAMA_MODEL_NAME
    
    # Get default cache directory
    # Can be overridden by HF_HOME or TRANSFORMERS_CACHE environment variables
    cache_dir = os.path.expanduser(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    )
    hub_cache_dir = os.path.join(cache_dir, "hub")
    
    # Convert model name to cache path format
    # e.g., "meta-llama/Meta-Llama-3-8B-Instruct" -> "models--meta-llama--Meta-Llama-3-8B-Instruct"
    cache_model_name = model_name.replace("/", "--")
    model_cache_path = os.path.join(hub_cache_dir, f"models--{cache_model_name}")
    
    info = {
        "cache_dir": hub_cache_dir,
        "model_cache_path": model_cache_path,
        "exists": os.path.exists(model_cache_path),
        "model_name": model_name
    }
    
    # Try to get more detailed cache info using huggingface_hub
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        info["cache_size"] = cache_info.size_on_disk_str if hasattr(cache_info, 'size_on_disk_str') else "unknown"
        info["num_repos"] = cache_info.num_repos if hasattr(cache_info, 'num_repos') else "unknown"
    except ImportError:
        # huggingface_hub not available, skip detailed info
        info["cache_size"] = "unknown (huggingface_hub not available)"
        info["num_repos"] = "unknown (huggingface_hub not available)"
    except Exception as e:
        # Other error, just record it
        info["cache_size"] = f"error: {str(e)}"
        info["num_repos"] = "unknown"
    
    return info


if __name__ == "__main__":
    # Test cache info
    print("Cache Information:")
    cache_info = get_cache_info()
    for key, value in cache_info.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*60)
    print("Testing model loading with cache...")
    
    try:
        tokenizer, model = load_model_with_cache()
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("\nIf model is not in cache, try running with USE_LOCAL_FILES_ONLY=False first")

