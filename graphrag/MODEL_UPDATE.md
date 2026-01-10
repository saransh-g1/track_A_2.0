# Model Update: Meta LLaMA 3.1 8B Instruct

## Summary

The codebase has been updated to use **Meta LLaMA 3.1 8B Instruct** instead of Llama 2.7B.

## Changes Made

### 1. Configuration (`config.py`)
- ✅ Updated `META_LLAMA_MODEL_NAME` to `"meta-llama/Meta-Llama-3.1-8B-Instruct"`
- ✅ Updated `EMAX_CONTEXT_LENGTH` to `8192` (Llama 3.1 supports 8K context vs 4K for Llama 2)

### 2. Prompt Templates
Updated all prompt formatting to use Llama 3.1 chat template format:

**Files Updated:**
- ✅ `phase1_2_meta_llama_encoder.py` - Uses `apply_chat_template()` for proper formatting
- ✅ `phase1_8_community_summarization.py` - Updated to chat template format
- ✅ `phase2_3_map_step.py` - Updated to chat template format
- ✅ `phase2_6_meta_llama_decoder.py` - Updated to chat template format

**Changes:**
- Now uses `tokenizer.apply_chat_template()` which automatically handles Llama 3.1 format
- Falls back to manual format if chat template not available
- Properly extracts only generated tokens (not input tokens)

### 3. Response Parsing
- ✅ Updated all `_generate_response()` methods to use `max_new_tokens` instead of `max_length`
- ✅ Properly extracts only generated tokens from output
- ✅ Removes Llama 3.1 specific tokens like `<|eot_id|>`

### 4. Documentation
- ✅ Updated `SETUP.md` with new model name
- ✅ Updated model access instructions for Llama 3.1
- ✅ Updated example configuration

## Model Requirements

### HuggingFace Access
You need to:
1. **Request Access** to Meta LLaMA 3.1 8B Instruct:
   - Go to: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
   - Click "Request Access" or "Agree and Access Repository"
   - Fill out the Meta LLaMA access form

2. **Set HuggingFace Token**:
   ```bash
   export HF_TOKEN="your_huggingface_token_here"
   # Or login via CLI:
   huggingface-cli login
   ```

### Model Size
- **Model Size**: ~16GB (8B parameters)
- **Memory Required**: 
  - Minimum: 16GB RAM (CPU mode)
  - Recommended: 32GB+ RAM or GPU with 16GB+ VRAM

## Benefits of Llama 3.1

1. **Better Performance**: Improved instruction following and reasoning
2. **Longer Context**: 8K tokens vs 4K (better for full novels)
3. **Better JSON Output**: Improved structured output generation
4. **Modern Format**: Uses latest chat template format

## Testing

After updating, test with:

```bash
cd graphrag
python offline_graph_construction.py --novel_path files/test_novel.txt
```

Check that:
- ✅ Model loads successfully
- ✅ Prompts format correctly
- ✅ JSON extraction works
- ✅ Responses are clean (no extra tokens)

## Rollback

If you need to rollback to Llama 2.7B:

1. Edit `config.py`:
   ```python
   META_LLAMA_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
   EMAX_CONTEXT_LENGTH = 4096
   ```

2. The chat template will automatically handle the format difference.

---

**Last Updated**: Model updated to Meta LLaMA 3.1 8B Instruct
**Status**: ✅ Complete

