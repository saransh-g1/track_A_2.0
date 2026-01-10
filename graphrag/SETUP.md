# Setup Guide for Track-A GraphRAG System

This guide covers all external dependencies and setup steps needed to run the system.

## 1. Python Environment

**Required Python Version**: Python 3.8 or higher (3.9+ recommended)

### Create Virtual Environment (Recommended)

```bash
cd graphrag
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## 2. Python Dependencies

Install all required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Key Dependencies Breakdown:

- **torch>=2.0.0** - PyTorch for Meta LLaMA model
- **transformers>=4.35.0** - HuggingFace transformers for Meta LLaMA
- **sentencepiece>=0.1.99** - Tokenizer for LLaMA models
- **tiktoken>=0.5.0** - Token counting for chunking
- **networkx>=3.2** - Graph operations
- **igraph>=0.11.0** - Graph processing (requires system dependencies)
- **leidenalg>=0.10.0** - Leiden algorithm for community detection
- **pydantic>=2.5.0** - Data validation
- **numpy>=1.24.0** - Numerical operations
- **pathway>=0.7.0** - Vector storage (may need alternatives)

## 3. System Dependencies (Required for igraph)

### Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y build-essential
sudo apt-get install -y libigraph0-dev
sudo apt-get install -y python3-dev
```

### macOS:
```bash
brew install igraph
```

### Windows:
Install from: https://igraph.org/c/html/latest/igraph-Installation.html

## 4. HuggingFace Account & Access Token

**CRITICAL**: Meta LLaMA models require HuggingFace access.

### Steps:

1. **Create HuggingFace Account**: https://huggingface.co/join

2. **Request Access to Meta LLaMA Models**:
   - Go to: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
   - Click "Request Access" or "Agree and Access Repository"
   - Fill out the Meta LLaMA access form

3. **Generate Access Token**:
   - Go to: https://huggingface.co/settings/tokens
   - Click "New token"
   - Create a token with "Read" permissions
   - Copy the token

4. **Set Environment Variable**:
   ```bash
   export HF_TOKEN="your_huggingface_token_here"
   # Or add to ~/.bashrc or ~/.zshrc for persistence
   ```

5. **Login via HuggingFace CLI (Alternative)**:
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   # Enter your token when prompted
   ```

## 5. GPU Setup (Optional but Recommended)

### For NVIDIA GPU with CUDA:

```bash
# Install CUDA-enabled PyTorch (adjust for your CUDA version)
# Check your CUDA version: nvidia-smi
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Verify GPU Access:
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Should show GPU name
```

**Note**: System works on CPU but will be significantly slower.

## 6. Model Configuration

The system uses `meta-llama/Meta-Llama-3.1-8B-Instruct` by default.

### To Change Model:

Edit `config.py`:
```python
META_LLAMA_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Change this
```

### Alternative Models (if you have access):
- `meta-llama/Meta-Llama-3.1-8B-Instruct` (default)
- `meta-llama/Meta-Llama-3.1-70B-Instruct` (if available)
- `meta-llama/Llama-3-8B-Instruct` (alternative)
- `meta-llama/Llama-2-7b-chat-hf` (older version)

## 7. Directory Structure Setup

The system will create these directories automatically, but you can create them manually:

```bash
cd graphrag
mkdir -p graph_storage
mkdir -p pathway_storage
```

## 8. Test Installation

Create a test script `test_setup.py`:

```python
#!/usr/bin/env python3
"""Test if all dependencies are installed correctly."""

print("Testing dependencies...")

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
except ImportError:
    print("✗ PyTorch not installed")

try:
    import transformers
    print(f"✓ Transformers {transformers.__version__}")
except ImportError:
    print("✗ Transformers not installed")

try:
    import igraph
    print(f"✓ igraph {igraph.__version__}")
except ImportError:
    print("✗ igraph not installed")

try:
    import leidenalg
    print(f"✓ leidenalg installed")
except ImportError:
    print("✗ leidenalg not installed")

try:
    import tiktoken
    print(f"✓ tiktoken installed")
except ImportError:
    print("✗ tiktoken not installed")

try:
    from transformers import AutoTokenizer
    print("✓ HuggingFace transformers working")
except ImportError:
    print("✗ HuggingFace transformers not working")

print("\nTesting HuggingFace access...")
import os
if os.environ.get("HF_TOKEN") or os.path.exists(os.path.expanduser("~/.huggingface/token")):
    print("✓ HuggingFace token found")
else:
    print("⚠ HuggingFace token not found - set HF_TOKEN environment variable")

print("\nSetup test complete!")
```

Run: `python test_setup.py`

## 9. First Run - Quick Test

### Step 1: Prepare a Test Novel

Create a small test file `test_novel.txt`:

```
CHAPTER 1

John walked into the room. He saw Mary sitting by the window.
She looked sad. John asked, "What's wrong?" Mary replied, "I lost my job today."
This caused John to feel concerned. He promised to help her find a new one.

CHAPTER 2

The next morning, everything had changed. The storm had passed, but new challenges awaited.
John met with his friend Tom to discuss Mary's situation.
```

### Step 2: Run Offline Construction

```bash
cd graphrag
python offline_graph_construction.py --novel_path test_novel.txt
```

**Expected Output:**
- Model will download on first run (can take time)
- Progress through all 9 phases
- Graph storage created

### Step 3: Run Query Processing

```bash
python online_query_processing.py --query "What happens to John?"
```

**Expected Output:**
- Query encoded
- Communities selected
- Final answer generated

## 10. Common Issues & Solutions

### Issue 1: "401 Unauthorized" or "Model Not Found"

**Solution**: 
- Verify HuggingFace token is set: `echo $HF_TOKEN`
- Re-request access to Meta LLaMA models
- Try logging in: `huggingface-cli login`

### Issue 2: "igraph installation failed"

**Solution**:
- Install system dependencies first (see section 3)
- On Ubuntu: `sudo apt-get install libigraph0-dev`
- On macOS: `brew install igraph`

### Issue 3: "CUDA out of memory"

**Solution**:
- Use smaller model: `Meta-Llama-3.1-8B-Instruct` instead of larger ones
- Reduce batch size in code
- Use CPU mode: Code will fallback automatically

### Issue 4: "ModuleNotFoundError: No module named 'phase1_1'"

**Solution**:
- Make sure you're in the `graphrag` directory
- All phase files should be in same directory
- Run from `graphrag/` directory: `python offline_graph_construction.py ...`

### Issue 5: "Pathway not found" or pathway errors

**Solution**:
- Pathway library may need additional setup
- Alternative: The code can work without pathway, using JSON storage
- Check `phase1_9_pathway_storage.py` for storage implementation

## 11. Memory Requirements

**Minimum:**
- RAM: 16GB (8GB if using CPU-only)
- Disk: 20GB free space (for model + data)

**Recommended:**
- RAM: 32GB+
- GPU: 16GB+ VRAM (for faster processing)
- Disk: 50GB+ free space

## 12. Environment Variables (Optional)

Create `.env` file in `graphrag/` directory:

```bash
# HuggingFace Token
HF_TOKEN=your_token_here

# Model Configuration
META_LLAMA_MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct

# Storage Paths
GRAPH_STORAGE_PATH=./graph_storage
PATHWAY_STORAGE_PATH=./pathway_storage

# CUDA Device (if multiple GPUs)
CUDA_VISIBLE_DEVICES=0
```

Then install python-dotenv:
```bash
pip install python-dotenv
```

## 13. Next Steps

1. ✅ Complete all setup steps above
2. ✅ Test with small novel file
3. ✅ Run offline graph construction
4. ✅ Run query processing
5. ✅ Check output files in `graph_storage/` and `pathway_storage/`

## 14. Production Considerations

For production use:

1. **Use GPU** - Significantly faster
2. **Batch Processing** - Process multiple novels
3. **Error Handling** - Add try-catch blocks
4. **Logging** - Add proper logging
5. **Caching** - Cache model loads
6. **Monitoring** - Track resource usage

---

**Need Help?** Check the ARCHITECTURE.md file for detailed system documentation.

