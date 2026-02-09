# Quick Start: RAG Upload Optimization

## 1-Minute Setup

### Step 1: Install Dependency

```bash
pip install PyMuPDF
```

That's it! The optimization is now enabled.

### Step 2: Upload Your PDF

**Option A: Using Jupyter Notebook**

```python
from tools.rag import RAGTool

# Initialize
tool = RAGTool(username="admin")  # or your username

# Upload (automatically uses optimized method)
result = tool.upload_document(
    collection_name="my_documents",
    document_path="path/to/your/large_manual.pdf"
)

# Check results
print(f"✓ Uploaded in {result['timing']['total']:.1f}s")
print(f"  Created {result['chunks_created']:,} chunks")
print(f"  Speedup estimate: {result.get('speedup_estimate', 'N/A')}")
```

**Option B: Using API**

```bash
curl -X POST "http://localhost:10006/api/tools/rag/upload" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "collection_name=my_documents" \
  -F "file=@path/to/your/large_manual.pdf"
```

### Step 3: Verify It Worked

Look for these log messages indicating optimization is active:

```
[RAG] Using optimized PDF uploader for: large_manual.pdf
[OPTIMIZED UPLOAD] Starting optimized upload (0.0%)
[OPTIMIZED UPLOAD] PDF has 523 pages (5.0%)
[OPTIMIZED UPLOAD] Extracting text from 27 batches in parallel (10.0%)
...
[OPTIMIZED UPLOAD] Upload complete in 45.3s (100.0%)
```

## Performance Check

Run the benchmark to see the speedup:

```bash
python tests/test_rag_upload_performance.py your_document.pdf
```

Expected output:
```
Performance Comparison
==============================================================================
Optimized time:  45.2s
Original time:   308.7s
Speedup:         6.8x faster

🚀 Excellent! Optimizations are working great!
```

## What Changed?

### Before (Old Method)
- **Time**: 5-10 minutes for 500-page PDF
- **Method**: Sequential page-by-page processing
- **Feedback**: None (appears frozen)
- **CPU**: 1 core used

### After (Optimized Method)
- **Time**: 45-60 seconds for 500-page PDF
- **Method**: Parallel batch processing
- **Feedback**: Real-time progress updates
- **CPU**: All cores used

## Troubleshooting

### Still Slow?

**Check 1: Is PyMuPDF installed?**
```python
import fitz  # Should not error
print("PyMuPDF is installed ✓")
```

**Check 2: Is optimization enabled?**
Look for this log line:
```
[RAG] Using optimized PDF uploader for: ...
```

If you see this instead, optimization is disabled:
```
[RAG] Processing document: ...
```

**Check 3: System resources**
- CPU: Should be near 100% on all cores during "Extracting text"
- RAM: Should use ~400-800 MB
- Disk: SSD recommended (10x faster than HDD)

### Error: "PyMuPDF not installed"

```bash
pip install PyMuPDF
```

If that fails:
```bash
pip install --upgrade pip
pip install PyMuPDF
```

### Want to Disable Optimization?

```python
result = tool.upload_document(
    collection_name="my_documents",
    document_path="document.pdf",
    use_optimized=False  # Use original method
)
```

## Configuration (Optional)

For even faster uploads, edit `config.py`:

```python
# Use GPU for embeddings (3-5x faster)
RAG_EMBEDDING_DEVICE = "cuda"  # Change from "cpu"

# Larger batch size (faster, more memory)
RAG_EMBEDDING_BATCH_SIZE = 64  # Change from 32

# Smaller chunks (fewer embeddings, faster)
RAG_CHUNK_SIZE = 256  # Change from 512
```

## Next Steps

1. ✅ Install PyMuPDF
2. ✅ Upload your first large PDF
3. ✅ Run benchmark test
4. 📖 Read full guide: `docs/RAG_OPTIMIZATION_GUIDE.md`
5. 🔧 Tune configuration if needed

## Support

- Full documentation: `docs/RAG_OPTIMIZATION_GUIDE.md`
- Performance test: `tests/test_rag_upload_performance.py`
- Logs: `data/logs/prompts.log`

---

**Bottom Line**: Install `PyMuPDF`, then upload PDFs as normal. They'll be **5-10x faster** automatically.
