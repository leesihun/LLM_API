# VibeThinker-1.5B Setup Guide for Ollama

## Quick Overview

VibeThinker-1.5B is a highly efficient 1.5B parameter model that **outperforms models 400x its size** (like DeepSeek R1) on mathematical reasoning benchmarks. It was trained for only $7,800 but achieves exceptional performance on competitive math and coding problems.

**Key Stats:**
- Parameters: 1.5B
- Training Cost: $7,800 (vs $294K for DeepSeek R1)
- Benchmarks: AIME24 (80.3), AIME25 (74.4), HMMT25 (50.4)
- Base: Qwen2.5-Math-1.5B
- License: MIT (free for commercial use)

---

## Installation Steps

### Option 1: Using Pre-Quantized GGUF (Recommended)

If someone has already converted VibeThinker to GGUF format:

```bash
# 1. Download the GGUF file (check Hugging Face for quantized versions)
# Look for models like: "VibeThinker-1.5B-Q4_K_M.gguf"

# 2. Update the FROM line in Modelfile.vibethinker to point to your GGUF file
# Example: FROM ./models/vibethinker-1.5b-q4_K_M.gguf

# 3. Create the Ollama model
ollama create vibethinker -f Modelfile.vibethinker

# 4. Test it
ollama run vibethinker "Solve: What is the derivative of x^3 + 2x^2 - 5x + 1?"
```

### Option 2: Convert from HuggingFace Yourself

```bash
# 1. Download the original model
git lfs install
git clone https://huggingface.co/WeiboAI/VibeThinker-1.5B

# 2. Convert to GGUF using llama.cpp converter
# (You'll need llama.cpp: https://github.com/ggerganov/llama.cpp)
python3 /path/to/llama.cpp/convert_hf_to_gguf.py \
    ./VibeThinker-1.5B \
    --outfile vibethinker-1.5b-f16.gguf \
    --outtype f16

# 3. Quantize for better performance (optional but recommended)
/path/to/llama.cpp/llama-quantize \
    vibethinker-1.5b-f16.gguf \
    vibethinker-1.5b-q4_K_M.gguf \
    Q4_K_M

# 4. Update Modelfile.vibethinker with the path
# FROM ./vibethinker-1.5b-q4_K_M.gguf

# 5. Create the model
ollama create vibethinker -f Modelfile.vibethinker
```

---

## Usage Examples

### Mathematical Reasoning
```bash
ollama run vibethinker "Solve the system of equations:
x^2 + y^2 = 25
x + y = 7
Find the value of x*y."
```

### Competitive Programming
```bash
ollama run vibethinker "Write a Python function to find the longest increasing subsequence in an array. Optimize for O(n log n) time complexity."
```

### Mathematical Proofs
```bash
ollama run vibethinker "Prove that the square root of 2 is irrational using proof by contradiction."
```

### Creative Mode (Temperature 1.0)
```bash
ollama run vibethinker --temperature 1.0 "Find three different approaches to solve the N-Queens problem."
```

---

## Parameter Tuning Guide

### Recommended Settings (Default in Modelfile)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `temperature` | 0.6 | Balanced reasoning (use 1.0 for exploration) |
| `num_ctx` | 40960 | Large context for extended reasoning chains |
| `top_p` | 0.95 | High-quality token selection |
| `top_k` | -1 | Disabled (model works better with pure top_p) |
| `repeat_penalty` | 1.05 | Mild penalty to reduce loops |

### When to Adjust

**For Precise/Deterministic Answers:**
```bash
ollama run vibethinker \
    --temperature 0.3 \
    --top-p 0.9 \
    "Calculate the exact value of 123456 * 789012"
```

**For Creative Problem-Solving:**
```bash
ollama run vibethinker \
    --temperature 1.0 \
    --top-p 0.95 \
    "Generate 5 different algorithms to solve the traveling salesman problem"
```

**For Long Reasoning Chains:**
```bash
ollama run vibethinker \
    --num-ctx 40960 \
    "Solve this multi-step calculus problem: [complex problem]"
```

---

## Integration with Your LLM_API Project

### Adding VibeThinker to settings.py

Edit `backend/config/settings.py`:

```python
# Add to your model options
AVAILABLE_MODELS = [
    "gemma3:12b",
    "gpt-oss:20b",
    "vibethinker:latest",  # Add this line
]

# Optional: Create a dedicated config for VibeThinker
VIBETHINKER_CONFIG = {
    "model_name": "vibethinker:latest",
    "temperature": 0.6,
    "num_ctx": 40960,
    "top_p": 0.95,
    "top_k": -1,
}
```

### Using in Your ReAct Agent

For math/code-heavy tasks, you might want to use VibeThinker in your ReAct workflow:

```python
from backend.utils.llm_factory import create_llm

# Create VibeThinker instance for reasoning tasks
vibethinker_llm = create_llm(
    model_name="vibethinker:latest",
    temperature=0.6,
    num_ctx=40960,
)

# Use in python_coder tool for better code generation
# Or use in a specialized "math_solver" tool
```

### Creating a Specialized Math Tool

You could create a new tool specifically for VibeThinker:

```python
# backend/tools/math_solver/tool.py
from backend.utils.llm_factory import create_llm

class MathSolverTool:
    def __init__(self):
        self.llm = create_llm(
            model_name="vibethinker:latest",
            temperature=0.6,
            num_ctx=40960,
        )

    def solve(self, problem: str) -> str:
        prompt = f"""Solve this mathematical problem step-by-step:

{problem}

Show your work and verify your answer."""
        return self.llm.invoke(prompt).content
```

---

## Performance Expectations

### What VibeThinker Excels At:
✅ Competition-level mathematics (AIME, AMC, HMMT)
✅ Algorithm design and optimization
✅ Code generation for complex logic
✅ Mathematical proofs and derivations
✅ Multi-step reasoning problems

### What to Use Other Models For:
- General conversation → gemma3:12b
- Broad knowledge tasks → gpt-oss:20b
- Task classification → gpt-oss:20b (current classifier)

### Benchmark Performance:
- **AIME24**: 80.3% (beats DeepSeek R1's 79.8%)
- **AIME25**: 74.4% (beats DeepSeek R1's 70.0%)
- **HMMT25**: 50.4% (beats DeepSeek R1's 41.7%)
- At only 1.5B parameters vs DeepSeek's 600B+!

---

## Troubleshooting

### "Model not found" error
```bash
# Check if model was created
ollama list | grep vibethinker

# If not, recreate it
ollama create vibethinker -f Modelfile.vibethinker
```

### Slow inference
```bash
# Use a more aggressive quantization
# Q4_K_M (good balance) or Q4_0 (faster, slight quality loss)

# Check Ollama is using GPU
ollama ps  # Should show GPU usage
```

### Poor quality responses
```bash
# Check your quantization level - Q4_K_M or higher recommended
# Q2/Q3 quantization may degrade reasoning ability

# Try adjusting temperature
ollama run vibethinker --temperature 0.6  # More focused
ollama run vibethinker --temperature 1.0  # More creative
```

### Context length errors
```bash
# Ensure num_ctx is set properly
ollama run vibethinker --num-ctx 40960 "Your long prompt here"
```

---

## Comparison with Your Current Models

| Model | Size | Best For | Cost |
|-------|------|----------|------|
| **vibethinker:latest** | 1.5B | Math, coding, reasoning | $7.8K training |
| gemma3:12b | 12B | General purpose | - |
| gpt-oss:20b | 20B | Broad knowledge, classification | - |
| bge-m3:latest | - | Embeddings (RAG) | - |

**Recommendation**: Use VibeThinker for the `python_coder` tool and any math-heavy agentic tasks.

---

## Resources

- **Official Repo**: https://github.com/WeiboAI/VibeThinker
- **HuggingFace**: https://huggingface.co/WeiboAI/VibeThinker-1.5B
- **Paper**: https://arxiv.org/abs/2511.06221
- **Ollama Docs**: https://ollama.com/docs

---

## Next Steps

1. ✅ Modelfile created (`Modelfile.vibethinker`)
2. ⬜ Download/convert model to GGUF format
3. ⬜ Update `FROM` path in Modelfile
4. ⬜ Run `ollama create vibethinker -f Modelfile.vibethinker`
5. ⬜ Test with sample math/code problems
6. ⬜ (Optional) Integrate into your LLM_API backend for specialized tasks

Happy reasoning! 🧮💻
