# Search Engine Testing

This directory contains tools for testing and comparing **5 carefully selected search engines** through LangChain, with emphasis on **free, no-API-key options**.

## Files

- **test_search_engines.ipynb** - Comprehensive Jupyter notebook testing 5 search engines
- **test_search_simple.py** - Simple Python script to verify dependencies and test DuckDuckGo
- **SEARCH_ENGINES_README.md** - This file

## 🎯 Focus: Free & Accessible Search

This testing suite prioritizes search engines that:
- ✓ Work without API keys (DuckDuckGo, SearxNG)
- ✓ Have optional/free API keys (Search1API)
- ⚠️ Documents known quality issues (Tavily)

## Quick Start

### 1. Install Dependencies

```bash
# Activate your virtual environment first
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac

# Install required packages
pip install langchain-community duckduckgo-search ddgs tavily-python langchain-search1api
```

### 2. Test Dependencies

Run the simple test script to verify everything is installed correctly:

```bash
python test_search_simple.py
```

This will:
- Check if all required packages are installed
- Run a simple DuckDuckGo search test
- Display sample results

### 3. Run Jupyter Notebook

```bash
jupyter notebook test_search_engines.ipynb
```

## Search Engines Tested (5 Total)

### 🥇 1. **DuckDuckGo** (No API Key Required) - RECOMMENDED
- ✓ Privacy-focused search
- ✓ Free to use, **no API key needed**
- ✓ **Best for getting started**
- ✓ Fast and reliable
- ⚠️ May have rate limits

### 🥈 2. **SearxNG** (No API Key Required) - RECOMMENDED
- ✓ Meta search engine (aggregates multiple sources)
- ✓ **No API key needed**
- ✓ Privacy-focused
- ✓ Maximum coverage
- ✓ Can use public instances or self-host
- Default public instance: https://searx.be

### 🥉 3. **Search1API** (Optional API Key)
- ✓ Can try **without API key**
- ✓ Optional free key for higher limits
- ✓ Good balance of features
- ⚠️ Requires: `pip install langchain-search1api`
- Get optional API key from: https://www.search1api.com/

### 4. **Mojeek** (Requires API Key)
- ✓ Independent search engine with own index
- ✓ Privacy-focused
- ✓ Not reliant on other search engines
- ⚠️ Requires API key
- Get API key from: https://www.mojeek.com/services/search/web-search-api/

### ⚠️ 5. **Tavily** (Requires API Key) - QUALITY CONCERNS
- ❌ **2x larger token usage** vs competitors (1928 vs 918 tokens)
- ❌ **Includes low-relevant information**
- ❌ **Higher costs** due to verbose results
- ⚠️ Requires post-processing/filtering
- ✓ Optimized for AI (in theory)
- Get API key from: https://tavily.com/
- **Note:** Included for comparison, but not recommended due to quality issues

## API Key Configuration

### Option 1: Environment Variables

Create a `.env` file (add to `.gitignore`!):

```bash
# Optional - only if you want to use these engines
TAVILY_API_KEY=your_tavily_key_here  # Not recommended due to quality issues
MOJEEK_API_KEY=your_mojeek_key_here
SEARCH1API_KEY=your_search1api_key_here  # Optional, can use without
SEARX_HOST=https://searx.be  # Or your self-hosted instance
```

**Note:** DuckDuckGo and SearxNG work without any API keys!

### Option 2: Direct in Notebook

Edit the configuration cell in the notebook to add your API keys directly.

## Usage Examples

### Basic Search (Python)

```python
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# DuckDuckGo (no API key)
search = DuckDuckGoSearchAPIWrapper(max_results=5)
results = search.results("artificial intelligence 2025", 5)

for result in results:
    print(f"Title: {result['title']}")
    print(f"URL: {result['link']}")
    print(f"Snippet: {result['snippet']}\n")
```

### Multi-Engine Fallback (Recommended)

```python
from langchain_community.utilities import (
    DuckDuckGoSearchAPIWrapper,
    SearxSearchWrapper
)

# Production-ready fallback strategy
def search_with_fallback(query, max_results=5):
    engines = [
        DuckDuckGoSearchAPIWrapper(max_results=max_results),  # Primary
        SearxSearchWrapper(searx_host="https://searx.be"),  # Fallback
    ]

    for engine in engines:
        try:
            results = engine.results(query, max_results)
            if results:
                return results
        except Exception:
            continue

    return []
```

The notebook includes more examples of multi-engine strategies.

## Performance Comparison

The notebook includes benchmarking to compare:
- **Speed** - How fast each engine returns results
- **Quality** - Number and relevance of results
- **Reliability** - Success rate and error handling
- **Cost** - Free tier limits and pricing

## Recommendations

**🥇 For Getting Started (BEST):**
- Use **DuckDuckGo** - free, no API key, reliable, works immediately

**🥈 For Maximum Coverage:**
- Use **SearxNG** - free, no API key, aggregates multiple engines

**🥉 For Production (Recommended Strategy):**
- Use **DuckDuckGo + SearxNG** fallback - zero cost, high reliability

**For Independent Index:**
- Use **Mojeek** - own search index, privacy-focused (requires API key)

**For Flexible Testing:**
- Use **Search1API** - optional API key, good for experimentation

**❌ NOT Recommended:**
- **Tavily** - quality issues, 2x token usage, includes low-relevant info
  - Only use if you absolutely need AI-specific features AND can post-process results

## Troubleshooting

### Issue: "No module named 'duckduckgo_search'"
**Solution:** Run `pip install duckduckgo-search ddgs`

### Issue: "Could not import ddgs python package"
**Solution:** Run `pip install ddgs`

### Issue: "Tavily API error" or poor quality results
**Solution:**
- Check your API key is valid and not expired
- Consider switching to DuckDuckGo or SearxNG for better quality and no costs

### Issue: Rate limit errors
**Solution:**
- Use multiple engines with fallback logic
- Implement caching for repeated queries
- Upgrade to paid tier if needed

### Issue: Encoding errors on Windows
**Solution:** The test script handles this automatically, but if you see encoding errors in custom code, add:
```python
import sys
import os
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
```

## Integration with Your API

The current `backend/tools/web_search.py` uses Tavily with websearch_ts fallback. **Recommended: Replace with DuckDuckGo + SearxNG** for better quality and zero cost:

```python
# Recommended: Multi-engine fallback (no API keys needed!)
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper, SearxSearchWrapper

def search_with_fallback(query, max_results=5):
    engines = [
        DuckDuckGoSearchAPIWrapper(max_results=max_results),  # Primary: Fast, reliable
        SearxSearchWrapper(searx_host="https://searx.be"),    # Fallback: Broader coverage
    ]

    for engine in engines:
        try:
            results = engine.results(query, max_results)
            if results:
                return results
        except Exception as e:
            print(f"Engine failed: {e}, trying next...")
            continue

    return []  # All engines failed
```

**Benefits over Tavily:**
- ✓ Zero cost (no API key needed)
- ✓ Better quality results
- ✓ No token bloat
- ✓ Automatic fallback for reliability

## Next Steps

1. Run the test script to verify setup
2. Open the Jupyter notebook
3. Add your API keys to the configuration
4. Run all cells to see comparison
5. Review performance metrics
6. Choose the best engine(s) for your use case
7. Integrate into your application

## Resources

- **LangChain Docs:** https://python.langchain.com/docs/integrations/tools/
- **DuckDuckGo Search:** https://github.com/deedy5/duckduckgo_search
- **SearxNG:** https://docs.searxng.org/
- **Search1API:** https://www.search1api.com/
- **Mojeek API:** https://www.mojeek.com/services/search/web-search-api/
- **Tavily API:** https://docs.tavily.com/ (not recommended due to quality issues)

## License

This testing code is part of your LLM API project and follows the same license.

---

## Summary

**Best Approach:** Use DuckDuckGo as primary + SearxNG as fallback = **Zero cost, high quality, no API key management!**

**Avoid:** Tavily (quality issues, 2x token usage, expensive)

---

**Last Updated:** November 4, 2025
**Tested With:** Python 3.11+, LangChain 0.3+
**Search Engines:** 5 (down from 6, removed Brave/Google/Bing, added Mojeek/Search1API)
