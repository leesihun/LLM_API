# Search Engine Testing

This directory contains tools for testing and comparing various search engines through LangChain.

## Files

- **test_search_engines.ipynb** - Comprehensive Jupyter notebook testing multiple search engines
- **test_search_simple.py** - Simple Python script to verify dependencies and test DuckDuckGo
- **SEARCH_ENGINES_README.md** - This file

## Quick Start

### 1. Install Dependencies

```bash
# Activate your virtual environment first
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac

# Install required packages
pip install langchain-community duckduckgo-search tavily-python ddgs
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

## Search Engines Tested

### 1. **DuckDuckGo** (No API Key Required)
- Privacy-focused search
- Free to use, no API key needed
- Great for development and testing
- May have rate limits

### 2. **Tavily** (Requires API Key)
- Optimized for AI/LLM applications
- Clean, structured results
- Includes answer generation
- Get API key from: https://tavily.com/

### 3. **Brave Search** (Requires API Key)
- Privacy-focused alternative to Google
- Generous free tier: 2,000 queries/month
- High-quality results
- Get API key from: https://brave.com/search/api/

### 4. **Google Search** (Requires API Key + CSE ID)
- Traditional Google search quality
- Requires Custom Search Engine setup
- Limited free tier (100 queries/day)
- Setup: https://developers.google.com/custom-search

### 5. **Bing Search** (Requires API Key)
- Microsoft's search engine
- Good quality results
- Requires Azure subscription
- Get API key from: https://portal.azure.com

### 6. **SearxNG** (No API Key, Requires Instance)
- Meta search engine (aggregates multiple sources)
- Privacy-focused
- Can use public instances or self-host
- Public instance: https://searx.be

## API Key Configuration

### Option 1: Environment Variables

Create a `.env` file (add to `.gitignore`!):

```bash
TAVILY_API_KEY=your_tavily_key_here
BRAVE_API_KEY=your_brave_key_here
GOOGLE_API_KEY=your_google_key_here
GOOGLE_CSE_ID=your_cse_id_here
BING_SUBSCRIPTION_KEY=your_bing_key_here
SEARX_HOST=https://searx.be
```

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

### Tavily Search (Python)

```python
from langchain_community.utilities import TavilySearchAPIWrapper

search = TavilySearchAPIWrapper(tavily_api_key="your-key-here")
results = search.results("latest AI developments", 5)
```

### Multi-Engine Search

The notebook includes an example of using multiple search engines together for better coverage.

## Performance Comparison

The notebook includes benchmarking to compare:
- **Speed** - How fast each engine returns results
- **Quality** - Number and relevance of results
- **Reliability** - Success rate and error handling
- **Cost** - Free tier limits and pricing

## Recommendations

**For Development/Testing:**
- Use **DuckDuckGo** - free, no API key required

**For AI/LLM Applications:**
- Use **Tavily** - optimized for AI with clean results

**For Privacy + Quality:**
- Use **Brave Search** - good balance of privacy and results

**For Enterprise/Production:**
- Use **Google** or **Bing** - highest quality results

**For Maximum Coverage:**
- Use **SearxNG** - aggregates multiple engines

## Troubleshooting

### Issue: "No module named 'duckduckgo_search'"
**Solution:** Run `pip install duckduckgo-search ddgs`

### Issue: "Could not import ddgs python package"
**Solution:** Run `pip install ddgs`

### Issue: "Tavily API error"
**Solution:** Check your API key is valid and not expired

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

The current `backend/tools/web_search.py` uses Tavily with websearch_ts fallback. You can enhance it to use multiple engines:

```python
# Example: Multi-engine fallback
engines = [
    TavilySearchAPIWrapper(tavily_api_key=TAVILY_KEY),
    DuckDuckGoSearchAPIWrapper(max_results=5),
]

for engine in engines:
    try:
        results = engine.results(query, max_results)
        if results:
            return results
    except Exception as e:
        continue
```

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
- **Tavily API:** https://docs.tavily.com/
- **Brave Search API:** https://brave.com/search/api/
- **DuckDuckGo Search:** https://github.com/deedy5/duckduckgo_search
- **SearxNG:** https://docs.searxng.org/

## License

This testing code is part of your LLM API project and follows the same license.

---

**Last Updated:** November 3, 2025
**Tested With:** Python 3.11+, LangChain 0.3+
