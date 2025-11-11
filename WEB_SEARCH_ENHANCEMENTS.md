# Web Search Contextual Enhancements

## Overview

Enhanced the web search functionality to include **temporal context** (current date/time) and **location context** for more accurate and relevant search results.

## Changes Made

### 1. Core Enhancements in `backend/tools/web_search.py`

#### **New Features**

##### A. Temporal Context
- Added `_get_temporal_context()` method that provides:
  - Current date (YYYY-MM-DD format)
  - Current time (HH:MM:SS format)
  - Day of week
  - Month and year
  - ISO datetime format

##### B. Query Enhancement
- Added `_enhance_query_with_context()` method that:
  - Detects temporal keywords: "today", "now", "current", "latest", "recent", "this week", "this month", "this year", "updated", "new", "breaking"
  - Detects location keywords: "near me", "nearby", "local", "in my area", "around here", "weather", "restaurants", "stores", "events"
  - Automatically appends current date to queries with temporal intent
  - Replaces location placeholders ("near me") with actual user location
  - Only enhances queries when relevant keywords are detected

##### C. Updated Search Method
**Signature:**
```python
async def search(
    query: str,
    max_results: int = 5,
    include_context: bool = True,
    user_location: Optional[str] = None
) -> Tuple[List[SearchResult], Dict[str, Any]]
```

**Returns:**
- `List[SearchResult]`: Search results as before
- `Dict[str, Any]`: Context metadata including:
  - All temporal context fields
  - `user_location`: User's location (if provided)
  - `query_enhanced`: Boolean indicating if query was enhanced
  - `original_query`: Original user query
  - `enhanced_query`: Enhanced query (if modified)

##### D. Updated Answer Generation
**Signature:**
```python
async def generate_answer(
    query: str,
    results: List[SearchResult],
    user_location: Optional[str] = None
) -> Tuple[str, List[str]]
```

**Enhancements:**
- System prompt now includes current date/time context
- Optionally includes user location in prompt
- LLM is aware of temporal/spatial context when generating answers

### 2. Schema Updates in `backend/models/schemas.py`

#### Updated `WebSearchRequest`:
```python
class WebSearchRequest(BaseModel):
    query: str
    max_results: int = 5
    include_context: bool = True  # NEW: Enable context enhancement
    user_location: Optional[str] = None  # NEW: User location
```

#### Updated `WebSearchResponse`:
```python
class WebSearchResponse(BaseModel):
    results: List[SearchResult]
    answer: str
    sources_used: List[str]
    context_used: Optional[Dict[str, Any]] = None  # NEW: Context metadata
```

### 3. Integration Updates

#### A. `backend/tasks/React.py`
Updated web search tool execution to:
- Use new search signature with context parameters
- Include context metadata in observations
- Log enhanced query information

#### B. `backend/core/agent_graph.py`
Updated LangGraph web search node to:
- Use new search signature
- Log context enhancement status

#### C. `backend/api/routes.py`
Updated `/api/tools/websearch` endpoint to:
- Accept `include_context` and `user_location` parameters
- Return context metadata in response
- Log query enhancement details

## Usage Examples

### Example 1: Temporal Enhancement

**User Query:** "latest news about AI"

**Process:**
1. System detects "latest" keyword (temporal intent)
2. Adds current date: "latest news about AI 2025-11-11"
3. Search results are more recent and relevant

### Example 2: Location Enhancement

**User Query:** "weather near me"
**User Location:** "Seoul, Korea"

**Process:**
1. System detects "near me" keyword (location intent)
2. Replaces with location: "weather in Seoul, Korea"
3. Search results are location-specific

### Example 3: Combined Enhancement

**User Query:** "current restaurants near me"
**User Location:** "New York, USA"

**Process:**
1. Detects both "current" (temporal) and "near me" (location)
2. Enhances to: "current restaurants in New York, USA"
3. Returns up-to-date, location-specific results

### Example 4: No Enhancement

**User Query:** "what is Python programming"

**Process:**
1. No temporal or location keywords detected
2. Query remains unchanged
3. Generic search results returned

## API Usage

### Direct Endpoint

```python
POST /api/tools/websearch
{
    "query": "latest AI trends",
    "max_results": 5,
    "include_context": true,
    "user_location": "Seoul, Korea"  // optional
}

Response:
{
    "results": [...],
    "answer": "...",
    "sources_used": [...],
    "context_used": {
        "current_date": "2025-11-11",
        "current_time": "12:54:06",
        "day_of_week": "Tuesday",
        "user_location": "Seoul, Korea",
        "query_enhanced": true,
        "enhanced_query": "latest AI trends 2025-11-11"
    }
}
```

### Programmatic Usage

```python
from backend.tools.web_search import web_search_tool

# Search with context
results, context = await web_search_tool.search(
    query="current weather near me",
    max_results=5,
    include_context=True,
    user_location="Tokyo, Japan"
)

# Generate answer with context
answer, sources = await web_search_tool.generate_answer(
    query="current weather near me",
    results=results,
    user_location="Tokyo, Japan"
)
```

## Testing

Run the test suite to verify functionality:

```bash
python test_websearch_context.py
```

**Test Coverage:**
1. Temporal context enhancement
2. Location context enhancement
3. Combined temporal + location enhancement
4. No enhancement for generic queries
5. Full search with context (requires Tavily API)

## Benefits

1. **Improved Relevance**: Searches automatically use current date for time-sensitive queries
2. **Location Awareness**: Automatically converts location placeholders to actual locations
3. **Better LLM Answers**: LLM is aware of current date/time when generating answers
4. **Transparency**: Context metadata shows exactly how queries were enhanced
5. **Backward Compatible**: Default behavior maintains compatibility with existing code
6. **Opt-in**: Context enhancement can be disabled via `include_context=False`

## Configuration

All enhancements are enabled by default. To disable:

```python
# Disable context enhancement
results, context = await web_search_tool.search(
    query="some query",
    include_context=False  # No temporal/location enhancement
)
```

## Future Enhancements

Potential improvements:
1. **Time zone support**: Use user's timezone instead of system timezone
2. **Seasonal context**: Add "season" field (Spring, Summer, etc.)
3. **Event awareness**: Detect and add context for major events/holidays
4. **Smart location detection**: Auto-detect location from IP or user profile
5. **Context caching**: Cache temporal context for performance
6. **Multi-language support**: Date/time in user's preferred language

## Technical Notes

- Temporal context is computed fresh on each search (no caching)
- Location context must be explicitly provided by the caller
- Query enhancement uses keyword matching (not LLM-based)
- Compatible with both Tavily API and websearch_ts fallback
- All enhancements are logged for debugging and transparency

## Version History

- **v1.0.0** (2025-11-11): Initial implementation
  - Temporal context injection
  - Location context injection
  - Enhanced answer generation
  - Schema updates
  - Integration with ReAct agent and API endpoints
