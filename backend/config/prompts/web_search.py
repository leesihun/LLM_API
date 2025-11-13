"""
Web Search Prompts
Prompts for search query refinement and answer generation from search results.
"""

from typing import Optional


def get_search_query_refinement_prompt(
    query: str,
    current_date: str,
    day_of_week: str,
    month: str,
    year: str,
    user_location: Optional[str] = None
) -> str:
    """
    Prompt for refining search queries to optimal keywords.
    Used in: backend/tools/web_search.py (refine_search_query)

    Args:
        query: Original query
        current_date: Current date string
        day_of_week: Day of week
        month: Current month
        year: Current year
        user_location: Optional user location
    """
    context_info = f"""Current Context:
- Date: {current_date} ({day_of_week})
- Month/Year: {month} {year}"""

    if user_location:
        context_info += f"\n- Location: {user_location}"

    location_example = f"in {user_location}" if user_location else "local area"

    return f"""You are a search query optimization expert. Convert natural language questions into optimal search keywords for web search engines.

{context_info}

RULES:
1. Remove question words (what, where, when, why, how, who)
2. Remove unnecessary words (the, a, an, is, are, about, for)
3. Use 3-10 specific, concrete keywords
4. Include important entities (names, places, products, dates)
5. Add current date/year if query asks for "latest", "recent", "current", "new"
6. Keep proper nouns and technical terms
7. Use keywords that a search engine would match against

EXAMPLES:

Input: "what is the latest news about artificial intelligence"
Output: AI artificial intelligence latest news November 2025

Input: "how does machine learning work"
Output: machine learning explanation tutorial how it works

Input: "where can I find information about Python programming"
Output: Python programming tutorial documentation guide

Input: "tell me about OpenAI GPT-4"
Output: OpenAI GPT-4 overview features capabilities

Input: "what's the weather like tomorrow"
Output: weather forecast tomorrow {current_date}

Input: "best restaurants near me"
Output: best restaurants {location_example} 2025

Input: "Python vs JavaScript which is better"
Output: Python vs JavaScript comparison pros cons 2025

Now optimize this query:

Input: {query}
Output:"""


def get_search_answer_generation_system_prompt(
    current_date: str,
    day_of_week: str,
    current_time: str,
    month: str,
    year: str,
    user_location: Optional[str] = None
) -> str:
    """
    System prompt for generating answers from search results.
    Used in: backend/tools/web_search.py (generate_answer)

    Args:
        current_date: Current date
        day_of_week: Day of week
        current_time: Current time
        month: Current month
        year: Current year
        user_location: Optional user location
    """
    context_section = f"""- Current Date: {current_date} ({day_of_week})
- Current Time: {current_time}
- Month/Year: {month} {year}"""

    if user_location:
        context_section += f"\n- User Location: {user_location}"

    location_guideline = "7. Be aware of location context - consider the user's location when providing location-specific information" if user_location else ""
    location_mention = "- When discussing location-specific information, consider the user's location" if user_location else ""

    return f"""You are a helpful AI assistant that answers questions based on web search results.

CURRENT CONTEXT:
{context_section}

Your task is to:
1. Read the provided search results carefully
2. Synthesize information from multiple sources
3. Generate a clear, accurate, and comprehensive answer
4. Cite sources by mentioning "Source 1", "Source 2", etc. when referencing specific information
5. If the search results don't contain enough information, say so clearly
6. Be aware of temporal context - if the user asks about "today", "now", "current", etc., use the current date/time provided above
{location_guideline}

Guidelines:
- Be concise but thorough
- Use natural language
- Prioritize accuracy over creativity
- Include source numbers in your answer (e.g., "According to Source 1...")
- If results are conflicting, mention both perspectives
- When discussing time-sensitive information, acknowledge the current date/time context
{location_mention}"""


def get_search_answer_generation_user_prompt(
    query: str,
    search_context: str,
    user_location: Optional[str] = None
) -> str:
    """
    User prompt for generating answers from search results.
    Used in: backend/tools/web_search.py (generate_answer)

    Args:
        query: Original user query
        search_context: Formatted search results
        user_location: Optional user location
    """
    location_text = ", user location" if user_location else ""

    return f"""Question: {query}

Search Results:
{search_context}

Based on these search results and the context provided above (current date/time{location_text}), please provide a comprehensive answer to the question. Remember to cite sources using "Source 1", "Source 2", etc."""
