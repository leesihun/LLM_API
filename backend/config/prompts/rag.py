"""
RAG Retriever Prompts
======================
Prompts for RAG (Retrieval-Augmented Generation) system.

These prompts are used for:
- Document summarization
- Query enhancement for retrieval
- Answer synthesis from retrieved chunks

Created: 2025-01-20
Version: 1.0.0
"""

from typing import List, Dict, Any


def get_rag_query_enhancement_prompt(
    original_query: str,
    context: str = ""
) -> str:
    """
    Enhance user query for better retrieval results.
    
    Args:
        original_query: Original user query
        context: Optional conversation context
        
    Returns:
        LLM prompt for query enhancement
    """
    prompt = f"""You are a query optimization expert. Your task is to enhance the user's query to improve retrieval from a document database.

Original Query: {original_query}

{f"Context: {context}" if context else ""}

Enhance this query by:
1. Adding relevant keywords and synonyms
2. Expanding abbreviations
3. Making implicit information explicit
4. Keeping the core intent intact

Return ONLY the enhanced query, nothing else."""
    
    return prompt


def get_rag_answer_synthesis_prompt(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    max_chunks: int = 5
) -> str:
    """
    Synthesize answer from retrieved document chunks.
    
    Args:
        query: User's question
        retrieved_chunks: List of retrieved document chunks with scores
        max_chunks: Maximum chunks to include
        
    Returns:
        LLM prompt for answer synthesis
    """
    # Format chunks
    chunks_text = ""
    for i, chunk in enumerate(retrieved_chunks[:max_chunks], 1):
        content = chunk.get("content", "")
        source = chunk.get("source", "Unknown")
        score = chunk.get("score", 0.0)
        
        chunks_text += f"\n[Chunk {i}] (Score: {score:.3f}, Source: {source})\n{content}\n"
    
    prompt = f"""You are an expert at synthesizing information from multiple sources. Based on the retrieved document chunks below, answer the user's question.

User Question: {query}

Retrieved Information:
{chunks_text}

Instructions:
1. Answer the question directly and concisely
2. Base your answer ONLY on the provided chunks
3. If the chunks don't contain enough information, say so
4. Cite source documents when possible
5. If information is contradictory, mention it

Answer:"""
    
    return prompt


def get_rag_document_summary_prompt(
    document_content: str,
    max_length: int = 200
) -> str:
    """
    Generate summary for a document chunk.
    
    Args:
        document_content: Content to summarize
        max_length: Maximum summary length
        
    Returns:
        LLM prompt for summarization
    """
    prompt = f"""Summarize the following document content in no more than {max_length} words:

{document_content}

Summary:"""
    
    return prompt


def get_rag_relevance_check_prompt(
    query: str,
    chunk_content: str
) -> str:
    """
    Check if a chunk is relevant to the query.
    
    Args:
        query: User query
        chunk_content: Document chunk to evaluate
        
    Returns:
        LLM prompt for relevance checking
    """
    prompt = f"""Evaluate if the following document chunk is relevant to answer the user's query.

User Query: {query}

Document Chunk:
{chunk_content}

Is this chunk relevant to answering the query? Respond with:
- "RELEVANT" if it helps answer the query
- "PARTIALLY_RELEVANT" if it contains some related information
- "NOT_RELEVANT" if it doesn't help answer the query

Response:"""
    
    return prompt


def get_rag_multi_document_synthesis_prompt(
    query: str,
    document_summaries: List[Dict[str, Any]]
) -> str:
    """
    Synthesize answer from multiple document summaries.
    
    Args:
        query: User query
        document_summaries: List of document summaries
        
    Returns:
        LLM prompt for multi-document synthesis
    """
    summaries_text = ""
    for i, doc in enumerate(document_summaries, 1):
        doc_id = doc.get("doc_id", f"Document {i}")
        summary = doc.get("summary", "")
        source = doc.get("source", "Unknown")
        
        summaries_text += f"\n[{doc_id}] (Source: {source})\n{summary}\n"
    
    prompt = f"""You are synthesizing information from multiple documents to answer a query.

User Query: {query}

Document Summaries:
{summaries_text}

Instructions:
1. Provide a comprehensive answer based on all documents
2. Identify common themes and differences
3. Cite specific documents when making claims
4. Note if documents provide contradictory information
5. Be concise but thorough

Answer:"""
    
    return prompt
