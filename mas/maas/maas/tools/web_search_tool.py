#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Web search tool for LLM function calling
"""
from maas.tools.search_engine_ddg import DDGAPIWrapper

# Tool schema for OpenAI function calling
WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information, facts, or data that you don't have in your training data. Use this when you need up-to-date information, specific facts, or to verify information.",
        "parameters": {
            "type": "object",
            "properties": {
                "search_query": {
                    "type": "string",
                    "description": "The search query to look up on the web. Be specific and concise."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of search results to return (default: 5)",
                    "default": 5
                }
            },
            "required": ["search_query"]
        }
    }
}

# Global search engine instance
_search_engine = None

def get_search_engine():
    """Get or create the search engine instance"""
    global _search_engine
    if _search_engine is None:
        _search_engine = DDGAPIWrapper()
    return _search_engine

async def web_search(search_query: str, max_results: int = 5) -> str:
    """
    Execute a web search and return results as a formatted string.
    
    Args:
        search_query: The query to search for
        max_results: Maximum number of results to return
        
    Returns:
        Formatted string with search results
    """
    search_engine = get_search_engine()
    try:
        results = await search_engine.run(
            query=search_query,
            max_results=max_results,
            as_string=True
        )
        return results
    except Exception as e:
        return f"Search failed: {str(e)}"

# Mapping of tool names to their implementations
TOOL_FUNCTIONS = {
    "web_search": web_search
}
