"""
GAIA-specific utility functions for DyLAN framework
"""
import re
import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MMLU'))
from prompt_lib import TEMPERATURE, MAX_TOKENS
from openai import OpenAI
from ddgs import DDGS

# Initialize OpenAI client (for v1.0+)
client = OpenAI()


def parse_gaia_answer(text):
    """Extract answer from GAIA response - looks for <answer> tags first"""
    if text is None:
        return None
    
    text = str(text)
    
    # First priority: Look for <answer>...</answer> tags
    answer_tag_pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(answer_tag_pattern, text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Second priority: Common answer patterns
    patterns = [
        r'(?:final answer|answer) is:?\s*([^\n\.]+)',
        r'(?:the answer|answer):\s*([^\n\.]+)',
        r'(?:therefore|thus|so),?\s+(?:the answer is)?\s*([^\n\.]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return match.group(1).strip()
    
    # Last resort: Take last line as answer
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    if lines:
        return lines[-1]
    
    return text.strip()


def gaia_string_match(expected_answer, response):
    """Flexible string matching for GAIA with numerical and parenthetical handling"""
    if expected_answer is None or response is None:
        return False
    
    # Normalize function: lowercase, strip, remove spaces
    normalize = lambda x: str(x).lower().strip().replace(" ", "")
    
    norm_expected = normalize(expected_answer)
    norm_resp = normalize(response)
    
    # Check if expected answer is numerical (after removing commas)
    norm_expected_no_comma = norm_expected.replace(",", "")
    is_numerical = re.match(r'^-?\d+\.?\d*$', norm_expected_no_comma) is not None
    
    if is_numerical:
        # For numerical answers, remove commas and do exact match
        norm_resp_no_comma = norm_resp.replace(",", "")
        return norm_resp_no_comma == norm_expected_no_comma
    else:
        # For string answers, allow flexible matching
        exact_match = norm_resp == norm_expected
        
        # If response has parentheses, check parenthesized text
        paren_match = re.search(r'\(([^)]+)\)', str(response))
        paren_content = normalize(paren_match.group(1)) if paren_match else ""
        
        # Check if expected answer appears as a substring (helps for 'Liu (Liu Sang)')
        substring_match = norm_expected in norm_resp or norm_expected in paren_content
        
        # Combined match logic for strings
        return exact_match or substring_match


def web_search(search_query: str, max_results: int = 5):
    """Perform web search and return formatted results"""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(search_query, max_results=max_results))
        if not results:
            return f"No search results found for query: {search_query}"
        formatted_output = "Search results:\n"
        for i, result in enumerate(results, 1):
            title = result.get('title', 'Untitled')
            link = result.get('href', '')
            snippet = result.get('body', 'No content available.')
            formatted_output += f"--- SOURCE {i}: {title} ---\nURL: {link}\n\nCONTENT:\n{snippet}\n\n"
        return formatted_output
    except Exception as e:
        print(f"[SEARCH ERROR] Failed to search for '{search_query}': {e}")
        return f"Search failed: Unable to retrieve results for query '{search_query}'. Error: {str(e)}"


def generate_answer_with_tools(answer_context, model):
    """Generate answer with tool calling support for GAIA dataset"""
    print("question context: ")
    print(answer_context)
    # Define tools for GAIA
    tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Perform a web search to find current information. Use this when you need to look up facts, recent events, or information not in your training data.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_query": {"type": "string", "description": "The search query to look up"},
                        "max_results": {"type": "integer", "description": "Maximum number of results to return (default: 5)", "default": 5}
                    },
                    "required": ["search_query"]
                }
            }
        }
    ]

    # Build request parameters
    messages = [m.copy() if isinstance(m, dict) else m for m in answer_context]
    request_params = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
    }

    # Add model-specific parameters
    if "gpt-5" in model.lower():
        request_params["reasoning_effort"] = "minimal"
    else:
        request_params["temperature"] = TEMPERATURE
        request_params["max_tokens"] = MAX_TOKENS
        request_params["n"] = 1

    try:
        response = client.chat.completions.create(**request_params)
    except Exception as e:
        print(f"[ERROR] API call failed: {e}")
        return "Unable to answer due to API error.", 0, 0

    # Token accounting
    def _safe_get_usage(resp):
        try:
            p = getattr(resp.usage, 'prompt_tokens') if hasattr(resp.usage, 'prompt_tokens') else resp.usage.get('prompt_tokens', 0)
        except Exception:
            p = 0
        try:
            c = getattr(resp.usage, 'completion_tokens') if hasattr(resp.usage, 'completion_tokens') else resp.usage.get('completion_tokens', 0)
        except Exception:
            c = 0
        return int(p or 0), int(c or 0)

    total_prompt_tokens, total_completion_tokens = _safe_get_usage(response)

    # Extract message and tool_calls
    choice0 = response.choices[0]
    response_message = getattr(choice0, 'message', None) or (choice0.get('message') if isinstance(choice0, dict) else None)
    content = None
    if response_message is not None:
        content = getattr(response_message, 'content', None) or (response_message.get('content') if isinstance(response_message, dict) else None)

    tool_calls = getattr(response_message, 'tool_calls', None) if response_message is not None else None
    if tool_calls is None and isinstance(response_message, dict):
        tool_calls = response_message.get('tool_calls')

    # If model requested tools, execute them and synthesize
    if tool_calls:
        # print(f"[TOOL CALLING] Model requested {len(tool_calls)} tool call(s)")
        # Add assistant's response to messages (as in async_llm)
        messages.append(response_message if isinstance(response_message, dict) else response_message.__dict__)

        # Process each tool call
        for tool_call in tool_calls:
            # Support both attribute and dict style
            if hasattr(tool_call, 'function'):
                function_name = getattr(tool_call.function, 'name', None)
                function_args = json.loads(getattr(tool_call.function, 'arguments', '{}'))
                call_id = getattr(tool_call, 'id', None)
            else:
                function_name = tool_call.get('function', {}).get('name')
                function_args = json.loads(tool_call.get('function', {}).get('arguments', '{}'))
                call_id = tool_call.get('id')

            # print(f"[TOOL CALLING] Executing: {function_name}({function_args})")
            if function_name == "web_search":
                try:
                    function_response = web_search(**function_args)
                except Exception as e:
                    function_response = f"Search failed: {e}"
                # print(f"[TOOL CALLING] Got {len(str(function_response))} characters from search")
            else:
                function_response = f"Unknown function: {function_name}"

            # Add function response to messages
            tool_msg = {
                "tool_call_id": call_id,
                "role": "tool",
                "name": function_name,
                "content": str(function_response),
            }
            messages.append(tool_msg)

        # Make second API call with tool results
        # print("[TOOL CALLING] Making second API call with tool results")
        second_request_params = {
            "model": model,
            "messages": messages,
        }
        if "gpt-5" in model.lower():
            second_request_params["reasoning_effort"] = "minimal"
        else:
            second_request_params["temperature"] = TEMPERATURE
            second_request_params["max_tokens"] = MAX_TOKENS
            second_request_params["n"] = 1

        try:
            second_response = client.chat.completions.create(**second_request_params)
        except Exception as e:
            print(f"[ERROR] Second API call failed: {e}")
            tool_summary = "\n\n".join([msg.get("content", "") for msg in messages if msg.get("role") == "tool"])
            return f"Search results obtained but unable to synthesize.\n\n{tool_summary}", total_prompt_tokens, total_completion_tokens

        p2, c2 = _safe_get_usage(second_response)
        total_prompt_tokens += p2
        total_completion_tokens += c2

        choice0b = second_response.choices[0]
        resp_msg_b = getattr(choice0b, 'message', None) or (choice0b.get('message') if isinstance(choice0b, dict) else None)
        final_content = getattr(resp_msg_b, 'content', None) or (resp_msg_b.get('content') if isinstance(resp_msg_b, dict) else None) or ""
    else:
        final_content = content or (response.choices[0].message.content if hasattr(response.choices[0], 'message') else (response.choices[0].get('message', {}).get('content') if isinstance(response.choices[0], dict) else ""))

    return final_content, int(total_prompt_tokens or 0), int(total_completion_tokens or 0)