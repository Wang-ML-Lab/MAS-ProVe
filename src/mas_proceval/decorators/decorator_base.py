import asyncio
import warnings
from ..clients.client_base import BaseClient

MAX_PARALLEL_SEARCH_CALLS = 3  # Can be overridden in the importing code

def llm_parallel_search_decorator(llm_func):
    """
    Decorator to run the LLM call function MAX_PARALLEL_SEARCH_CALLS times in parallel,
    send the results to the server via client, and pick the best response.
    """
    async def gather_calls(*args, **kwargs):
        # Call the original LLM func in parallel
        print(f" Generating {MAX_PARALLEL_SEARCH_CALLS} parallel candidates...")
        tasks = [llm_func(*args, **kwargs) for _ in range(MAX_PARALLEL_SEARCH_CALLS)]
        responses = await asyncio.gather(*tasks)
        print(f" Generated {len(responses)} candidates")
        return responses

    def send_to_server(responses, client: BaseClient, task_type=None, question=""):
        # Format responses as trajectories for the judge server
        # Each response is a tuple: (content, usage, tool_calls_info)
        print(f"Formatting {len(responses)} candidates for judging...")
        partial_trajectories = []
        for i, function_result in enumerate(responses):
            if isinstance(function_result, tuple) or isinstance(function_result, list): 
                # Protocol: if the function's result is a tuple or list, the first one should be the actual reasoning response.
                content = function_result[0]
            elif isinstance(function_result, dict):
                content = function_result.get("response", "NA")
                if content == "NA":
                    warnings.warn(f"Wrapped function's intermediate result is a dict but does not contain a 'response' key. Got keys: {function_result.keys()}")
            else:
                content = function_result
            partial_trajectories.append({
                "context": "",
                "current-step": content
            })
            print(f"        Candidate {i+1}: {len(content)} chars")
        
        # Send to server for ranking
        print(f"Sending to judge server (task_type={task_type})...")
        result = client.send_request(task_type, "judge", partial_trajectories, question)
        print(f"Received rankings from server: {result.get('rankings', [])}")
        return result
    
    async def async_decorator_wrapper(*args, **kwargs):
        """
        Async wrapper that handles the decorated function call.
        kwargs must include 'client', 'task_type', and should have 'question'
        """
        print(f"DECORATOR CALLED")
        task_type = kwargs.pop('task_type', None)
        client = kwargs.pop('client', None)
        # Extract question - it's the second positional arg for both direct() and debate_refine()
        question = kwargs.get('question', args[1] if len(args) > 1 else "")
        print(f"      Task type: {task_type}")
        print(f"      Question: {question[:100]}..." if len(question) > 100 else f"      Question: {question}")
        
        if client is None:
            raise ValueError("A 'client' must be provided as a kwarg to the decorated function.")

        # Run the LLM calls in parallel & collect results
        responses = await gather_calls(*args, **kwargs)

        # Send all results to the server judge and get rankings
        server_result = send_to_server(responses, client, task_type, question)
        rankings = server_result.get("rankings", list(range(len(responses))))
        best_idx = rankings[0] if rankings else 0
        
        print(f"   Best candidate: #{best_idx + 1} (rank position 1/{len(responses)})")
        print(f"   DECORATOR COMPLETE\n")

        return responses[best_idx]
    
    return async_decorator_wrapper