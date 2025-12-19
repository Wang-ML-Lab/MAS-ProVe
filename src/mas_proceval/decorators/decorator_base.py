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
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions - replace failed responses with placeholder
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                error_msg = str(response)
                print(f"   WARNING: Candidate {i+1} failed with error: {error_msg[:100]}...")
                # Replace with a safe placeholder response
                responses[i] = {"response": "I cannot respond to this request due to content policy restrictions."}
        
        print(f" Generated {len(responses)} candidates")
        return responses

    def send_to_server(responses, task_type=None, question="", trajectory=None):
        client = BaseClient(host='127.0.0.1', port=5555)
        # Format responses as trajectories for the judge server
        # Each response is a tuple: (content, usage, tool_calls_info)
        print(f"Formatting {len(responses)} candidates for judging...")
        
        # Build context from trajectory
        context_str = ""
        if trajectory and len(trajectory) > 0:
            context_str = "\n\n".join([f"Step {i+1}: {step}" for i, step in enumerate(trajectory)])
            print(f"        Context: {len(trajectory)} previous steps")
        
        partial_trajectories = []
        for i, function_result in enumerate(responses):
            if isinstance(function_result, tuple) or isinstance(function_result, list): 
                # Protocol: if the function's result is a tuple or list, the first one should be the actual reasoning response.
                content = function_result[0]
            elif isinstance(function_result, dict):
                # Extract first non-None, non-empty value from dict, or stringify the whole dict
                content = next((v for v in function_result.values() if v), str(function_result))
            else:
                content = function_result
            partial_trajectories.append({
                "context": context_str,
                "current-step": content
            })
            print(f"        Candidate {i+1}: {len(content)} chars")
            print(f"        Candidate {i+1} content: {content[:200]}..." if len(content) > 200 else f"        Candidate {i+1} content: {content}")
        
        # Send to server for ranking
        print(f"Sending to judge server (task_type={task_type})...")
        result = client.send_request(task_type, "judge", partial_trajectories, question)
        print(f"Received rankings from server: {result.get('rankings', [])}")
        return result
    
    async def async_decorator_wrapper(*args, **kwargs):
        """
        Async wrapper that handles the decorated function call.
        kwargs should include 'task_type', 'question', and optionally 'trajectory'.
        The decorator creates its own client instance for thread-safe async operations.
        """
        print(f"DECORATOR CALLED")
        task_type = kwargs.pop('task_type', None)
        trajectory = kwargs.pop('trajectory', None)
        # Extract question - it's the second positional arg for both direct() and debate_refine()
        question = kwargs.get('question', args[1] if len(args) > 1 else "")
        print(f"      Task type: {task_type}")
        print(f"      Question: {question[:100]}..." if len(question) > 100 else f"      Question: {question}")

        # Run the LLM calls in parallel & collect results
        responses = await gather_calls(*args, **kwargs)

        # Send all results to the server judge and get rankings
        # send_to_server creates its own client instance for this call
        server_result = send_to_server(responses, task_type, question, trajectory)
        rankings = server_result.get("rankings", list(range(len(responses))))
        best_idx = rankings[0] if rankings else 0
        
        print(f"   Best candidate: #{best_idx + 1} (rank position 1/{len(responses)})")
        print(f"   DECORATOR COMPLETE\n")

        return responses[best_idx]
    
    return async_decorator_wrapper