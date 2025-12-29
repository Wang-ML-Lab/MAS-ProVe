import asyncio
import warnings
from ..clients.client_base import BaseClient

# Define default if not present
MAX_PARALLEL_SEARCH_CALLS = 3 

def llm_parallel_search_decorator(llm_func):
    """
    Decorator to run the LLM call function MAX_PARALLEL_SEARCH_CALLS times in parallel,
    send the results to the server via client, and pick the best response.
    """
    async def gather_calls(*args, **kwargs):
        # 1. Run the LLM generations in parallel
        # print(f" Generating {MAX_PARALLEL_SEARCH_CALLS} parallel candidates...")
        tasks = [llm_func(*args, **kwargs) for _ in range(MAX_PARALLEL_SEARCH_CALLS)]
        
        # This await is non-blocking because it uses asyncio.gather
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions - replace failed responses with placeholder
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                error_msg = str(response)
                print(f"   WARNING: Candidate {i+1} failed with error: {error_msg[:100]}...")
                responses[i] = {"response": "I cannot respond to this request due to content policy restrictions."}
        
        return responses

    def send_to_server_sync(responses, task_type=None, question="", trajectory=None):
        """
        Synchronous wrapper for the blocking BaseClient.
        This function will be run in a separate thread.
        """
        # Ensure we use the correct port. Your server code said 5556, but client default is 5555.
        # I am setting it to 5556 to match the server code you provided previously.
        client = BaseClient(host='127.0.0.1', port=5556)
        
        # Build context string
        context_str = ""
        if len(responses) > 0 and isinstance(responses[0], dict) and "context" in responses[0]:
            context_str = responses[0]["context"]
        elif trajectory and len(trajectory) > 0:
            context_str = "\n\n".join([f"Step {i+1}: {step}" for i, step in enumerate(trajectory)])
        
        def flatten_to_str(obj):
            if isinstance(obj, str): return obj
            elif isinstance(obj, (list, tuple)): return '\n'.join(flatten_to_str(item) for item in obj)
            elif isinstance(obj, dict):
                if "response" in obj and obj["response"]: return flatten_to_str(obj["response"])
                else:
                    for v in obj.values():
                        if v: return flatten_to_str(v)
                return str(obj)
            return str(obj)

        partial_trajectories = []
        for i, function_result in enumerate(responses):
            content = flatten_to_str(function_result)
            partial_trajectories.append({
                "context": context_str,
                "current-step": content
            })
            print(f"        Candidate {i+1}: {len(content)} chars")
        # This calls your BaseClient, which blocks. 
        # But since we are in a thread, it won't block the main loop.
        result = client.send_request(task_type, "judge", partial_trajectories, question)
        return result
    
    async def async_decorator_wrapper(*args, **kwargs):
        """
        Async wrapper that handles the decorated function call.
        """
        task_type = kwargs.pop('task_type', None)
        trajectory = kwargs.pop('trajectory', None)
        question = kwargs.get('question', args[1] if len(args) > 1 else "")

        # 1. Generate candidates (Async/Parallel)
        responses = await gather_calls(*args, **kwargs)

        # 2. Send to judge (Blocking I/O offloaded to Thread)
        # CRITICAL FIX: For MAS-Zero
        # server_result = send_to_server(responses, task_type, question, trajectory)
        try:
            server_result = await asyncio.to_thread(
                send_to_server_sync, 
                responses, 
                task_type, 
                question, 
                trajectory
            )
        except Exception as e:
            print(f"Error communicating with judge server: {e}")
            # Fallback if judge fails
            server_result = {"rankings": list(range(len(responses)))}

        rankings = server_result.get("rankings", list(range(len(responses))))
        best_idx = rankings[0] if rankings else 0
        
        return responses[best_idx]
    
    return async_decorator_wrapper