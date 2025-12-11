import asyncio
from ..clients.client_base import BaseClient

MAX_PARALLEL_SEARCH_CALLS = 5  # Can be overridden in the importing code

def llm_parallel_search_decorator(llm_func):
    """
    Decorator to run the LLM call function MAX_PARALLEL_SEARCH_CALLS times in parallel,
    send the results to the server via client, and pick the best response.
    """
    async def gather_calls(*args, **kwargs):
        # Call the original LLM func in parallel
        tasks = [llm_func(*args, **kwargs) for _ in range(MAX_PARALLEL_SEARCH_CALLS)]
        responses = await asyncio.gather(*tasks)
        return responses

    def send_to_server(responses, client: BaseClient, task_type=None):
        # Sends all candidate responses to server (expects 'server_client' conforming to your client API)
        return client.send_request(task_type, "judge", responses)
    
    def decorator_wrapper(*args, **kwargs):
        """
        kwargs must include 'client', and optionally 'task_type'
        """
        task_type = kwargs.pop('task_type', None)
        client = kwargs.pop('client', None)
        if client is None:
            raise ValueError("A 'client' must be provided as a kwarg to the decorated function.")

        # Run the LLM calls in parallel & collect results
        responses = asyncio.run(gather_calls(*args, **kwargs))

        # Send all results to the server judge and get rankings
        server_result = send_to_server(responses, client, task_type)
        rankings = server_result.get("rankings", list(range(len(responses))))
        best_idx = rankings[0] if rankings else 0

        return responses[best_idx]

    return decorator_wrapper