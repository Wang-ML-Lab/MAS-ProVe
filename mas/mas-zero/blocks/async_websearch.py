import inspect


# %%%%%%%%%%%%%%%%%%%% WebSearch with CoT %%%%%%%%%%%%%%%%%%%%
async def forward(self, taskInfo, extra_info):
    # Instruction for the WebSearch with Chain-of-Thought approach
    # The agent will execute web search and use results for reasoning
    cot_instruction = self.cot_instruction
    
    # Use the original question/task as the search query
    # Note: The meta-agent can modify this search_query in generated code for further iterations
    # or customize it during code generation if needed
    search_query = taskInfo.content.strip()
    
    print(f"WebSearch agent using query: {search_query}")
    
    # Execute web search with the generated query
    web_search_results = "Web search unavailable."
    try:
        from ddgs import DDGS
        
        with DDGS() as ddgs:
            results = list(ddgs.text(search_query, max_results=5))
        
        formatted_output = "Search results:\n"
        for i, result in enumerate(results, 1):
            title = result.get('title', 'Untitled')
            link = result.get('href', '')
            snippet = result.get('body', 'No content available.')
            formatted_output += f"--- SOURCE {i}: {title} ---\nURL: {link}\n\nCONTENT:\n{snippet}\n\n"
        
        web_search_results = formatted_output
        print(f"Web search completed - found {len(results)} results")
    except Exception as e:
        print(f"Web search failed: {e}")
    
    # Add web search results to the instruction instead of extra_info
    enhanced_instruction = f"{cot_instruction}\n\n### Web Search Results ###\n{web_search_results}\n\nUse the above search results to help answer the question."
    
    # Instantiate a new LLM agent specifically for CoT with web search
    websearch_agent = LLMAgentBase(['thinking', 'answer'], 'WebSearch Agent', model=self.node_model, temperature=0.0)

    # Prepare the inputs for the CoT agent
    cot_agent_inputs = [taskInfo]

    # Get the response from the CoT agent with enhanced instruction
    thinking, answer = await websearch_agent(cot_agent_inputs, extra_info, enhanced_instruction)
    final_answer = self.make_final_answer(thinking, answer)

    # Return only the final answer
    return final_answer


func_string = inspect.getsource(forward)

WebSearch = {
    "thought": "Web search allows models to access up-to-date information from the internet and provide answers with sourced citations. The agent first generates an optimal search query based on the task, executes a web search using DuckDuckGo, and then applies Chain of Thought reasoning using the retrieved information. This combined approach enables the model to provide well-researched answers grounded in current information while showing its reasoning process.",
    "name": "WebSearch",
    "code": """{func_string}""".format(func_string=func_string)
}
