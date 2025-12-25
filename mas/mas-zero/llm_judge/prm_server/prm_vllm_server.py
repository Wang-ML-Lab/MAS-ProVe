import requests


def post_http_request(prompt: dict, api_url: str) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    response = requests.post(api_url, headers=headers, json=prompt)
    return response


def request_for_process_rewards(query, step_responses, model, api_url="http://localhost:8000/pooling"):
    conversation_str = assemble_conversation_str(query, step_responses, model)
    prompt = {"model": model, "input": conversation_str}
    pooling_response = post_http_request(prompt=prompt, api_url=api_url)

    if pooling_response.status_code != 200:
        raise ValueError(
            f"Request failed with status code {pooling_response.status_code}: {pooling_response.text}")

    rewards = [x[1] for x in pooling_response.json()["data"][0]["data"]]

    assert len(rewards) == len(
        step_responses), "Rewards length mismatch with step responses"

    return rewards


def assemble_conversation_str(query, step_responses, model):
    # should work for both Qwen2.5-Math-PRM-7B and Qwen2.5-Math-PRM-32B
    if "qwen2.5-math-prm" in model.lower():
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model, trust_remote_code=True)
        system = "Please reason step by step, and put your final answer within \\boxed{}."
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
            # {"role": "assistant", "content": data['response'] + "<extra_0>"},
            {"role": "assistant", "content": "<extra_0>".join(
                step_responses) + "<extra_0>"},
        ]

        conversation_str = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    else:
        raise ValueError(f"Unsupported model: {model}")

    return conversation_str


if __name__ == "__main__":
    # Example usage
    prm_model_path = "/research/projects/mllab/public_llms/reward_models/qwen_rms/Qwen2.5-Math-PRM-7B"
    datas = [
        {"problem": "What is 2 + 2?", "response": "4"},
        {"problem": "What is 3 + 5?", "response": "8"}
    ]

    # Load the model and tokenizer
    rewards = request_for_process_rewards(
        query=datas[0]["problem"],
        step_responses=[data["response"] for data in datas],
        model=prm_model_path,
        api_url="http://localhost:8000/pooling"
    )

    print(rewards)
