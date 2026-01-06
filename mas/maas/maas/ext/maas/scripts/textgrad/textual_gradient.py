import argparse
import json
import os
import backoff
import openai
import yaml
from pathlib import Path

from maas.ext.maas.scripts.textgrad.prompt_aime import get_init_archive_aime, get_prompt_aime
from maas.const import METAGPT_ROOT, CONFIG_ROOT
TEXT_GRAD_PROMPT = """
Please act as an expert with extensive experience in Natural Language Processing and model tuning. Your task is to generate a detailed and specific prompt tailored to the {dataset} dataset.

The original prompt content is as follows: {prompt_name} = {prompt_content}

Please adhere to the following requirements:
1. Clearly describe the core characteristics and challenges of the dataset, analyze the type of tasks involved, and make targeted prompt adjustments accordingly.
2. Strictly preserve all placeholders in the original prompt content without any modifications, ensuring that the number and names of the placeholders remain unchanged for subsequent replacements.
3. Return the newly generated prompt content in the 'prompt' field. This field should include only the new prompt content (excluding the name) and should not contain any triple quotation marks.
4.  Include your single prompt in XML tags in your reply. 
"""

def load_llm_config(model_name: str) -> dict:
    """
    Loads the LLM configuration for the specified model from one of the predefined YAML files.
    
    The function searches for the configuration file in the following locations (in order):
        1. METAGPT_ROOT / "config/config2.yaml"
        2. CONFIG_ROOT / "config2.yaml"
        
    The expected YAML structure is:
    
    llm:
      api_type: "openai"  # or azure / ollama / groq etc.
      model: "gpt-5-mini"
      base_url: "https://openai.com/api/v1"
      api_key: "your_api_key_here" 
    models:
      gpt-4o-mini:
        api_type: "openai"
        model: "gpt-5-mini"
        base_url: "https://openai.com/api/v1"
        api_key: "your_api_key_here"
        
    Args:
        model_name (str): The name of the model whose configuration is needed.
        
    Returns:
        dict: A dictionary with keys "base_url" and "api_key" for the specified model.
        
    Raises:
        FileNotFoundError: If no configuration file is found in the predefined paths.
        ValueError: If the configuration for the specified model is missing required keys.
    """
    config_paths = [
        METAGPT_ROOT / "config" / "config2.yaml",
        CONFIG_ROOT / "config2.yaml"
    ]
    
    config = None
    for path in config_paths:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            break
    
    if config is None:
        raise FileNotFoundError("No configuration file found in the predefined paths.")
    
    if "models" in config and model_name in config["models"]:
        model_config = config["models"][model_name]
    else:
        model_config = config.get("llm", {})
    
    if "base_url" not in model_config or "api_key" not in model_config:
        raise ValueError(f"Configuration for model '{model_name}' must include 'base_url' and 'api_key'.")
    
    return {"base_url": model_config["base_url"], "api_key": model_config["api_key"]}

default_model = "gpt-5-mini"
llm_config = load_llm_config(default_model)

client = openai.OpenAI(
    api_key=llm_config["api_key"],
    base_url=llm_config["base_url"],
)

@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt_reflect(
        msg_list,
        model,
        temperature=0.8
):
    response = client.chat.completions.create(
        model=model,
        messages=msg_list,
        reasoning_efffort = "minimal",
        stop=None,
        response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    json_dict = json.loads(content)
    assert not json_dict is None
    return json_dict

def search(args):
    archive_humaneval = get_init_archive_humaneval()
    archive_gsm8k = get_init_archive_gsm8k()
    archive_math = get_init_archive_math()
    system_prompt, prompt = get_prompt_humaneval(archive_humaneval)
    msg_list_reasoning = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    system_prompt, prompt = get_prompt_gsm8k(archive_gsm8k)
    msg_list_planning = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    system_prompt, prompt = get_prompt_math(archive_math)
    msg_list_memory = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    try: 
        next_solution_reasoning = get_json_response_from_gpt_reflect(msg_list_reasoning, args.model)
        next_solution_planning = get_json_response_from_gpt_reflect(msg_list_planning, args.model)
        next_solution_memory = get_json_response_from_gpt_reflect(msg_list_memory, args.model)
        with open('output_humaneval.jsonl', 'a') as jsonl_file:
                jsonl_file.write(json.dumps(next_solution_reasoning) + '\n')
        for key, value in next_solution_reasoning.items():
            print(f"{key}: {value}")
        with open('output_gsm8k.jsonl', 'a') as jsonl_file:
                jsonl_file.write(json.dumps(next_solution_planning) + '\n')
        for key, value in next_solution_planning.items():
            print(f"{key}: {value}")
        with open('output_math.jsonl', 'a') as jsonl_file:
                jsonl_file.write(json.dumps(next_solution_memory) + '\n')
        for key, value in next_solution_memory.items():
            print(f"{key}: {value}")
        
    except Exception as e:
        print("During LLM generate new solution:")
        print(e)
