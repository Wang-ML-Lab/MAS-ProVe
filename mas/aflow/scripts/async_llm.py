# -*- coding: utf-8 -*-
# @Date    : 2025-03-31
# @Author  : Zhaoyang
# @Desc    : 

from openai import AsyncOpenAI
from scripts.formatter import BaseFormatter, FormatError

import yaml
import os
import json
from pathlib import Path
from typing import Dict, Optional, Any, List, Callable

class LLMConfig:
    def __init__(self, config: dict):
        self.model = config.get("model", "gpt-4o-mini")
        self.temperature = config.get("temperature", 1)
        self.key = config.get("key", None)
        self.base_url = config.get("base_url", "https://oneapi.deepwisdom.ai/v1")
        self.top_p = config.get("top_p", 1)
        self.reasoning_effort = config.get("reasoning_effort", "minimal")  # New parameter

class LLMsConfig:
    """Configuration manager for multiple LLM configurations"""
    
    _instance = None  # For singleton pattern if needed
    _default_config = None
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize with an optional configuration dictionary"""
        self.configs = config_dict or {}
    
    @classmethod
    def default(cls):
        """Get or create a default configuration from YAML file"""
        if cls._default_config is None:
            # Look for the config file in common locations
            config_paths = [
                Path("config/config2.yaml"),
                Path("config2.yaml"),
                Path("./config/config2.yaml")
            ]
            
            config_file = None
            for path in config_paths:
                if path.exists():
                    config_file = path
                    break
            
            if config_file is None:
                raise FileNotFoundError("No default configuration file found in the expected locations")
            
            # Load the YAML file
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # Your YAML has a 'models' top-level key that contains the model configs
            if 'models' in config_data:
                config_data = config_data['models']
                
            cls._default_config = cls(config_data)
        
        return cls._default_config
    
    def get(self, llm_name: str) -> LLMConfig:
        """Get the configuration for a specific LLM by name"""
        if llm_name not in self.configs:
            raise ValueError(f"Configuration for {llm_name} not found")
        
        config = self.configs[llm_name]
        
        # Get API key - if it's an environment variable name, fetch from os.environ
        api_key = config.get("api_key")
        if api_key and not api_key.startswith("sk-"):
            # Try to get from environment variable
            api_key = os.environ.get(api_key, api_key)
        
        # Create a config dictionary with the expected keys for LLMConfig
        llm_config = {
            "model": llm_name,  # Use the key as the model name
            "temperature": config.get("temperature", 1),
            "key": api_key,  # Map api_key to key
            "base_url": config.get("base_url", "https://oneapi.deepwisdom.ai/v1"),
            "top_p": config.get("top_p", 1),  # Add top_p parameter
            "reasoning_effort": config.get("reasoning_effort", "minimal")  # New parameter
        }
        
        # Create and return an LLMConfig instance with the specified configuration
        return LLMConfig(llm_config)
    
    def add_config(self, name: str, config: Dict[str, Any]) -> None:
        """Add or update a configuration"""
        self.configs[name] = config
    
    def get_all_names(self) -> list:
        """Get names of all available LLM configurations"""
        return list(self.configs.keys())
    
class ModelPricing:
    """Pricing information for different models in USD per 1K tokens"""
    PRICES = {
        # GPT-4o models
        "gpt-4o": {"input": 0.0025, "output": 0.01},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4o-mini-2024-07-18": {"input": 0.00015, "output": 0.0006},
        "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
        "o3":{"input":0.003, "output":0.015},
        "o3-mini": {"input": 0.0011, "output": 0.0025},
        "gpt-5-mini": {"input": 0.00025, "output": 0.002},
    }
    
    @classmethod
    def get_price(cls, model_name, token_type):
        """Get the price per 1K tokens for a specific model and token type (input/output)"""
        # Try to find exact match first
        if model_name in cls.PRICES:
            return cls.PRICES[model_name][token_type]
        
        # Try to find a partial match (e.g., if model name contains version numbers)
        for key in cls.PRICES:
            if key in model_name:
                return cls.PRICES[key][token_type]
        
        # Return default pricing if no match found
        return 0

class TokenUsageTracker:
    """Tracks token usage and calculates costs"""
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0
        self.usage_history = []
    
    def add_usage(self, model, input_tokens, output_tokens):
        """Add token usage for a specific API call"""
        input_cost = (input_tokens / 1000) * ModelPricing.get_price(model, "input")
        output_cost = (output_tokens / 1000) * ModelPricing.get_price(model, "output")
        total_cost = input_cost + output_cost
        
        usage_record = {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "prices": {
                "input_price": ModelPricing.get_price(model, "input"),
                "output_price": ModelPricing.get_price(model, "output")
            }
        }
        
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += total_cost
        self.usage_history.append(usage_record)
        
        return usage_record
    
    def get_summary(self):
        """Get a summary of token usage and costs"""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost": self.total_cost,
            "call_count": len(self.usage_history),
            "history": self.usage_history
        }

class AsyncLLM:
    def __init__(self, config, system_msg:str = None, tools: List[dict] = None, tool_functions: Dict[str, Callable] = None):
        """
        Initialize the AsyncLLM with a configuration
        
        Args:
            config: Either an LLMConfig instance or a string representing the LLM name
                   If a string is provided, it will be looked up in the default configuration
            system_msg: Optional system message to include in all prompts
            tools: Optional list of tool definitions for function calling
            tool_functions: Optional dictionary mapping function names to actual callable functions
        """
        # Handle the case where config is a string (LLM name)
        if isinstance(config, str):
            llm_name = config
            config = LLMsConfig.default().get(llm_name)
        
        # At this point, config should be an LLMConfig instance
        self.config = config
        self.aclient = AsyncOpenAI(api_key=self.config.key, base_url=self.config.base_url)
        self.sys_msg = system_msg
        self.usage_tracker = TokenUsageTracker()
        self.tools = tools
        self.tool_functions = tool_functions or {}
        
    async def __call__(self, prompt):
        message = []
        if self.sys_msg is not None:
            message.append({
                "content": self.sys_msg,
                "role": "system"
            })

        message.append({"role": "user", "content": prompt})
        
        # Build request parameters
        request_params = {
            "model": self.config.model,
            "messages": message,
        }
        
        # Add tools if available
        if self.tools is not None:
            # print(f"[TOOL CALLING] Tools are enabled with {len(self.tools)} tool(s)")
            request_params["tools"] = self.tools
            request_params["tool_choice"] = "auto"
        
        # Add model-specific parameters
        if self.config.model.startswith("gpt-5"):
            request_params["reasoning_effort"] = self.config.reasoning_effort
            request_params["top_p"] = self.config.top_p
        else:
            request_params["temperature"] = self.config.temperature
            request_params["top_p"] = self.config.top_p
        
        # Make the initial API call
        response = await self.aclient.chat.completions.create(**request_params)
        
        # Track initial token usage
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        usage_record = self.usage_tracker.add_usage(
            self.config.model,
            input_tokens,
            output_tokens
        )
        
        # Handle tool calls if present
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        
        if tool_calls:
            # print(f"[TOOL CALLING] Model requested {len(tool_calls)} tool call(s)")
            # Add assistant's response to messages
            message.append(response_message)
            
            # Process each tool call
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                # print(f"[TOOL CALLING] Tool call: {function_name} with args: {function_args}")
                
                # Execute the function
                if function_name in self.tool_functions:
                    # print(f"[TOOL CALLING] Executing function: {function_name}")
                    function_response = self.tool_functions[function_name](**function_args)
                    # print(f"[TOOL CALLING] Function {function_name} returned {len(str(function_response))} characters")
                    
                    # Add function response to messages
                    message.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": str(function_response),
                    })
                else:
                    print(f"Warning: Function {function_name} not found in tool_functions")
            
            # Make a second API call with the function results
            # print(f"[TOOL CALLING] Making second API call with tool results")
            second_request_params = {
                "model": self.config.model,
                "messages": message,
            }
            
            if self.config.model.startswith("gpt-5"):
                second_request_params["reasoning_effort"] = self.config.reasoning_effort
                second_request_params["top_p"] = self.config.top_p
            else:
                second_request_params["temperature"] = self.config.temperature
                second_request_params["top_p"] = self.config.top_p
            
            second_response = await self.aclient.chat.completions.create(**second_request_params)
            
            # Track second call token usage
            input_tokens = second_response.usage.prompt_tokens
            output_tokens = second_response.usage.completion_tokens
            usage_record = self.usage_tracker.add_usage(
                self.config.model,
                input_tokens,
                output_tokens
            )
            
            ret = second_response.choices[0].message.content
        else:
            ret = response_message.content
        
        print(ret)
        
        # Print token usage information
        print(f"Token usage: {input_tokens} input + {output_tokens} output = {input_tokens + output_tokens} total")
        print(f"Cost: ${usage_record['total_cost']:.6f} (${usage_record['input_cost']:.6f} for input, ${usage_record['output_cost']:.6f} for output)")
        
        return ret
    
    async def call_with_format(self, prompt: str, formatter: BaseFormatter):
        """
        Call the LLM with a prompt and format the response using the provided formatter
        
        Args:
            prompt: The prompt to send to the LLM
            formatter: An instance of a BaseFormatter to validate and parse the response
            
        Returns:
            The formatted response data
            
        Raises:
            FormatError: If the response doesn't match the expected format
        """
        # Prepare the prompt with formatting instructions
        formatted_prompt = formatter.prepare_prompt(prompt)
        # Call the LLM
        response = await self.__call__(formatted_prompt)
        
        # Validate and parse the response
        is_valid, parsed_data = formatter.validate_response(response)
        
        if not is_valid:
            error_message = formatter.format_error_message()
            raise FormatError(f"{error_message}. Raw response: {response}")
        
        return parsed_data
    
    def get_usage_summary(self):
        """Get a summary of token usage and costs"""
        return self.usage_tracker.get_summary()    
    

def create_llm_instance(llm_config):
    """
    Create an AsyncLLM instance using the provided configuration
    
    Args:
        llm_config: Either an LLMConfig instance, a dictionary of configuration values,
                            or a string representing the LLM name to look up in default config
    
    Returns:
        An instance of AsyncLLM configured according to the provided parameters
    """
    # Case 1: llm_config is already an LLMConfig instance
    if isinstance(llm_config, LLMConfig):
        return AsyncLLM(llm_config)
    
    # Case 2: llm_config is a string (LLM name)
    elif isinstance(llm_config, str):
        return AsyncLLM(llm_config)  # AsyncLLM constructor handles lookup
    
    # Case 3: llm_config is a dictionary
    elif isinstance(llm_config, dict):
        # Create an LLMConfig instance from the dictionary
        llm_config = LLMConfig(llm_config)
        return AsyncLLM(llm_config)
    
    else:
        raise TypeError("llm_config must be an LLMConfig instance, a string, or a dictionary")