import time
from typing import Any

import openai
from openai import OpenAI, AsyncOpenAI

from .sampler_base import SamplerBase, MessageList


class ChatCompletionSampler(SamplerBase):
    """
    Sample from OpenAI's chat completion API
    """

    def __init__(
            self,
            model: str = "gpt-3.5-turbo",
            system_message: str | None = None,
            temperature: float = 0.5,
            # max_tokens: int = 1024,
    ):
        self.api_key_name = "OPENAI_API_KEY"
        self.client = OpenAI()
        # using api_key=os.environ.get("OPENAI_API_KEY")  # please set your API_KEY
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        # self.max_tokens = max_tokens
        self.image_format = "url"

    def _handle_image(
            self, image: str, encoding: str = "base64", format: str = "png", fovea: int = 768
    ):
        new_image = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{format};{encoding},{image}",
            },
        }
        return new_image

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList, temperature=None, response_format=None) -> str:
        if self.system_message:
            message_list = [self._pack_message("system", self.system_message)] + message_list
        trial = 0
        while True:
            try:
                for message_id, message in enumerate(message_list):
                    if type(message['content']) != str:
                        message_list[message_id]['content'] = str(message['content'])
                # print('message_list: ',message_list)

                if response_format == 'normal':
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=message_list,
                        temperature=temperature if temperature is not None else self.temperature,
                        # max_tokens=self.max_tokens
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=message_list,
                        temperature=temperature if temperature is not None else self.temperature,
                        # max_tokens=self.max_tokens, 
                        response_format={"type": "json_object"}
                    )
                # print('response: ',response)
                return response.choices[0].message.content, response.usage
            # NOTE: BadRequestError is triggered once for MMMU, please uncomment if you are reruning MMMU
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return ""
            except Exception as e:
                exception_backoff = 2 ** trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
                if trial == 3:  # basically mean it is bad request after 3 trials
                    print("Bad Request Error", e)
                    return ""
            # unknown error shall throw exception


class AsyncChatCompletionSampler(ChatCompletionSampler):
    """
    Sample from OpenAI's chat completion API
    """

    def __init__(
            self,
            model: str = "gpt-3.5-turbo",
            system_message: str | None = None,
            temperature: float = 0.5,
            max_tokens: int = 1024,
    ):
        self.api_key_name = "OPENAI_API_KEY"
        self.client = AsyncOpenAI()
        # using api_key=os.environ.get("OPENAI_API_KEY")  # please set your API_KEY
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "url"

    async def __call__(self, message_list: MessageList, temperature=None, response_format=None) -> str:
        if self.system_message:
            message_list = [self._pack_message("system", self.system_message)] + message_list
        trial = 0
        while True:
            try:
                for message_id, message in enumerate(message_list):
                    if type(message['content']) != str:
                        message_list[message_id]['content'] = str(message['content'])
                # print('message_list: ',message_list)

                if response_format == 'normal':
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=message_list,
                        temperature=temperature if temperature is not None else self.temperature,
                        max_tokens=self.max_tokens
                    )
                else:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=message_list,
                        temperature=temperature if temperature is not None else self.temperature,
                        # max_tokens=self.max_tokens,
                        response_format={"type": "json_object"}
                    )
                # print('response: ',response)
                return response.choices[0].message.content, response.usage
            # NOTE: BadRequestError is triggered once for MMMU, please uncomment if you are reruning MMMU
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return ""
            except Exception as e:
                exception_backoff = 2 ** trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
                if trial == 3:  # basically mean it is bad request after 3 trials
                    print("Bad Request Error", e)
                    return ""
            # unknown error shall throw exception
            

