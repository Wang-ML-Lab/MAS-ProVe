import time
from typing import Any

import openai
from openai import OpenAI, AsyncOpenAI

from .sampler_base import SamplerBase, MessageList


class AsyncChatCompletionSampler(SamplerBase):
    """
    Sample from OpenAI's chat completion API
    """

    def __init__(
            self,
            model: str = "gpt-5-nano",
            system_message: str | None = None,
            
    ):
        self.api_key_name = "OPENAI_API_KEY"
        self.client = AsyncOpenAI()
        # using api_key=os.environ.get("OPENAI_API_KEY")  # please set your API_KEY
        self.model = model
        self.system_message = system_message
        self.reasoning_effort = "minimal"
    
    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}
    
    async def __call__(self, message_list: MessageList, response_format=None) -> str:
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
                        reasoning_effort=self.reasoning_effort,
                        response_format={"type": "json_object"}
                    )
                else:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=message_list,
                        reasoning_effort=self.reasoning_effort,
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
