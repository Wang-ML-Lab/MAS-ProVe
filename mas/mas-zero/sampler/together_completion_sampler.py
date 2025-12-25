import json
import re
import time
from collections import OrderedDict
from typing import Any

import openai
from together import Together

from shared_vars import get_global
from utils import extract_xml
from .sampler_base import SamplerBase

Message = dict[str, Any]  # keys role, content
MessageList = list[Message]


class ChatCompletionSampler(SamplerBase):
    """
    Sample from OpenAI's chat completion API
    """

    def __init__(
            self,
            system_message: str | None = None,
            temperature: float = 0.5,
            model: str | None = None,
            # max_tokens: int = 1024,
    ):
        try:
            self.api_key_name = "OPENAI_API_KEY"
            self.system_message = system_message
            self.temperature = temperature
            # self.max_tokens = max_tokens
            self.client = Together()
            self.model = model
        except Exception as e:
            print(f'warning Together AI: {e}')

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def xml_to_json(self, ori_answer):
        output_dict = OrderedDict()  # <-- keep insertion order
        tag_names = re.findall(r"</?(\w+)>", ori_answer)
        ordered_unique_tags = list(OrderedDict.fromkeys(tag_names))

        for tag in ordered_unique_tags:
            if all(t not in tag for t in ['A', 'B', 'C', 'D', 'sub', 'S_y', 'TOO_HARD']) and tag not in ['a', 'script', 'rst_prolog', 'generated', 'format']:
                tag_text = extract_xml(ori_answer, tag)
                output_dict[tag] = tag_text
        print('output_dict: ', output_dict.keys())

        json_string = json.dumps(output_dict, indent=4)
        return json_string

    def __call__(self, message_list: MessageList, temperature=None, response_format=None) -> str:
        if self.system_message:
            message_list = [self._pack_message("system", self.system_message)] + message_list
        trial = 0
        global_format_choice = get_global("global_format_choice")

        while True:
            try:
                for message_id, message in enumerate(message_list):
                    if type(message['content']) != str:
                        message_list[message_id]['content'] = str(message['content'])
                # print('message_list: ',message_list)

                if global_format_choice == 'json':
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=message_list,
                        temperature=temperature if temperature is not None else self.temperature,
                        response_format={"type": "json_object"}
                    )
                    json_string = response.choices[0].message.content

                elif global_format_choice == 'xml':
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=message_list,
                        temperature=temperature if temperature is not None else self.temperature,
                        min_tokens=3000,
                    )
                    ori_answer = response.choices[0].message.content
                    # print('ori_answer: ',ori_answer)
                    json_string = self.xml_to_json(ori_answer)
                else:
                    raise NotImplementedError

                # exit()
                # print('message_list: ',message_list)
                # print('json_string: ',json_string)

                return json_string, response.usage
            # NOTE: BadRequestError is triggered once for MMMU, please uncomment if you are reruning MMMU
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return ""
            except Exception as e:
                exception_backoff = 2 ** trial  # expontial back off
                print(
                    f"Together AI: Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
                if trial == 3:  # basically mean it is bad request after 3 trials
                    print("Bad Request Error", e)
                    return ""
            # unknown error shall throw exception
