import json
import os
import random
import re

from openai import OpenAI

from prompt_lib import TEMPERATURE, MAX_TOKENS

client = OpenAI()


def parse_single_choice(reply):
    pattern = r'\(([ABCDabcd])\)'
    matches = re.findall(pattern, reply)

    solution = None
    for match_str in matches[::-1]:
        solution = match_str.upper()
        if solution:
            break

    if solution is None:
        alter_pattern = r'([ABCDabcd])\)'
        alter_matches = re.findall(alter_pattern, reply)
        for match_str in alter_matches[::-1]:
            solution = match_str.upper()
            if solution:
                break

    return solution


def generate_answer(answer_context, model):
    # print("question context: ")
    # print(answer_context)

    request_params = {
        "model": model,
        "messages": answer_context,
        "n": 1,
    }

    if "gpt-5" in model.lower():
        request_params["reasoning_effort"] = "minimal"
    else:
        request_params["temperature"] = TEMPERATURE
        request_params["max_tokens"] = MAX_TOKENS

    try:
        response = client.chat.completions.create(**request_params)
        return response.choices[0].message.content, response.usage.prompt_tokens, response.usage.completion_tokens
    except Exception as e:
        # print(f"[ERROR] API call failed: {e}")
        return "Unable to answer due to API error.", 0, 0


def most_frequent(clist, cmp_func):
    counter = 0
    num = clist[0]

    for item in clist:
        current_frequency = sum(cmp_func(item, other) for other in clist)
        if current_frequency > counter:
            counter = current_frequency
            num = item

    return num, counter


def _normalize_answer(answer):
    if answer is None:
        return None
    answer = str(answer).strip().upper()
    match = re.search(r'\(([ABCD])\)', answer)
    if match:
        return match.group(1)
    match = re.search(r'\b([ABCD])\b', answer)
    if match:
        return match.group(1)
    return answer


def get_gpqa_qa_pairs(sub_dir, min_file, max_file):
    ret_list = []
    for subdir, _, files in os.walk(sub_dir):
        for file in files:
            if not file.endswith('.json'):
                continue

            filename_without_ext = os.path.splitext(file)[0]
            if not filename_without_ext.isdigit():
                continue

            file_num = int(filename_without_ext)
            if min_file <= file_num <= max_file:
                with open(os.path.join(subdir, file), 'r', encoding='utf-8') as fp:
                    try:
                        problem_data = json.load(fp)
                    except Exception as e:
                        print(f"Error loading JSON from {file}", e)
                        continue

                    question = problem_data["question"]
                    answer = _normalize_answer(problem_data["answer"])
                    ret_list.append((question, answer))

    return sorted(ret_list, key=lambda x: x[0])