import json
import re
import os
import ast
import random
from enum import Enum
from typing import Any, List, Tuple
from collections import Counter

class CodeDataset(Enum):
    HUMAN_EVAL = "HumanEval"
    MBPP = "MBPP"

def extract_random_prompt(log_path: str):
    parent_dir = os.path.dirname(log_path)

    prompt_file_path = os.path.join(parent_dir, "template", "op_prompt.py")
    
    if not os.path.exists(prompt_file_path):
        raise FileNotFoundError(f"Prompt does not exist: {prompt_file_path}")
    
    with open(prompt_file_path, 'r', encoding='utf-8') as f:
        file_content = f.read()
    
    tree = ast.parse(file_content)
    prompt_dict = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.endswith('_PROMPT'):
                    if isinstance(node.value, ast.Constant):
                        prompt_content = node.value.value
                    elif isinstance(node.value, ast.Str):
                        prompt_content = node.value.s
                    else:
                        continue
                    prompt_dict[target.id] = prompt_content

    if not prompt_dict:
        return None, None

    prompt_name, prompt_content = random.choice(list(prompt_dict.items()))
    return prompt_name, prompt_content

def update_prompt_in_file(log_path: str, prompt_name: str, prompt_content: str):
    parent_dir = os.path.dirname(log_path)
    prompt_file_path = os.path.join(parent_dir, "template", "op_prompt.py")
    
    if not os.path.exists(prompt_file_path):
        raise FileNotFoundError(f"Prompt does not exist: {prompt_file_path}")
    
    with open(prompt_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    pattern = rf'{re.escape(prompt_name)}\s*=\s*"""(.*?)"""'
    match = re.search(pattern, content, flags=re.DOTALL)

    if match:
        old_prompt = match.group(1)
        old_placeholders = re.findall(r'{([^{}]+)}', old_prompt)
        new_placeholders = re.findall(r'{([^{}]+)}', prompt_content)
        if Counter(old_placeholders) != Counter(new_placeholders):
            return
    else:
        pass

    new_assignment = f'{prompt_name} = """\n{prompt_content}\n"""'
    
    if re.search(pattern, content, flags=re.DOTALL):
        new_content = re.sub(pattern, new_assignment, content, flags=re.DOTALL)
    else:
        new_content = content.rstrip() + "\n\n" + new_assignment + "\n"
    
    with open(prompt_file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

def extract_test_cases_from_jsonl(entry_point: str, dataset: CodeDataset = CodeDataset.HUMAN_EVAL):
    if dataset == CodeDataset.HUMAN_EVAL.value:
        file_path = "maas/ext/maas/data/humaneval_public_test.jsonl"
        # Retain the original hardcoded test cases
        hardcoded_cases = {
            "find_zero": "",
            "decode_cyclic": "",
            "decode_shift": "",
            "by_length": "",
            "add": "",
            "triangle_area": "",
            "correct_bracketing": "",
            "solve": "",
            "sum_squares": "",
            "starts_one_ends": "",
        }
    elif dataset == CodeDataset.MBPP.value:
        file_path = "maas/ext/maas/data/mbpp_public_test.jsonl"
        hardcoded_cases = {
            "remove_odd": "",
            "replace_spaces": "",
            "snake_to_camel": "",
            "Split": "",
            "swap_List": "",
            "square_Sum": "",
            "sort_sublists": "",
            "unique_sublists": "",
        }
    # Check if there are hardcoded test cases
    if entry_point in hardcoded_cases:
        return hardcoded_cases[entry_point]

    # If there are no hardcoded test cases, read from the file
    with open(file_path, "r") as file:
        for line in file:
            data = json.loads(line)
            if data.get("entry_point") == entry_point:
                return data.get("test")

    return None


def extract_test_cases(docstring: str) -> List[Tuple[str, List[Any], Any]]:
    # Use regular expressions to match test cases, now capturing function names and any output
    pattern = r">>> (\w+)\((.*?)\)\n\s*(.*?)(?=\n|$)"
    matches = re.findall(pattern, docstring, re.DOTALL)

    test_cases = []
    for match in matches:
        func_name, input_str, expected_output = match

        # Process input
        input_list = []
        for item in input_str.split(","):
            item = item.strip()
            try:
                # Try to convert input to numeric type
                if "." in item:
                    input_list.append(float(item))
                else:
                    input_list.append(int(item))
            except ValueError:
                # If unable to convert to numeric, keep as string
                input_list.append(item.strip("'\""))

        # Process output
        try:
            # Try to convert output to numeric or boolean value
            if expected_output.lower() == "true":
                expected_output = True
            elif expected_output.lower() == "false":
                expected_output = False
            elif "." in expected_output:
                expected_output = float(expected_output)
            else:
                expected_output = int(expected_output)
        except ValueError:
            # If unable to convert, keep as string
            expected_output = expected_output.strip("'\"")

        test_cases.append([func_name, input_list, expected_output])

    return test_cases


def test_cases_2_test_functions(solution: str, test_cases: str):
    tester_function = f"""
{solution}

{test_cases}
"""
    return tester_function


def test_case_2_test_function(solution: str, test_case: str, entry_point: str):
    tester_function = f"""
{solution}


def check(candidate):
    {test_case}

def test_check():
    check({entry_point})

test_check()
"""
    return tester_function
