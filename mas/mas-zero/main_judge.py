import argparse
import os
import json
import re
import asyncio

from common import EQUALITY_TEMPLATE, MCQ_EQUALITY_TEMPLATE, GAIA_EQUALITY_TEMPLATE, ANSWER_PATTERN
from llm_judge import self_verifier_list_wise
from sampler.chat_completion_sampler import ChatCompletionSampler
from sampler.o_chat_completion_sampler import OChatCompletionSampler
from sampler.together_completion_sampler import ChatCompletionSampler as ToChatCompletionSampler
from sampler.vllm_completion_sampler import ChatCompletionSampler as VllmChatCompletionSampler
from sampler.gpt5_chat_completion_sampler import AsyncChatCompletionSampler as Gpt5ChatCompletionSampler

async def check_equality(dataset, question, correct, candidate):
    FORMAT_INST = lambda \
        request_keys: f"""Reply EXACTLY with the following JSON format.\n{str(request_keys)}\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed JSON object!\n\n"""

    output_description = "Return ONLY 'yes' or 'no' and DO NOT return anything other than these two."
    thinking_description = "Give your detialed thinking, Specifically, what is expression 1 and what is expression 2."

    output_fields_and_description = {key: f"Your {key}. {thinking_description}" if 'thinking' in key else f"Your {key}. {output_description}" for key in
                                     ['thinking', 'equal']}

    system_prompt = 'You are a helpful assistant. ' + FORMAT_INST(output_fields_and_description)

    if dataset == 'aime24':

        prompt = EQUALITY_TEMPLATE % {"expression1": correct, "expression2": candidate}

        msg = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        while True:
            try:
                response, _ = await equality_checker(msg)
                json_dict = json.loads(response)
                break
            except Exception as e:
                print(f'Error: {e}')

        print(f'json_dict: {json_dict}')
        score = json_dict['equal'].lower().strip() == "yes"
    
    elif dataset == 'aime25':

        prompt = EQUALITY_TEMPLATE % {"expression1": correct, "expression2": candidate}

        msg = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        while True:
            try:
                response, _ = await equality_checker(msg)
                json_dict = json.loads(response)
                break
            except Exception as e:
                print(f'Error: {e}')

        print(f'json_dict: {json_dict}')
        score = json_dict['equal'].lower().strip() == "yes"

    elif dataset == 'gaia':

        prompt = GAIA_EQUALITY_TEMPLATE % {"expression1": correct, "expression2": candidate}

        msg = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        while True:
            try:
                response, _ = await equality_checker(msg)
                json_dict = json.loads(response)
                break
            except Exception as e:
                print(f'Error: {e}')

        print(f'json_dict: {json_dict}')
        score = json_dict['equal'].lower().strip() == "yes"

    return score


parser = argparse.ArgumentParser()
parser.add_argument('--judge_method', type=str)
parser.add_argument('--baseline', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--max_sample', type=int)
parser.add_argument('--min_sample', type=int, default=0)
parser.add_argument('--max_response_per_sample', type=int)
parser.add_argument('--node_model', type=str, default="gpt-4o_chatgpt")
parser.add_argument('--model', type=str, default="gpt-4o_chatgpt")
parser.add_argument('--majority_vote', action='store_true')
parser.add_argument("--save_dir", type=str, default="async_results")
args = parser.parse_args()

if __name__ == "__main__":

    dataset = args.dataset
    judge_method = args.judge_method
    max_sample = args.max_sample
    min_sample = args.min_sample
    max_response_per_sample = args.max_response_per_sample
    model = args.model
    node_model = args.node_model
    majority_vote = args.majority_vote

    special_ids = []
    root_dir = f'{args.save_dir}/question/meta_agent/{args.baseline}'

    # all results
    if judge_method == 'external':
        prm_model_path = "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B"
        result_path = f'{root_dir}/{dataset}/{model}_{node_model}_Skywork-o1-Open-PRM-Qwen-2.5-7B.results'
    else:
        result_path = f'{root_dir}/{dataset}/{model}_{node_model}_{judge_method}.results_{max_response_per_sample}'
    if os.path.exists(result_path):
        os.remove(result_path)  # remove the file, do not repeat

    print('result_path: ', result_path)

    model_sampler_map = {
        "o3-mini": OChatCompletionSampler(
            model="o3-mini",
        ),
        "gpt-4o_chatgpt": ChatCompletionSampler(
            model="gpt-4o",
        ),
        "qwen-2.5-32b-instr": VllmChatCompletionSampler(
            model="qwen-2.5-32b-instr",
        ),
        "qwen3-30b-a3b": VllmChatCompletionSampler(
            model="qwen3-30b-a3b",
        ),
        "qwq-32b": ToChatCompletionSampler(
            model="Qwen/Qwen2.5-32B-Instruct",
        ),
        "llama-3.3-70b-instr": ToChatCompletionSampler(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        ),
        "deepseek-v3": ToChatCompletionSampler(
            model="deepseek-ai/DeepSeek-V3"
        ),
        "gpt-4.1-nano":ChatCompletionSampler(
            model="gpt-4.1-nano",
        ),
        "gpt-5-nano":Gpt5ChatCompletionSampler(
            model="gpt-5-nano",
        ),
        "gpt-5-mini":Gpt5ChatCompletionSampler(
            model="gpt-5-mini",
        )
    }

    # we always use gpt-4o for post-process and equilty check
    post_processer = model_sampler_map['gpt-4o_chatgpt']
    # Use async sampler for concurrent processing
    equality_checker = model_sampler_map['gpt-5-mini']
    sampler = model_sampler_map[model]

    correct_example = []

    async def process_example(example_id):
        """Process a single example and return whether it's correct"""
        print(f'-------- example_id {example_id} --------')

        # Old version: Load from response file with full response text
        response_path = f'{root_dir}/{dataset}/{example_id}/{model}_{node_model}_gpt-4o_chatgpt_0_plan_response'
        # reponse_path = f'{root_dir}/{dataset}/{example_id}/{model}_{model}_{model}_0__reponse' #sometimes miss "plan"
        try:
            with open(response_path, 'r') as json_file:
                responses = json.load(json_file)
        except Exception as e:
            print(f'example_id {example_id} response file {response_path} does not exisit')
            special_ids.append(f'example_id {example_id} response file {response_path} does not exisit')
            return False

        if len(responses) < max_response_per_sample:
            print(f'responses length {len(responses)} is lower than {max_response_per_sample}')
            special_ids.append(f'example_id {example_id}: responses length {len(responses)} is lower than {max_response_per_sample}')
            # just a warning is fine
            # pass instead of return, continue processing

        question = responses[0]['problem']  # all responses have the same answer

        # accumulate
        extracted_answers = []
        correct_answers = []

        for response in responses:
            if isinstance(response['n'], int):
                filter_response = response['response']
            else:
                continue
            # filter_response = response['response']
            # TODO: for gpqa, in some cases, it gives the final answer instead of final selection
            if '<TOO_HARD>' in filter_response:
                filter_response = filter_response[:filter_response.index('<TOO_HARD>')]
                # print(f'<TOO_HARD> detected: response: {response['response']}; filter_response: {filter_response}')

            match = re.search(ANSWER_PATTERN, filter_response)
            extracted_answer = match.group(1) if match else None
            extracted_answers.append(
                extracted_answer.strip() if extracted_answer is not None else extracted_answer)  # for exact match, "strip()" can make a significant difference

            correct_answer = response['correct_answer']
            correct_answers.append(correct_answer)
        # print(len(responses), len(extracted_answers), len(correct_answers))
        # Load extracted answers from mem.json and filter only those selected from greedy
        # mem_path = f'{root_dir}/{dataset}/{example_id}/{model}_{node_model}_gpt-4o_chatgpt_0_plan_mem.json'
        
        # try:
        #     with open(mem_path, 'r') as json_file:
        #         mem_data = json.load(json_file)
            
        #     # Filter only entries that contain "Selected from greedy"
        #     filtered_data = [item for item in mem_data if any("Selected from greedy" in str(v) for v in item.values())]
            
        #     if not filtered_data:
        #         print(f'example_id {example_id}: No "Selected from greedy" entries found in mem.json')
        #         special_ids.append(f'example_id {example_id}: No "Selected from greedy" entries found')
        #         return False
            
        #     # Extract answers (the dictionary keys)
        #     extracted_answers = [list(item.keys())[0] for item in filtered_data]
            
        #     # Get correct answer from first response
        #     correct_answer = responses[0]['correct_answer']
        #     correct_answers = [correct_answer] * len(extracted_answers)
            
        # except Exception as e:
        #     print(f'example_id {example_id} mem file {mem_path} does not exist or error loading: {e}')
        #     special_ids.append(f'example_id {example_id} mem file error: {e}')
        #     return False
        
        # if len(extracted_answers) < max_response_per_sample:
        #     print(f'filtered extracted_answers length {len(extracted_answers)} is lower than {max_response_per_sample}')
        #     special_ids.append(f'example_id {example_id}: filtered extracted_answers length {len(extracted_answers)} is lower than {max_response_per_sample}')
        #     # just a warning is fine
        #     # pass instead of return, continue processing

        print('extracted_answers: ', extracted_answers)
        print('correct_answers: ', correct_answers)
        is_correct = False

        if judge_method == 'oracle':

            for round_id, (correct_answer, extracted_answer) in enumerate(zip(correct_answers, extracted_answers)):
                # if round_id < 5:
                #     continue
                if round_id == max_response_per_sample: break  # Do not go futher

                print('round_id: ', round_id)
                print('extracted_answer: ', extracted_answer)

                score = await check_equality(dataset, question, correct_answer, extracted_answer)

                if score == 1:
                    print(f'correct: correct_answer: {correct_answer} vs. extracted_answer: {extracted_answer}')
                    with open(result_path, "a+") as fh:
                        fh.write(
                            f'experiemnt {example_id}: 1 (round_{round_id}); correct_answer: {correct_answer} vs. extracted_answer: {extracted_answer}\n')
                    is_correct = True
                    break

        elif judge_method == 'external':
            from llm_judge import prm

            post_process_path = f'{root_dir}/{dataset}/{example_id}/{model}_{node_model}_{model}_0_plan_post_process.json'
            print('post_process_path: ', post_process_path)

            try:
                chosen_id = prm.run_judge(prm_model_path, result_path, post_process_path, responses, sampler, post_processer, extracted_answers, dataset)
            except Exception as e:
                special_ids.append(f'Error: {e}; skip')
                print(f'Error: {e}; skip')
                return False
            print(f'chosen_id: {chosen_id}')

            correct_answer = correct_answers[chosen_id]
            extracted_answer = extracted_answers[chosen_id]

            score = await check_equality(dataset, question, correct_answer, extracted_answer)

            if score == 1:  # if the chosen one is correct
                print(f'correct: correct_answer: {correct_answer} vs. extracted_answer: {extracted_answer}')
                with open(result_path, "a+") as fh:
                    fh.write(
                        f'experiemnt {example_id}: 1 ({responses[chosen_id]["n"]}); correct_answer: {correct_answer} vs. extracted_answer: {extracted_answer}\n')
                is_correct = True

        if judge_method == 'self':
            # TODO: consider a list-wise judge

            post_process_path = f'{root_dir}/{dataset}/{example_id}/{model}_{model}_{model}_0_plan_sub_task_post_process.json'
            log_path = f'{root_dir}/{dataset}/{example_id}/{model}_{model}_{model}_0_plan_sub_self_verifier_log'
            score_path = f'{root_dir}/{dataset}/{example_id}/{model}_{model}_{model}_0_plan_score.json'

            chosen_id = self_verifier_list_wise.run_self_verifier(post_process_path, log_path, score_path, responses, sampler, post_processer,
                                                                  extracted_answers, dataset, max_response_per_sample, majority_vote)

            print('chosen_id: ', chosen_id)
            correct_answer = correct_answers[chosen_id]
            extracted_answer = extracted_answers[chosen_id]

            score = await check_equality(dataset, question, correct_answer, extracted_answer)

            if score == 1:
                print(f'correct: correct_answer: {correct_answer} vs. extracted_answer: {extracted_answer}')
                with open(result_path, "a+") as fh:
                    fh.write(
                        f'experiemnt {example_id}: 1 ({responses[chosen_id]["n"]}); correct_answer: {correct_answer} vs. extracted_answer: {extracted_answer}\n')
                    is_correct = True

        return is_correct

    # Process all examples concurrently
    async def process_all_examples():
        tasks = [process_example(example_id) for example_id in range(min_sample, max_sample + 1)]
        results = await asyncio.gather(*tasks)
        return results

    # Run all examples and collect results
    results = asyncio.run(process_all_examples())
    correct_example = [1 if result else 0 for result in results]

    for special_id in special_ids:
        print('special_id: ', special_id)

    acc = sum(correct_example) / len(correct_example)
    print(f'coorect {sum(correct_example)}; Total: {len(correct_example)}; Acc: {acc}')

    with open(result_path, "a+") as fh:
        fh.write(f'coorect {sum(correct_example)}; Total: {len(correct_example)}; Acc: {acc}\n')
