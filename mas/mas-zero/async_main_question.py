import argparse
import asyncio
import copy

import pandas as pd
from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio

# import async_search_iter as search
import async_search as search
import async_search_sub as search
from prompts.swe.patch_oracle import AGENTLESS_REPAIR
from sampler import init_model
from utils import extract_xml
from utils import load_questions


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--valid_size', type=int, default=128)
    parser.add_argument('--test_size', type=int, default=800)
    parser.add_argument('--shuffle_seed', type=int, default=0)
    parser.add_argument('--n_repeats', type=int, default=1)
    parser.add_argument('--multiprocessing', action='store_true', default=True)
    parser.add_argument('--max_workers', type=int, default=48)
    parser.add_argument('--debug', action='store_true', default=True)
    parser.add_argument('--save_dir', type=str, default='async_results/')
    parser.add_argument('--expr_name', type=str)
    parser.add_argument('--n_generation', type=int, default=10)
    parser.add_argument('--max_round', type=int, default=5)
    parser.add_argument('--max_sc', type=int, default=5)
    parser.add_argument('--debug_max', type=int, default=3)
    parser.add_argument('--option', type=str, default='')
    parser.add_argument('--meta_model',
                        type=str)
    parser.add_argument('--node_model',
                        type=str)
    parser.add_argument('--verifier_model',
                        type=str,
                        default="o3-mini")
    # gpt-4o
    parser.add_argument('--shorten_context', action='store_true')
    parser.add_argument('--merge_context', action='store_true')

    parser.add_argument(
        "--blocks", type=str, nargs="*", help="Number of examples to use (overrides default)"
    )
    parser.add_argument('--dataset', type=str)
    parser.add_argument(
        "--given_examples", type=int, nargs="*", help="Number of examples to use (overrides default)"
    )
    parser.add_argument(
        "--use_oracle_verifier", action='store_true', default=False
    )
    parser.add_argument(
        "--defer_verifier", action='store_true'
    )
    parser.add_argument(
        "--no_decompose", action='store_true'
    )
    parser.add_argument(
        "--no_meta_reward", action='store_true'
    )
    args = parser.parse_args()

    return args


async def run_aime_search(example, example_id, meta_model, node_model, verifier_model, n, dataset, extra_info,
                          blocks, n_generation, save_dir, option, defer_verifier, debug_max):
    expr_name = f'question/meta_agent/{dataset}/{example_id}/{meta_model}_{node_model}_{verifier_model}_{n}'
    print('args.expr_name: ', expr_name)

    questions = [example['problem']]
    answers = [example['answer']]

    task_queue = []
    for q in questions:
        taskInfo = ('task', 'User', q, None, None, None, -1)
        task_queue.append(taskInfo)

    extra_info["answers"] = answers
    extra_info["questions"] = questions
    extra_info["example_id"] = example_id
    extra_info["instance_id"] = example_id
    extra_info["response_dict"] = []

    # search
    await search.search(extra_info, task_queue, meta_model, blocks, verifier_model, n_generation,
                        save_dir, expr_name, option, dataset, defer_verifier, debug_max)

async def run_gaia_search(example, example_id, meta_model, node_model, verifier_model, n, dataset, extra_info,
                                     blocks, n_generation, save_dir, option, defer_verifier, debug_max):
    expr_name = f'question/meta_agent/{dataset}/{example_id}/{meta_model}_{node_model}_{verifier_model}_{n}'
    print('args.expr_name: ', expr_name)

    questions = [example['Question']]
    answers = [example['answer']]

    task_queue = []
    for q in questions:
        taskInfo = ('task', 'User', q, None, None, None, -1)
        task_queue.append(taskInfo)

    extra_info["answers"] = answers
    extra_info["questions"] = questions
    extra_info["example_id"] = example_id
    extra_info["instance_id"] = example.get("task_id", f"gaia_{example_id}")
    extra_info["response_dict"] = []

    # search
    await search.search(extra_info, task_queue, meta_model, blocks, verifier_model, n_generation,
                                  save_dir, expr_name, option, dataset, defer_verifier, debug_max)


async def main(args):
    blocks = args.blocks
    meta_model = args.meta_model
    node_model = args.node_model
    verifier_model = args.verifier_model
    use_oracle_verifier = args.use_oracle_verifier
    max_round = args.max_round
    max_sc = args.max_sc

    # SEARCHING_MODE = True
    # technique = args.dataset.split('/')[0]
    # data_scorer = DataScorer(args.dataset, technique)

    print('verifier_model: ', verifier_model)
    # print('technique: ', technique)
    print('node_model: ', node_model)

    json_model = ['gpt']
    xml_model = ['qwen', 'llama-3.3', 'deepseek']

    extra_info = {}
    if any(kw in node_model for kw in json_model):
        format_inst_template = ("Reply EXACTLY with the following JSON format.\n{request_keys}\n"
                                "DO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed JSON object!\n\n")
        extra_info["format_choice"] = "json"

    elif any(kw in node_model for kw in xml_model):
        format_inst_template = ("Reply EXACTLY with the following XML format.\n{request_keys}\n"
                                "DO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed XML object!\n\n")
        extra_info["format_choice"] = 'xml'

    else:
        raise NotImplementedError

    init_model(verifier_model)
    init_model(node_model)
    init_model(meta_model)

    extra_info["FORMAT_INST"] = format_inst_template
    extra_info["shorten_context"] = args.shorten_context
    extra_info["merge_context"] = args.merge_context
    extra_info["COST_TOTAL"] = 0.0
    extra_info["no_decompose"] = args.no_decompose
    extra_info["no_meta_reward"] = args.no_meta_reward

    print('shorten_context: ', args.shorten_context)
    print('merge_context: ', args.merge_context)
    print('global_no_meta_reward: ', args.no_meta_reward)
    print('global_no_decompose: ', args.no_decompose)

    code_snippet = None
    for n in range(args.n_repeats):

        if 'swe_bench' in args.dataset:

            cot_instruction = "Put your thinking process in the 'thinking' entry and the final patch in the 'answer' entry."  # TODO: may need something for xml

            debate_role = ['Computer Science Professor', 'Software Engineer']

            # output_description = "Return ONLY an integer. DO NOT return anything other than the integer answer."
            output_description = (
                "If the question is asked for a patch to fix an issue, Return ONLY the solution and DO NOT return anything other than the patch;"
                " If the question is asked for more than a patch, Return what the question asked and make sure the answer is complete.")

            # Load SWE-bench dataset
            examples = load_dataset("princeton-nlp/SWE-bench_Lite_oracle", split="test")

            for example_id, example in enumerate(examples):

                if args.given_examples:
                    if example_id not in args.given_examples: continue

                    # if example_id <= 1: continue

                args.expr_name = f'question/meta_agent/{args.dataset}/{example_id}/{meta_model}_{node_model}_{verifier_model}_{n}'
                print('args.expr_name: ', args.expr_name)

                instance_id = example['instance_id']
                example_text = example['text']

                if extra_info["format_choice"] == 'xml':  # conflict with xml TODO: ADAS and OURS may also need to change
                    example_text = example_text.replace('<patch>', '<answer>').replace('</patch>', '</answer>')
                    example_text = example_text.replace('Please respond with a single patch file in the following format.',
                                                        'If asked for <answer> field, the <answer> field should be a single patch file in the following format')
                    # example_text += '\n\nIf asked for <thinking> field, you should put your thinking in the <thinking> field.'
                    cot_instruction = "Put your thinking process in the <thinking> field and the final patch in the <answer> field."  # TODO: may need something for xml

                code_snippet = extract_xml(example_text, 'code').strip()
                print('code_snippet: ', code_snippet)

                questions = [example_text + '\n\n' + AGENTLESS_REPAIR]

                answers = [None]

                print('instance_id: ', instance_id)

                task_queue = []
                for q in questions:
                    taskInfo = ('task', 'User', q, None, None, None, -1)
                    task_queue.append(taskInfo)

                extra_info["output_description"] = output_description
                # extra_info["score_compute"] = data_scorer.score
                extra_info["max_round"] = max_round
                extra_info["max_sc"] = max_sc
                extra_info["debate_role"] = debate_role
                extra_info["cot_instruction"] = cot_instruction
                extra_info["node_model"] = node_model
                extra_info["answers"] = answers
                extra_info["questions"] = questions
                extra_info["use_oracle_verifier"] = use_oracle_verifier
                extra_info["example_id"] = example_id
                extra_info["response_dict"] = []
                extra_info["dataset"] = args.dataset
                extra_info["instance_id"] = instance_id
                extra_info["code_snippet"] = code_snippet

                # search
                await search.search(args, extra_info, task_queue, meta_model, blocks, verifier_model)

        elif 'aime24' in args.dataset:

            cot_instruction = "Please think step by step and then solve the task."
            # output_description = "Return ONLY an integer. DO NOT return anything other than the integer answer."
            output_description = (
                "If the question is asked for a numeric result, Return ONLY an integer and DO NOT return anything other than the integer answer; "
                "If the question is asked for more than numeric results, Return what the question asked and make sure the answer is complete.")

            debate_role = ['Math Professor', 'Grade School Teacher']

            dataset = load_dataset("simplescaling/aime24_nofigures")
            df = pd.DataFrame(dataset['train'])
            examples = [row.to_dict() for _, row in df.iterrows()]

            extra_info["node_model"] = node_model
            extra_info["verifier_model"] = verifier_model
            extra_info["output_description"] = output_description
            extra_info["max_round"] = max_round
            extra_info["max_sc"] = max_sc
            extra_info["debate_role"] = debate_role
            extra_info["cot_instruction"] = cot_instruction
            extra_info["use_oracle_verifier"] = use_oracle_verifier
            extra_info["dataset"] = args.dataset
            extra_info["code_snippet"] = code_snippet

            # Global semaphore to control total concurrent API calls across all searches and evaluations
            # With max_workers=48, limit to 40 to stay under rate limits
            global_api_semaphore = asyncio.Semaphore(40)
            
            # 控制并发数量的信号量，最多同时运行5个任务
            search_semaphore = asyncio.Semaphore(32)

            async def run_task_with_semaphore(*a, **kw):
                async with search_semaphore:
                    # Pass global semaphore to search function
                    return await run_aime_search(*a, **kw)

            tasks = []
            for example_id, example in enumerate(examples):

                if args.given_examples:
                    if example_id not in args.given_examples:
                        continue

                _info = copy.deepcopy(extra_info)
                tasks.append(run_task_with_semaphore(
                    example, example_id, meta_model, node_model, verifier_model, n, args.dataset, _info,
                    blocks, args.n_generation, args.save_dir, args.option, args.defer_verifier, args.debug_max
                ))

                # if len(tasks) >= 1:
                #     break

            await tqdm_asyncio.gather(*tasks)


        elif 'aime25' in args.dataset:

            cot_instruction = "Please think step by step and then solve the task."
            # output_description = "Return ONLY an integer. DO NOT return anything other than the integer answer."
            output_description = (
                "If the question is asked for a numeric result, Return ONLY an integer and DO NOT return anything other than the integer answer; "
                "If the question is asked for more than numeric results, Return what the question asked and make sure the answer is complete.")

            debate_role = ['Math Professor', 'Grade School Teacher']

            dataset = load_dataset("simplescaling/aime25_nofigures")
            df = pd.DataFrame(dataset['train'])
            examples = [row.to_dict() for _, row in df.iterrows()]

            extra_info["node_model"] = node_model
            extra_info["verifier_model"] = verifier_model
            extra_info["output_description"] = output_description
            extra_info["max_round"] = max_round
            extra_info["max_sc"] = max_sc
            extra_info["debate_role"] = debate_role
            extra_info["cot_instruction"] = cot_instruction
            extra_info["use_oracle_verifier"] = use_oracle_verifier
            extra_info["dataset"] = args.dataset
            extra_info["code_snippet"] = code_snippet

            # 控制并发数量的信号量，最多同时运行5个任务
            semaphore = asyncio.Semaphore(32)

            async def run_task_with_semaphore(*a, **kw):
                async with semaphore:
                    return await run_aime_search(*a, **kw)

            tasks = []
            for example_id, example in enumerate(examples):

                if args.given_examples:
                    if example_id not in args.given_examples:
                        continue

                _info = copy.deepcopy(extra_info)
                tasks.append(run_task_with_semaphore(
                    example, example_id, meta_model, node_model, verifier_model, n, args.dataset, _info,
                    blocks, args.n_generation, args.save_dir, args.option, args.defer_verifier, args.debug_max
                ))

                # if len(tasks) >= 1:
                #     break

            await tqdm_asyncio.gather(*tasks)

        elif 'gpqa_diamond' in args.dataset:

            cot_instruction = "Please think step by step and then solve the task."
            # output_description = "Return ONLY the alphabet choice, i.e. A or B or C or D."
            output_description = ("If the question is asked for a multiple-choice result, Return ONLY the alphabet choice, i.e. A or B or C or D; "
                                  "If the question is asked for more than multiple-choice results, "
                                  "return what the question asked and make sure the answer is complete.")
            # need to consider sub-task output as well (no fixed form for sub-tasks)
            debate_role = ['Biology Expert', 'Physics Expert', 'Chemistry Expert', 'Science Generalist']

            # set seed 0 for valid set
            questions = load_questions('dataset/gpqa_diamond.csv', seed=0)
            answers = [question.correct_index for question in questions]

            examples = [{'problem': questions[i], 'answer': answers[i]} for i in range(len(questions))]

            for example_id, example in enumerate(examples):
                instance_id = example_id

                if args.given_examples:
                    if example_id not in args.given_examples:
                        continue

                args.expr_name = f'question/meta_agent/{args.dataset}/{example_id}/{meta_model}_{node_model}_{verifier_model}_{n}'
                print('args.expr_name: ', args.expr_name)

                questions = [example['problem']]
                answers = [example['answer']]

                final_question = []
                task_queue = []
                for q in questions:
                    task_content = f"What is the correct answer to this question: {q.question}" \
                                   + f"\n\nChoices:\n(A) {q.choice1}\n(B) {q.choice2}\n(C) {q.choice3}\n(D) {q.choice4}"
                    taskInfo = ('task', 'User', task_content, None, None, None, -1)
                    task_queue.append(taskInfo)
                    final_question.append(task_content)

                extra_info["output_description"] = output_description
                # extra_info["score_compute"] = data_scorer.score
                extra_info["max_round"] = max_round
                extra_info["max_sc"] = max_sc
                extra_info["debate_role"] = debate_role
                extra_info["cot_instruction"] = cot_instruction
                extra_info["node_model"] = node_model
                extra_info["answers"] = answers
                extra_info["questions"] = final_question  # 注意：此处原始代码使用了 final_question 变量
                extra_info["use_oracle_verifier"] = use_oracle_verifier
                extra_info["example_id"] = example_id
                extra_info["response_dict"] = []
                extra_info["dataset"] = args.dataset
                extra_info["instance_id"] = instance_id
                extra_info["code_snippet"] = code_snippet
                # extra_info["mas_trajectory"] = []

                # search
                await search.search(extra_info, task_queue, meta_model, blocks, verifier_model, args.n_generation,
                                  args.save_dir, args.expr_name, args.option, args.dataset, args.defer_verifier, args.debug_max)

        elif 'gaia' in args.dataset:

            cot_instruction = "Please think step by step and then solve the task."
            output_description = (
                """If you are asked for a number, dont use comma to write your number neither use units such as $ or
percent sign unless specified otherwise.
If you are asked for a string, dont use articles, neither abbreviations (e.g. for cities), and write the
digits in plain text unless specified otherwise.
If you are asked for a comma separated list, apply the above rules depending of whether the element
to be put in the list is a number or a string.""")

            debate_role = ['Research Assistant', 'Fact Checker', 'Domain Expert']

            # Load GAIA dataset from local JSON file
            data_files = {"test": "dataset/GAIA.json"}
            dataset = load_dataset("json", data_files=data_files, split="test")
            examples = [row for row in dataset]

            extra_info["node_model"] = node_model
            extra_info["verifier_model"] = verifier_model
            extra_info["output_description"] = output_description
            extra_info["max_round"] = max_round
            extra_info["max_sc"] = max_sc
            extra_info["debate_role"] = debate_role
            extra_info["cot_instruction"] = cot_instruction
            extra_info["use_oracle_verifier"] = use_oracle_verifier
            extra_info["dataset"] = args.dataset
            extra_info["code_snippet"] = code_snippet

            # Control concurrent tasks
            semaphore = asyncio.Semaphore(52)

            async def run_task_with_semaphore(*a, **kw):
                async with semaphore:
                    return await run_gaia_search(*a, **kw)

            tasks = []
            for example_id, example in enumerate(examples):

                if args.given_examples:
                    if example_id not in args.given_examples:
                        continue

                _info = copy.deepcopy(extra_info)
                
                tasks.append(run_task_with_semaphore(
                    example, example_id, meta_model, node_model, verifier_model, n, args.dataset, _info,
                    blocks, args.n_generation, args.save_dir, args.option, args.defer_verifier, args.debug_max
                ))

            await tqdm_asyncio.gather(*tasks)

        else:
            raise NotImplementedError


if __name__ == '__main__':
    args = parse_arguments()

    asyncio.run(main(args))
