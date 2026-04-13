import os
import re
from datetime import datetime

import torch
from math_verify import ExprExtractionConfig, LatexExtractionConfig, StringExtractionConfig, parse, verify

LOG_PATH = os.environ.get("REWARD_LOG_PATH", "reward.log")

choices = ["a", "b", "c", "d"]
problem_pattern = r"<\|im_start\|>user\n(.*?)<\|im_end\|>"
response_prefix = r"<\|im_start\|>assistant\n"

# 优化了code/boxed的格式限制
# format reward 如果调用了code，出现错误 format reward = 0
def get_response_from_query(q: str):
    ends_of_sentence = ["<|im_end|>", "<｜end▁of▁sentence｜>", "<|endoftext|>"]
    pos = re.search(response_prefix, q)
    if pos is None:
        return ""
    response = q[pos.end() :]
    for e in ends_of_sentence:
        response = response.replace(e, "")
    return response.strip()


def get_query_from_query(q: str):
    try:
        matches = re.findall(problem_pattern, q, re.DOTALL)
        return matches[0]
    except:
        return q


def extract_answer_with_tags(text):
    match = re.search(r"(<answer>.*?</answer>)", text)
    if match:
        return match.group(1)
    return None

def accuracy_reward_func(completion, answer):
    reward = 0.0
    response = extract_answer_with_tags(completion)
    if response != None:
        response = response
    else:
        try:
            response = completion.split("<answer>")[-1]
        except:
            response = completion.split("\n")[-1]

    content, sol = response, answer
    answer_parsed = content
    sol = f"${str(sol)}$"
    gold_parsed = parse(sol)
    if len(gold_parsed) != 0:
        answer_parsed = parse(
            content,
            extraction_config=[StringExtractionConfig(), LatexExtractionConfig(), ExprExtractionConfig()],
        )
        try:
            reward = float(verify(answer_parsed, gold_parsed))
        except Exception:
            pass

        if reward == 0.0:
            try:
                content_match = re.search(r"<answer>(.*?)</answer>", completion)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                student_answer = student_answer.replace("</answer>", "").replace("<answer>", "").strip()
                for answer in gold_parsed:
                    if str(answer).lower() in choices:
                        if str(answer).lower() in student_answer.lower():
                            choices_other = [choice for choice in choices if choice != str(answer).lower()]
                            if all(choice not in student_answer.lower() for choice in choices_other):
                                reward = 1.0
            except Exception:
                pass
    else:
        reward = 1.0
        print("Failed to parse gold solution: ", sol)

    return reward, answer_parsed


def format_reward_func(completion, **kwargs):
    pattern = (
        r"^(?=(?:.*<think>){1})(?=(?:.*<\/think>){1})"
        r"(?=(?:.*<answer>){1})(?=(?:.*<\/answer>){1})"
        r"(?!.*<think>.*<think>)"
        r"(?!.*<\/think>.*<\/think>)"
        r"(?!.*<answer>.*<answer>)"
        r"(?!.*<\/answer>.*<\/answer>)"
        r".*<think>(.+?)</think>\s*<answer>.+?</answer>.*$"
        # r"<answer>[\s\S]*?\$\\boxed\{([^}]*)\}\$\s*</answer>"
    )
    pattern_answer_boxed=r"<answer>[\s\S]*?\$\\boxed[^$]*\$\s*</answer>"
    matches = re.search(pattern, completion, re.DOTALL)
    pattern_code_block = r'<code>.*?</code>'
    pattern_code = (
        # code必须包含```python代码块，且必须在think内部
        r"(?=<think>(?:(?!<think>).)*<code>\s*```python\s*(?:.*?\s*)?```\s*<\/code>(?:(?!<think>).)*<\/think>)"
        
        # 每个code块后必须紧跟一个interpreter
        r"(?=.*<code>\s*```python\s*(?:.*?\s*)?```\s*<\/code>\s*<interpreter>.*<\/interpreter>)"
        
        # 禁止code/interpreter出现在think外部
        # r"(?!.*<code>(?:(?!<think>).)*<\/code>(?:(?!<think>).)*)"
        # r"(?!.*<interpreter>(?:(?!<think>).)*<\/interpreter>(?:(?!<think>).)*)"
        
        # 确保code-interpreter成对出现，且顺序正确
        r"(?!.*<code>(?:(?!<interpreter>).)*<code>)"
        r"(?!.*<interpreter>(?:(?!<code>).)*<interpreter>)"
    ) 
    matches_answer_boxed = re.search(pattern_answer_boxed, completion, re.DOTALL) 
    format_reward = 0.0
    if matches:
        format_reward += 0.5
        matches_code_block = re.search(pattern_code_block, completion, re.DOTALL)
        if matches_code_block:
            matches_code = re.search(pattern_code, completion, re.DOTALL)
            if not matches_code:
                format_reward = 0.0 # 如果调用了code，格式出错，扣0.5分
        
    return format_reward

def format_code_func(completion):
    pattern_format =r"(.*?<code>\s*```python\s*(.*?)\s*```\s*</code>\s*?<interpreter>(?:(?!<code>).*?)*?</interpreter>.*?)+"
    matches_format = re.search(pattern_format, completion, re.DOTALL)

    reward_score = 0.0
    if matches_format is None:
        return reward_score, 0.0

    # import pdb;pdb.set_trace()

    # 正则表达式模式
    pattern_interpreter = r"<interpreter>(.*?)</interpreter>"
    # 使用 re.findall 查找所有匹配的内容
    matches_res = re.findall(pattern_interpreter, completion, re.DOTALL)

    reward_score = 0.5
    right_times = len(matches_res)

    if right_times == 0:
        return 0.0, 0.0

    for i, match_tmp in enumerate(matches_res, 0):
        # 使用正则表达式进行不区分大小写的查找
        if re.search(r"error", match_tmp, re.IGNORECASE):
            right_times -= 1
    right_rate = right_times / len(matches_res) * 1.0
    reward_run_acc = reward_score * right_rate

    return reward_score , reward_run_acc


def reward_func(queries, prompts, labels, global_step,reward_log=None):
    # queries is prompts + responses

    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    rewards = []
    accuracy_rewards = []
    format_rewards = []
    original_code_rewards = []
    code_rewards=[]
    code_penaltys = []
    code_penalty = 0.5
    #分割符
    LOG_PATH = os.environ.get("REWARD_LOG_PATH", "reward.log")
    # print("*"*30)
    # print(f"log path {LOG_PATH}")
    # print("*"*30)
    if reward_log != None and LOG_PATH=="reward.log":
        LOG_PATH=reward_log
    # print(f"log path {LOG_PATH}")
    # #分割符
    with open(LOG_PATH, "a") as f:
        f.write(f"----------------------------- {current_time} -----------------------------\n")
        for query, prompt, answer in zip(queries, prompts, labels):
            try:
                response = get_response_from_query(query)
                if response == "":
                    f.write("Error: " + query + "\n")
                    rewards.append(0.0)
                    accuracy_rewards.append(0.0)
                    format_rewards.append(0.0)
                    original_code_rewards.append(0.0)
                    code_rewards.append(0.0)
                    code_penaltys.append(0.0)
                else:
                    query1 = get_query_from_query(query)

                    accuracy_reward, answer_parsed = accuracy_reward_func(response, answer)
                    format_reward = format_reward_func(response)

                    origal_reward_code, reward_code = format_code_func(response)
                    
                    reward_tmp = format_reward + accuracy_reward + reward_code
                    rewards.append(reward_tmp)
                    accuracy_rewards.append(accuracy_reward)
                    format_rewards.append(format_reward)
                    original_code_rewards.append(origal_reward_code)
                    code_rewards.append(reward_code)
                    code_penaltys.append(code_penalty)
                    f.write(f"===============================================================\n")
                    f.write("Query: " + query1.replace("<|image_pad|>","") + "\n")
                    f.write("Response: " + response + "\n")
                    f.write("GT Answer: " + answer + "\n")
                    f.write(f"Accuracy Reward: {accuracy_reward}\tFormat Reward: {format_reward} \t original code reward: {origal_reward_code} \t code reward: {reward_code} \t reward: {reward_tmp} \n\n\n\n")
                    f.write(f"===============================================================\n")
            except:
                f.write("Error: " + query + "\n")
                rewards.append(0.0)
                accuracy_rewards.append(0.0)
                format_rewards.append(0.0)
                original_code_rewards.append(origal_reward_code)
                code_rewards.append(reward_code)
                code_penaltys.append(code_penalty)

    return {
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "accuracy_rewards": torch.tensor(accuracy_rewards, dtype=torch.float32),
        "format_rewards": torch.tensor(format_rewards, dtype=torch.float32),
        "original_code_rewards": torch.tensor(original_code_rewards, dtype=torch.float32),
        "code_rewards": torch.tensor(code_rewards, dtype = torch.float32),
        "code_penalty": torch.tensor(code_penaltys, dtype = torch.float32)
    }
