"""
AIME-specific utility functions for DyLAN framework
"""
import re
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MMLU'))
from prompt_lib import TEMPERATURE, MAX_TOKENS
from openai import OpenAI

# Initialize OpenAI client (for v1.0+)
client = OpenAI()


def _strip_string(string):
    """Strip and normalize math answer string"""
    # linebreaks
    string = string.replace("\n", "")
    # remove inverse spaces
    string = string.replace("\\!", "")
    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    # remove dollar signs
    string = string.replace("\\$", "")
    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    # remove spaces
    string = string.replace(" ", "")
    
    if string != "":
        if string[-1] == ".":
            string = string[:-1]
        if string[-1] == "/":
            string = string[:-1]
        # Remove leading zeros for pure integer/float answers (but not for '0' itself)
        # Only if string is all digits (possibly with a leading minus sign)
        s = string
        if s.startswith('-'):
            sign = '-'
            s = s[1:]
        else:
            sign = ''
        if s.isdigit():
            # Remove leading zeros, but keep '0' as is
            s = s.lstrip('0')
            if s == '':
                s = '0'
            string = sign + s
    return string


def extract_math_answer(pred_str):
    """Extract answer from AIME response - looks for <answer></answer> tags first"""
    if pred_str is None:
        return None
    
    pred_str = str(pred_str)
    
    # First priority: Look for <answer>...</answer> tags
    answer_tag_pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(answer_tag_pattern, pred_str, re.IGNORECASE | re.DOTALL)
    if match:
        answer = match.group(1).strip()
        answer = _strip_string(answer)
        return answer
    
    # Second priority: Look for \boxed{...} (backward compatibility)
    if 'boxed' in pred_str:
        ans = pred_str.split('boxed')[-1]
        if len(ans) == 0:
            return ""
        if (ans[0] == '{'):
            stack = 1
            a = ''
            for c in ans[1:]:
                if (c == '{'):
                    stack += 1
                    a += c
                elif (c == '}'):
                    stack -= 1
                    if (stack == 0): break
                    a += c
                else:
                    a += c
        else:
            a = ans.split('$')[0].strip()
        a = _strip_string(a)
        return a
    
    # Third priority: Look for "The answer is ..." or "the answer is ..."
    if('The answer is ' in pred_str):
        pred = pred_str.split('The answer is ')[-1].strip()
        pred = _strip_string(pred)
        return pred
    elif('the answer is ' in pred_str):
        pred = pred_str.split('the answer is ')[-1].strip()
        pred = _strip_string(pred)
        return pred
    
    # Last resort: extract last number
    pattern = '-?\d+\.?\d*'
    pred = re.findall(pattern, pred_str)
    if(len(pred) >= 1):
        pred = pred[-1]
    else:
        pred = ''
    
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]
        if pred[-1] == "/":
            pred = pred[:-1]
    
    pred = _strip_string(pred)
    return pred


def is_equiv(str1, str2, verbose=False):
    """Check if two math answers are equivalent"""
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str(str1))
        ss2 = _strip_string(str(str2))
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str(str1) == str(str2)


def generate_answer_aime(answer_context, model):
    """Generate answer for AIME (no tool calling)"""
    print("question context: ")
    print(answer_context)
    
    # Build request parameters
    request_params = {
        "model": model,
        "messages": answer_context,
    }
    
    # Add model-specific parameters
    if "gpt-5" in model.lower():
        request_params["reasoning_effort"] = "minimal"
    else:
        request_params["temperature"] = TEMPERATURE
        request_params["max_tokens"] = MAX_TOKENS
        request_params["n"] = 1
    
    try:
        # Use new OpenAI v1.0+ API
        response = client.chat.completions.create(**request_params)
        
        # Extract data from response object
        content = response.choices[0].message.content
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        
        return content, prompt_tokens, completion_tokens
        
    except Exception as e:
        print(f"[ERROR] API call failed: {e}")
        return "Unable to answer due to API error.", 0, 0
