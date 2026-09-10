import re
from typing import List, Optional
from slm_post_train.rewards.registry import register_reward


@register_reward("xml_format")
def xml_format_reward(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """Scores 1.0 if output matches <think>...</think><answer>...</answer> structure, else 0.0."""
    scores = []
    pattern = re.compile(r"^<think>.*?</think>\s*<answer>.*?</answer>$", re.DOTALL)
    for completion in completions:
        text = completion.strip()
        scores.append(1.0 if pattern.match(text) else 0.0)
    return scores


@register_reward("exact_match")
def exact_match_reward(
    prompts: List[str],
    completions: List[str],
    answer: Optional[List[str]] = None,
    **kwargs
) -> List[float]:
    """Extracts text inside <answer>...</answer> and compares to ground truth answer."""
    scores = []
    answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    for i, completion in enumerate(completions):
        if not answer or i >= len(answer):
            scores.append(0.0)
            continue
        expected = str(answer[i]).strip().lower()
        match = answer_pattern.search(completion)
        if match:
            extracted = match.group(1).strip().lower()
            scores.append(1.0 if extracted == expected else 0.0)
        else:
            scores.append(0.0)
    return scores


@register_reward("code_execution")
def code_execution_reward(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """Safely checks if python code block can be executed without error."""
    code_block_pattern = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL)
    scores = []
    for completion in completions:
        match = code_block_pattern.search(completion)
        if not match:
            scores.append(0.0)
            continue
        code_str = match.group(1)
        safe_globals = {
            "__builtins__": {
                "range": range,
                "len": len,
                "min": min,
                "max": max,
                "sum": sum,
                "int": int,
                "float": float,
                "str": str,
                "bool": bool,
                "list": list,
                "dict": dict,
                "set": set,
                "print": print,
            }
        }
        try:
            exec(code_str, safe_globals)
            scores.append(1.0)
        except Exception:
            scores.append(0.0)
    return scores
