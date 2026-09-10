import multiprocessing
import re
from typing import Any, List, Optional, Union
from slm_post_train.rewards.registry import register_reward


def _extract_text(completion: Union[str, list, dict, Any]) -> str:
    """Extract string content from string or conversational message formats."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        if completion and isinstance(completion[-1], dict) and "content" in completion[-1]:
            return str(completion[-1]["content"])
        return "".join(
            item.get("content", str(item)) if isinstance(item, dict) else str(item)
            for item in completion
        )
    if isinstance(completion, dict):
        if "content" in completion:
            return str(completion["content"])
        return str(completion)
    return str(completion)


def _safe_exec_worker(conn, code_str: str) -> None:
    """Target function executed in isolated child process for code execution."""
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
        conn.send(True)
    except Exception:
        conn.send(False)
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _run_code_with_timeout(code_str: str, timeout: float = 2.0) -> bool:
    """Run code string with timeout using child process to prevent hanging on infinite loops."""
    try:
        ctx = multiprocessing.get_context("fork")
    except ValueError:
        ctx = multiprocessing.get_context()

    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_safe_exec_worker, args=(child_conn, code_str))
    proc.start()
    child_conn.close()

    proc.join(timeout=timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=0.5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        parent_conn.close()
        return False

    success = False
    try:
        if parent_conn.poll():
            success = bool(parent_conn.recv())
    except Exception:
        success = False
    finally:
        parent_conn.close()

    return success


@register_reward("xml_format")
def xml_format_reward(prompts: List[Any], completions: List[Any], **kwargs) -> List[float]:
    """Scores 1.0 if output matches <think>...</think><answer>...</answer> structure, else 0.0."""
    scores = []
    pattern = re.compile(r"^<think>.*?</think>\s*<answer>.*?</answer>$", re.DOTALL)
    for raw_completion in completions:
        text = _extract_text(raw_completion).strip()
        scores.append(1.0 if pattern.match(text) else 0.0)
    return scores


@register_reward("exact_match")
def exact_match_reward(
    prompts: List[Any],
    completions: List[Any],
    answer: Optional[List[str]] = None,
    **kwargs
) -> List[float]:
    """Extracts text inside <answer>...</answer> and compares to ground truth answer."""
    scores = []
    answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    for i, raw_completion in enumerate(completions):
        if not answer or i >= len(answer):
            scores.append(0.0)
            continue
        expected = str(answer[i]).strip().lower()
        completion_text = _extract_text(raw_completion)
        match = answer_pattern.search(completion_text)
        if match:
            extracted = match.group(1).strip().lower()
            scores.append(1.0 if extracted == expected else 0.0)
        else:
            scores.append(0.0)
    return scores


@register_reward("code_execution")
def code_execution_reward(
    prompts: List[Any],
    completions: List[Any],
    timeout: float = 2.0,
    **kwargs
) -> List[float]:
    """Safely checks if python code block can be executed without error within timeout."""
    code_block_pattern = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL)
    scores = []
    for raw_completion in completions:
        completion_text = _extract_text(raw_completion)
        match = code_block_pattern.search(completion_text)
        if not match:
            scores.append(0.0)
            continue
        code_str = match.group(1)
        passed = _run_code_with_timeout(code_str, timeout=timeout)
        scores.append(1.0 if passed else 0.0)
    return scores
