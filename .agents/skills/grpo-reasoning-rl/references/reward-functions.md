# GRPO Reward Functions & Verifiers Reference

Comprehensive guide for authoring, registering, and securing reward functions for Group Relative Policy Optimization (GRPO) in `slm-post-train`.

---

## Table of Contents
1. [Overview & Role in GRPO](#overview--role-in-grpo)
2. [Reward Registry Architecture](#reward-registry-architecture)
3. [Function Signature & Contract](#function-signature--contract)
4. [Built-In Reward Verifiers](#built-in-reward-verifiers)
   - [XML Structure Verifier (`xml_format`)](#xml-structure-verifier-xml_format)
   - [Ground Truth Verifier (`exact_match`)](#ground-truth-verifier-exact_match)
   - [Subprocess Code Execution Verifier (`code_execution`)](#subprocess-code-execution-verifier-code_execution)
5. [Authoring Custom Reward Functions](#authoring-custom-reward-functions)
6. [Reward Hacking & Mitigation Strategies](#reward-hacking--mitigation-strategies)
7. [Subprocess Isolation & Security Sandbox](#subprocess-isolation--security-sandbox)

---

## Overview & Role in GRPO

In Group Relative Policy Optimization (GRPO), the model acts as the policy $\pi_\theta$. For each training prompt $q$, the model samples a group of $G$ independent rollouts $\{o_1, o_2, \dots, o_G\}$. 

Each rollout $o_i$ is evaluated by one or more reward verifier functions $R(q, o_i) \in \mathbb{R}$. The rewards within the group are normalized into relative advantages:

$$A_i = \frac{R(q, o_i) - \text{mean}(\{R(q, o)\}_{j=1}^G)}{\text{std}(\{R(q, o)\}_{j=1}^G) + \epsilon}$$

Unlike traditional PPO, GRPO eliminates the critic/value network, drastically reducing GPU memory requirements. The reward functions serve as the sole training feedback signal driving reasoning emergence.

---

## Reward Registry Architecture

Reward functions are managed through an explicit decorator-based registry in `src/slm_post_train/rewards/registry.py`.

```python
from slm_post_train.rewards.registry import register_reward, get_reward_function, list_registered_rewards

@register_reward("my_reward")
def my_reward_func(prompts, completions, **kwargs):
    ...
```

### Registry Functions

| Function | Signature | Description |
| :--- | :--- | :--- |
| `register_reward(name)` | `(str) -> Callable` | Decorator that registers a reward function under the given name. |
| `get_reward_function(name)` | `(str) -> Callable` | Retrieves a registered reward callable by name. Raises `KeyError` if unregistered. |
| `list_registered_rewards()` | `() -> List[str]` | Lists all registered reward function names in the runtime environment. |

When a GRPO training job starts (`run_grpo`), the trainer reads the `rewards` list from YAML:
```yaml
rewards:
  - "xml_format"
  - "exact_match"
```
Each name is resolved via `get_reward_function(name)` and passed directly to TRL's `GRPOTrainer`.

---

## Function Signature & Contract

Every reward function passed to `GRPOTrainer` must adhere to the standard TRL reward function contract:

```python
def reward_function(
    prompts: List[Any],
    completions: List[Any],
    **kwargs
) -> List[float]:
```

### Contract Requirements

1. **Input Lists**: `prompts` and `completions` are lists of length $B \times G$ (batch size $\times$ number of generations per prompt).
2. **Completion Formats**: Depending on the tokenizer and dataset format, `completions` can be:
   - A list of raw strings: `["<think>...", "<think>..."]`
   - A list of conversational message dictionaries: `[[{"role": "assistant", "content": "..."}]]`
   Use the helper `_extract_text` to normalize input.
3. **Keyword Arguments (`**kwargs`)**: Any extra columns present in the prompt dataset (e.g. `answer`, `ground_truth`, `solution`, `test_cases`) are forwarded to the reward function as keyword arguments. Always accept `**kwargs` in your signature to avoid `TypeError`.
4. **Return Value**: Must return a `List[float]` whose length **strictly equals** `len(completions)`.

### Completion Normalization Helper

```python
from typing import Any, Union

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
```

---

## Built-In Reward Verifiers

Pre-built reward verifiers reside in `src/slm_post_train/rewards/standard.py`.

### XML Structure Verifier (`xml_format`)

Enforces that the reasoning model adheres to explicit `<think>` and `<answer>` XML tags.

```python
@register_reward("xml_format")
def xml_format_reward(prompts: List[Any], completions: List[Any], **kwargs) -> List[float]:
    """Scores 1.0 if output matches <think>...</think><answer>...</answer> structure, else 0.0."""
    scores = []
    pattern = re.compile(r"^<think>.*?</think>\s*<answer>.*?</answer>$", re.DOTALL)
    for raw_completion in completions:
        text = _extract_text(raw_completion).strip()
        scores.append(1.0 if pattern.match(text) else 0.0)
    return scores
```

- **Scoring**: `1.0` if strictly matching `^<think>.*?</think>\s*<answer>.*?</answer>$`, else `0.0`.
- **Purpose**: Encourages chain-of-thought separation from the final answer.

### Ground Truth Verifier (`exact_match`)

Extracts the answer enclosed within `<answer>...</answer>` and compares it case-insensitively with the ground truth `answer` column from the dataset.

```python
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
```

- **Scoring**: `1.0` if extracted answer matches ground truth, else `0.0`.
- **Dataset Dependency**: Requires dataset to include an `answer` or `solution` field.

### Subprocess Code Execution Verifier (`code_execution`)

Extracts fenced python code blocks (` ```python ... ``` `) and runs them in a sandboxed, timeout-guarded child process.

```python
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
```

- **Scoring**: `1.0` if the code runs to completion without exception within `timeout` seconds, else `0.0`.

---

## Authoring Custom Reward Functions

To author a custom reward function for domain-specific tasks (e.g., Sudoku validator, math symbolic equality):

### Example: Sudoku Solution Verifier

```python
import re
from typing import Any, List, Optional
from slm_post_train.rewards.registry import register_reward
from slm_post_train.rewards.standard import _extract_text

def is_valid_sudoku(grid: List[List[int]]) -> bool:
    """Validates 9x9 Sudoku row, column, and 3x3 block constraints."""
    for row in grid:
        nums = [n for n in row if n != 0]
        if len(nums) != len(set(nums)) or len(row) != 9:
            return False
    for col in range(9):
        nums = [grid[row][col] for row in range(9) if grid[row][col] != 0]
        if len(nums) != len(set(nums)):
            return False
    for r in range(0, 9, 3):
        for c in range(0, 9, 3):
            block = [grid[r + i][c + j] for i in range(3) for j in range(3) if grid[r + i][c + j] != 0]
            if len(block) != len(set(block)):
                return False
    return True

@register_reward("sudoku_validity")
def sudoku_validity_reward(
    prompts: List[Any],
    completions: List[Any],
    **kwargs
) -> List[float]:
    """Scores 1.0 if the completion parses into a valid Sudoku grid."""
    scores = []
    answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    for raw in completions:
        text = _extract_text(raw)
        match = answer_pattern.search(text)
        if not match:
            scores.append(0.0)
            continue
        try:
            # Parse 9 lines of 9 digits
            lines = [line.strip().split() for line in match.group(1).strip().splitlines() if line.strip()]
            grid = [[int(x) for x in line] for line in lines]
            if len(grid) == 9 and is_valid_sudoku(grid):
                scores.append(1.0)
            else:
                scores.append(0.0)
        except Exception:
            scores.append(0.0)
    return scores
```

To use it, ensure the module containing `@register_reward("sudoku_validity")` is imported before `run_grpo()`, and add `"sudoku_validity"` to the `rewards` list in your YAML config.

---

## Reward Hacking & Mitigation Strategies

Online RL policies readily exploit weaknesses in reward specifications. Common hacking patterns and defenses include:

### 1. Structure Gaming (Empty or Repetitive Thinking)
- **Problem**: The policy outputs `<think></think><answer>42</answer>` to collect the XML format reward without performing actual reasoning. Or it repeats a single token thousands of times to artificially pad thinking length.
- **Mitigation**:
  - Add a minimum non-whitespace token count or character count inside `<think>` (e.g. `>= 20` characters).
  - Use regex that prohibits immediate empty tags: `^<think>(?!\s*</think>).+?</think>\s*<answer>.+?</answer>$`.
  - Add repetition penalty rewards (e.g., n-gram diversity score).

### 2. Length Bias & Verbosity Drift
- **Problem**: Models learn that longer reasoning chains correlate with higher task accuracy, expanding generations until hitting `max_completion_length` and causing truncation.
- **Mitigation**:
  - Add a soft length penalty:
    ```python
    @register_reward("conciseness")
    def conciseness_reward(prompts, completions, **kwargs):
        scores = []
        for raw in completions:
            tokens = len(_extract_text(raw).split())
            if tokens > 500:
                scores.append(max(0.0, 1.0 - (tokens - 500) / 500.0))
            else:
                scores.append(1.0)
        return scores
    ```

### 3. The Zero-Variance Collapse
- **Problem**: In GRPO, advantages are normalized across the $G$ generations for each prompt:
  $$\frac{r_i - \bar{r}}{\sigma_r + \epsilon}$$
  If every generated rollout fails ($r = [0, 0, 0, 0]$) or every rollout succeeds ($r = [1, 1, 1, 1]$), the variance $\sigma_r$ is 0, advantages become 0, and the policy receives zero gradient update.
- **Mitigation**:
  - Ensure sampling temperature is high enough (e.g., $0.7 - 0.9$) to encourage exploratory diversity.
  - Combine dense continuous shaping rewards (e.g., partial format credit, partial matching) with sparse binary verifiers.

---

## Subprocess Isolation & Security Sandbox

Reward functions that execute code generated by the LLM (`code_execution`) pose security and stability risks:
- Malicious or accidental `import os; os.system(...)`
- Infinite loops causing training hangs (`while True: pass`)
- Memory exhaustion bombs (`[0] * 10**10`)

### Hardened Execution Sandbox Pattern

The standard sandbox in `src/slm_post_train/rewards/standard.py` enforces isolation using `multiprocessing`:

1. **Restricted `__builtins__`**: Only safe primitives (`range`, `len`, `min`, `max`, `sum`, `int`, `float`, `str`, `bool`, `list`, `dict`, `set`, `print`) are available in the evaluation namespace. Dangerous modules (`import`, `open`, `eval`, `exec`) are removed.
2. **Dedicated Child Process**: Executed via `multiprocessing.Process` with OS pipe communication.
3. **Process Join Timeout**: `proc.join(timeout=timeout)`. If execution exceeds `timeout` (default 2.0s), the process is forcefully terminated via `proc.terminate()` followed by `proc.kill()`.
4. **IPC Exception Handling**: Communication pipes are guarded with `try...finally: parent_conn.close()` to prevent file descriptor leaks during high-step training.
