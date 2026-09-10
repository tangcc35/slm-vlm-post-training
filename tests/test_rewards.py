import time
import pytest
from slm_post_train.rewards.registry import register_reward, get_reward_function, list_registered_rewards
from slm_post_train.rewards.standard import (
    xml_format_reward,
    exact_match_reward,
    code_execution_reward,
    _extract_text,
)


def test_reward_registry():
    @register_reward("test_custom_reward")
    def dummy_reward(prompts, completions, **kwargs):
        return [1.0] * len(completions)

    fn = get_reward_function("test_custom_reward")
    assert fn(["p"], ["c"]) == [1.0]
    assert "test_custom_reward" in list_registered_rewards()


def test_get_reward_function_not_found():
    with pytest.raises(KeyError, match="not found"):
        get_reward_function("nonexistent_reward_function_name")


def test_standard_rewards_auto_registered():
    import slm_post_train.rewards
    registered = list_registered_rewards()
    assert "xml_format" in registered
    assert "exact_match" in registered
    assert "code_execution" in registered


def test_xml_format_reward():
    prompts = ["Calculate 1+1"] * 2
    completions = [
        "<think>1+1=2</think><answer>2</answer>",
        "The answer is 2 without tags",
    ]
    scores = xml_format_reward(prompts=prompts, completions=completions)
    assert scores[0] == 1.0
    assert scores[1] == 0.0


def test_exact_match_reward():
    prompts = ["What is 2+2?"] * 3
    completions = [
        "<think>2+2</think><answer>4</answer>",
        "<answer> 4 </answer>",
        "<answer>5</answer>",
    ]
    scores = exact_match_reward(prompts=prompts, completions=completions, answer=["4", "4", "4"])
    assert scores[0] == 1.0
    assert scores[1] == 1.0
    assert scores[2] == 0.0


def test_exact_match_reward_missing_or_empty_answers():
    prompts = ["What is 2+2?"] * 2
    completions = [
        "<answer>4</answer>",
        "<answer>4</answer>",
    ]
    scores_none = exact_match_reward(prompts=prompts, completions=completions, answer=None)
    assert scores_none == [0.0, 0.0]

    scores_short = exact_match_reward(prompts=prompts, completions=completions, answer=["4"])
    assert scores_short == [1.0, 0.0]


def test_code_execution_reward():
    prompts = ["Write a function"] * 2
    completions = [
        "```python\ndef solve():\n    return 42\nresult = solve()\n```",
        "```python\nraise ValueError('error')\n```",
    ]
    scores = code_execution_reward(prompts=prompts, completions=completions)
    assert scores[0] == 1.0
    assert scores[1] == 0.0


def test_code_execution_reward_no_code_block():
    prompts = ["Write a function"]
    completions = ["Just some text without backticks"]
    scores = code_execution_reward(prompts=prompts, completions=completions)
    assert scores == [0.0]


def test_extract_text_helper():
    # String format
    assert _extract_text("hello world") == "hello world"

    # Single message dict
    assert _extract_text({"role": "assistant", "content": "42"}) == "42"
    assert _extract_text({"content": "content only"}) == "content only"
    assert _extract_text({"other_key": "val"}) == "{'other_key': 'val'}"

    # Conversation list format
    assert (
        _extract_text([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "bye"}])
        == "bye"
    )
    assert _extract_text([{"content": "first"}, {"content": "second"}]) == "second"
    assert _extract_text(["line 1", "line 2"]) == "line 1line 2"


def test_rewards_with_conversational_completions():
    # xml_format with conversation list
    conv_completions = [
        [{"role": "assistant", "content": "<think>reasoning</think><answer>42</answer>"}],
        {"role": "assistant", "content": "plain text"},
    ]
    xml_scores = xml_format_reward(prompts=["p1", "p2"], completions=conv_completions)
    assert xml_scores == [1.0, 0.0]

    # exact_match with message dict
    em_completions = [
        {"content": "<answer>Paris</answer>"},
        {"content": "<answer>London</answer>"},
    ]
    em_scores = exact_match_reward(
        prompts=["Capital of France?"] * 2,
        completions=em_completions,
        answer=["paris", "paris"],
    )
    assert em_scores == [1.0, 0.0]

    # code_execution with message list
    code_completions = [
        [{"role": "assistant", "content": "```python\nx = 10\ny = 20\n```"}],
        [{"role": "assistant", "content": "```python\n1 / 0\n```"}],
    ]
    ce_scores = code_execution_reward(prompts=["p1", "p2"], completions=code_completions)
    assert ce_scores == [1.0, 0.0]


def test_code_execution_reward_timeout():
    prompts = ["Write an infinite loop"]
    completions = ["```python\nwhile True:\n    pass\n```"]
    start = time.time()
    scores = code_execution_reward(prompts=prompts, completions=completions, timeout=1.0)
    elapsed = time.time() - start
    assert scores == [0.0]
    assert elapsed < 3.0

