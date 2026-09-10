from typing import Callable, Dict, List

_REWARD_REGISTRY: Dict[str, Callable] = {}


def register_reward(name: str):
    """Decorator to register a reward function."""
    def decorator(fn: Callable) -> Callable:
        _REWARD_REGISTRY[name] = fn
        return fn
    return decorator


def get_reward_function(name: str) -> Callable:
    """Retrieve a registered reward function by name."""
    if name not in _REWARD_REGISTRY:
        raise KeyError(
            f"Reward function '{name}' not found. Available rewards: {list(_REWARD_REGISTRY.keys())}"
        )
    return _REWARD_REGISTRY[name]


def list_registered_rewards() -> List[str]:
    """Return all registered reward function names."""
    return list(_REWARD_REGISTRY.keys())
