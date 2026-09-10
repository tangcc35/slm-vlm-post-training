from slm_post_train.rewards.registry import register_reward, get_reward_function, list_registered_rewards
import slm_post_train.rewards.standard  # auto-register standard rewards

__all__ = ["register_reward", "get_reward_function", "list_registered_rewards"]
