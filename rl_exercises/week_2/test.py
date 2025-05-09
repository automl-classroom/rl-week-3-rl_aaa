import numpy as np
from my_env import MyEnv, PartialObsWrapper

# if __name__ == "__main__":
#     env = MyEnv()
#     obs, info = env.reset()
#     print(f"Initial observation: {obs}")

#     for i in range(5):
#         action = i % 2  # alternate 0, 1
#         obs, reward, terminated, truncated, info = env.step(action)
#         print(f"Step {i+1}: action={action}, obs={obs}, reward={reward}, truncated={truncated}")
#         env.render()
#         if terminated or truncated:
#             break

#     print("Reward matrix:\n", env.get_reward_per_action())
#     print("Transition matrix:\n", env.get_transition_matrix())


def test_partial_obs_wrapper(env_cls, noise=0.5, seed=42, steps=10):
    """
    Tests the PartialObsWrapper by running a short episode and logging both
    true states and noisy observations.

    Parameters:
    - env_cls: A callable that returns a new unwrapped env instance.
    - noise: Probability of observation noise.
    - seed: RNG seed for reproducibility.
    - steps: Number of steps to simulate.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)

    base_env = env_cls()
    wrapped_env = PartialObsWrapper(base_env, noise=noise, seed=seed)
    obs, _ = wrapped_env.reset(seed=seed)

    print(f"{'Step':>4} | {'Action':>6} | {'True Pos':>9} | {'Obs':>5} | {'Noisy?':>7}")
    print("-" * 42)

    for step in range(1, steps + 1):
        action = random.choice([0, 1])
        # Save current true state before step
        # true_pos = wrapped_env.env.position
        # print(true_pos)

        obs, reward, terminated, truncated, _ = wrapped_env.step(action)
        new_pos = wrapped_env.env.position

        is_noisy = obs != new_pos
        print(f"{step:>4} | {action:>6} | {new_pos:>9} | {obs:>5} | {str(is_noisy):>7}")

        if terminated or truncated:
            break


test_partial_obs_wrapper(MyEnv, noise=0.5, seed=42, steps=10)
