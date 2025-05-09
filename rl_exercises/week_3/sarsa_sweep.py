"""Run multiple SARSA episodes using Hydra-configured components.
# uv run rl-week-3-rl_aaa/rl_exercises/week_3/sarsa_sweep.py -m
This script uses Hydra to instantiate the environment, policy, and SARSA agent from config files,
then runs multiple episodes and returns the average total reward.
"""
# import sys
import os
import matplotlib.pyplot as plt  # Add at the top

# # Add the root directory containing 'hypersweeper' to PYTHONPATH
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
# sys.path.insert(0, project_root)
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import csv

# generated with chatGPT
def log_trial_results(config, reward, file_path="sarsa_rs/sweep_results.csv"):
    """Log results to a single CSV file."""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Check if the file exists, if not, create it and write header
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            # Writing the header for the first time
            writer.writerow(["alpha", "gamma", "epsilon", "mean_reward"])
        writer.writerow([config.agent.alpha, config.agent.gamma, config.policy.epsilon, reward])

def run_episodes(agent, env, num_episodes=5):
    """Run multiple episodes using the SARSA algorithm.

    Each episode is executed with the agent's current policy. The agent updates its Q-values
    after every step using the SARSA update rule.

    Parameters
    ----------
    agent : object
        An agent implementing `predict_action` and `update_agent`.
    env : gym.Env
        The environment in which the agent interacts.
    num_episodes : int, optional
        Number of episodes to run, by default 5.

    Returns
    -------
    float
        Mean total reward across all episodes.
    list
        List of total rewards per episode.
    """

    episode_rewards = []
    for episode in range(num_episodes):
        # Reset the environment and agent for each episode
        state, _ = env.reset()
        done = False
        action = agent.predict_action(state)
        total = 0
        while not done:
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            next_action = agent.predict_action(next_state)
            agent.update_agent(state, action, reward, next_state, next_action, done)
            total += reward
            state, action = next_state, next_action
        episode_rewards.append(total)
    # Calculate the mean total reward across all episodes
    mean_total = sum(episode_rewards) / len(episode_rewards)
    # Return the mean total reward and all episode rewards
    return mean_total, episode_rewards


# Decorate the function with the path of the config file and the particular config to use
@hydra.main(
    config_path="../configs/agent/", config_name="sarsa_sweep", version_base="1.1"
)
def main(cfg: DictConfig) -> dict:
    print(OmegaConf.to_yaml(cfg))
    """Main function to run SARSA with Hydra-configured components.

    This function sets up the environment, policy, and agent using Hydra-based
    configuration, seeds them for reproducibility, and runs multiple episodes.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration containing `env`, `policy`, `agent`, and optionally `seed`.

    Returns
    -------
    float
        Mean total reward across the episodes.
    """

    # Hydra-instantiate the env
    env = instantiate(cfg.env)
    # instantiate the policy (passing in env!)
    policy = instantiate(cfg.policy, env=env)
    # 3) instantiate the agent (passing in env & policy)
    agent = instantiate(cfg.agent, env=env, policy=policy)

    # 4) (optional) reseed for reproducibility
    if cfg.seed is not None:
        env.reset(seed=cfg.seed)
        env.action_space.seed(cfg.seed)

    # 5) run & return reward chatgpt:
    mean_reward, all_rewards = run_episodes(agent, env, cfg.num_episodes)
    print(f"Mean Reward: {mean_reward}")
    # Plot the episode rewards
    # plt.plot(all_rewards)
    # plt.xlabel("Episode")
    # plt.ylabel("Total Reward")
    # plt.title("SARSA Reward per Episode")
    # plt.grid(True)
    # plt.savefig("reward_curve.png")
    # plt.show()
    log_trial_results(cfg, all_rewards, file_path="./sarsa_rs/sweep_results.csv")
    # Save the plot

    return mean_reward


if __name__ == "__main__":
    main()
