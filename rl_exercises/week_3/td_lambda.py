# Refrence:
# https://medium.com/data-science/reinforcement-learning-td-%CE%BB-introduction-686a5e4f4e60
# gridenv Q-learning code
# chatGPT
from collections import defaultdict

import numpy as np


def obs_to_key(obs):
    if isinstance(obs, tuple):
        obs = obs[0]
    return (tuple(obs["agent_pos"]), obs["direction"])


def td_lambda_learning(
    environment,
    num_episodes,
    discount_factor=0.99,
    alpha=0.5,
    epsilon=0.1,
    lambda_param=0.8,
    epsilon_decay="const",
    decay_starts=0,
    eval_every=100,
    render_eval=False,
):
    """
    Tabular TD(λ) (accumulating traces) with ε‑greedy behaviour policy.
    ...
    """
    # initialize Q and trace matrices
    Q = defaultdict(lambda: np.zeros(environment.action_space.n))
    E = defaultdict(lambda: np.zeros(environment.action_space.n))

    # schedule ε
    def get_decay_schedule(start, start_decay, total, mode):
        if mode == "const":
            return np.ones(total) * start
        elif mode == "linear":
            return np.hstack(
                [
                    np.ones(start_decay) * start,
                    np.linspace(start, 0, total - start_decay),
                ]
            )
        elif mode == "log":
            return np.hstack(
                [
                    np.ones(start_decay) * start,
                    np.logspace(np.log10(start), np.log10(1e-6), total - start_decay),
                ]
            )
        else:
            raise ValueError

    eps_schedule = get_decay_schedule(
        epsilon, decay_starts, num_episodes, epsilon_decay
    )

    # bookkeeping
    train_rewards, train_lengths = [], []
    test_rewards, test_lengths = [], []

    for ep in range(1, num_episodes + 1):
        # reset env, traces
        state = environment.reset()
        E.clear()
        eps = eps_schedule[min(ep - 1, len(eps_schedule) - 1)]

        # pick initial action with ε‑greedy
        def policy(obs):
            p = np.ones(environment.action_space.n) * eps / environment.action_space.n
            best = np.argmax(Q[obs_to_key(obs)])
            p[best] += 1 - eps
            return p

        action = np.random.choice(environment.action_space.n, p=policy(state))
        done = False
        t = 0
        total_r = 0

        while not done:
            next_state, reward, done, _, _ = environment.step(action)
            total_r += reward

            next_action = np.random.choice(
                environment.action_space.n, p=policy(next_state)
            )

            # compute TD error (SARSA form)
            # Inside the TD(lambda) loop:
            td_target = (
                reward + discount_factor * Q[obs_to_key(next_state)][next_action]
            )
            delta = td_target - Q[obs_to_key(state)][action]

            # Decay all eligibility traces first
            for s in list(E.keys()):
                for a in range(environment.action_space.n):
                    E[s][a] *= discount_factor * lambda_param

            # Increment current state-action's trace
            current_key = obs_to_key(state)
            E[current_key][action] += 1

            # Update Q-values using the current traces
            for s in list(E.keys()):
                for a in range(environment.action_space.n):
                    if E[s][a] != 0:
                        Q[s][a] += alpha * delta * E[s][a]

            state, action = next_state, next_action
            t += 1

        train_rewards.append(total_r)
        train_lengths.append(t)

        if ep % eval_every == 0:
            s = environment.reset()
            done_eval = False
            r_eval = 0
            steps = 0
            while not done_eval:
                a = np.argmax(Q[obs_to_key(s)])
                s, r, done_eval, _, _ = environment.step(a)
                if render_eval:
                    environment.render()
                r_eval += r
                steps += 1
            test_rewards.append(r_eval)
            test_lengths.append(steps)
            print(f"Episode {ep}/{num_episodes} — Eval reward: {r_eval}")

    return (train_rewards, train_lengths), (test_rewards, test_lengths)
