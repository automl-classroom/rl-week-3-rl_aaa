from __future__ import annotations

import gym
import numpy as np
from gym import spaces
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from minigrid.minigrid_env import MiniGridEnv


class PositionDirectionWrapper(gym.ObservationWrapper):  # chatGPT
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Dict(
            {
                "agent_pos": spaces.Box(
                    low=0,
                    high=max(self.env.width, self.env.height),
                    shape=(2,),
                    dtype=np.int32,
                ),
                "direction": spaces.Discrete(4),
            }
        )

    def observation(self, obs):
        return {
            "agent_pos": np.array(self.env.agent_pos, dtype=np.int32),
            "direction": self.env.agent_dir,
        }


class FourRoomsEnv(MiniGridEnv):
    """
    ## Description

    Classic four room reinforcement learning environment. The agent must
    navigate in a maze composed of four rooms interconnected by 4 gaps in the
    walls. To obtain a reward, the agent must reach the green goal square. Both
    the agent and the goal square are randomly placed in any of the four rooms.

    ## Mission Space

    "reach the goal"

    ## Action Space

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |
    | 3   | pickup       | Unused       |
    | 4   | drop         | Unused       |
    | 5   | toggle       | Unused       |
    | 6   | done         | Unused       |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `MiniGrid-FourRooms-v0`

    """

    def __init__(self, agent_pos=None, goal_pos=None, max_steps=100, **kwargs):
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos

        self.size = 19
        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            width=self.size,
            height=self.size,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "reach the goal"

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 2
        room_h = height // 2

        # For each row of rooms
        for j in range(0, 2):
            # For each column
            for i in range(0, 2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h

                # Bottom wall and door
                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, room_h)
                    pos = (xR, self._rand_int(yT + 1, yB))
                    self.grid.set(*pos, None)

                # Bottom wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, room_w)
                    pos = (self._rand_int(xL + 1, xR), yB)
                    self.grid.set(*pos, None)

        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            # assuming random start direction
            self.agent_dir = self._rand_int(0, 4)
        else:
            self.place_agent()

        if self._goal_default_pos is not None:
            goal = Goal()
            self.put_obj(goal, *self._goal_default_pos)
            goal.init_pos, goal.cur_pos = self._goal_default_pos
        else:
            self.place_obj(Goal())


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from td_lambda import td_lambda_learning

    # Initialize the environment
    env = FourRoomsEnv(render_mode="human", max_steps=200)
    env = PositionDirectionWrapper(env)

    # Run TD(λ)
    (train_rewards, train_lengths), (test_rewards, test_lengths) = td_lambda_learning(
        environment=env,
        num_episodes=10000,
        discount_factor=0.99,
        alpha=0.1,
        epsilon=0.3,
        lambda_param=0.9,
        epsilon_decay="linear",
        decay_starts=3000,
        eval_every=500,
        render_eval=True,
    )

    # Plotting training rewards
    plt.plot(train_rewards)
    plt.title("Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid()
    plt.show()

    # Plotting evaluation rewards
    plt.plot(range(100, 1001, 100), test_rewards, marker="o")
    plt.title("Evaluation Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Evaluation Reward")
    plt.grid()
    plt.show()
