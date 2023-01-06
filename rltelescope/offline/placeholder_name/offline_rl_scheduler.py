"""
Author(s) - Maggie Voetberg (maggiev@fnal.gov) 2023

Use the data generated offlline to produce weights that can be used in an online scheduling sitution
Data generation is an adaption of Astrotact, courtesy of Eric Nielson
Uses a torch framework to train both a standard fcnet and a lstm network.

Reward - Defined as "T_eff", with is a combination of sky brightness and airmass
Measures only each site, does not take the entire schedule into account.

Invalid actions are actions that result in an observation that is not visible or under the
quality standards set by the observatory used to generate the training data.

Actions are selected by allowing the scheduler to pick any RA/Declination within the object range
And "snapping" it to the closest site with an observation recorded for that point in time.
    Closeness is measured by <TBD>
"""

import argparse
import os

import pandas as pd
import numpy as np
import gym
from functools import cached_property

import ray.rllib.algorithms.es as es
import ray
import tqdm

import warnings

warnings.filterwarnings("ignore")


class RLEnv(gym.Env):
    def __init__(self, training_directory):
        self.training_directory = ""

    def load_stored_data(self):
        pass

    def reward(self):
        pass

    @cached_property
    def action_space(self):
        pass

    @cached_property
    def observation_space(self):
        pass

    def step(self):
        # Take the next action
        # Find the closest step to it in the time set
        # Verify it's valid, return the reward
        pass

    def reset(self):
        # I don't think there's really any reset I have to do here....
        pass


class RLTrainer:
    def __init__(self, agent_type, rl_env_config, out_path, iterations):
        self.agent = self.make_agent(agent_type, rl_env_config)
        self.out_path = out_path
        self.iterations = iterations

    def make_agent(self, agent_type, env_config):

        agent_config = es.DEFAULT_CONFIG.copy()
        agent_config["env_config"] = env_config
        agent_config["num_workers"] = 50

        if agent_type == "lstm":
            agent_config["model_config"]["use_lstm"] = True

        agent_config["framework"] = "torch"
        agent = es.ES(config=agent_config, env=RLEnv)
        return agent

    def __call__(self):

        checkpoints_outpath = f"{self.out_path}/checkpoints/"
        if not os.path.exists(checkpoints_outpath):
            os.makedirs(checkpoints_outpath)

        ray.init()

        history = {}
        print(f"Beginning training for {self.iterations} iterations")
        for i in tqdm.trange(self.iterations):
            # Training loop

            step_history = self.agent.train()
            self.agent.save(checkpoints_outpath)

            history[i] = step_history
            pd.DataFrame(history).T.to_csv(f"{self.out_path}/history.csv")

        ray.shutdown()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-i", "--iterations", type=int, default=80)
    args.add_argument("-o", "--out_path", type=str, default="../../../results/test_exp")
    args.add_argument("--lstm", action="raise_true")
    a = args.parse_args()

    agent_type = "lstm" if a.lstm else None
    trainer = RLTrainer(
        agent_type, rl_env_config={}, out_path=a.out_path, iterations=a.iterations
    )
    trainer()
