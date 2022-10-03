'''
Uses the weights as generated by the rl agent as coefficients to produce an observation for a ground telescope.
Configures each trained model and outputs both the weight producing model and the
'''

import pandas as pd
import numpy as np
import configparser
import os

class Scheduler:
    def __init__(self, config):
        assert os.path.exists(config)
        self.config = configparser.ConfigParser()
        self.config.read(config)

        self.actions = self.generate_action_table()
        self.invalid_reward = self.config.getfloat("reward", "invalid_reward")

        schedule_cols = ["mjd", "ra", "decl", "band", "exposure_time", "reward"]
        self.schedule = pd.DataFrame(columns=schedule_cols)

    def generate_action_table(self):
        # Based on the config params
        actions = pd.DataFrame(columns=["ra", 'decl', 'band'])

        min_ra = self.config.getfloat("actions", "min ra")
        max_ra = self.config.getfloat("actions", "max ra")
        n_ra = self.config.getint("actions", "num ra steps")
        ra_range = np.linspace(min_ra, max_ra, num=n_ra)

        min_decl = self.config.getfloat("actions", "min decl")
        max_decl = self.config.getfloat("actions", "max decl")
        n_decl = self.config.getint("actions", "num decl steps")
        decl_range = np.linspace(min_decl, max_decl, num=n_decl)

        bands = ""
        for ra in ra_range:
            for decl in decl_range:
                for band in bands:
                    new_action = {"ra": ra, "decl": decl, "band": band}
                    new_action = pd.DataFrame(new_action)

                    actions = pd.concat([actions, new_action])

        actions["exposure_time"] = self.config.getfloat("actions",
                                                        "exposure_time")
        return actions

    def update(self, obsprog):
        raise NotImplementedError
        # obsprog.reset()
        # done = False
        # while not done:
        #     observation = obsprog.observation()
        #     eq_params = self.rl_agent(observation)
        #
        #     action = self.calculate_action(eq_params)
        #     new_observation = self.feed_action(obsprog, action)
        #     reward = self.reward(new_observation)
        #     done = self.check_endtime(obsprog, action)
        #
        #     self.rl_agent.update_reward(reward)
        #     self.update_schedule(action, reward)

    def feed_action(self, obsprog, action):
        obsprog.update_observation(**action)
        new_observation = obsprog.state
        return new_observation

    def save(self, outpath):
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        schedule_name = f"{outpath.rstrip('/')}/schedule.csv"
        self.schedule.to_csv(schedule_name)

    def reward(self, observation):
        reward_type = self.config.get("reward", "reward_type")
        if self.invalid_action(observation):
            reward = self.invalid_reward
        else:
            reward = self.teff_reward(observation) if reward_type=="teff" else \
                self.full_exposure_reward(observation)

        return reward

    def teff_reward(self, observation):
        return observation['teff'] if observation['teff'] is not None else \
            self.invalid_reward

    def full_exposure_reward(self, observation):

        pass

    def invalid_action(self, observation):
        # TODO
        invalid = False
        return invalid

    def calculate_action(self, **action_params):
        raise NotImplementedError
        # # TODO
        # action_coeff = eq_params.coeff
        # action_powers = eq_params.powers
        #
        # actions_weights = ""
        # action = {}
        #
        # return action

    def update_schedule(self, action, reward):
        action["reward"] = reward
        new_action = pd.DataFrame(action)
        self.schedule = pd.concat([self.schedule, new_action])

    def check_endtime(self, obsprog, action):
        done = False
        length = self.config.getfloat("schedule", "length")
        # TODO Units
        end_time = obsprog.start_time + length
        if action["mjd"]>=end_time:
            done = True
        return done