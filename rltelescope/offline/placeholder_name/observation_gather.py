"""

"""
import os.path

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.time import Time

import sys

sys.path.append("../..")

from observation_program import ObservationProgram


class ObservationGather(ObservationProgram):
    def __init__(
        self, obsprog_config, n_observation_sites, n_schedules, observation_length
    ):
        duration = 1
        self.observation_length = observation_length
        self.n_sites = n_observation_sites
        self.n_schedules = n_schedules
        super().__init__(obsprog_config, duration)
        self.schedules = {"mjd": [], "decl": [], "ra": []}
        self.actions = {}
        self.dates = []

        self.get_actions()
        self.get_dates()

    def reset_schedule(self):
        self.schedules = {"mjd": [], "decl": [], "ra": []}

    def get_dates(self):
        date_range = range(55197, 58849)
        self.dates = [
            date_range[index]
            for index in np.random.randint(0, len(date_range), self.n_schedules)
        ]

    def get_actions(self):
        possible_bands = ["g"]  # ["u", "g", "r", "i", "z", "Y"]
        possible_ras_degrees = np.linspace(0, 360, 72)
        possible_delcs_degrees = np.linspace(-90, 90, 36)

        band_selections = np.random.randint(0, len(possible_bands), self.n_sites)
        ra_selections = np.random.randint(0, len(possible_ras_degrees), self.n_sites)
        delc_selections = np.random.randint(
            0, len(possible_delcs_degrees), self.n_sites
        )

        self.actions = {
            "band": [possible_bands[index] for index in band_selections],
            "ra": [int(possible_ras_degrees[index]) for index in ra_selections],
            "decl": [int(possible_delcs_degrees[index]) for index in delc_selections],
        }

    def make_schedules(self):
        self.get_dates()
        self.get_actions()

        self.reset_schedule()

        time = Time(self.dates * u.day, format="mjd")
        start_mjd = self.observatory.sun_set_time(time, which="next").mjd
        end_mjd = self.observatory.sun_rise_time(time, which="next").mjd
        time_steps = [
            abs(int((end - start) / self.observation_length))
            for start, end in zip(start_mjd, end_mjd)
        ]
        for time_step, start_time in zip(time_steps, start_mjd):
            times = [
                start_time + (self.observation_length * i) for i in range(time_step)
            ]
            for time in times:
                self.schedules["mjd"] += [time for _ in range(self.n_sites)]
                self.schedules["decl"] += self.actions["decl"]
                self.schedules["ra"] += self.actions["ra"]

        self.schedules["exposure_time_days"] = [
            self.observation_length for _ in range(len(self.schedules["mjd"]))
        ]
        self.schedules = pd.DataFrame(self.schedules)

    def single_schedule_results(self, schedule):
        schedule = schedule.to_dict(orient="list")
        schedule = {key: np.asarray(values) for key, values in schedule.items()}
        return pd.DataFrame(self.calculate_obversation(schedule))

    def __call__(self, batches, save_path="offline_observations.csv"):
        self.make_schedules()

        batch_size = int(len(self.schedules) / batches)

        base_dir = os.path.dirname(save_path)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        pd.DataFrame().to_csv(save_path)

        for item in range(batches):
            batch_lower_index = item * batch_size
            batch_higher_index = (item + 1) * batch_size
            batch_schedule = self.schedules.iloc[batch_lower_index:batch_higher_index]

            batch_observations = self.single_schedule_results(batch_schedule)
            observations = pd.concat([pd.read_csv(save_path), batch_observations])
            observations.to_csv(save_path)


if __name__ == "__main__":
    obsprog_config = os.path.abspath("../../train_configs/default_obsprog.conf")
    observation_length_days = 0.003472222

    obsgather = ObservationGather(
        obsprog_config,
        n_observation_sites=50,
        n_schedules=50,
        observation_length=observation_length_days,
    )

    train_names = [f"offline_data/train_observations/train_{n}.csv" for n in range(2)]
    test_names = [f"offline_data/test_observations/test_{n}.csv" for n in range(1)]
    val_names = [f"offline_data/val_observations/val_{n}.csv" for n in range(1)]
    file_names = train_names + test_names + val_names

    for file_name in file_names:
        obsgather(1, save_path=file_name)
