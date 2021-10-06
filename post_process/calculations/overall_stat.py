# overall comparison between runs
import os
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter

from post_process.calculations.mission_calcs import MissionCalcs
import post_process.calculations.calculation_utils as cutils
import post_process.visualization.visualization_utils as vutils


class OverallStat():

    def __init__(self, path, folder_name, n=None , episode_duration =16):
        self.path = path
        self.folder_name = folder_name
        self.n = n
        self.episode_average_log = cutils.load_episode_average_log(self.path,
                                                                   self.folder_name)

        if n is None:
            self.n = self.episode_average_log.shape[0] - 1

        self.average_episode_length = None
        self.average_episode_reward = None
        self.average_mission_time = None
        self.total_not_mission = None
        self.total_crash = None
        self.crash_episodes = None
        self.average_distance_travelled_stat = None
        self.average_episode_reward_stat = None
        self.episode_duration = episode_duration
        if self.episode_average_log.shape[0] <= self.n:
            raise ValueError("Provided 'n' value is larger than the total number of episodes")

    def get_overall_stat(self):
        self.average_episode_length = self. \
            get_run_average(self.episode_average_log.episode_length)

        self.average_episode_reward = self. \
            get_run_average(self.episode_average_log.episode_reward)

        self.average_mission_time = self. \
            get_run_average(self.episode_average_log.mission_time)

        self.average_speed_all = self. \
            get_run_average(self.episode_average_log.episode_average_speed_all)

        self.average_speed_controlled = self. \
            get_run_average(self.episode_average_log.episode_average_speed_controlled)

        self.average_speed_human = self. \
            get_run_average(self.episode_average_log.episode_average_speed_human)

        self.average_speed_controlled = self. \
            get_run_average(self.episode_average_log.episode_average_speed_controlled)

        self.average_speed_human = self. \
            get_run_average(self.episode_average_log.episode_average_speed_human)

        average_distance_travelled_series = \
            self.episode_average_log.episode_length.multiply(self.episode_average_log.episode_average_speed_all)

        average_distance_travelled_human_series = \
            self.episode_average_log.episode_length.multiply(self.episode_average_log.episode_average_speed_human)

        average_distance_travelled_controlled_series = \
            self.episode_average_log.episode_length.multiply(self.episode_average_log.episode_average_speed_controlled)

        self.average_distance_travelled_stat = {"mean": average_distance_travelled_series.mean(),
                                                "stddev": average_distance_travelled_series.std()}

        self.average_distance_travelled_human_stat = {"mean": average_distance_travelled_human_series.mean(),
                                                "stddev": average_distance_travelled_human_series.std()}

        self.average_distance_travelled_controlled_stat = {"mean": average_distance_travelled_controlled_series.mean(),
                                                "stddev": average_distance_travelled_controlled_series.std()}

        self.average_episode_reward_stat = {"mean": self.episode_average_log.episode_reward.mean(),
                                                "stddev": self.episode_average_log.episode_reward.std()}

        # Todo: move this to MissionCalcs
        # mission_calcs = MissionCalcs(self.path, self.folder_name)
        mission_times = self.episode_average_log.mission_time.values

        # TODO: move this somewhere else
        # TODO: episode_duration is not accurate here also check if duration is >1
        episode_lengths = self.episode_average_log.episode_length.values
        if self.episode_duration is None:
            episode_duration = max(episode_lengths)
        else:
            episode_duration = self.episode_duration

        # TODO: fix crashed mission
        crashed_mission = (episode_lengths - mission_times) < 3
        not_mission_flag_all = np.logical_or(crashed_mission, mission_times == -1)
        # not_mission_flag = np.logical_or(crashed_mission[-self.n:], mission_times[-self.n:] == -1)
        not_mission_episodes = np.where(not_mission_flag_all[-self.n:])[0] + 1
        self.total_not_mission = len(not_mission_episodes)

        # TODO: fix crashes
        # crash_episodes_flag_all = np.logical_or(episode_lengths < episode_duration , not_mission_flag_all)
        if 'crashed_av' in self.episode_average_log:
            crash_episodes_flag_all = self.episode_average_log.crashed_av == 1
        else:
            crash_episodes_flag_all = episode_lengths < episode_duration

        # crash_episodes_numbers_all = np.where(episode_lengths < episode_duration)[0] + 1
        crash_episodes_numbers_all = np.where(crash_episodes_flag_all)[0] + 1
        # crash_episodes = np.where(episode_lengths[-self.n:] < episode_duration)[0] + 1
        crash_episodes = np.where(crash_episodes_flag_all[-self.n:])[0] + 1


        self.crash_episodes_flag_all = crash_episodes_flag_all
        self.not_mission_episodes_flag_all = not_mission_flag_all

        self.crash_episodes_numbers_all = crash_episodes_numbers_all
        self.crash_episodes = crash_episodes
        self.total_crash = len(crash_episodes)

    def get_run_average(self, df):
        arr = df.values
        type = df.name

        # episodes with length=1
        episode_lengths = self.episode_average_log.episode_length.values
        mask = np.where(episode_lengths == 1)
        # remove episodes with length=1
        arr = np.delete(arr, mask)

        if (type == 'mission_time'):
            arr = np.delete(arr, np.where(arr == -1))

        avg = np.mean(arr[-self.n:])

        return avg
