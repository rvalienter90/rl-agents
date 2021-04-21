# calculations about the mission time of an episode
import os
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter

import post_process.calculations.calculation_utils as cutils
import post_process.visualization.visualization_utils as vutils

class MissionCalcs():


    def __init__(self, path, folder_name):
        self.path = path
        self.folder_name = folder_name

        self.episode_average_log = cutils.load_episode_average_log(self.path,
                                                                   self.folder_name)
        self.mission_times = self.episode_average_log.mission_time
        self.simulation_length = self.mission_times.size
        self.total_not_mission = None
        self.not_mission_episodes = None
        self.successive_successful_missions = None
        self.not_mission_freq = None


    def mission_not_accomplished(self):
        # filtering - 1 s from episode logs and analyzing them

        mission_times = self.mission_times.values
        # indexes of episodes that mission was not accomplished
        self.not_mission_episodes = np.where(mission_times==-1)[0] + 1

        # total number of episodes that mission was not accomplished
        self.total_not_mission = len(self.not_mission_episodes)
        self.successive_successful_missions =  np.diff(self.not_mission_episodes)
        self.not_mission_freq = 1 / self.successive_successful_missions
        # print(">>>>>>>> DEBUG: total_not_mission, not_mission_episodes",
        #       self.total_not_mission, len(self.not_mission_episodes))

    def add_to_tensorboard(self):
        writer = SummaryWriter(os.path.join(self.path, self.folder_name))
        for i, value in enumerate(self.successive_successful_missions):
            episode = self.not_mission_episodes[i]
            writer.add_scalar('mission_calcs/successive_successful_missions',
                              value, episode)

        for i, value in enumerate(self.not_mission_freq):
            episode = self.not_mission_episodes[i]
            writer.add_scalar('mission_calcs/not_mission_freq',
                              value, episode)

    def debug_plot_results(self):
        vutils.simple_2d_plot(self.successive_successful_missions, self.not_mission_episodes[:-1],
                              title='successive_successful_missions ' + self.folder_name)

        vutils.simple_2d_plot(self.not_mission_freq, self.not_mission_episodes[:-1],
                              title='not_mission_freq ' + self.folder_name)


