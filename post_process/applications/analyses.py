import os
import numpy as np
from tensorboardX import SummaryWriter

import post_process.calculations.calculation_utils as cutils
import post_process.visualization.visualization_utils as vutils
from post_process.calculations.mission_calcs import MissionCalcs
from post_process.calculations.overall_stat import OverallStat
import post_process.applications.applications as apps
import pandas as pd


def main():
    plots_output_path = os.path.join("..", "..", "scripts", "out", "plots")
    # rodo
    base_path = os.path.join("D:/Rodolfo/Data/Behavior/simulations")
    folder_path = "trainned_neutral_200s"
    add_to_tensorboard_folders = []
    # add_to_tensorboard_folders.append("1100s/train")
    # modes = ['plt_folder_stats_episode','plt_folder_stats' , 'plt_folder_stats_train']
    modes = ['plt_folder_stats']
    # modes.append("add_to_tensorboard")

    #######################################################################
    '''
    This goes over all folders in path and adds mission related calculations to TensorBoard
    '''
    if "add_to_tensorboard" in modes:
        for folder in add_to_tensorboard_folders:
            add_to_tensorboard_path = os.path.join(base_path, folder)
            apps.add_mission_calc_to_tensorboard(add_to_tensorboard_path)

    #######################################################################
    '''
       These are folder summary plots
       '''
    if "plt_folder_stats" in modes:
        # simulation_path_base = os.path.join(base_path, folder_path, "train")
        # pltfolder(simulation_path_base, plots_output_path_base=plots_output_path, plt_name=folder_path + "_train",
        #           n=3000)
        # try:
        simulation_path_base = os.path.join(base_path, folder_path, "test")
        pltfolder(simulation_path_base, plots_output_path_base=plots_output_path, plt_name=folder_path + "_test",
                  n=900)
        # except:
        #     print("**********************No test***************************")
    if "plt_folder_stats_episode" in modes:
        simulation_path_base = os.path.join(base_path, folder_path, "test")
        episodes = [2970, 2995]
        plt_name = folder_path + "_test"
        # plt100_5(simulation_path_base, plots_output_path_base=plots_output_path, plt_name=plt_name, episodes = episodes)
        # plt100_6(simulation_path_base, plots_output_path_base=plots_output_path, plt_name=plt_name, episode = episodes[0])
        plt_train(simulation_path_base, plots_output_path_base=plots_output_path, plt_name=plt_name, span=200)

    if "plt_folder_stats_train" in modes:
        simulation_path_base = os.path.join(base_path, folder_path, "train")
        plt_name = folder_path + "_train"
        pltavg_every_n_episodes(simulation_path_base, plots_output_path_base=plots_output_path, plt_name=plt_name, n=20,
                                span=100)
        plt_train(simulation_path_base, plots_output_path_base=plots_output_path, plt_name=plt_name, span=1000)


def pltfolder(simulation_path_base, plots_output_path_base=None, plt_name=None, n=900, absolute=True):
    overall_stats = apps.get_overall_stats(simulation_path_base, n)

    plots_output_path = os.path.join(plots_output_path_base, plt_name)
    try:
        if not os.path.exists(plots_output_path_base):
            os.mkdir(plots_output_path_base)
        if not os.path.exists(plots_output_path):
            os.mkdir(plots_output_path)
    except OSError:
        pass

    apps.compare_folder(overall_stats, metrics=["total_not_mission", "total_crash", "average_distance_travelled"],
                        plots_output_path=plots_output_path,n=n, absolute=absolute)


def plt_train(simulation_path_base, plots_output_path_base=None, plt_name=None, span=1000, show=False):
    plots_output_path = os.path.join(plots_output_path_base, plt_name)
    subfolders = cutils.get_subfolders(simulation_path_base)

    try:
        if not os.path.exists(plots_output_path_base):
            os.mkdir(plots_output_path_base)
        if not os.path.exists(plots_output_path):
            os.mkdir(plots_output_path)
    except OSError:
        pass

    colors = [np.random.rand(3, ) for p in range(0, len(subfolders))]

    espisodes_data = cutils.load_espisodes_data(simulation_path_base)
    rewards_data_array, rewards_data_array_std = cutils.espisodes_data_array_from_dict(espisodes_data,
                                                                                       stat="episode_reward", span=span,
                                                                                       std=True)
    duration_data_array_ema, duration_data_array_ema_std = cutils.espisodes_data_array_from_dict(espisodes_data,
                                                                                                 stat="episode_length",
                                                                                                 span=span, std=True)

    espisodes_distance_traveled_all, espisodes_distance_traveled_all_std = cutils.espisodes_distance_traveled(
        espisodes_data, stat="episode_average_speed_all", span=span, std=True)
    espisodes_distance_traveled_controlled, espisodes_distance_traveled_controlled_std = cutils.espisodes_distance_traveled(
        espisodes_data, stat="episode_average_speed_controlled", span=span, std=True)
    espisodes_distance_traveled_human, espisodes_distance_traveled_human_std = cutils.espisodes_distance_traveled(
        espisodes_data, stat="episode_average_speed_human", span=span, std=True)

    # rewards_data_array_ma =  pd.DataFrame.rolling_mean(rewards_data_array, period)
    # rewards_data_array_ema = pd.ewma(rewards_data_array, span=period , axis =0)
    # rewards_data_array_ema = pd.ewma(rewards_data_array, alpha=0.9)[-1]
    legends = []
    for subfolder in subfolders:
        # distance_v1v2 = np.sqrt(np.square((time_step_logs_array[0][0,:] - time_step_logs_array[0][1, :])) + np.square((time_step_logs_array[1][0, :] - time_step_logs_array[1][1, :])))
        # distances.append(distance_v1v2)
        legends.append(subfolder.split("_")[-1])

    colors = [np.random.rand(3, ) for p in range(0, len(subfolders))]
    fig = vutils.plot_time_array(rewards_data_array, ini=1, end=None, x=None, y_label="reward", x_label="episodes",
                                 title=" reward vs episodes ",
                                 y_limit=None, x_limit=None, colors=colors, show=show, legend=legends,
                                 data_std=rewards_data_array_std)

    out_file = plt_name + "reward_vs_episodes"
    fig_out_path = os.path.join(plots_output_path, out_file)
    fig.savefig(fig_out_path + ".png", dpi=300)

    fig2 = vutils.plot_time_array(duration_data_array_ema, ini=1, end=None, x=None, y_label="duration",
                                  x_label="episodes",
                                  title=" duration vs episodes ",
                                  y_limit=None, x_limit=None, colors=colors, show=show, legend=legends,
                                  data_std=duration_data_array_ema_std)

    out_file = plt_name + "duration_vs_episodes"
    fig_out_path = os.path.join(plots_output_path, out_file)
    fig2.savefig(fig_out_path + ".png", dpi=300)

    fig3 = vutils.plot_time_array(espisodes_distance_traveled_all, ini=1, end=None, x=None, y_label="distance",
                                  x_label="episodes",
                                  title=" distance_all vs episodes ",
                                  y_limit=None, x_limit=None, colors=colors, show=show, legend=legends,
                                  data_std=espisodes_distance_traveled_all_std)

    out_file = plt_name + "distancea_all_vs_episodes"
    fig_out_path = os.path.join(plots_output_path, out_file)
    fig3.savefig(fig_out_path + ".png", dpi=300)

    fig4 = vutils.plot_time_array(espisodes_distance_traveled_controlled, ini=1, end=None, x=None, y_label="distance",
                                  x_label="episodes",
                                  title=" distance_controlled vs episodes ",
                                  y_limit=None, x_limit=None, colors=colors, show=show, legend=legends,
                                  data_std=espisodes_distance_traveled_controlled_std)

    out_file = plt_name + "distancea_controlled_vs_episodes"
    fig_out_path = os.path.join(plots_output_path, out_file)
    fig4.savefig(fig_out_path + ".png", dpi=300)

    fig5 = vutils.plot_time_array(espisodes_distance_traveled_human, ini=1, end=None, x=None, y_label="distance",
                                  x_label="episodes",
                                  title=" distance_human vs episodes ",
                                  y_limit=None, x_limit=None, colors=colors, show=show, legend=legends,
                                  data_std=espisodes_distance_traveled_human_std)

    out_file = plt_name + "distancea_human_vs_episodes"
    fig_out_path = os.path.join(plots_output_path, out_file)
    fig5.savefig(fig_out_path + ".png", dpi=300)


def pltavg_every_n_episodes(simulation_path_base, plots_output_path_base=None, plt_name=None, n=20, span=100, std=True,
                            span_std=20):
    plots_output_path = os.path.join(plots_output_path_base, plt_name)
    try:
        if not os.path.exists(plots_output_path_base):
            os.mkdir(plots_output_path_base)
        if not os.path.exists(plots_output_path):
            os.mkdir(plots_output_path)
    except OSError:
        pass

    overall_stats = apps.get_overall_stats(simulation_path_base)
    avg_crashes = []
    legends = []
    lens = []
    for idx, crashes in enumerate(overall_stats["crash_episodes_flag_all"]):
        avg_crash = cutils.average_binary_array(crashes, n)
        avg_crashes.append(avg_crash)
        lens.append(len(avg_crash))
        pathx = overall_stats["paths"][idx]
        legends.append(pathx.split("_")[-1])
    l = min(lens)

    avg_crashes_filter = [avg_crash_filter[0:l - 1] for avg_crash_filter in avg_crashes]
    alpha = 2 / (span + 1)
    avg_crashes_filter_std = [np.std(cutils.rolling_window(np.array(data), span_std), 1) for data in avg_crashes_filter]
    avg_crashes_filter_ema = [cutils.ewma_vectorized_safe(data, alpha) for data in avg_crashes_filter]
    avg_crashes_np = np.array(avg_crashes_filter_ema)
    if std:
        avg_crashes_filter_std_zeros = np.zeros((np.shape(avg_crashes_filter_ema)))
        avg_crashes_filter_std_zeros[:, span_std - 1:] = avg_crashes_filter_std

    avg_crashes_np_std = np.array(avg_crashes_filter_std_zeros)
    colors = [np.random.rand(3, ) for p in range(0, len(overall_stats["paths"]))]
    fig = vutils.plot_time_array(avg_crashes_np, ini=1, end=None, x=None, y_label="num_of_crashes",
                                 x_label="episodes*" + str(n),
                                 title="num_of_crashes every" + str(n) + "episodes",
                                 y_limit=None, x_limit=None, colors=colors, show=False, legend=legends,
                                 data_std=avg_crashes_np_std)

    out_file = plt_name + "num_of_crashes_every_" + str(n) + "_episodes"
    fig_out_path = os.path.join(plots_output_path, out_file)
    fig.savefig(fig_out_path + ".png", dpi=300)


def plt100_5(simulation_path_base, plots_output_path_base=None, plt_name=None, episodes=[900, 902]):
    """"
    input : simulation path and outpath
    output: plot for each exepriment: x-distance of 12 and -1 vs. timestep from 1 test episode
    """
    subfolders = cutils.get_subfolders(simulation_path_base)
    vehicles = [-1, 12]
    colors = [np.random.rand(3, ) for p in range(0, len(vehicles))]

    plots_output_path = os.path.join(plots_output_path_base, plt_name)
    try:
        if not os.path.exists(plots_output_path_base):
            os.mkdir(plots_output_path_base)
        if not os.path.exists(plots_output_path):
            os.mkdir(plots_output_path)
    except OSError:
        pass

    colors = [np.random.rand(3, ) for p in range(0, len(subfolders))]
    # episode = 1
    episode_lenth = 16
    distances_array_cumulative = np.zeros((len(subfolders), episode_lenth))

    for episode in range(episodes[0], episodes[1]):
        distances = []
        legends = []

        for subfolder in subfolders:
            if subfolder == "plot_images":
                continue
            path = os.path.join(simulation_path_base, subfolder)
            time_step_path = os.path.join(path, "raw_logfiles", "timestep_logs")
            stats = ["positionx", "positiony"]

            time_step_logs = cutils.load_time_step_data(time_step_path, vehicles=vehicles, episode=episode + 1)
            time_step_logs_array = cutils.time_step_array_from_dict(time_step_logs, stats=stats)
            distance_v1v2 = np.sqrt(
                np.square((time_step_logs_array[0][0, :] - time_step_logs_array[0][1, :])) + np.square(
                    (time_step_logs_array[1][0, :] - time_step_logs_array[1][1, :])))
            distances.append(distance_v1v2)
            legends.append(subfolder.split("_")[-1:])

        distancesfix = []
        for dist in distances:
            while len(dist) < episode_lenth:
                dist = np.append(dist, 0)
            distancesfix.append(dist)

        distances_array = np.array(distancesfix)
        distances_array_cumulative += distances_array
        # fig = vutils.plot_time_array(distances_array, ini=1, end=None, x=None, y_label="distance", x_label="time_step", title='x-distance of 12 and -1 vs. timestep',
        #                        y_limit=None, x_limit=None, colors=colors, show=False , legend = legends)
        #
        # out_file = plt_name + "_episode_" + str(episode)
        # fig_out_path = os.path.join(plots_output_path, out_file)
        # fig.savefig(fig_out_path + ".png", dpi=300)

    distances_array_cumulative_avg = distances_array_cumulative / (episodes[1] - episodes[0])
    fig = vutils.plot_time_array(distances_array_cumulative_avg, ini=1, end=None, x=None, y_label="distance",
                                 x_label="time_step",
                                 title='x-distance of 12 and -1 vs. timestep avg',
                                 y_limit=None, x_limit=None, colors=colors, show=False, legend=legends)

    out_file = plt_name + "_episode_avg" + str(episodes[0]) + "_to_" + str(episodes[1])
    fig_out_path = os.path.join(plots_output_path, out_file)
    fig.savefig(fig_out_path + ".png", dpi=300)


def plt100_6(simulation_path_base, plots_output_path_base=None, plt_name=None, episode=1):
    """"
    input : simulation path and outpath
    output: plot for 1 test episode: 4 subplots, (a) speed of 4 agents + mission (b) lane number of mission (c) has_merged (d) rewards of 4 agents (e)   vs. timestep
    """
    subfolders = cutils.get_subfolders(simulation_path_base)
    vehicles = [-1, 12, 13, 14, 15]
    colors = [np.random.rand(3, ) for p in range(0, len(vehicles))]
    plots_output_path = os.path.join(plots_output_path_base, plt_name)
    try:
        if not os.path.exists(plots_output_path_base):
            os.mkdir(plots_output_path_base)
        if not os.path.exists(plots_output_path):
            os.mkdir(plots_output_path)
    except OSError:
        pass

    for subfolder in subfolders:
        if subfolder == "plot_images":
            continue
        path = os.path.join(simulation_path_base, subfolder)
        time_step_path = os.path.join(path, "raw_logfiles", "timestep_logs")

        # episode = 1
        stats = ["timestep_reward", "vehicle_speed", "positiony", "mission_accomplished"]
        legends = ["vehicles" + str(c) for c in vehicles]
        time_step_logs = cutils.load_time_step_data(time_step_path, vehicles=vehicles, episode=episode + 1)
        time_step_logs_array = cutils.time_step_array_from_dict(time_step_logs, stats=stats)

        fig = vutils.time_step_plot_image(path, time_step_logs_array, episode=episode, ini=1, end=None, x=None,
                                          y_label=stats, x_label="time_step",
                                          title=None,
                                          y_limit="Auto", x_limit=None, colors=colors, legend=legends, show=False)

        out_file = plt_name + "_episode_" + str(episode) + "_" + subfolder
        fig_out_path = os.path.join(plots_output_path, out_file)
        fig.savefig(fig_out_path + ".png", dpi=300)


if __name__ == '__main__':
    main()
