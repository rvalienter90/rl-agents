import os
import numpy as np
from tensorboardX import SummaryWriter
import copy
import post_process.calculations.calculation_utils as cutils
import post_process.visualization.visualization_utils as vutils
from post_process.calculations.mission_calcs import MissionCalcs
from post_process.calculations.overall_stat import OverallStat
from matplotlib import pyplot as plt
# importing library
import csv

BLACK = "#000000"

DARK_ORANGE = "#ff8c00"
VIVID_ORANGE = "#FF5E0E"
RED_ORANFE = "#FF4500"
LIGHT_ORANGE = "#ffa500"
VERY_LIGHT_ORANGE = "#ffddb3"

DARK_GRAY = "#595959"
MEDIUM_GRAY = "#999999"
LIGHT_GRAY = "#b9b9b9"
VERY_LIGHT_GRAY = "#cecece"
VERY_VERY_LIGHT_GRAY = "#eaeaea"

DARK_BLUE = "#42838f"
LIGHT_BLUE = "#aaccff"

DARK_NEON_BLUE = "#0066ff"
NEON_BLUE = "#00ccff"

DARK_GREEN = "#006400"
LIGHT_GREEN = "#90EE90"

LIGHT_GREEN_BLUE = "#87decd"

# DARK_GREEN_V2 = "#4b742f"
# DARK_GREEN_V2 = "#548235"
DARK_GREEN_V2 = "#658f49"
LIGHT_GREEN_V2 = "#a9c09a"

PURPLE = "#9900cc"

SLAC = "#947F73"
LILAC = "#7E7193"

def add_mission_calc_to_tensorboard(path):
    # this function adds not_mission info to Tensorboard for all simulations in path

    subfolders = cutils.get_subfolders(path)

    for subfolder in subfolders:
        print("running add_mission_calc_to_tensorboard for subfolder = ", subfolder)

        mission_calcs = MissionCalcs(path, subfolder)
        mission_calcs.mission_not_accomplished()

        mission_calcs.add_to_tensorboard()
        # mission_calcs.debug_plot_results()

def get_overall_stats(path, n=None, single=False , subfolders = None,episode_duration =16):
    # this function compares all the runs in path by averaging them over the last n episodes
    if subfolders is None:
        if single:
           subfolders = [os.path.split(path)[1]]
           path = os.path.split(path)[0]
        else:

           subfolders = cutils.get_subfolders(path)

    overall_stats = {"paths": [],
                     "average_episode_length": [],
                     "average_episode_reward": [],
                     "average_mission_time": [],
                     "average_speed_all": [],
                     "average_speed_controlled": [],
                     "average_speed_human": [],
                     "reward_params": [],
                     "total_not_mission": [],
                     "total_crash": [],
                     "crash_episodes": [],
                     "crash_episodes_flag_all": [],
                     "average_distance_travelled": [],
                     "average_distance_travelled_human": [],
                     "average_distance_travelled_controlled": [],
                     "average_distance_travelled_stat": [],
                     "average_distance_travelled_human_stat": [],
                     "average_distance_travelled_controlled_stat": [],
                     "average_episode_reward_stat": [],
                     "not_mission_episodes_flag_all": []
                     }

    for subfolder in subfolders:
        if subfolder == "test":
            continue
        print("running get_overall_stats for subfolder = ", subfolder)
        if single:
            full_path =os.path.join(path, subfolder)
        else:
            full_path = os.path.join(path, subfolder)

        overall_stat = OverallStat(path, subfolder, n,episode_duration=episode_duration)
        overall_stat.get_overall_stat()

        overall_stats["paths"].append(full_path)
        overall_stats["average_episode_length"]. \
            append(overall_stat.average_episode_length)
        overall_stats["average_episode_reward"]. \
            append(overall_stat.average_episode_reward)
        overall_stats["average_mission_time"]. \
            append(overall_stat.average_mission_time)
        overall_stats["average_speed_all"]. \
            append(overall_stat.average_speed_all)
        overall_stats["average_speed_controlled"]. \
            append(overall_stat.average_speed_controlled)
        overall_stats["average_speed_human"]. \
            append(overall_stat.average_speed_human)
        overall_stats["total_not_mission"]. \
            append(overall_stat.total_not_mission)
        overall_stats["total_crash"]. \
            append(overall_stat.total_crash)
        overall_stats["crash_episodes"]. \
            append(overall_stat.crash_episodes)
        overall_stats["crash_episodes_flag_all"]. \
            append(overall_stat.crash_episodes_flag_all)
        overall_stats["average_distance_travelled"]. \
            append(overall_stat.average_speed_all * overall_stat.average_episode_length)
        overall_stats["average_distance_travelled_human"]. \
            append(overall_stat.average_speed_human * overall_stat.average_episode_length)
        overall_stats["average_distance_travelled_controlled"]. \
            append(overall_stat.average_speed_controlled * overall_stat.average_episode_length)
        overall_stats["average_distance_travelled_stat"]. \
            append(overall_stat.average_distance_travelled_stat)
        overall_stats["average_distance_travelled_controlled_stat"]. \
            append(overall_stat.average_distance_travelled_controlled_stat)
        overall_stats["average_distance_travelled_human_stat"]. \
            append(overall_stat.average_distance_travelled_human_stat)
        overall_stats["average_episode_reward_stat"]. \
            append(overall_stat.average_episode_reward_stat)
        overall_stats["not_mission_episodes_flag_all"]. \
                append(overall_stat.not_mission_episodes_flag_all)

    return overall_stats

def compare_stats_action_hist(stats, metrics=None, x_name=None, location_x=None, y_name=None, location_y=None,
                              plots=None, location_plot=None, base_dict=None, fig_out_path=None):
    figsize = [20, 3]
    for z_name in metrics:
        x_vals = []
        y_vals = []
        plots_vals = []
        metric = []
        dict_map = {
            "binary": 0,
            "xy_discrete": 1,
            "discrete": 2
        }
        if base_dict:
            location_x = location_y = location_plot = base_dict

        for i, value in enumerate(stats[z_name]):
            # if stats[condition][condition_location] != condition_value:
            #     continue
            metric.append(value)
            val1 = stats[location_x][i][x_name]
            if isinstance(val1, list):
                val1 = val1[0]
            x_vals.append(val1)
            val2 = stats[location_y][i][y_name]
            if isinstance(val2, list):
                val2 = val2[0]
            val2 = dict_map[val2]
            y_vals.append(val2)
            plots_vals. \
                append(stats[location_plot][i][plots])

        # TODO: clean this up and put them in a single 3D array
        x_vals = np.array(x_vals)
        y_vals = np.array(y_vals)
        plots_vals = np.array(plots_vals)
        metric = np.array(metric)
        max_z = max(metric)
        # max_z = 100
        plot_vals_set = set(plots_vals)
        axs = []
        if len(plot_vals_set) == 2:
            fig = plt.figure(figsize=(figsize[0], figsize[1] * len(plot_vals_set)))
            ax1 = fig.add_subplot(121, projection='3d')
            axs.append(ax1)
            ax2 = fig.add_subplot(122, projection='3d')
            axs.append(ax2)
        elif len(plot_vals_set) == 3:
            fig = plt.figure(figsize=(figsize[0], figsize[1] * len(plot_vals_set)))
            axs[0] = fig.add_subplot(131, projection='3d')
            axs[1] = fig.add_subplot(132, projection='3d')
            axs[2] = fig.add_subplot(133, projection='3d')

        # fig, axs = plt.subplots(1,len(plot_vals_set), figsize=(figsize[0], figsize[1] * len(plot_vals_set)))
        for idx, val in enumerate(plot_vals_set):

            # filter based on cooperative_rewards
            x = x_vals[np.where(plots_vals == val)]
            y = y_vals[np.where(plots_vals == val)]
            z = metric[np.where(plots_vals == val)]

            plt_file_name = os.path.split(fig_out_path)[1]
            plot_name = plots + " = " + str(val) + " - " + plt_file_name

            tmp_info = []
            for idx2, path in enumerate(stats["paths"]):
                if plots_vals[idx2] == val:
                    scenario_name = path.split("_")[-1]
                    tmp_info.append(scenario_name)

            fig1 = vutils.simple_3d_bar_plot(x, y, z,
                                             x_name,
                                             y_name,
                                             z_name,
                                             plot_name, zlim3d=max_z, show=True, info=tmp_info)

            print(">>>>>>>>>>>>>>>>>>>>>>> INFO: ", plot_name, x_name, x, y_name, y, z_name, z)

            if len(plot_vals_set) > 1:
                fig2 = vutils.simple_3d_bar_plot(x, y, z,
                                                 x_name,
                                                 y_name,
                                                 z_name,
                                                 plot_name, zlim3d=max_z, ax=axs[idx])

            if fig_out_path:
                fig_out_path_final = fig_out_path + "_" + z_name + "_" + plots + "=" + str(val)
                fig1.savefig(fig_out_path_final + ".png", dpi=300)

        if len(plot_vals_set) > 1 and fig_out_path:
            fig_out_path_final = fig_out_path + "_" + z_name + "_subplots_" + plots
            fig.savefig(fig_out_path_final + ".png", dpi=300)
    print("end")

def compare_stats_obs_features(stats, metrics=None, x_name=None, location_x=None, y_name=None, location_y=None,
                               plots=None, location_plot=None, base_dict=None, fig_out_path=None):
    figsize = [20, 3]
    for z_name in metrics:
        x_vals = []
        y_vals = []
        plots_vals = []
        metric = []
        dict_mission_vehicle_observation = {
            False: 0,
            True: 1,
        }
        dict_features = {
            tuple(["presence", "x", "y", "vx", "vy", "is_controlled"]): 0,
            tuple(["presence", "x", "y", "vx", "vy"]): 1,
            tuple(["x", "y", "vx", "vy", "is_controlled"]): 2
        }
        dict_order = {
            'sorted': 0,
            'sorted_by_id': 1,
            'shuffled': 2
        }

        if base_dict:
            location_x = location_y = location_plot = base_dict

        # location_x = location_y = location_plot=  "observation"
        # x_name = "features"
        # y_name = "order"
        # plots = "mission_vehicle_observation"
        for i, value in enumerate(stats[z_name]):
            # if stats[condition][condition_location] != condition_value:
            #     continue
            metric.append(value)
            val1 = stats[location_x][i][x_name]
            val1 = dict_features[tuple(val1)]
            if isinstance(val1, list):
                val1 = val1[0]
            x_vals.append(val1)
            val2 = stats[location_y][i][y_name]
            if isinstance(val2, list):
                val2 = val2[0]
            val2 = dict_order[val2]
            y_vals.append(val2)

            val3 = stats[location_plot][i][plots]
            plots_vals.append(dict_mission_vehicle_observation[val3])

        # TODO: clean this up and put them in a single 3D array
        x_vals = np.array(x_vals)
        y_vals = np.array(y_vals)
        plots_vals = np.array(plots_vals)
        metric = np.array(metric)
        max_z = max(metric)
        # max_z = 100
        plot_vals_set = set(plots_vals)
        axs = []
        if len(plot_vals_set) == 2:
            fig = plt.figure(figsize=(figsize[0], figsize[1] * len(plot_vals_set)))
            ax1 = fig.add_subplot(121, projection='3d')
            axs.append(ax1)
            ax2 = fig.add_subplot(122, projection='3d')
            axs.append(ax2)
        elif len(plot_vals_set) == 3:
            fig = plt.figure(figsize=(figsize[0], figsize[1] * len(plot_vals_set)))
            axs[0] = fig.add_subplot(131, projection='3d')
            axs[1] = fig.add_subplot(132, projection='3d')
            axs[2] = fig.add_subplot(133, projection='3d')

        # fig, axs = plt.subplots(1,len(plot_vals_set), figsize=(figsize[0], figsize[1] * len(plot_vals_set)))
        for idx, val in enumerate(plot_vals_set):

            # filter based on cooperative_rewards
            x = x_vals[np.where(plots_vals == val)]
            y = y_vals[np.where(plots_vals == val)]
            z = metric[np.where(plots_vals == val)]

            plt_file_name = os.path.split(fig_out_path)[1]
            plot_name = plots + " = " + str(val) + " - " + plt_file_name

            tmp_info = []
            for idx2, path in enumerate(stats["paths"]):
                if plots_vals[idx2] == val:
                    scenario_name = path.split("_")[-1]
                    tmp_info.append(scenario_name)

            fig1 = vutils.simple_3d_bar_plot(x, y, z,
                                             x_name,
                                             y_name,
                                             z_name,
                                             plot_name, zlim3d=max_z, show=True, info=tmp_info)

            print(">>>>>>>>>>>>>>>>>>>>>>> INFO: ", plot_name, x_name, x, y_name, y, z_name, z)

            if len(plot_vals_set) > 1:
                fig2 = vutils.simple_3d_bar_plot(x, y, z,
                                                 x_name,
                                                 y_name,
                                                 z_name,
                                                 plot_name, zlim3d=max_z, ax=axs[idx])

            if fig_out_path:
                fig_out_path_final = fig_out_path + "_" + z_name + "_" + plots + "=" + str(val)
                fig1.savefig(fig_out_path_final + ".png", dpi=300)

        if len(plot_vals_set) > 1 and fig_out_path:
            fig_out_path_final = fig_out_path + "_" + z_name + "_subplots_" + plots
            fig.savefig(fig_out_path_final + ".png", dpi=300)
    print("end")

def compare_stats(stats, metrics=None, x_name=None, location_x=None, y_name=None, location_y=None, plots=None,
                  location_plot=None, base_dict=None, fig_out_path=None, condition_value=None, condition_name=None):
    '''
    [in] stats should be created by
    overall_stats = get_overall_stats(path, 2000)
    overall_stats = append_from_json(overall_stats)
    '''
    figsize = [20, 3]
    for z_name in metrics:
        x_vals = []
        y_vals = []
        plots_vals = []
        metric = []
        if base_dict:
            location_x = location_y = location_plot = base_dict

        path_ok = []
        for i, value in enumerate(stats[z_name]):
            if condition_name:
                if stats[location_x][i][condition_name] != condition_value:
                    continue
            metric.append(value)
            val1 = stats[location_x][i][x_name]
            if isinstance(val1, list):
                val1 = val1[0]
            x_vals.append(val1)
            val2 = stats[location_y][i][y_name]
            if isinstance(val2, list):
                val2 = val2[0]
            y_vals.append(val2)

            tmp_val = stats[location_plot][i][plots]
            if isinstance(tmp_val, list):
                tmp_val = tmp_val[0]
            plots_vals. \
                append(tmp_val)
            path_ok.append(stats["paths"][i])

        # TODO: clean this up and put them in a single 3D array
        x_vals = np.array(x_vals)
        y_vals = np.array(y_vals)
        plots_vals = np.array(plots_vals)
        metric = np.array(metric)
        max_z = max(metric)
        # max_z = 100
        plot_vals_set = set(plots_vals)
        axs = []
        if len(plot_vals_set) == 2:
            fig = plt.figure(figsize=(figsize[0], figsize[1] * len(plot_vals_set)))
            ax1 = fig.add_subplot(121, projection='3d')
            axs.append(ax1)
            ax2 = fig.add_subplot(122, projection='3d')
            axs.append(ax2)
        elif len(plot_vals_set) == 3:
            fig = plt.figure(figsize=(figsize[0], figsize[1] * len(plot_vals_set)))
            ax1 = fig.add_subplot(131, projection='3d')
            ax2 = fig.add_subplot(132, projection='3d')
            ax3 = fig.add_subplot(133, projection='3d')
            axs.append(ax1)
            axs.append(ax2)
            axs.append(ax3)

        elif len(plot_vals_set) == 4:
            fig = plt.figure(figsize=(figsize[0], figsize[1] * len(plot_vals_set)))
            ax1 = fig.add_subplot(141, projection='3d')
            ax2 = fig.add_subplot(142, projection='3d')
            ax3 = fig.add_subplot(143, projection='3d')
            ax4 = fig.add_subplot(144, projection='3d')
            axs.append(ax1)
            axs.append(ax2)
            axs.append(ax3)
            axs.append(ax4)
        elif len(plot_vals_set) == 5:
            fig = plt.figure(figsize=(figsize[0], figsize[1] * len(plot_vals_set)))
            axs[0] = fig.add_subplot(151, projection='3d')
            axs[1] = fig.add_subplot(152, projection='3d')
            axs[2] = fig.add_subplot(153, projection='3d')
            axs[3] = fig.add_subplot(154, projection='3d')
            axs[4] = fig.add_subplot(155, projection='3d')
        # fig, axs = plt.subplots(1,len(plot_vals_set), figsize=(figsize[0], figsize[1] * len(plot_vals_set)))
        for idx, val in enumerate(plot_vals_set):

            # filter based on cooperative_rewards
            x = x_vals[np.where(plots_vals == val)]
            y = y_vals[np.where(plots_vals == val)]
            z = metric[np.where(plots_vals == val)]

            plt_file_name = os.path.split(fig_out_path)[1]
            plot_name = plots + " = " + str(val) + " - " + plt_file_name

            tmp_info = []
            for idx2, path in enumerate(path_ok):
                if plots_vals[idx2] == val:
                    scenario_name = path.split("_")[-1]
                    tmp_info.append(scenario_name)

            fig1 = vutils.simple_3d_bar_plot(x, y, z,
                                             x_name,
                                             y_name,
                                             z_name,
                                             plot_name, zlim3d=max_z, show=False, info=tmp_info)

            if len(plot_vals_set) > 1:
                fig2 = vutils.simple_3d_bar_plot(x, y, z,
                                                 x_name,
                                                 y_name,
                                                 z_name,
                                                 plot_name, zlim3d=max_z, ax=axs[idx], info=tmp_info)

            if fig_out_path:
                fig_out_path_final = fig_out_path + "_" + z_name + "_" + plots + "=" + str(val)
                if condition_name:
                    fig_out_path_final += "_" + condition_name + str(condition_value)
                fig1.savefig(fig_out_path_final + ".png", dpi=300)

        if len(plot_vals_set) > 1 and fig_out_path:
            fig_out_path_final = fig_out_path + "_" + z_name + "_subplots_" + plots
            if condition_name:
                fig_out_path_final += "_" + condition_name + str(condition_value)
            fig.savefig(fig_out_path_final + ".png", dpi=300)
    print("end")

def append_from_json(stats):
    '''
    [in] stats: this should be created overall_stats = get_overall_stats(path, 2000)
    reads the json and append the stats with reward_params and configs of merging vehicle
    '''
    stats['merging_vehicle'] = []
    stats['observation'] = []
    stats['metadata'] = []
    for path in stats["paths"]:
        metadata = cutils.read_metadata_file(path)
        reward_params = metadata['env']['reward']
        stats['reward_params'].append(reward_params)
        merging_vehicle = metadata['env']['merging_vehicle']
        stats['merging_vehicle'].append(merging_vehicle)
        observation = metadata['env']['observation']
        stats['observation'].append(observation)
        stats['metadata'].append(metadata)

    return stats

def add_total_distance_travelled(stats):

    stats['total_distance_travelled'] = []
    # for path in stats["paths"]:
    #
    #
    a = 2

    return stats

# output {'exp_number': [metric1_val, metric2_val,...] ; ....}
def get_experiments(stats, metrics=None, train = True,n=900,max_distance= 400):
    exp_dict= {}
    exp_dict["metrics"] = metrics
    for i, path in enumerate(stats["paths"]):
        if train:
            exp = path.split("_")[-1]
        else:
            exp = path.split("_")[-1]
            exp = exp.split("-")[0]
        metric_vals = []
        for metric in metrics:
            if metric !="average_distance_travelled":
                val = stats[metric][i]/n*100
            else:
                # val = stats[metric][i] / max_distance * 100
                val = stats[metric][i]
            metric_vals.append(val)
        exp_dict[exp] = metric_vals

    return exp_dict
def compare_folder(stats, metrics=None, plots_output_path=None, n=900,persentage=True, set_limit=True):
    figsize = [10, 6]
    # plt.rcdefaults()
    if persentage:
        x_limit = 100
    else:
        x_limit =n
    header = ['Name', 'metric']
    for z_name in metrics:
        # metric = []

        all_stats_non_aggressive = []
        legend_non_aggressive = []
        paths = []
        for i, value in enumerate(stats[z_name]):
            # metric.append(value)

            if persentage:
                # value =value/n*100
                if z_name != "average_distance_travelled":
                    value = float(value)/n*100

                else:
                    value = float(value)/400*100

            all_stats_non_aggressive.append(value)
            # legend_name = "plt_" + stats["paths"][i].split("_")[-1]
            legend_name = "plt_" + stats["paths"][i]

            legend_non_aggressive.append(legend_name)
            paths.append(stats["paths"][i].split("_")[-1])

        order = legend_non_aggressive

        # figManager = plt.get_current_fig_manager()
        # figManager.window.showMaximized()
        # plt_name = plots_output_path.split("_")[-1]
        plt_name = os.path.split(plots_output_path)[-1]
        plt.ioff()
        # fig, ax = plt.subplots(figsize=figsize)
        # y_pos = np.arange(len(all_stats_aggressive_order))
        # ax.barh(y_pos, all_stats_aggressive_order, align='center')
        # ax.set_yticks(y_pos)
        # ax.set_yticklabels(order)
        # ax.invert_yaxis()  # labels read top-to-bottom
        # ax.set_xlabel(z_name)
        # ax.set_title('Aggressive')
        # print(order)
        # print('Aggressive', z_name, all_stats_aggressive_order)
        # plt.subplots_adjust(left=0.3, bottom=0.1, right=0.95, top=0.9)

        # plt.margins(0.9)
        # out_file = plt_name + 'Aggressive' + z_name
        # fig_out_path = os.path.join(plots_output_path, out_file)
        # fig.savefig(fig_out_path + ".png", dpi=300)

        fig, ax = plt.subplots(figsize=figsize)
        y_pos = np.arange(len(all_stats_non_aggressive))
        ax.barh(y_pos, all_stats_non_aggressive, align='center')

        for i, v in enumerate(all_stats_non_aggressive):
            ax.text(v + 3, i + .25, str(v), color='blue', fontweight='bold')
        if set_limit:
            ax.set_xlim(0, x_limit)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(order)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel(z_name)
        ax.set_title(plt_name)
        print(order)
        print(plt_name, z_name, all_stats_non_aggressive)
        plt.subplots_adjust(left=0.3, bottom=0.1, right=0.95, top=0.9)

        # plt.margins(0.9)


        out_file = plt_name + "_" + z_name
        fig_out_path = os.path.join(plots_output_path, out_file)

        data = [legend_non_aggressive, all_stats_non_aggressive]
        datat = np.array(data).T
        # datat_2d_t = np.array(data).T.tolist()
        datat_2d_t = np.array(datat).tolist()
        with open(fig_out_path + '.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Name', 'metric'])
            writer.writerows(datat)

        # file = open(fig_out_path + '.csv', 'w', newline='')
        # writer = csv.DictWriter(file, fieldnames=header)
        # writer.writeheader()

        fig.savefig(fig_out_path + ".png", dpi=300)
        # plt.show()

def compare_behavior(stats, metrics=None, plots_output_path=None, condition_name=None, condition_value=None):
    figsize = [10, 6]
    # plt.rcdefaults()

    for z_name in metrics:
        # metric = []

        all_stats_aggressive = []
        legend_aggressive = []
        all_stats_non_aggressive = []
        legend_non_aggressive = []
        paths = []
        for i, value in enumerate(stats[z_name]):
            # metric.append(value)
            if condition_name and condition_value:
                if stats["merging_vehicle"][i][condition_name][1] != condition_value:
                    continue

            if stats["merging_vehicle"][i]["vehicles_type"] == "highway_env.vehicle.behavior.CustomVehicleAggressive":
                all_stats_aggressive.append(value)
                legend_name = "coop_" + str(stats["reward_params"][i]["cooperative_flag"]) + "_symp_" + str(
                    stats["reward_params"][i][
                        "sympathy_flag"]) + "_percep_" + str(stats["observation"][i]["cooperative_perception"])
                legend_aggressive.append(legend_name)
                paths.append(stats["paths"][i].split("_")[-1])
            else:
                all_stats_non_aggressive.append(value)
                legend_name = "coop_" + str(stats["reward_params"][i]["cooperative_flag"]) + "_symp_" + str(
                    stats["reward_params"][i][
                        "sympathy_flag"]) + "_percep_" + str(stats["observation"][i]["cooperative_perception"])
                legend_non_aggressive.append(legend_name)
                paths.append(stats["paths"][i].split("_")[-1])

        all_stats_aggressive_order = []
        all_stats_non_aggressive_order = []
        if len(legend_non_aggressive) == 4:
            order = ['coop_False_symp_False_percep_False', 'coop_False_symp_False_percep_True',
                     'coop_True_symp_False_percep_True', 'coop_True_symp_True_percep_True']
        else:
            order = ['coop_False_symp_False_percep_True',
                     'coop_True_symp_False_percep_True', 'coop_True_symp_True_percep_True']

        order_name = copy.deepcopy(order)
        for ido, o in enumerate(order):
            if legend_aggressive:
                idx1 = [idx for idx, l in enumerate(legend_aggressive) if l == o]
                all_stats_aggressive_order.append(all_stats_aggressive[idx1[0]])
            if legend_non_aggressive:
                idx1 = [idx for idx, l in enumerate(legend_non_aggressive) if l == o]
                all_stats_non_aggressive_order.append(all_stats_non_aggressive[idx1[0]])
            order_name[ido] = order[ido] + paths[idx1[0]]

        order = order_name
        if legend_aggressive and legend_non_aggressive:
            # figManager = plt.get_current_fig_manager()
            # figManager.window.showMaximized()
            plt.ioff()
            fig, ax = plt.subplots(figsize=figsize)
            y_pos = np.arange(len(all_stats_aggressive_order))
            ax.barh(y_pos, all_stats_aggressive_order, align='center')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(order)
            ax.invert_yaxis()  # labels read top-to-bottom
            ax.set_xlabel(z_name)
            ax.set_title('Aggressive')
            print(order)
            print('Aggressive', z_name, all_stats_aggressive_order)
            plt.subplots_adjust(left=0.3, bottom=0.1, right=0.95, top=0.9)

            # plt.margins(0.9)
            plt_name = plots_output_path.split[-1]
            out_file = plt_name + 'Aggressive' + z_name
            fig_out_path = os.path.join(plots_output_path, out_file)
            fig.savefig(fig_out_path + ".png", dpi=300)

            fig, ax = plt.subplots(figsize=figsize)
            y_pos = np.arange(len(all_stats_non_aggressive_order))
            ax.barh(y_pos, all_stats_non_aggressive_order, align='center')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(order)
            ax.invert_yaxis()  # labels read top-to-bottom
            ax.set_xlabel(z_name)
            ax.set_title('Non Aggressive')
            print(order)
            print('Non Aggressive', z_name, all_stats_non_aggressive_order)

            plt.subplots_adjust(left=0.3, bottom=0.1, right=0.95, top=0.9)

            # plt.margins(0.9)
            out_file = plt_name + 'NonAgressive_' + z_name
            fig_out_path = os.path.join(plots_output_path, out_file)
            fig.savefig(fig_out_path + ".png", dpi=300)

        if not legend_aggressive:
            # figManager = plt.get_current_fig_manager()
            # figManager.window.showMaximized()
            # plt_name = plots_output_path.split("_")[-1]
            plt_name = os.path.split(plots_output_path)[-1]
            plt.ioff()
            # fig, ax = plt.subplots(figsize=figsize)
            # y_pos = np.arange(len(all_stats_aggressive_order))
            # ax.barh(y_pos, all_stats_aggressive_order, align='center')
            # ax.set_yticks(y_pos)
            # ax.set_yticklabels(order)
            # ax.invert_yaxis()  # labels read top-to-bottom
            # ax.set_xlabel(z_name)
            # ax.set_title('Aggressive')
            # print(order)
            # print('Aggressive', z_name, all_stats_aggressive_order)
            # plt.subplots_adjust(left=0.3, bottom=0.1, right=0.95, top=0.9)

            # plt.margins(0.9)
            # out_file = plt_name + 'Aggressive' + z_name
            # fig_out_path = os.path.join(plots_output_path, out_file)
            # fig.savefig(fig_out_path + ".png", dpi=300)

            fig, ax = plt.subplots(figsize=figsize)
            y_pos = np.arange(len(all_stats_non_aggressive_order))
            ax.barh(y_pos, all_stats_non_aggressive_order, align='center')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(order)
            ax.invert_yaxis()  # labels read top-to-bottom
            ax.set_xlabel(z_name)
            ax.set_title(plt_name)
            print(order)
            print(plt_name, z_name, all_stats_non_aggressive_order)
            plt.subplots_adjust(left=0.3, bottom=0.1, right=0.95, top=0.9)

            # plt.margins(0.9)
            out_file = plt_name + "_" + z_name
            if condition_name and condition_value:
                out_file += "_" + condition_name + "_" + str(condition_value)
            fig_out_path = os.path.join(plots_output_path, out_file)
            fig.savefig(fig_out_path + ".png", dpi=300)
        # plt.show()

def avg_every_n_episodes(simulation_path_base, n=20, span=100, std=True, span_std=20, subfolders=None,
                         metric="crash_episodes_flag_all"):
    overall_stats = get_overall_stats(simulation_path_base, subfolders=subfolders)
    overall_stats = append_from_json(overall_stats)
    avg_crashes = []
    lens = []
    for idx, crashes in enumerate(overall_stats["crash_episodes_flag_all"]):
        avg_crash = cutils.average_binary_array(crashes, n)
        avg_crashes.append(avg_crash)
        lens.append(len(avg_crash))
        pathx = overall_stats["paths"][idx]
    l = min(lens)

    avg_crashes_filter = [avg_crash_filter[0:l - 1] for avg_crash_filter in avg_crashes]
    alpha = 2 / (span + 1)
    avg_crashes_filter_std = [np.std(cutils.rolling_window(np.array(data), span_std), 1) for data in avg_crashes_filter]
    avg_crashes_filter_ema = [cutils.ewma_vectorized_safe(data, alpha) for data in avg_crashes_filter]
    avg_crashes_np = np.array(avg_crashes_filter_ema)
    if std:
        avg_crashes_filter_std_zeros = np.zeros((np.shape(avg_crashes_filter_ema)))
        avg_crashes_filter_std_zeros[:, span_std - 1:] = avg_crashes_filter_std

    if std:
        return avg_crashes_np, np.array(avg_crashes_filter_std_zeros)
    else:
        return avg_crashes_np
