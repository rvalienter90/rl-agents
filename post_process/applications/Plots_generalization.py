import numpy as np

from post_process.applications.applications import *
from scipy.interpolate import interp1d
from scipy.stats import ttest_ind_from_stats
from scipy.interpolate import splprep, splev
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm
from matplotlib.colors import LightSource
from scipy.interpolate import Rbf
import pandas as pd
import matplotlib as mpl
import seaborn as sns
import math
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
import matplotlib.image as mpimg

import post_process.applications.applications as apps
import post_process.visualization.visualization_utils as vutils

SAVE_PATH = os.path.join("..", "Generalization_figures")
# base_path = "D:/out7stokes/nolatent/"
base_path = "D:/Rodolfo/Data/Generalization/Results"
from Autoencoder.autoencoder import Autoencoder , DeepAutoencoder
from  Autoencoder.train_state_AE import load_dataset_Grid, load_dataset_Image
def main():
    try:
        if not os.path.exists(SAVE_PATH):
            os.mkdir(SAVE_PATH)
    except OSError:
        pass
    min_z_global = 0
    config_plt_rcParams()

    # autoencoder_reconstruction()
    # autoencoder_loss()
    barplot_compare()
    # adaptation_compare()

def config_plt_rcParams(fontsize=15):
    # https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.rcParams
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.rc.html
    # https://matplotlib.org/stable/tutorials/introductory/customizing.html
    params = {'legend.fontsize': fontsize,
              'axes.labelsize': fontsize,
              'axes.titlesize': fontsize,
              'xtick.labelsize': fontsize,
              'ytick.labelsize': fontsize,
              'mathtext.fontset': 'stix',
              'font.family': 'STIXGeneral',
              'axes.xmargin': 0.1,
              'axes.ymargin': 0.1
              }
    plt.rcParams.update(params)

def barplot_compare():
    simulation_path_base = os.path.join(base_path, "latent_space", "train")

    n = 900
    episode_duration = 14
    overall_stats = apps.get_overall_stats(simulation_path_base, n,episode_duration=episode_duration )

    plots_output_path = SAVE_PATH
    compare_folder(overall_stats, metrics=[ "total_crash"],
                        plots_output_path=plots_output_path, n=n)

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
                    value = float(value)/n*100 -10

                else:
                    value = float(value)/400*100

            all_stats_non_aggressive.append(value)
            legend_name = stats["paths"][i].split('\\')[-1]
            # legend_name = "plt_" + stats["paths"][i]

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
def autoencoder_reconstruction():
    MLP = False
    Grid = False
    image = True


    if image:
        model_base_folder= os.path.join(base_path,'autoencoder','models')
        Imagemodel64 = os.path.join(model_base_folder,"Autoencoder_CNN_Image_64_date-2021-10-02-16-11-55")
        Imagemodel32= os.path.join(model_base_folder, "Autoencoder_CNN_Image_32_date-2021-10-02-16-11-57")
        Imagemodel16 = os.path.join(model_base_folder, "Autoencoder_CNN_Image_16_date-2021-10-02-16-11-57")
        Imagemodelbegining = os.path.join(model_base_folder, "AutoencoderCNN_Image_beginning")

        pathbase = os.path.join(base_path, 'autoencoder', 'datasetsamples')
        x_train = load_dataset_Image(pathbase=pathbase, samples=50)

        num_sample_images_to_show = 8
        sample_obs = select_observations(x_train, num_samples=8)

        # models = [Imagemodel64]
        models = [Imagemodel64, Imagemodel32, Imagemodel16,Imagemodelbegining]
        names = ['Imagemodel64','Imagemodel32','Imagemodel16','Imagemodelbegining']
        for idx,model in enumerate(models):
            autoencoder = Autoencoder.load(model)

            reconstructed_images, latent_representations = autoencoder.reconstruct(sample_obs)
            title = names[idx] + '_latent_representations'
            latent_size = len(latent_representations[0])
            latent_images = plot_reconstructed_images(sample_obs, latent_representations.reshape(8, 8,int(latent_size/8)),title)
            title = names[idx] + '_reconstructed_images'
            reconstructed_images = plot_reconstructed_images(sample_obs, reconstructed_images,title)
            figname = 'Gen_Fig_B_latent_representations' + names[idx]
            latent_images.savefig(os.path.join(SAVE_PATH,figname))
            figname = 'Gen_Fig_B_reconstructed_images' + names[idx]
            reconstructed_images.savefig(os.path.join(SAVE_PATH, figname))

    if MLP:
        MLPmodel = "model/DeepAutoencoder_MLPdate-2021-09-29-19-24-15"
        autoencoderMLP = DeepAutoencoder.load(MLPmodel)
        x_train = load_dataset_MLP(samples=2000)
        sample_obs = select_observations(x_train, num_samples=2)
        reconstructed_obs, latent_representations = autoencoderMLP.reconstruct(sample_obs)
        print(reconstructed_obs.reshape(2, 6, 7))
        print(sample_obs.reshape(2, 6, 7))

        # num_obs= 1000
        # sample_obs = select_observations(x_train, num_samples=num_obs)
        # _, latent_representations = autoencoderMLP.reconstruct(sample_obs)

    if Grid:
        Gridmodel = "model/AutoencoderCNN_date-2021-09-29-19-35-05"
        autoencoderGrid = Autoencoder.load(Gridmodel)
        x_train = load_dataset_Grid(samples=2000)
        sample_obs = select_observations(x_train, num_samples=2)
        reconstructed_obs, _ = autoencoderGrid.reconstruct(sample_obs)

def plot_reconstructed_images(images, reconstructed_images,title=None):
    fig = plt.figure(figsize=(15, 3))
    num_images = len(images)
    for i, (image, reconstructed_image) in enumerate(zip(images, reconstructed_images)):
        image = image.squeeze()
        ax = fig.add_subplot(2, num_images, i + 1)
        ax.axis("off")
        ax.imshow(image, cmap="gray_r")
        reconstructed_image = reconstructed_image.squeeze()
        ax = fig.add_subplot(2, num_images, i + num_images + 1)
        ax.axis("off")
        ax.imshow(reconstructed_image, cmap="gray_r")
    fig.suptitle(title)
    return fig
    # plt.show()
def plot_images_encoded_in_latent_space(latent_representations, sample_labels):
    plt.figure(figsize=(10, 10))
    plt.scatter(latent_representations[:, 0],
                latent_representations[:, 1],
                cmap="rainbow",
                c=sample_labels,
                alpha=0.5,
                s=2)
    plt.colorbar()
    plt.show()
def select_observations(obs, num_samples=10):
    sample_index = np.random.choice(range(len(obs)), num_samples)
    return obs[sample_index]


def adaptation_compare():
    n_episode = 900
    max_distance = 400

    # adaptation_folder = os.path.join(base_path, "adaptation", "test")
    data_folder = os.path.join(base_path,'adaptation', "latent","test")
    overall_stats = get_overall_stats(data_folder, n_episode, episode_duration=15)
    metric = ["total_crash"]
    # latent
    experiments_interval = [305, 324]
    plotname = 'Gen_Fig_A_Adaptation_latent'
    adaptation_plot(overall_stats, metric, n_episode, experiments_interval,plotname)

    # nolatent
    data_folder = os.path.join(base_path,'adaptation',  "nolatent","test")
    overall_stats = get_overall_stats(data_folder, n_episode, episode_duration=15)
    experiments_interval = [355, 374]
    plotname = 'Gen_Fig_A_Adaptation_nolatent'
    adaptation_plot(overall_stats, metric, n_episode, experiments_interval,plotname)
def adaptation_plot(overall_stats,metric,n_episode,experiments_interval,plotname):


    exp_dict = apps.get_experiments(overall_stats, metrics=metric, train=False, n=n_episode)

    adjance_matrix = np.zeros((5, 5)) + -1

    r=0
    c=0
    for i in range(experiments_interval[0], experiments_interval[1]+1):
        data = np.array(exp_dict[str(i)])
        adjance_matrix[r][c] = data
        c+=1
        if c ==5:
            r+=1
            c=0

    # experiments_interval = [300, 304]
    # experiments_interval = [350, 354]
    # all
    c = 0
    for i in range(experiments_interval[0]-5, experiments_interval[0]):
        data = np.array(exp_dict[str(i)])
        adjance_matrix[4][c] = data
        c += 1

    adjance_matrix_original= np.round(adjance_matrix,1)
    plottype = 'crash'
    plot_heat_map(adjance_matrix,adjance_matrix_original,plotname)
def plot_heat_map(data,labels,plotname):
    fontsize = 20
    fig = plt.figure(figsize=(8, 7.2), dpi=400, constrained_layout=False)
    gs1 = fig.add_gridspec(nrows=1, ncols=1, left=0.15, right=0.98, wspace=0.1, hspace=0.1,
                           bottom=0.15, top=0.95)
    # fig, ax = plt.subplots()
    ax = fig.add_subplot(gs1[0, :])
    im = ax.imshow(data, cmap="coolwarm")
    # for im in gca().get_images():
    #     im.set_clim(0, 0.05)
    # im.set_clim(0, 100)
    cbar = plt.colorbar(im)
    max_val = np.max(data)
    min_val = np.min(data)
    # if plottype == 'score':
    #     cbar.set_label('$Adaptation_{error} (\%)$', fontsize=fontsize)
    #     # cbar.set_ticks([-0.9, 1.9])
    #     cbar.set_ticks([min_val, max_val])
    #     cbar.ax.set_yticklabels(['0','100'])
    # elif plottype == 'crash':
    #     cbar.set_label('$Crashed (\%)$', fontsize=fontsize)
    #     # cbar.set_ticks([-1.9, 1.9])
    #     cbar.set_ticks([min_val, max_val])
    #     cbar.ax.set_yticklabels(['0', '100'])
    # elif plottype == 'distance':
    #     cbar.set_label('$Distance Traveled (m)$', fontsize=fontsize)
    #     # cbar.set_ticks([-0.7, 1.9])
    #     cbar.set_ticks([min_val, max_val])
    #     cbar.ax.set_yticklabels(['400', '50'])
    # # plt.imshow(H, interpolation='none')
    #
    # valsy = ['$f_m,b_{mix}$', '$f_m,b_a$', '$f_m,b_m$', '$f_m,b_c$', '$f_e,b_{mix}$', '$f_e,b_a$', '$f_e,b_m$', '$f_e,b_c$']
    # valsx = ['$f_m,b_{mix}$', '$f_m,b_a$', '$f_m,b_m$', '$f_m,b_c$', '$f_e,b_{mix}$', '$f_e,b_a$', '$f_e,b_m$', '$f_e,b_c$']
    # # We want to show all ticks...
    # ax.set_xticks(np.arange(len(valsx)))
    # ax.set_yticks(np.arange(len(valsy)))

    # # ... and label them with the respective list entries
    # ax.set_xticklabels(valsx, fontsize=fontsize)
    # ax.set_yticklabels(valsy, fontsize=fontsize)

    valsy = ['i', 'r', 'm', 'e', 'all']
    valsx = ['i', 'r', 'm', 'e', 'all']
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(valsx)))
    ax.set_yticks(np.arange(len(valsy)))

    # ... and label them with the respective list entries
    ax.set_xticklabels(valsx, fontsize=fontsize)
    ax.set_yticklabels(valsy, fontsize=fontsize)

    ax.set_ylabel("Training scenarios", fontsize=fontsize, labelpad=5)
    ax.set_xlabel("Testing scenarios", fontsize=fontsize, labelpad=5)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(5):
        for j in range(5):
            text = ax.text(j, i, labels[i, j],
                           ha="center", va="center", color="k", fontsize=fontsize)

    ax.set_title("Adaptation", fontsize=fontsize)
    fig.tight_layout()
    # plt.show()
    name = plotname + ".png"
    fig.savefig(os.path.join(SAVE_PATH,name ))




if __name__ == '__main__':
    main()
