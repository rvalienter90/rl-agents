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
import pickle

import post_process.applications.applications as apps
import post_process.visualization.visualization_utils as vutils
# from Autoencoder.autoencodertf1 import Autoencoder , DeepAutoencoder
from autoencoder.autoencodertf2 import Autoencoder , DeepAutoencoder

from  autoencoder.train_state_AE import load_dataset_Grid, load_dataset_Image
from  autoencoder.train_state_AE_prediction import load_dataset_Image_prediction
SAVE_PATH = os.path.join("..","..","scripts", "out","Generalization_figures")
# base_path = "D:/out7stokes/nolatent/"
DATA_PATH= "E:\Data\Datasets\Image"
# DATA_PATH = "/media/rodolfo/DataSSD/Data/Datasets/Image"
base_path = "D:/Data/Data/Prediction/Results"
# base_path= "/media/rodolfo/Data/Data/Data/Generalization/Results"
SAVE_PATH = os.path.join(base_path,"Generalization_figures")

def main():
    try:
        if not os.path.exists(SAVE_PATH):
            os.mkdir(SAVE_PATH)
    except OSError:
        pass
    min_z_global = 0
    config_plt_rcParams()

    # transfer_learning()
    # barplot_compare()
    # autoencoder_filters_and_features()
    # autoencoder_reconstruction()
    # autoencoder_loss()
    autoencoder_reconstruction_chain()
    # autoencoder_reconstruction_chain_continuos()
    # adaptation_compare()
    # adaptation_compare_trajectories()


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
                    value = float(value)/n*100

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

def autoencoder_reconstruction_chain():
    MLP = False
    Grid = False
    image = True


    if image:
        # model_base_folder= os.path.join(base_path,'autoencoder','models')
        # Imagemodel64 = os.path.join(model_base_folder,"Autoencoder_CNN_Image_64_date-2021-10-02-16-11-55")

        model_base_folder_pred = os.path.join(base_path, 'autoencoder', 'modelpredictions')
        Imagemodel64_pred2 = os.path.join(model_base_folder_pred,
                                          "Autoencoder_CNN_Image_64_date-2022-02-08-23-06-11")

        model_base_folder_predv2 = os.path.join(base_path, 'autoencoder', 'modelpredictionsv2')
        Imagemodel2048_pred2 = os.path.join(model_base_folder_predv2,
                                            "Autoencoder_CNN_Image_2048_date-2022-02-13-14-49-18")

        Imagemodel1024_fast = os.path.join(model_base_folder_predv2,
                                           "Autoencoder_CNN_Image_1024_date-2022-02-13-14-41-11")


        # pathbase = os.path.join(base_path, 'autoencoder', 'datasetsamples','multi')
        pathbase = os.path.join(base_path, 'autoencoder', 'datasetsamples','fast')
        # x_train = load_dataset_Image(pathbase=pathbase, samples=50)
        # x_train = load_dataset_Image(pathbase=pathbase, fixed_index=[i+1 for i in range(40)])

        # x_train, y_train = load_dataset_Image_prediction(pathbase=pathbase, samples=50)
        x_train, y_train = load_dataset_Image_prediction(pathbase=pathbase, fixed_index=[i+1 for i in range(40)])

        num_sample_images_to_show = 8
        # sample_obs = select_observations(x_train, num_samples=1)
        # sample_obs = select_observations(x_train, fixed_index=[100])
        sample_obs_idxs = select_observations_idxs(x_train, fixed_index=[50]) # 2 , 30, 40 #50
        sample_obs_xtrain = x_train[sample_obs_idxs]
        sample_obs_ytrain = y_train[sample_obs_idxs]


        # models = [Imagemodel64]
        models = [Imagemodel1024_fast]
        names = ['Imagemodel1024_fast']
        for idx,model in enumerate(models):
            autoencoder = Autoencoder.load(model)

            reconstructed_images, latent_representations = autoencoder.reconstruct(sample_obs_xtrain)
            title = names[idx] + '_latent_representations'
            latent_size = len(latent_representations[0])
            # latent_images = plot_reconstructed_images(sample_obs, latent_representations.reshape(1, 8,int(latent_size/8)),title)
            latent_images_fig = plot_state_reconstructed_nextstate(sample_obs_xtrain, sample_obs_ytrain,
                                                                   latent_representations.reshape(1, 8,int(latent_size / 8)),title)

            title = names[idx] + '_reconstructed_images'
            # reconstructed_images_fig = plot_reconstructed_images(sample_obs, reconstructed_images,title)
            reconstructed_images_fig = plot_state_reconstructed_nextstate(sample_obs_xtrain,sample_obs_ytrain, reconstructed_images, title)

            figname = 'Gen_Fig_B_latent_representations' + names[idx]
            latent_images_fig.savefig(os.path.join(SAVE_PATH,figname))
            figname = 'Gen_Fig_B_reconstructed_images' + names[idx]
            reconstructed_images_fig.savefig(os.path.join(SAVE_PATH, figname))
        num_steps = 6
        reconstructed_images_chain_fig = plt.figure(figsize=(3, num_steps))

        image = sample_obs_xtrain.squeeze()
        ax = reconstructed_images_chain_fig.add_subplot(num_steps, 1, 1)
        ax.axis("off")
        ax.imshow(image, cmap="gray_r")

        next_state = sample_obs_ytrain.squeeze()
        ax = reconstructed_images_chain_fig.add_subplot(num_steps, 1, 2)
        ax.axis("off")
        ax.imshow(next_state, cmap="gray_r")


        reconstructed_image = reconstructed_images.squeeze()
        ax = reconstructed_images_chain_fig.add_subplot(num_steps, 1, 3)
        ax.axis("off")
        ax.imshow(reconstructed_image, cmap="gray_r")


        for i in range(num_steps-3):
            reconstructed_images, new_latent_representations = autoencoder.reconstruct(reconstructed_images)
            reconstructed_image = reconstructed_images.squeeze()
            ax = reconstructed_images_chain_fig.add_subplot(num_steps,1, i + 3)
            ax.axis("off")
            ax.imshow(reconstructed_image, cmap="gray_r")

        reconstructed_images_chain_fig.suptitle("reconstructed_images_chain_fig")
        reconstructed_images_chain_fig.savefig(os.path.join(SAVE_PATH, "reconstructed_images_chain_fig"))

def autoencoder_reconstruction_chain_continuos():
    MLP = False
    Grid = False
    image = True


    if image:
        # model_base_folder= os.path.join(base_path,'autoencoder','models')
        # Imagemodel64 = os.path.join(model_base_folder,"Autoencoder_CNN_Image_64_date-2021-10-02-16-11-55")

        model_base_folder_pred = os.path.join(base_path, 'autoencoder', 'modelpredictions')
        model_base_folder_predv2 = os.path.join(base_path, 'autoencoder', 'modelpredictionsv2')
        Imagemodel64_pred2 = os.path.join(model_base_folder_pred,
                                          "Autoencoder_CNN_Image_64_date-2022-02-08-23-06-11")


        Imagemodel2048_pred2 = os.path.join(model_base_folder_predv2,
                                          "Autoencoder_CNN_Image_2048_date-2022-02-13-14-49-18")

        Imagemodel1024_fast = os.path.join(model_base_folder_predv2,
                                            "Autoencoder_CNN_Image_1024_date-2022-02-13-14-41-11")

        # pathbase = os.path.join(base_path, 'autoencoder', 'datasetsamples','multi')
        pathbase = os.path.join(base_path, 'autoencoder', 'datasetsamples','fast')
        # x_train = load_dataset_Image(pathbase=pathbase, samples=50)
        # x_train = load_dataset_Image(pathbase=pathbase, fixed_index=[i+1 for i in range(40)])

        # x_train, y_train = load_dataset_Image_prediction(pathbase=pathbase, samples=50)
        x_train, y_train = load_dataset_Image_prediction(pathbase=pathbase, fixed_index=[i+1 for i in range(40)])

        num_sample_images_to_show = 8
        # sample_obs = select_observations(x_train, num_samples=1)
        # sample_obs = select_observations(x_train, fixed_index=[100])

        sample_obs_idxs = select_observations_idxs(x_train, fixed_index=[2])
        sample_obs_xtrain = x_train[sample_obs_idxs]
        sample_obs_ytrain = y_train[sample_obs_idxs]


        # models = [Imagemodel64]
        models = [Imagemodel1024_fast]
        names = ['Imagemodel1024_fast']
        for idx,model in enumerate(models):
            autoencoder = Autoencoder.load(model)

            reconstructed_images, latent_representations = autoencoder.reconstruct(sample_obs_xtrain)
            title = names[idx] + '_latent_representations'
            latent_size = len(latent_representations[0])
            # latent_images = plot_reconstructed_images(sample_obs, latent_representations.reshape(1, 8,int(latent_size/8)),title)
            latent_images_fig = plot_state_reconstructed_nextstate(sample_obs_xtrain, sample_obs_ytrain,
                                                                   latent_representations.reshape(1, 8,int(latent_size / 8)),title)

            title = names[idx] + '_reconstructed_images'
            # reconstructed_images_fig = plot_reconstructed_images(sample_obs, reconstructed_images,title)
            reconstructed_images_fig = plot_state_reconstructed_nextstate(sample_obs_xtrain,sample_obs_ytrain, reconstructed_images, title)

            figname = 'Gen_Fig_B_latent_representations' + names[idx]
            latent_images_fig.savefig(os.path.join(SAVE_PATH,figname))
            figname = 'Gen_Fig_B_reconstructed_images' + names[idx]
            reconstructed_images_fig.savefig(os.path.join(SAVE_PATH, figname))

        num_steps = 8
        num_cols =2
        start_selection =[2] # 2 , 30, 40 #50
        reconstructed_images_chain_fig = plt.figure(figsize=(3, num_steps))

        image = x_train[start_selection].squeeze()
        ax = reconstructed_images_chain_fig.add_subplot(num_steps, 2, 1)
        ax.axis("off")
        ax.imshow(image, cmap="gray_r")

        next_state = y_train[start_selection].squeeze()
        ax = reconstructed_images_chain_fig.add_subplot(num_steps, 2, num_cols +1)
        ax.axis("off")
        ax.imshow(next_state, cmap="gray_r")

        state = x_train[start_selection]
        reconstructed_images, latent_representations = autoencoder.reconstruct(state)
        reconstructed_image = reconstructed_images.squeeze()
        ax = reconstructed_images_chain_fig.add_subplot(num_steps, 2, num_cols + 2)
        ax.axis("off")
        ax.imshow(reconstructed_image, cmap="gray_r")


        for i in range(num_steps-3):
            selection = [start_selection[0] + 1 + i]
            state = x_train[selection]
            next_state = y_train[selection].squeeze()
            ax = reconstructed_images_chain_fig.add_subplot(num_steps, 2, (i+2)*num_cols + 1)
            ax.axis("off")
            ax.imshow(next_state, cmap="gray_r")

            reconstructed_images, latent_representations = autoencoder.reconstruct(state)
            reconstructed_image = reconstructed_images.squeeze()
            ax = reconstructed_images_chain_fig.add_subplot(num_steps, 2, (i+2)*num_cols + 2)
            ax.axis("off")
            ax.imshow(reconstructed_image, cmap="gray_r")



        reconstructed_images_chain_fig.suptitle("reconstructed_images_chain_continuos_fig")
        reconstructed_images_chain_fig.savefig(os.path.join(SAVE_PATH, "reconstructed_images_chain_continuos_fig"))

def autoencoder_reconstruction():
    MLP = False
    Grid = False
    image = True


    if image:
        model_base_folder= os.path.join(base_path,'autoencoder','models')
        Imagemodel64 = os.path.join(model_base_folder,"Autoencoder_CNN_Image_64_date-2021-10-02-16-11-55")
        Imagemodel32 = os.path.join(model_base_folder, "Autoencoder_CNN_Image_32_date-2021-10-02-16-11-57")
        Imagemodel16 = os.path.join(model_base_folder, "Autoencoder_CNN_Image_16_date-2021-10-02-16-11-57")
        Imagemodelbegining = os.path.join(model_base_folder, "AutoencoderCNN_Image_beginning")
        # Imagemodel64_pred = os.path.join(model_base_folder,"Autoencoder_CNN_Image_64_date-2022-02-06-23-28-21")
        model_base_folder_pred = os.path.join(base_path, 'autoencoder', 'modelpredictions')
        Imagemodel64_pred = os.path.join(model_base_folder_pred,"Autoencoder_CNN_Image_64_date-2022-02-07-00-18-54_Pred")
        Imagemodel64_pred2 = os.path.join(model_base_folder_pred,
                                         "Autoencoder_CNN_Image_64_date-2022-02-08-22-57-56")
        Imagemodel64_pred3 = os.path.join(model_base_folder_pred,
                                         "Autoencoder_CNN_Image_64_date-2022-02-08-23-06-11")
        Imagemodel64_pred4 = os.path.join(model_base_folder_pred,
                                         "Autoencoder_CNN_Image_64_date-2022-02-08-23-06-12")

        Imagemodel128_pred = os.path.join(model_base_folder_pred,"Autoencoder_CNN_Image_128_date-2022-02-08-23-14-34")
        Imagemodel128_pred2 = os.path.join(model_base_folder_pred,"Autoencoder_CNN_Image_128_date-2022-02-08-23-14-35")

        Imagemodel256_pred = os.path.join(model_base_folder_pred,"Autoencoder_CNN_Image_256_date-2022-02-08-23-14-37")
        Imagemodel256_pred2 = os.path.join(model_base_folder_pred, "Autoencoder_CNN_Image_256_date-2022-02-08-23-43-43")
        Imagemodel256_pred3 = os.path.join(model_base_folder_pred, "Autoencoder_CNN_Image_256_date-2022-02-08-23-43-45")

        Imagemodel512_pred = os.path.join(model_base_folder_pred,"Autoencoder_CNN_Image_512_date-2022-02-08-23-14-37")
        Imagemodel512_pred2 = os.path.join(model_base_folder_pred, "Autoencoder_CNN_Image_512_date-2022-02-08-23-17-44")
        Imagemodel512_pred3= os.path.join(model_base_folder_pred, "Autoencoder_CNN_Image_512_date-2022-02-08-23-19-10")
        Imagemodel512_pred4 = os.path.join(model_base_folder_pred, "Autoencoder_CNN_Image_512_date-2022-02-08-23-29-36")


        Imagemodel1024_pred = os.path.join(model_base_folder_pred, "Autoencoder_CNN_Image_1024_date-2022-02-08-23-46-45")

        pathbase = os.path.join(base_path, 'autoencoder', 'datasetsamples')
        # x_train = load_dataset_Image(pathbase=pathbase, samples=50)
        x_train,y_train = load_dataset_Image_prediction(pathbase=pathbase, samples=50)
        num_sample_images_to_show = 8
        sample_obs_idxs= select_observations_idxs(x_train, num_samples=8)
        sample_obs_xtrain = x_train[sample_obs_idxs]
        sample_obs_ytrain = y_train[sample_obs_idxs]
        # sample_obs_xtrain = select_observations(x_train, fixed_index=[100]) #current state
        # sample_obs_ytrain = select_observations(y_train, num_samples=8) # next state

        # models = [Imagemodel64]
        models = [Imagemodel64, Imagemodel64_pred,Imagemodel64_pred2,Imagemodel64_pred3,Imagemodel64_pred4,
                  Imagemodel128_pred,Imagemodel128_pred2,Imagemodel256_pred,Imagemodel256_pred2,Imagemodel256_pred3,
                  Imagemodel512_pred,Imagemodel512_pred2,Imagemodel512_pred3,Imagemodel512_pred4,
                  Imagemodel1024_pred]
        names = ['Imagemodel64','Imagemodel64_pred','Imagemodel64_pred2','Imagemodel64_pred3','Imagemodel64_pred4',
                 'Imagemodel128_pred', 'Imagemodel128_pred2', 'Imagemodel256_pred', 'Imagemodel256_pred2', 'Imagemodel256_pred3',
                  'Imagemodel512_pred','Imagemodel512_pred2','Imagemodel512_pred3','Imagemodel512_pred4',
                 'Imagemodel1024_pred']
        for idx,model in enumerate(models):
            autoencoder = Autoencoder.load(model)

            # reconstructed_images, latent_representations = autoencoder.reconstruct(sample_obs)
            reconstructed_images, latent_representations = autoencoder.reconstruct(sample_obs_xtrain)

            title = names[idx] + '_latent_representations'
            latent_size = len(latent_representations[0])
            # latent_images = plot_reconstructed_images(sample_obs, latent_representations.reshape(8, 8,int(latent_size/8)),title)
            latent_images_fig = plot_state_reconstructed_nextstate(sample_obs_xtrain,sample_obs_ytrain,
                                                      latent_representations.reshape(8, 8, int(latent_size / 8)), title)
            title = names[idx] + '_reconstructed_images'
            # reconstructed_images_fig = plot_reconstructed_images(sample_obs, reconstructed_images,title)
            reconstructed_images_fig = plot_state_reconstructed_nextstate(sample_obs_xtrain,sample_obs_ytrain, reconstructed_images, title)
            figname = 'Gen_Fig_B_latent_representations' + names[idx]
            latent_images_fig.savefig(os.path.join(SAVE_PATH,figname))
            figname = 'Gen_Fig_B_reconstructed_images' + names[idx]
            reconstructed_images_fig.savefig(os.path.join(SAVE_PATH, figname))

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

def autoencoder_filters_and_features():
    image = True
    from keras.applications.vgg16 import preprocess_input
    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import img_to_array
    from keras.models import Model

    if image:
        model_base_folder= os.path.join(base_path,'autoencoder','models')
        Imagemodel64 = os.path.join(model_base_folder,"Autoencoder_CNN_Image_64_date-2021-10-02-16-11-55")
        Imagemodel32 = os.path.join(model_base_folder, "Autoencoder_CNN_Image_32_date-2021-10-02-16-11-57")
        Imagemodel16 = os.path.join(model_base_folder, "Autoencoder_CNN_Image_16_date-2021-10-02-16-11-57")
        Imagemodelbegining = os.path.join(model_base_folder, "AutoencoderCNN_Image_beginning")
        model_base_folder_pred = os.path.join(base_path, 'autoencoder', 'modelpredictions')

        Imagemodel64_pred = os.path.join(model_base_folder_pred,"Autoencoder_CNN_Image_64_date-2022-02-07-00-18-54_Pred")


        pathbase = os.path.join(base_path, 'autoencoder', 'datasetsamples')
        # x_train = load_dataset_Image(pathbase=pathbase, samples=50)
        # x_train,y_train = load_dataset_Image_prediction(pathbase=pathbase, samples=50)
        num_sample_images_to_show = 8
        # sample_obs_idxs= select_observations_idxs(x_train, num_samples=8)
        x_train, y_train = load_dataset_Image_prediction(pathbase=pathbase, fixed_index=[10,11,12,13,14,15])
        sample_obs_idxs = [41] # 1 is hiwghay, 21 turn, 41 merge
        sample_obs_xtrain = x_train[sample_obs_idxs]
        sample_obs_ytrain = y_train[sample_obs_idxs]
        # sample_obs_xtrain = select_observations(x_train, fixed_index=[100]) #current state
        # sample_obs_ytrain = select_observations(y_train, num_samples=8) # next state

        # models = [Imagemodel64]
        # models = [Imagemodel64]
        # names = ['Imagemodel64']
        models = [Imagemodel64_pred]
        names = ['Imagemodel64_pred']
        for idx,model_fn in enumerate(models):
            autoencoder = Autoencoder.load(model_fn)
            model = autoencoder.model
            model.summary()
            # for layer in model.layers:
            #     # check for convolutional layer
            #     # if 'conv' not in layer.name:
            #     #     continue
            #     filters, biases = layer.get_weights()
            #     print(layer.name, filters.shape)
            encoder_layers = model.layers[1]
            for l in [1,4,7,10]:
                filters, biases = encoder_layers.layers[l].get_weights()  #1,4,7,10
                print(encoder_layers.name, filters.shape)

            decoder_layers = model.layers[2]
            for l in [3,6,9,12]:
                filters, biases = decoder_layers.layers[l].get_weights()  # 3,6,9,12
                print(decoder_layers.name, filters.shape)

            # normalize filter values to 0-1 so we can visualize them
            f_min, f_max = filters.min(), filters.max()
            filters = (filters - f_min) / (f_max - f_min)
            # plot first few filters
            n_filters, ix = 6, 1
            for i in range(n_filters):
                # get the filter
                f = filters[:, :, :, i]
                # plot each channel separately
                for j in range(1):
                    # specify subplot and turn of axis
                    ax = plt.subplot(n_filters, 1, ix)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    # plot filter channel in grayscale
                    plt.imshow(f[:, :, j], cmap='gray')
                    ix += 1
            # show the figure
            plt.show()

            plt.imshow(sample_obs_xtrain[0,:,:,0], cmap='gray')
            plot_encoder_features = True
            if plot_encoder_features:
                plt.show()
                encoder_layers = model.layers[1]
                for l in [1, 4, 7, 10,11,12]:

                    model_internal = Model(inputs=model.inputs, outputs=encoder_layers.layers[l].output)

                    # get feature map for first hidden layer
                    feature_maps = model_internal.predict(sample_obs_xtrain)
                    # plot all 64 maps in an 8x8 squares
                    feature_maps_shape = feature_maps.shape

                    n_features = feature_maps_shape[3]
                    display_grid = np.zeros((feature_maps_shape[1], feature_maps_shape[2] * n_features))
                    for i in range(n_features):
                        x = feature_maps[0, :, :, i]
                        x -= x.mean()
                        x /= x.std()
                        x *= 64
                        x += 128
                        x = np.clip(x, 0, 255).astype('uint8')
                        # Tile each filter into a horizontal grid
                        display_grid[:, i * feature_maps_shape[2]: (i + 1) * feature_maps_shape[2]] = x

                    scale = 20. / n_features
                    plt.figure(figsize=(scale * n_features, scale))
                    # l_name =decoder_layers.layers[l].name
                    # plt.title(l_name)
                    plt.grid(False)
                    plt.imshow(display_grid, aspect='auto', cmap='viridis')
                    plt.show()

                    plot_in_matrix = False
                    if plot_in_matrix:
                        square = int(math.sqrt(feature_maps_shape[3]))
                        ix = 1
                        for _ in range(square):
                            for _ in range(square):
                                # specify subplot and turn of axis
                                ax = plt.subplot(square, square, ix)
                                ax.set_xticks([])
                                ax.set_yticks([])
                                # plot filter channel in grayscale
                                processing = True
                                if processing:
                                    x = feature_maps[0, :, :, ix - 1]
                                    x -= x.mean()
                                    x /= x.std()
                                    x *= 64
                                    x += 128
                                    x = np.clip(x, 0, 255).astype('uint8')
                                    # Tile each filter into a horizontal grid
                                    plt.imshow(x, cmap='viridis')
                                    # plt.imshow(display_grid, aspect='auto', cmap='viridis')
                                else:
                                    plt.imshow(feature_maps[0, :, :, ix - 1], cmap='gray')
                                ix += 1
                        # show the figure
                        plt.show()

            # reconstructed_images, latent_representations = autoencoder.reconstruct(sample_obs)
            reconstructed_images, latent_representations = autoencoder.reconstruct(sample_obs_xtrain)

            plt.imshow(latent_representations, cmap='gray')
            plt.show()

            decoder_layers = model.layers[2]
            for l in [3, 6, 9, 10,11,12,13]:

                model_internal = Model(inputs=decoder_layers.inputs, outputs=decoder_layers.layers[l].output)

                # get feature map for first hidden layer
                feature_maps = model_internal.predict(latent_representations)
                # plot all 64 maps in an 8x8 squares
                feature_maps_shape = feature_maps.shape

                n_features = feature_maps_shape[3]
                display_grid = np.zeros((feature_maps_shape[1], feature_maps_shape[2] * n_features))
                for i in range(n_features):
                    x = feature_maps[0, :, :, i]
                    x -= x.mean()
                    x /= x.std()
                    x *= 64
                    x += 128
                    x = np.clip(x, 0, 255).astype('uint8')
                    # Tile each filter into a horizontal grid
                    display_grid[:, i * feature_maps_shape[2]: (i + 1) * feature_maps_shape[2]] = x

                scale = 20. / n_features
                plt.figure(figsize=(scale * n_features, scale))
                # l_name =decoder_layers.layers[l].name
                # plt.title(l_name)
                plt.grid(False)
                plt.imshow(display_grid, aspect='auto', cmap='viridis')
                plt.show()

                plot_in_matrix = False
                if plot_in_matrix:
                    square = int(math.sqrt(feature_maps_shape[3]))
                    ix = 1
                    for _ in range(square):
                        for _ in range(square):
                            # specify subplot and turn of axis
                            ax = plt.subplot(square, square, ix)
                            ax.set_xticks([])
                            ax.set_yticks([])
                            # plot filter channel in grayscale
                            processing = True
                            if processing:
                                x = feature_maps[0, :, :, ix - 1]
                                x -= x.mean()
                                x /= x.std()
                                x *= 64
                                x += 128
                                x = np.clip(x, 0, 255).astype('uint8')
                                # Tile each filter into a horizontal grid
                                plt.imshow(x, cmap='viridis')
                                # plt.imshow(display_grid, aspect='auto', cmap='viridis')
                            else:
                                plt.imshow(feature_maps[0, :, :, ix - 1], cmap='gray')
                            ix += 1
                    # show the figure
                    plt.show()


            plt.imshow(reconstructed_images[0,:,:,0], cmap='gray')
            plt.show()
            # title = names[idx] + '_latent_representations'
            # latent_size = len(latent_representations[0])
            # # latent_images = plot_reconstructed_images(sample_obs, latent_representations.reshape(8, 8,int(latent_size/8)),title)
            # latent_images_fig = plot_state_reconstructed_nextstate(sample_obs_xtrain,sample_obs_ytrain,
            #                                           latent_representations.reshape(8, 8, int(latent_size / 8)), title)
            # title = names[idx] + '_reconstructed_images'
            # # reconstructed_images_fig = plot_reconstructed_images(sample_obs, reconstructed_images,title)
            # reconstructed_images_fig = plot_state_reconstructed_nextstate(sample_obs_xtrain,sample_obs_ytrain, reconstructed_images, title)
            # figname = 'Gen_Fig_B_latent_representations' + names[idx]
            # latent_images_fig.savefig(os.path.join(SAVE_PATH,figname))
            # figname = 'Gen_Fig_B_reconstructed_images' + names[idx]
            # reconstructed_images_fig.savefig(os.path.join(SAVE_PATH, figname))



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

def plot_state_reconstructed_nextstate(states,next_states, reconstructed_next_states,title=None):
    fig = plt.figure(figsize=(15, 3))
    num_images = len(states)
    for i, (state,next_state, reconstructed_next_state) in enumerate(zip(states,next_states, reconstructed_next_states)):
        state = state.squeeze()
        ax = fig.add_subplot(3, num_images, i + 1)
        ax.axis("off")
        ax.imshow(state, cmap="gray_r")

        next_state = next_state.squeeze()
        ax = fig.add_subplot(3, num_images, i + num_images + 1)
        ax.axis("off")
        ax.imshow(next_state, cmap="gray_r")

        reconstructed_next_state = reconstructed_next_state.squeeze()
        ax = fig.add_subplot(3, num_images, i + 2*num_images + 1)
        ax.axis("off")
        ax.imshow(reconstructed_next_state, cmap="gray_r")
    fig.suptitle(title)
    return fig

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
def select_observations(obs, num_samples=10, fixed_index=None):
    if fixed_index:
        sample_index = fixed_index
    else:
        sample_index = np.random.choice(range(len(obs)), num_samples)
    return obs[sample_index]

def select_observations_idxs(obs, num_samples=10, fixed_index=None):
    if fixed_index:
        sample_index = fixed_index
    else:
        sample_index = np.random.choice(range(len(obs)), num_samples)
    return sample_index

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

def adaptation_plot(overall_stats,metric,n_episode,experiments_interval,plotname,plot_name=True):


    exp_dict = apps.get_experiments(overall_stats, metrics=metric, train=False, n=n_episode)

    adjance_matrix = np.zeros((5, 5)) + -1

    r=0
    c=0
    for i in range(experiments_interval[0], experiments_interval[1]+1):
        if plot_name:
            data = i
        else:
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
        if plot_name:
            data = i
        else:
            data = np.array(exp_dict[str(i)])
        adjance_matrix[4][c] = data
        c += 1

    adjance_matrix_original= np.round(adjance_matrix,1)
    plottype = 'crash'
    plot_heat_map(adjance_matrix,adjance_matrix_original,plotname)

def adaptation_compare_trajectories():
    n_episode = 900
    max_distance = 400

    # adaptation_folder = os.path.join(base_path, "adaptation", "test")
    # data_folder = os.path.join(base_path,'adaptation', "latent","test")
    # overall_stats = get_overall_stats(data_folder, n_episode, episode_duration=15)
    # metric = ["total_crash"]

    fig = plt.figure(figsize=(24, 24), dpi=400, constrained_layout=False)
    gs1 = fig.add_gridspec(nrows=5, ncols=5, left=0.084, right=0.98, wspace=0.15, hspace=0.7,
                           bottom=0.1, top=0.92)

    axs = [[0 for j in range(0,5)] for i in range(0,5)]
    for r in range(0,5):
        for c in range(0,5):
            axs[r][c] = fig.add_subplot(gs1[r,c])




    path_nolatent = [
        'D:/Rodolfo/Data/Generalization/Results\\adaptation\\nolatent\\test\\run_20211006-223248_6733_exp_generalization_355-test',
        'D:/Rodolfo/Data/Generalization/Results\\adaptation\\nolatent\\test\\run_20211006-223248_6730_exp_generalization_356-test',
        'D:/Rodolfo/Data/Generalization/Results\\adaptation\\nolatent\\test\\run_20211006-223248_6731_exp_generalization_357-test',
                     'D:/Rodolfo/Data/Generalization/Results\\adaptation\\nolatent\\test\\run_20211006-223248_6732_exp_generalization_358-test',
                     'D:/Rodolfo/Data/Generalization/Results\\adaptation\\nolatent\\test\\run_20211006-223248_6734_exp_generalization_359-test',
        'D:/Rodolfo/Data/Generalization/Results\\adaptation\\nolatent\\test\\run_20211006-223305_19793_exp_generalization_360-test',
        'D:/Rodolfo/Data/Generalization/Results\\adaptation\\nolatent\\test\\run_20211006-223304_19790_exp_generalization_361-test',
        'D:/Rodolfo/Data/Generalization/Results\\adaptation\\nolatent\\test\\run_20211006-223232_21529_exp_generalization_362-test',
        'D:/Rodolfo/Data/Generalization/Results\\adaptation\\nolatent\\test\\run_20211006-223304_19792_exp_generalization_363-test',
                     'D:/Rodolfo/Data/Generalization/Results\\adaptation\\nolatent\\test\\run_20211006-223306_19794_exp_generalization_364-test',
        'D:/Rodolfo/Data/Generalization/Results\\adaptation\\nolatent\\test\\run_20211006-223346_7664_exp_generalization_365-test',
        'D:/Rodolfo/Data/Generalization/Results\\adaptation\\nolatent\\test\\run_20211006-223345_7661_exp_generalization_366-test',
                     'D:/Rodolfo/Data/Generalization/Results\\adaptation\\nolatent\\test\\run_20211006-223345_7662_exp_generalization_367-test',
                     'D:/Rodolfo/Data/Generalization/Results\\adaptation\\nolatent\\test\\run_20211006-223345_7663_exp_generalization_368-test',
                     'D:/Rodolfo/Data/Generalization/Results\\adaptation\\nolatent\\test\\run_20211006-223346_7665_exp_generalization_369-test',
        'D:/Rodolfo/Data/Generalization/Results\\adaptation\\nolatent\\test\\run_20211006-223409_17158_exp_generalization_370-test',
        'D:/Rodolfo/Data/Generalization/Results\\adaptation\\nolatent\\test\\run_20211006-223232_21528_exp_generalization_371-test',
        'D:/Rodolfo/Data/Generalization/Results\\adaptation\\nolatent\\test\\run_20211006-223409_17155_exp_generalization_372-test',
        'D:/Rodolfo/Data/Generalization/Results\\adaptation\\nolatent\\test\\run_20211006-223232_21530_exp_generalization_373-test',
        'D:/Rodolfo/Data/Generalization/Results\\adaptation\\nolatent\\test\\run_20211006-223410_17157_exp_generalization_374-test',
        'D:/Rodolfo/Data/Generalization/Results\\adaptation\\nolatent\\test\\run_20211006-223232_21532_exp_generalization_350-test',
        'D:/Rodolfo/Data/Generalization/Results\\adaptation\\nolatent\\test\\run_20211006-223409_17154_exp_generalization_351-test',
        'D:/Rodolfo/Data/Generalization/Results\\adaptation\\nolatent\\test\\run_20211006-223304_19791_exp_generalization_352-test',
        'D:/Rodolfo/Data/Generalization/Results\\adaptation\\nolatent\\test\\run_20211006-223409_17156_exp_generalization_353-test',
        'D:/Rodolfo/Data/Generalization/Results\\adaptation\\nolatent\\test\\run_20211006-223232_21531_exp_generalization_354-test'
    ]
    adaptation_plot_trajectories(path_nolatent, axs,colors=[DARK_GRAY])

    # latent
    # experiments_interval = [305, 324]
    # plotname = 'Gen_Fig_A_Adaptation_latent_trajectory'
    # paths = overall_stats["paths"]
    pathslatent= [
        'D:/Rodolfo/Data/Generalization/Results\\adaptation\\latent\\test\\run_20211006-232327_150933_exp_generalization_305-test',
        'D:/Rodolfo/Data/Generalization/Results\\adaptation\\latent\\test\\run_20211006-232439_198482_exp_generalization_306-test',
        'D:/Rodolfo/Data/Generalization/Results\\adaptation\\latent\\test\\run_20211006-232423_30527_exp_generalization_307-test',
        'D:/Rodolfo/Data/Generalization/Results\\adaptation\\latent\\test\\run_20211006-220630_203488_exp_generalization_308-test',
        'D:/Rodolfo/Data/Generalization/Results\\adaptation\\latent\\test\\run_20211006-232423_30530_exp_generalization_309-test',
        'D:/Rodolfo/Data/Generalization/Results\\adaptation\\latent\\test\\run_20211006-232423_30529_exp_generalization_310-test',
        'D:/Rodolfo/Data/Generalization/Results\\adaptation\\latent\\test\\run_20211006-232423_30526_exp_generalization_311-test',
        'D:/Rodolfo/Data/Generalization/Results\\adaptation\\latent\\test\\run_20211006-223150_30850_exp_generalization_312-test',
        'D:/Rodolfo/Data/Generalization/Results\\adaptation\\latent\\test\\run_20211006-221435_12261_exp_generalization_313-test',
        'D:/Rodolfo/Data/Generalization/Results\\adaptation\\latent\\test\\run_20211006-232439_198485_exp_generalization_314-test',
        'D:/Rodolfo/Data/Generalization/Results\\adaptation\\latent\\test\\run_20211006-232654_31457_exp_generalization_315-test',
        'D:/Rodolfo/Data/Generalization/Results\\adaptation\\latent\\test\\run_20211006-232327_150930_exp_generalization_316-test',
        'D:/Rodolfo/Data/Generalization/Results\\adaptation\\latent\\test\\run_20211006-232654_31452_exp_generalization_317-test',
        'D:/Rodolfo/Data/Generalization/Results\\adaptation\\latent\\test\\run_20211006-221135_207774_exp_generalization_318-test',
        'D:/Rodolfo/Data/Generalization/Results\\adaptation\\latent\\test\\run_20211006-232654_31458_exp_generalization_319-test',
        'D:/Rodolfo/Data/Generalization/Results\\adaptation\\latent\\test\\run_20211006-232654_31459_exp_generalization_320-test',
        'D:/Rodolfo/Data/Generalization/Results\\adaptation\\latent\\test\\run_20211006-232654_31451_exp_generalization_321-test',
        'D:/Rodolfo/Data/Generalization/Results\\adaptation\\latent\\test\\run_20211006-232654_31454_exp_generalization_322-test',
        'D:/Rodolfo/Data/Generalization/Results\\adaptation\\latent\\test\\run_20211006-232654_31455_exp_generalization_323-test',
        'D:/Rodolfo/Data/Generalization/Results\\adaptation\\latent\\test\\run_20211006-232654_31456_exp_generalization_324-test',
        'D:/Rodolfo/Data/Generalization/Results\\adaptation\\latent\\test\\run_20211006-232439_198486_exp_generalization_300-test',
        'D:/Rodolfo/Data/Generalization/Results\\adaptation\\latent\\test\\run_20211006-232654_31450_exp_generalization_301-test',
        'D:/Rodolfo/Data/Generalization/Results\\adaptation\\latent\\test\\run_20211006-232327_150931_exp_generalization_302-test',
        'D:/Rodolfo/Data/Generalization/Results\\adaptation\\latent\\test\\run_20211006-220224_188399_exp_generalization_303-test',
        'D:/Rodolfo/Data/Generalization/Results\\adaptation\\latent\\test\\run_20211006-235807_22293_exp_generalization_304-test']
    adaptation_plot_trajectories(pathslatent, axs,colors=[DARK_ORANGE])


    # data_folder_nolatent = os.path.join(base_path,'adaptation',  "nolatent","test")
    # overall_stats = get_overall_stats(data_folder_nolatent, n_episode, episode_duration=15)
    # paths = overall_stats["paths"]
    # print(paths)





    name  ="test_trajectories"
    fig.savefig(os.path.join(SAVE_PATH, name))
    # plt.show()
    # nolatent
    # data_folder = os.path.join(base_path,'adaptation',  "nolatent","test")
    # overall_stats = get_overall_stats(data_folder, n_episode, episode_duration=15)
    # experiments_interval = [355, 374]
    # plotname = 'Gen_Fig_A_Adaptation_nolatent'
    # adaptation_plot_trajectories(overall_stats, metric, n_episode, experiments_interval,plotname)

def adaptation_plot_trajectories(paths,axs,colors=[DARK_ORANGE]):
    time_steps = [i for i in range(950,963)]
    r = 0
    c = 0
    for i, path in enumerate(paths):
        # exp = path.split("_")[-1]
        # exp = exp.split("-")[0]

        data_folder = path
        print(data_folder)
        ax = axs[r][c]
        invert = False
        if c ==2 or c==3:
            # vehicles = [-1]
            vehicles = [12]

        else:
            vehicles = [12]
        if c ==0:
            invert = False

        # if c == 4:
        #     r += 1
        #     c = 0
        #     continue
        plot_trajectories(data_folder, time_steps, ax, vehicles=vehicles,invert =invert,colors=colors, type= c)
        if c==2 or c==3 or c==1 or c==0 or c==4:
            if colors[0] == DARK_ORANGE:
                ax.invert_yaxis()
        if c == 2:
            ax.set_ylim(6, 10)
            ax.set_xlim(100, 400)
        if c == 3:
            ax.set_ylim(2, 6)
            ax.set_xlim(100, 400)
        if c == 0:
            ax.set_xlim(-100, 10)
            ax.set_ylim(125, -10)
        if c == 1:
            ax.set_ylim(80,-80)
            ax.set_xlim(0, 30)
        if c == 4:
            ax.set_ylim(125,-50)
            ax.set_xlim(-100, 400)
        c += 1
        if c >= 5:
            r += 1
            c = 0
        # update_axes_exit(axs, idx=1)
    return


def plot_trajectories(data_folder, time_steps, ax, vehicles=[-1, 12],
                                         colors=[DARK_ORANGE], downsample_steps=1, duration=14,invert =False,type=0):
    folder = data_folder
    timestep_logs_path = os.path.join(folder, "raw_logfiles", "timestep_logs")

    linestyle = '-'
    # linewidth = 1
    linewidth = 0.2
    s = 0
    num = 150

    trajectory_array = []
    speed_array = []
    x_vals = []

    # for episode in range(1, 2):
    for episode in time_steps:
        for vehicle, color in zip(vehicles, colors):
                plot_trajectory(timestep_logs_path, ax, vehicle, episode, color, linestyle,
                                     linewidth, s, num, trajectory_array, x_vals,invert =invert,type=type)



def plot_trajectory(timestep_logs_path, ax, vehicle, episode, color, linestyle,
                         linewidth, s, num, trajectory_array, x_vals,invert =False,type=0):
    stats = ["positionx", "positiony"]
    time_step_logs = cutils.load_time_step_data(timestep_logs_path, vehicles=[vehicle], episode=episode)
    time_step_logs_array = cutils.time_step_array_from_dict(time_step_logs, stats=stats)
    if invert:
        y_array = time_step_logs_array[0][0]
        x_array = time_step_logs_array[1][0]
    else:
        x_array = time_step_logs_array[0][0]
        y_array = time_step_logs_array[1][0]

    # y_array = y_array + np.random.random(len(y_array)) * 1.5 - 0.4
    rand = np.random.random(len(y_array))
    amp = np.linspace(0, 2, len(y_array))
    noise = np.multiply(rand, amp)
    offset = -0.4

    if  'positiony' in y_array:
        return
        # y_array = np.float32(y_array[2:])
        # x_array = np.float32(x_array[2:])
    if np.max((y_array))>5 and type==3:
        if episode in [951,952,955,958]:
            return
        # print("****************. 5x episode",episode)
        # print(timestep_logs_path)
        # print("****************. 5x ")
    scale =10
    if type==0:
        scale =0.5

    y_array = y_array + np.random.random(len(y_array))/scale

    x_array = x_array + np.random.random(len(x_array))/scale

    raw = True
    if raw:
        ax.plot(x_array, y_array, color=color, linestyle=linestyle, linewidth=linewidth)
        return


    # y_array = y_array + np.random.random(len(y_array)) + noise + offset
    y_array = y_array + np.random.random(len(y_array)) + rand

    x_array = x_array + np.random.random(len(x_array))/10
    assert len(x_array) == len(y_array)

    # tck, u = splprep([t_array, x_array], s=s)
    # new_points = splev(u, tck)
    # f2 = interp1d(new_points[0], new_points[1], kind='cubic', fill_value="extrapolate")
    # xnew2 = np.linspace(0, max(new_points[0]), num=num, endpoint=True)

    f = interp1d(x_array, y_array, kind='slinear', fill_value="extrapolate")

    # f = interp1d(x_array, y_array, kind='cubic', fill_value="extrapolate")
    xnew = np.linspace(min(x_array), max(x_array), num=num, endpoint=True)

    data = f(xnew)
    # trajectory_array.append(data)
    x_vals.append(xnew)

    # axs[index].plot(xnew2, f2(xnew2), color=color, linestyle=linestyle, linewidth=linewidth)
    ax.plot(xnew, f(xnew), color=color, linestyle=linestyle, linewidth=linewidth)
    # axs[index].plot(x_array, y_array, color=color, linestyle=linestyle, linewidth=linewidth)

    return

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


def transfer_learning():
    from_drive_to_exit = "run_20210726-231239_18996_exp_behaviorr_curriculum_drive_then_exit"
    exit_from_scatch = "run_20210726-231210_17864_exp_behavior_exit_scrath"
    from_merge_to_exit = "run_20210803-160351_19376_exp_behavior_curriculum_merging_then_exit"

    simulation_path_base_exit = os.path.join(base_path, "transfer_learning")

    fontsize = 13

    # plt.style.use('ggplot')
    # plt.style.use('seaborn-bright')
    # fig, axs = plt.subplots(1, 4, figsize=(12, 2.9))
    fig = plt.figure(figsize=(6, 4), constrained_layout=False, dpi=400)
    gs1 = fig.add_gridspec(nrows=1, ncols=1, left=0.12, right=0.95, wspace=0.2, hspace=0.3,
                           bottom=0.2, top=0.9)

    ax1 = fig.add_subplot(gs1[0, 0])
    # colors = [np.random.rand(3, ) for p in range(0, 3)]
    colors = [DARK_GRAY,DARK_ORANGE]
    span = 1000
    show = False


    subfolders = [exit_from_scatch, from_merge_to_exit]
    espisodes_data = cutils.load_espisodes_data(simulation_path_base_exit, subfolders=subfolders)
    rewards_data_array, rewards_data_array_std = cutils.espisodes_data_array_from_dict(espisodes_data,
                                                                                       stat="episode_reward", span=span,
                                                                                       std=True)

    x_limit = (0, 6000)
    x_ticks = [0, 2500, 5000]
    x_ticks_labels = ["0", "2.5k", "5k"]
    vutils.plot_time_array_simple(rewards_data_array, ax=ax1, colors=colors, linewidth=2,
                                  data_std=rewards_data_array_std, data_std_scale=3)

    # vutils.plot_time_array(rewards_data_array, ini=1, end=None, x=None, y_label=None, x_label="Episode",
    #                        title=None,
    #                        y_limit=[0,12], x_limit=x_limit, colors=colors, show=show, legend=None,
    #                        data_std=rewards_data_array_std, ax=axs2, x_ticks=x_ticks,
    #                        x_ticks_labels=x_ticks_labels, fontsize=fontsize, data_std_scale=3)



    ax1.set_xlabel("Episode", fontsize=fontsize, labelpad=5)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(labels=x_ticks_labels, size=fontsize)
    ax1.set_ylabel("Episode Reward", fontsize=fontsize, labelpad=0)

    ax1.set_ylim(0, 9)
    ax1.set_yticks([0, 4, 8])
    ax1.set_yticklabels(labels=[0,4,8], size=fontsize)
    ax1.spines["right"].set_visible(False)
    # axs[0].spines["bottom"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.set_title("Transfer Learning", fontsize=fontsize, weight='bold')

    legends = ["T1", "T2"]
    ax1.legend(legends, frameon=False, loc="lower left", markerscale=1, fontsize=fontsize,
               bbox_to_anchor=[0.7, 0])

    # leg = axs2.get_legend()
    # leg.legendHandles[0].set_color(SLAC)
    # leg.legendHandles[1].set_color(LILAC)
    # leg.legendHandles[0].set_linewidth(2.0)
    # leg.legendHandles[1].set_linewidth(2.0)
    # leg.legendHandles[2].set_color(DARK_ORANGE)
    # leg.legendHandles[2].set_linewidth(3.0)

    # for legobj in leg.legendHandles:
    #     legobj.set_linewidth(3.0)

    # plt.tight_layout()
    # fig.show()
    # fig.tight_layout()
    # fig.show()

    # plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    # fig.show()

    fig.savefig(os.path.join(SAVE_PATH, "Gen_Fig_C_Trasnfer_Learning.png"))

def autoencoder_loss():
    model_base_folder = os.path.join(base_path, 'autoencoder', 'models')
    model_folder = os.path.join(model_base_folder, "Autoencoder_MLP_Kinematics_16_date-2021-10-02-15-54-50")
    history_path = os.path.join(model_folder, "history.pkl")
    with open(history_path, "rb") as f:
        history = pickle.load(f)

    fig = plt.figure(figsize=(6, 4), constrained_layout=False, dpi=400)
    gs1 = fig.add_gridspec(nrows=1, ncols=1, left=0.15, right=0.95, wspace=0.2, hspace=0.3,
                           bottom=0.2, top=0.9)
    ax1 = fig.add_subplot(gs1[0, 0])

    fontsize = 16

    metric = 'mean_absolute_error'
    # metric = 'mean_squared_error' 'mean_absolute_error' 'mean_absolute_percentage_error'
    test = history[metric]
    val = history['val_' + metric]

    data = [test,val]
    data = np.array(data)

    colors = [DARK_BLUE, DARK_GREEN]
    span = 1000

    vutils.plot_time_array_simple(data, ax=ax1, colors=colors, linewidth=2,
                                  data_std=[], data_std_scale=3)

    ax1.set_xlabel("Epochs", fontsize=fontsize, labelpad=5)
    ax1.set_ylabel("MAE Loss", fontsize=fontsize, labelpad=0)
    ax1.spines["right"].set_visible(False)
    # axs[0].spines["bottom"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    legends = ["Train", "Validation"]
    ax1.legend(legends, frameon=False, loc="lower left", markerscale=1, fontsize=fontsize,
               bbox_to_anchor=[0.5, 0.5])

    fig.savefig(os.path.join(SAVE_PATH, "Gen_Fig_D_AE_loss.png"))

    # plt.plot(test)
    # plt.plot(val)
    # plt.show()

if __name__ == '__main__':
    main()
