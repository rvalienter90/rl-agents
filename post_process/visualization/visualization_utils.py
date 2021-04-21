import os
from numpy import genfromtxt
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import post_process.calculations.calculation_utils as cutils

import matplotlib.image as mpimg
import cv2
import time , math
import os, fnmatch
#
#
# def subplots_3d_surf_bar(x, y, z, x_label='x', y_label='y', z_label='z', title='3d plots'):
#

def simple_3d_bar_plot(x, y, z, x_label='x', y_label='y', z_label='z',
                       title='simple 3d bar plot' , zlim3d = None,xlim3d = None,ylim3d = None, show=False,  ax=None, info=None):
    # Turn interactive plotting off
    plt.ioff()

    if ax is None:
        fig = plt.figure()
        # fig, ax = plt.subplots()
        ax = fig.gca(projection='3d')
    else:
        fig = None



    zz = np.zeros_like(z)
    dx = np.ones_like(x) * 0.5
    dy = np.ones_like(y) * 0.5
    dz = z  # Height of each bar

    ax.bar3d(x, y, zz, dx, dy, dz, shade=True)

    ax.set(xlabel=x_label, ylabel=y_label, zlabel=z_label, title=title)
    if not zlim3d:
        zlim3d=max(z) + 1
    ax.set_zlim3d(0, zlim3d)
    if xlim3d:
        # ax.set_xlim3d(-22,0)
        ax.set_xlim3d(xlim3d)
    if ylim3d:
        # ax.set_ylim3d(0, 12)
        ax.set_ylim3d(ylim3d)
    ax.view_init(20, 45)

    # adding text to the bars
    for i, values in enumerate(zip(x, y)):
        x0 = values[0] + 1.1
        y0 = values[1] + 0.45
        z0 = z[i] - 0.5

        if info is None:
            tmp_text = "{:.0f}".format(z[i])
        else:
            tmp_text = "{:.0f}".format(z[i]) + "-" + info[i]
        ax.text(x0, y0, z0, tmp_text, 'y', color='white', fontsize=7,
                bbox=dict(facecolor='black', alpha=1))

    if show:
        # figManager = plt.get_current_fig_manager()
        # figManager.window.showMaximized()
        # figManager.full_screen_toggle()
        # fig.canvas.window().statusBar().setVisible(False)
        plt.show()

    return fig


def simple_surf_plot(x, y, z, x_label='x', y_label='y', z_label='z', title='simple surf plot'):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True, cmap='copper')

    ax.set(xlabel=x_label, ylabel=y_label, zlabel=z_label, title=title)
    ax.view_init(30, 45)

    plt.show()
    return  fig


def simple_2d_plot(y, x=None, y_label='y', x_label='x', title='simple 2D plot'):
    if x is None:
        x = np.linspace(1, len(y), len(y))

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel=x_label, ylabel=y_label, title=title)
    ax.grid()
    plt.show()

    return fig

# def plot_2dbar(array, legend =None, title='2dbar plot',
#                     y_limit=None, x_limit=None, colors=None, show=False, ax=None , figsize = [8,3]):
#     """
#     1 plot n variables vs time (from ini to end)
#
#     Input
#     :param array n_variables vs x
#
#     :return figure
#     """
#     if show is False:
#         # Turn interactive plotting off
#         plt.ioff()
#     if ax is None:
#         fig, ax = plt.subplots(figsize=figsize)
#     else:
#         fig = None
#
#     if colors is None:
#         colors = [np.random.rand(3, ) for p in range(0, len(array))]
#
#     ax.set(xlabel=x_label, ylabel=y_label, title=title)
#     # array_p = array.reshape(len(array[0]),-1)
#     # ax.plot(x_axes, array_p)
#     for p in range(0, len(array)):
#         data = array[p][ini-1:end]
#         ax.plot(x_axes,data, color=colors[p])
#     if legend:
#         ax.legend(legend)
#
#     if show:
#         plt.show()
#
#     return fig


def plot_time_array(array, ini=1, end=None, x=None, y_label="data", x_label="time_step", title='time step plot',
                    y_limit=None, x_limit=None, colors=None, legend=None, show=False, ax=None , data_mean =None , data_std = None , x_axes =None , remove_border =True , x_ticks =None , x_ticks_labels = None , fontsize=10 , linewidth =1.5):
    """
    1 plot n variables vs time (from ini to end)

    Input
    :param array n_variables vs x

    :return figure
    """
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    # plt.style.use('ggplot')
    plt.style.use('bmh')

    if show is False:
        # Turn interactive plotting off
        plt.ioff()
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    if x_axes is None:
        if x:
            if end is None:
                end = len(x)
            x_axes = x[ini:end]
        else:
            if end is None:
                end = len(array[0])
            x_axes = np.arange(ini, end + 1)

    if y_limit:
        ax.set_ylim(y_limit)
    if x_limit:
        ax.set_xlim(x_limit)
    # if legend is None:
    #     legend = [str(i) for i in range(0,len(array))]

    if colors is None:
        colors = [np.random.rand(3, ) for p in range(0, len(array))]

    if remove_border:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)



    ax.set(xlabel=x_label, ylabel=y_label, title=title)
    ax.set_ylabel(y_label , fontsize =fontsize)
    ax.set_xlabel(x_label, fontsize =fontsize)
    # array_p = array.reshape(len(array[0]),-1)
    # ax.plot(x_axes, array_p)
    # for p in range(0, len(array)):
    for p in [0,2,1]:
        data = array[p][ini-1:end]
        # Calculate mean and standard deviation
        # data_mean = np.mean(data)
        # data_std = np.std(data)
        # data_std = np.std(cutils.rolling_window(data, 1000), 1)
        if p ==2:
            linewidthplot =3
        else:
            linewidthplot = linewidth
        ax.plot(x_axes,data, color=colors[p] ,linewidth=linewidthplot)

        # ax.fill_between(x_axes, data_mean - data_std, data_mean + data_std, color="gainsboro")
        if data_std is not None:
            data_std_i = data_std[p][ini-1:end]/3
            ax.fill_between(x_axes, data - data_std_i, data + data_std_i,
                            alpha=0.25, facecolor=colors[p],
                            linewidth=4, linestyle='dashdot', antialiased=True)
    if legend:
        ax.legend(legend, frameon=False ,fontsize =fontsize)
        # legend = ax.legend()
        # legend.get_frame().set_facecolor('none')

    if x_ticks == "Off":
        ax.axes.xaxis.set_visible(False)
    elif x_ticks:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks_labels , fontsize =fontsize)
        # ax.set_yticklabels( fontsize =fontsize)

    if show:
        plt.show()

    return fig


def plot_time_array(array, ini=1, end=None, x=None, y_label="data", x_label="time_step", title='time step plot',
                    y_limit=None, x_limit=None, colors=None, legend=None, show=False, ax=None , data_mean =None , data_std = None , x_axes =None , remove_border =True , x_ticks =None , x_ticks_labels = None , fontsize=10 , linewidth =1.5):
    """
    1 plot n variables vs time (from ini to end)

    Input
    :param array n_variables vs x

    :return figure
    """
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    # plt.style.use('ggplot')
    plt.style.use('bmh')

    if show is False:
        # Turn interactive plotting off
        plt.ioff()
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    if x_axes is None:
        if x:
            if end is None:
                end = len(x)
            x_axes = x[ini:end]
        else:
            if end is None:
                end = len(array[0])
            x_axes = np.arange(ini, end + 1)

    if y_limit:
        ax.set_ylim(y_limit)
    if x_limit:
        ax.set_xlim(x_limit)
    # if legend is None:
    #     legend = [str(i) for i in range(0,len(array))]

    if colors is None:
        colors = [np.random.rand(3, ) for p in range(0, len(array))]

    if remove_border:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)



    ax.set(xlabel=x_label, ylabel=y_label, title=title)
    ax.set_ylabel(y_label , fontsize =fontsize)
    ax.set_xlabel(x_label, fontsize =fontsize)
    # array_p = array.reshape(len(array[0]),-1)
    # ax.plot(x_axes, array_p)
    for p in range(0, len(array)):
    # for p in [0,2,1]:
        data = array[p][ini-1:end]
        # Calculate mean and standard deviation
        # data_mean = np.mean(data)
        # data_std = np.std(data)
        # data_std = np.std(cutils.rolling_window(data, 1000), 1)
        if p ==2:
            linewidthplot =3
        else:
            linewidthplot = linewidth
        ax.plot(x_axes,data, color=colors[p] ,linewidth=linewidthplot)

        # ax.fill_between(x_axes, data_mean - data_std, data_mean + data_std, color="gainsboro")
        if data_std is not None:
            data_std_i = data_std[p][ini-1:end]/3
            ax.fill_between(x_axes, data - data_std_i, data + data_std_i,
                            alpha=0.25, facecolor=colors[p],
                            linewidth=4, linestyle='dashdot', antialiased=True)
    if legend:
        ax.legend(legend, frameon=False ,fontsize =fontsize)
        # legend = ax.legend()
        # legend.get_frame().set_facecolor('none')

    if x_ticks == "Off":
        ax.axes.xaxis.set_visible(False)
    elif x_ticks:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks_labels , fontsize =fontsize)
        # ax.set_yticklabels( fontsize =fontsize)

    if show:
        plt.show()

    return fig



def barplot_annotate_brackets_vertical(num1, num2, data, center, height, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None):
    """
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    plt.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    plt.text(*mid, text, **kwargs)

def barplot_annotate_brackets_horizontal(num1, num2, data, center, height, yerr=None, dh=.03, barh=.01, fs=None, maxasterix=None, dx_tex=0.2, dy_tex=0.2, ax =None, ptext = "",r_line=0,l_line=0):
    """
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    if ax:
        ax_x0, ax_x1 = ax.get_xlim()
    else:
        ax_x0, ax_x1 = plt.gca().get_xlim()

    dh *= (ax_x1 - ax_x0)
    barh *= (ax_x1 - ax_x0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y - r_line, y+barh, y+barh, y - l_line]
    # mid = ((lx+rx)/2, y+barh)
    mid = ( y + barh + dx_tex, (lx + rx) / 2 + dy_tex)

    if ax:
        ax.plot(bary, barx, c='black',linewidth = 0.5)
    else:
        plt.plot(bary, barx, c='black')


    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    if ax:
        px = (mid[0] + 100, mid[1] )
        ps = (mid[0] +20, mid[1] -0.2)
        ax.text(*px, text, **kwargs, rotation=-90, size=10, fontweight=1,  weight=1000)

        ax.text(*ps , ptext, **kwargs, rotation=-90, size=10, fontweight=1)
    else:
        plt.text(*mid, text, **kwargs, rotation=-90)


def subplot_plot_time_array(array3D, ini=1, end=None, x=None, y_label=None, x_label="time_step", title='time step plot',
                            y_limit=None, x_limit=None, colors=None, legend=None, show=False, tight_layaut =True , figsize = [20,3]):
    """
    m subplots of  n variables vs time (from ini to end)

    Input
    :param array3D mxnxt  m_plots vs n_variables vs time

    :return figure
    """

    if show is False:
        # Turn interactive plotting off
        plt.ioff()

    fig, axs = plt.subplots(len(array3D),1, figsize = (figsize[0],figsize[1]*len(array3D[0])))

    # TODO fix fig size
    # plt.figure(figsize=(6, 6*len(array3D)))
    if y_label is None:
        y_label = ["data" + str(i) for i in range(0, len(array3D))]

    if colors is None:
        colors = [np.random.rand(3, ) for p in range(0, len(array3D[0]))]

    legendx =None
    for plots in range(0, len(array3D)):
        if plots == len(array3D) - 1:
            legendx = legend
        if y_limit == "Auto":
            # TODO chekc reward nan in the first time
            data = array3D[plots]
            maxy = np.max(data)
            miny = np.min(data)

            if math.isnan(miny) or math.isnan(maxy):
                y_limittemp = None
            else:
                y_limittemp = [miny - 0.1*miny, maxy + 0.1*maxy]
        else:
            y_limittemp = y_limit
        plot_time_array(array3D[plots], ini=ini, end=end, x=x, y_label=y_label[plots], x_label=x_label,
                              title=None, y_limit=y_limittemp, x_limit=x_limit, colors=colors,
                              legend=legendx, show=False, ax=axs[plots])

    if tight_layaut:
        fig.tight_layout()
    # fig.show()
    if show:
        plt.show()
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        # fig.show()

    return fig

def get_videoimage_at_timestep(path, episode,time_step, max_time_step=15, show= False):
    """
    video capture of episode at time_step

    Input
    :path path to the run

    :return figure , rgb frame of video
    """

    # path = os.path.join("..", "..", "1.mp4")
    video_path=""
    for path, folders, files in os.walk(path):
        for file in files:
            # if fnmatch.fnmatch(file, '*.mp4'):
            if file.endswith(".mp4"):
                num_extract = ((file.split("video"))[-1].split("."))[0]
                num= int(num_extract)
                if num == episode:
                    video_path = os.path.join(path,file)
                    break
        break
    if video_path == "":
        print("No video found")
        return None

    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    # print("Number of frames: ", video_length)
    frame_freq = video_length / max_time_step
    frame_number = max(1, int(frame_freq * (time_step)))
    frame_number = min(frame_number, video_length)
    cap.set(1, frame_number - 1)
    res, frame = cap.read()

    try:
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except:
        print("no  image")
    finally:
        framergb = np.zeros((200,800,3))
    if show:
        fig,ax = plt.subplots(1,1)
        ax.imshow(framergb, interpolation='nearest')
        plt.pause(1)
        # cv2.imshow('window_name', frame)  # show frame on window
        # while True:
        #     ch = 0xFF & cv2.waitKey(1)  # Wait for a second
        #     if ch == "q":
        #         break

    # in matplotlib rgb
    return framergb


def save_videoimage_and_plot(path,array3D, episode, ini=1, end=None, x=None, y_label=None, x_label="time_step", title='time step plot',
                            y_limit=None, x_limit=None, colors=None, legend=None, show=False):
    """
    will save in path all the frames and subplots for the specific episode from timestep =ini to timestep=end

    Input
    :path path to the run

    :return
    """
    if show is False:
        # Turn interactive plotting off
        plt.ioff()

    plots_output_path = os.path.join(path,"video_plot_images")
    frames_output_path = os.path.join(plots_output_path, "frames")
    try:
        if not os.path.exists(plots_output_path):
            os.mkdir(plots_output_path)
        if not os.path.exists(frames_output_path):
             os.mkdir(frames_output_path)
    except OSError:
        pass

    if end is None:
        end = len(array3D[0][0])

    if colors is None:
        colors = [np.random.rand(3, ) for p in range(0, len(array3D[0]))]

    for i in range(ini+1, end+1):
        fig = subplot_plot_time_array(array3D, ini=ini, end=i, x=None, y_label=y_label, x_label=x_label,
                                title=None,
                                y_limit="Auto", x_limit=[ini-1, end+1], colors=colors, legend=legend, show=False)

        frame = get_videoimage_at_timestep(path, episode,i,  max_time_step=len(array3D[0][0]), show= False)

        out_file = "\episode_" + str(episode) + "_time_step_" + str(i)
        fig_out_path = plots_output_path +  out_file + "_fig"
        fig.savefig(fig_out_path + ".png", dpi=300)
        # fig.savefig(out_path, bbox_inches='tight', format='pdf', dpi=1000)
        # fig.savefig(fig_out_path + ".eps", bbox_inches='tight', format='eps', dpi=1000)

        frame_out_path = frames_output_path + out_file + "_frame.png"
        framebgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(frame_out_path, framebgr)





def time_step_plot_animation_video(path,array3D, episode, ini=1, end=None, x=None, y_label=None, x_label="time_step", title='time step plot',
                            y_limit="Auto", x_limit=None, colors=None, legend=None, show=False, tight_layaut =True , figsize = [20,3]):
    """
    will save in path all the frames and subplots for the specific episode from timestep =ini to timestep=end

    Input
    :path path to the run

    :return
    """
    if show is False:
        # Turn interactive plotting off
        plt.ioff()
    plots_output_path = os.path.join(path,"video_plot_images")
    frames_output_path = os.path.join(plots_output_path, "frames")
    try:
        if not os.path.exists(plots_output_path):
            os.mkdir(plots_output_path)
        if not os.path.exists(frames_output_path):
             os.mkdir(frames_output_path)
    except OSError:
        pass

    if end is None:
        end = len(array3D[0][0])

    if colors is None:
        colors = [np.random.rand(3, ) for p in range(0, len(array3D[0]))]

    if show:
        plt.ion()
        plt.show(block=False)
    fig, axs = plt.subplots(len(array3D) + 1, 1, figsize=(figsize[0], figsize[1] * len(array3D[0])))


    for i in range(ini+1, end+1):

        legendx = None
        for plots in range(0, len(array3D)):
            if plots == len(array3D) - 1:
                legendx = legend
            if y_limit == "Auto":
                # TODO chekc reward nan in the first time
                data = array3D[plots]
                maxy = np.max(data)
                miny = np.min(data)

                if math.isnan(miny) or math.isnan(maxy):
                    y_limittemp = None
                else:
                    y_limittemp = [miny - 0.1 * miny, maxy + 0.1 * maxy]
            else:
                y_limittemp = y_limit
            plot_time_array(array3D[plots], ini=ini, end=i, x=x, y_label=y_label[plots], x_label=x_label,
                            title=None, y_limit=y_limittemp, x_limit=[ini-1, end+1], colors=colors,
                            legend=legendx, show=False, ax=axs[plots])
        if tight_layaut:
            fig.tight_layout()

        # fig = subplot_plot_time_array(array3D, ini=ini, end=i, x=None, y_label=y_label, x_label=x_label,
        #                         title=None,
        #                         y_limit="Auto", x_limit=[ini-1, end+1], colors=colors, legend=legend, show=False)


        frame = get_videoimage_at_timestep(path, episode,i, max_time_step=len(array3D[0][0]), show= False)

        axs[len(array3D)].imshow(frame, interpolation='nearest')
        # plt.pause(1 / 15)

        out_file = "\episode_" + str(episode) + "_time_step_" + str(i)
        fig_out_path = plots_output_path +  out_file + "_fig"
        fig.savefig(fig_out_path + ".png", dpi=300)
        # fig.savefig(out_path, bbox_inches='tight', format='pdf', dpi=1000)
        # fig.savefig(fig_out_path + ".eps", bbox_inches='tight', format='eps', dpi=1000)

        frame_out_path = frames_output_path + out_file + "_frame.png"
        framebgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(frame_out_path, framebgr)

        if show:
            plt.show()
            plt.pause(1 / 15)

def time_step_plot_image(path,array3D, episode, ini=1, end=None, x=None, y_label=None, x_label="time_step", title='time step plot',
                            y_limit="Auto", x_limit=None, colors=None, legend=None, show=False, tight_layaut =True , figsize = [20,3]):
    """
    will save in path all the frames and subplots for the specific episode from timestep =ini to timestep=end

    Input
    :path path to the run

    :return
    """
    if show is False:
        # Turn interactive plotting off
        plt.ioff()
    plots_output_path = os.path.join(path,"video_plot_images")
    # frames_output_path = os.path.join(plots_output_path, "frames")
    try:
        if not os.path.exists(plots_output_path):
            os.mkdir(plots_output_path)
        # if not os.path.exists(frames_output_path):
        #      os.mkdir(frames_output_path)
    except OSError:
        pass

    if end is None:
        end = len(array3D[0][0])

    if colors is None:
        colors = [np.random.rand(3, ) for p in range(0, len(array3D[0]))]

    if show:
        plt.ion()
        plt.show(block=False)
    fig, axs = plt.subplots(len(array3D) + 1, 1, figsize=(figsize[0], figsize[1] * len(array3D[0])))



    legendx = None
    for plots in range(0, len(array3D)):
        if plots == len(array3D) - 1:
            legendx = legend
        if y_limit == "Auto":
            # TODO chekc reward nan in the first time
            data = array3D[plots]
            maxy = np.max(data)
            miny = np.min(data)

            if math.isnan(miny) or math.isnan(maxy):
                y_limittemp = None
            else:
                y_limittemp = [miny - 0.1 * miny, maxy + 0.1 * maxy]
        else:
            y_limittemp = y_limit
        plot_time_array(array3D[plots], ini=ini, end=end, x=x, y_label=y_label[plots], x_label=x_label,
                        title=None, y_limit=y_limittemp, x_limit=[ini-1, end+1], colors=colors,
                        legend=legendx, show=False, ax=axs[plots])
    if tight_layaut:
        fig.tight_layout()

    # fig = subplot_plot_time_array(array3D, ini=ini, end=i, x=None, y_label=y_label, x_label=x_label,
    #                         title=None,
    #                         y_limit="Auto", x_limit=[ini-1, end+1], colors=colors, legend=legend, show=False)


    frame = get_videoimage_at_timestep(path, episode,end, max_time_step=len(array3D[0][0]), show= False)

    axs[len(array3D)].imshow(frame, interpolation='nearest')
    # plt.pause(1 / 15)

    out_file = "\episode_" + str(episode) + "_time_step_" + str(end)
    fig_out_path = plots_output_path +  out_file + "_fig"
    fig.savefig(fig_out_path + ".png", dpi=300)
    # fig.savefig(out_path, bbox_inches='tight', format='pdf', dpi=1000)
    # fig.savefig(fig_out_path + ".eps", bbox_inches='tight', format='eps', dpi=1000)

    # frame_out_path = frames_output_path + out_file + "_frame.png"
    # framebgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(frame_out_path, framebgr)

    if show:
        plt.show()
        plt.pause(1 / 15)

    return fig


def time_step_plot_animation(time_step_path, ini=1, end=5, vehicles=[1], stats=["timestep_reward"], episode=1,
                             save_frames=True):
    A = np.random.rand(5, 5)

    plt.ion()
    plt.show(block=False)
    fig, axs = plt.subplots(2, len(stats))
    fig.suptitle('Reward components')
    time_step_data = cutils.load_time_step_data(time_step_path, vehicles, episode)
    x_axes = np.arange(ini, end)

    # plt.xlim((0, 20))

    for a in range(0, len(stats)):
        axs[0][a].set_xlim([ini - 1, end + 1])
        miny = float('inf')
        maxy = -float('inf')
        for v in vehicles:
            data = np.array(time_step_data[v][stats[a]])
            mintemp = np.min(data)
            maxtemp = np.max(data)
            miny = min(miny, mintemp)
            maxy = max(maxy, maxtemp)
        axs[0][a].set_ylim([miny - 0.1 * miny, maxy + 0.1 * maxy])

    colors = [np.random.rand(3, ) for v in vehicles]
    labels = ["v" + str(i) for i in vehicles]
    # for ax in axs.flat:
    #     ax.set(xlabel='time_step', ylabel='values')

    path = os.path.join("..", "..", "1.mp4")

    cap = cv2.VideoCapture(path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print("Number of frames: ", video_length)

    axi = plt.subplot(212)

    ret, frame = cap.read()

    axi.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), interpolation='nearest')
    # if framenum%15 ==0:
    for i in x_axes:
        framenum = 0
        # axs[len(stats)].imshow(A, interpolation='nearest')
        for a in range(0, len(stats)):
            for v in vehicles:
                data = time_step_data[v][stats[a]]
                # plt.gca().cla() # optionally clear axes
                # axs[a].plot(x_axes[ini:i], data_crop_np, c=colors[idx], label = labels[idx])
                axs[0][a].plot(x_axes[ini - 1:i + 1], np.array(data[ini - 1:i + 1]), c=colors[vehicles.index(v)])
                # axs[a].set_title(stats[a])
                axs[0][a].set(ylabel=stats[a])
        # leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
        # leg.get_frame().set_alpha(0.5)
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        axs[0][a].set(xlabel="time_steps")
        plt.legend(labels)
        plt.draw()
        # plt.pause(0.001)

        while framenum < 15:
            ret, frame = cap.read()
            if ret == True:
                axi.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), interpolation='nearest')
                plt.pause(1 / 15)
            else:
                break
            framenum += 1

    # plt.legend(labels)
    # plt.draw()
    # plt.pause(1)
    exit()

    # # Hide x labels and tick labels for top plots and y ticks for right plots.
    # for ax in axs.flat:
    #     ax.label_outer()

