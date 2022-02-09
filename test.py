import  os, pickle
import post_process.visualization.visualization_utils as vutils
from post_process.applications.applications import *
def plot_loss():
    with open("history.pkl", "rb") as f:
        history = pickle.load(f)

    print("hi")

    fig = plt.figure(figsize=(6, 4), constrained_layout=False, dpi=400)
    gs1 = fig.add_gridspec(nrows=1, ncols=1, left=0.15, right=0.95, wspace=0.2, hspace=0.3,
                           bottom=0.2, top=0.9)
    ax1 = fig.add_subplot(gs1[0, 0])

    fontsize = 16

    metric = 'loss'
    # metric = 'mean_squared_error' 'mean_absolute_error' 'mean_absolute_percentage_error'
    test = history[metric]
    val = history['val_' + metric]

    data = [test, val]
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

    plt.show()


