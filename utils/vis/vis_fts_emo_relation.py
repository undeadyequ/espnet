import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.cm as cm

def vis_fts_emo_relation(id_fts_pred_lab_f):
    """
    id_fts_pred_lab_f:
    id, fts1, fts2, fts3, ... pred, lab
    wav1 ..
    """
    df = pd.read_csv(id_fts_pred_lab_f)

    id_fts_pred_lab_dict = df.set_index("id").T.to_dict("list")
    FEAT, PRED, LAB = 0, 1, 2
    ANG, HAP, SAD = 0, 1, 2
    id_fts_pred_lab_sorted_dict = sorted(id_fts_pred_lab_dict.items, key=lambda kv: kv[1][PRED][ANG])

    fts0 = [ft_pred_lab[FEAT][0] for id, ft_pred_lab in id_fts_pred_lab_sorted_dict]

    fts1 = [ft_pred_lab[FEAT][1] for id, ft_pred_lab in id_fts_pred_lab_sorted_dict]
    ang = [ft_pred_lab[PRED][0] for id, ft_pred_lab in id_fts_pred_lab_sorted_dict]

    fig = plt.figure()
    ax0 = fig.add_subplot(2, 3, 1)
    ax1 = fig.add_subplot(2, 3, 1)
    ...
    ax0.scatter(fts0, ang)
    ax0.line(fts0, ang)
    ax1.scatter(fts1, ang)
    ax1.line(fts1, ang)
    ...
    plt.savefig("fts_ang.png")


def vis_fts_emo_scatter(id_fts_pred_lab_f, out_img=None):
    """
    id_fts_pred_lab_f:
    id, fts1, fts2, fts3, ... lab
    wav1 ..
    """
    df = pd.read_csv(id_fts_pred_lab_f)
    id_fts_lab_dict = df.set_index("id").T.to_dict("list")
    #id_fts_pred_lab_sorted_dict = sorted(id_fts_pred_lab_dict.items, key=lambda kv: kv[1][PRED][ANG])

    colnames = ["id", "emo",
                "rmse", "rmse_std", "rmse_range",
                "harmonic", "harmonic_std"
                "pitch", "pitch_std", "pitch_range"]

    rmse = [ft_lab[1] for id, ft_lab in id_fts_lab_dict.items()]
    rmse_std = [ft_lab[2] for id, ft_lab in id_fts_lab_dict.items()]
    rmse_rng = [ft_lab[3] for id, ft_lab in id_fts_lab_dict.items()]

    harm = [ft_lab[4] for id, ft_lab in id_fts_lab_dict.items()]
    harm_std = [ft_lab[5] for id, ft_lab in id_fts_lab_dict.items()]

    pitch = [ft_lab[6] for id, ft_lab in id_fts_lab_dict.items()]
    pitch_std = [ft_lab[7] for id, ft_lab in id_fts_lab_dict.items()]
    pitch_rng = [ft_lab[8] for id, ft_lab in id_fts_lab_dict.items()]

    emo = [ft_lab[0] for id, ft_lab in id_fts_lab_dict.items()]

    #fts1 = [ft_pred_lab[FEAT][1] for id, ft_pred_lab in id_fts_pred_lab_sorted_dict]
    #ang = [ft_pred_lab[PRED][0] for id, ft_pred_lab in id_fts_pred_lab_sorted_dict]

    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)

    #ax1 = fig.add_subplot(2, 3, 1)

    plot_axis(ax1, rmse, rmse_std, emo, xlab="rmse", ylab="rmse_std", legend_title="emo")
    plot_axis(ax2, rmse, rmse_rng, emo, xlab="rmse", ylab="rmse_range", legend_title="emo")
    plot_axis(ax3, pitch, pitch_std, emo, xlab="pitch", ylab="pitch_std", legend_title="emo")
    plot_axis(ax4, pitch, pitch_rng, emo, xlab="pitch", ylab="pitch_range", legend_title="emo")
    plot_axis(ax5, harm, harm_std, emo, xlab="harm", ylab="harm_std", legend_title="emo")

    scatter1 = ax1.scatter(rmse, rmse_std, c=emo)
    legend1 = ax1.legend(*scatter1.legend_elements(), loc="upper left", title="emo", prop={'size': 5})
    ax1.add_artist(legend1)
    ax1.set_xlabel("rmse")
    ax1.set_ylabel("rmse_std")

    if out_img is None:
        out_img = id_fts_pred_lab_f[:-3] + "png"
    plt.savefig(out_img)


def plot_axis(ax1, x, y, c, xlab="", ylab="", legend_title=""):

    scatter1 = ax1.scatter(x, y, c=c)
    legend1 = ax1.legend(*scatter1.legend_elements(), loc="upper left", title=legend_title, prop={'size': 5})
    ax1.add_artist(legend1)
    ax1.set_xlabel(xlab)
    ax1.set_ylabel(ylab)
