from sklearn.manifold import TSNE
from sklearn.datasets import load_iris,load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os


def vis_emolabs(pred_emo_dist, emo_gd, v_dim=2, method="tsne"):
    """

    Args:
        pred_emo_dist: Array-like, [n_sample, emo_num3]
        emo_gd:

    Returns:
    """
    emos = ["A"]*6
    emo_nums = len(pred_emo_dist[0])
    pred_emo_tsne = TSNE(n_components=2, random_state=emo_nums).fit_transform(pred_emo_dist)
    #X_pca = PCA(n_components=2).fit_transform(digits.data)

    plt.figure(figsize=(10, 5))
    plt.scatter(pred_emo_tsne[:, 0], pred_emo_tsne[:, 1], c=emo_gd)
    plt.legend(emos)
    plt.savefig("vis_emolabs.png")