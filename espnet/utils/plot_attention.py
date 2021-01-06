import os

# define function for plot prob and att_ws
def plot_and_save(array, figname, figsize=(6, 4), dpi=150):
    import matplotlib.pyplot as plt

    shape = array.shape
    if len(shape) == 1:
        # for eos probability
        plt.figure(figsize=figsize, dpi=dpi)
        plt.plot(array)
        plt.xlabel("Frame")
        plt.ylabel("Probability")
        plt.ylim([0, 1])
    elif len(shape) == 2:
        # for tacotron 2 attention weights, whose shape is (out_length, in_length)
        plt.figure(figsize=figsize, dpi=dpi)
        plt.imshow(array, aspect="auto")
        plt.xlabel("Input")
        plt.ylabel("Output")
    elif len(shape) == 4:
        # for transformer attention weights,
        # whose shape is (#leyers, #heads, out_length, in_length)
        plt.figure(figsize=(figsize[0] * shape[0], figsize[1] * shape[1]), dpi=dpi)
        for idx1, xs in enumerate(array):
            for idx2, x in enumerate(xs, 1):
                plt.subplot(shape[0], shape[1], idx1 * shape[1] + idx2)
                plt.imshow(x, aspect="auto")
                plt.xlabel("Input")
                plt.ylabel("Output")
    else:
        raise NotImplementedError("Support only from 1D to 4D array.")
    plt.tight_layout()
    if not os.path.exists(os.path.dirname(figname)):
        # NOTE: exist_ok = True is needed for parallel process decoding
        os.makedirs(os.path.dirname(figname), exist_ok=True)
    plt.savefig(figname)
    plt.close()