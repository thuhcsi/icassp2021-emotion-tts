import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_alignment(alignment, path, info=None, title=None, text=None):
    """
    # Arguments
        text: the text str where each char is used to draw yticks
    """
    if text is None:
        figsize = None
        yticks = None
        ytick_labels = None
    else:
        yticks = np.arange(len(text))
        ytick_labels = list(text)
        ytick_labels[-1] = len(text)
        figsize = (0.02 * alignment.shape[1], 0.10 * len(text))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(alignment,
                   aspect='auto',
                   origin='lower',
                   interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.yticks(yticks, ytick_labels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, format='png')
    plt.close('all')


def plot_mel(mel, path, info=None, title=None, gt_mel=None):
    nrows = 1 if gt_mel is None else 2
    fig, ax = plt.subplots(nrows, squeeze=False)

    def plot(mel, ax, y_label='pred_freq'):
        im = ax.imshow(mel,
                       aspect='auto',
                       origin='lower',
                       interpolation='none')
        ax.set_ylabel(y_label)
        fig.colorbar(im, ax=ax)

    plot(mel.T, ax[0][0])  # mel shape [time_step, num_mels]
    ax[0][0].set_title(title)

    if gt_mel is not None:
        plot(gt_mel.T, ax[1][0], y_label='truth_freq')

    plt.xlabel(info or 'time step')
    plt.tight_layout()
    plt.savefig(path, format='png')
    plt.close('all')
