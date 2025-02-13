import datetime
import pathlib
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# ------------------
# plot fonts
# ------------------

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('figure', dpi=300)

# ------------------
# plot colors
# ------------------

cmap = mpl.colormaps['viridis']
pretty_colors = cmap(np.linspace(0, 1, 21))

# ------------------
# plot labels
# ------------------

pretty_target_labels = {
    'voc': r'$V_{oc}$',
    'voc mean': r'$\overline{V_{oc}^{grid}}$',
    'voc max': r'$V_{oc,max}^{grid}$',
    'voc std': r'$V_{oc,std}^{grid}$',
    'fom': r'$\rm{FoM}^{grid}$',
    'log jsc': r'$J_{sc}$',
    'Cu2O fraction': r'$f_{Cu_2O}$'
}

def pretty_target_label(target: str, grid: str) -> str:
    if target in pretty_target_labels:
        return pretty_target_labels[target].replace('grid', grid)
    return f'{target} ({grid})'

# ------------------
# true pred axes
# ------------------

def true_pred_ax(ax: plt.Axes, title: str) -> None:
    ax.plot([0, 1], [0, 1], zorder=-2, ls='--', color='k')
    ax.set_xlabel(r'$y_{norm}$')
    ax.set_ylabel(r'$\hat{y}_{norm}$')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_aspect('equal')
