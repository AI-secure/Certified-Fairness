import numpy as np
import matplotlib.pyplot as plt
from os import path
import os
import random

import matplotlib as mpl
import seaborn as sns


def init_style(sns_style='whitegrid', font_size_base=16, linewdith_base=1.0, font="Times New Roman"):
    sns.set_style(sns_style)
    colors = sns.color_palette('muted')
    mpl.rcParams["font.family"] = font
    mpl.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams["font.size"] = font_size_base
    mpl.rcParams["grid.linewidth"] = linewdith_base / 2.0
    mpl.rcParams["axes.linewidth"] = linewdith_base
    mpl.rcParams['xtick.major.size'] = 4
    mpl.rcParams['xtick.major.width'] = 1.
    mpl.rcParams['ytick.major.size'] = 4
    mpl.rcParams['ytick.major.width'] = 1.
    return colors



def make_plot(use_loss, hellinger_distances, upper_bounds_grammian, upper_bounds_l, upper_bounds_m, upper_bounds_l_wo_br, sampled_distances, sampled_scores, ori_score, path,
              dists_2=None, sampled_scores_2=None):
    # plot params
    font_size = 25
    linewidth = 1.0
    colors = init_style(font_size_base=font_size, linewdith_base=linewidth, sns_style='darkgrid')

    GRAMIAN_UPPER = 'Gramian Upper Bd.'
    GRAMIAN_LOWER = 'Gramian Lower Bd.'

    EMP_SCORE_LABEL = r'$\mathbb{E}_P[\ell(h(X),\,Y)]$'

    UPPER_BOUND_MARKERS = '^'
    LOWER_BOUND_MARKERS = 'v'

    # make figure
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()

    if upper_bounds_grammian is not None:
        ax.plot(hellinger_distances, upper_bounds_grammian, label=GRAMIAN_UPPER, linestyle='-', lw=1, c='black',
                marker=UPPER_BOUND_MARKERS, markevery=1)
    if upper_bounds_l is not None:
        ax.plot(hellinger_distances, upper_bounds_l, label=GRAMIAN_UPPER, linestyle='-', lw=1, c='blue',
                marker=UPPER_BOUND_MARKERS, markevery=1)
    if upper_bounds_m is not None:
        ax.plot(hellinger_distances, upper_bounds_m, label=GRAMIAN_UPPER, linestyle='-', lw=1, c='red',
                marker=UPPER_BOUND_MARKERS, markevery=1)

    if upper_bounds_l_wo_br is not None:
        ax.plot(hellinger_distances, upper_bounds_l_wo_br, label=GRAMIAN_UPPER, linestyle='-', lw=1, c='green',
                marker=UPPER_BOUND_MARKERS, markevery=1)



    if sampled_scores_2 is not None:
        ax.scatter(sampled_distances, sampled_scores, marker='o', alpha=1.0, color='red', s=2,
                   label=r'$\mathbb{E}_Q[\ell(h(X),\,Y)]$')
        ax.scatter(dists_2, sampled_scores_2, marker='o', alpha=0.5, color='dimgray', s=2,
                   label=r'$\mathbb{E}_Q[\ell(h(X),\,Y)]$')
    else:
        ax.scatter(sampled_distances, sampled_scores, marker='o', alpha=0.7, color='dimgray', s=2,
                   label=r'$\mathbb{E}_Q[\ell(h(X),\,Y)]$')

    ax.scatter([0], ori_score, label=EMP_SCORE_LABEL, marker='x', color=colors[1], linewidth=2, s=200)

    # if use_loss:
    #     ax.set_ylabel('JSD Loss')
    # else:
    #     ax.set_ylabel('Classification Error')
    # ax.set_xlabel('Hellinger Distance')
    ax.set_ylim((-0.02, 1.02))
    ax.set_xlim((-0.02, 0.62))
    # plt.legend(loc='best',bbox_to_anchor=(1.05, 1))
    fig.tight_layout()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if path is not None:
        plt.savefig(path, bbox_inches='tight', pad_inches=0.1, dpi=100)
        plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight', pad_inches=0.1, dpi=200)
        print(f'saved figure as {path}')
        plt.close(fig)
        return





# plot base rate vs. demographic parity
def make_plot_2(hellinger_distances, lower_bounds, upper_bounds, base_rates, parities, losses, path):
    # plot params
    font_size = 16
    linewidth = 1.0
    colors = init_style(font_size_base=font_size, linewdith_base=linewidth, sns_style='darkgrid')

    GRAMIAN_UPPER = 'Gramian Upper Bd.'
    GRAMIAN_LOWER = 'Gramian Lower Bd.'

    EMP_SCORE_LABEL = r'$\mathbb{E}_P[\ell(h(X),\,Y)]$'

    UPPER_BOUND_MARKERS = '^'
    LOWER_BOUND_MARKERS = 'v'

    # make figure
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()
    # ax.plot(hellinger_distances, lower_bounds, label=GRAMIAN_LOWER, linestyle='-', lw=2.5, c='black',
    #         marker=LOWER_BOUND_MARKERS, markevery=2)
    # ax.plot(hellinger_distances, upper_bounds, label=GRAMIAN_UPPER, linestyle='-', lw=2.5, c='black',
    #         marker=UPPER_BOUND_MARKERS, markevery=2)


    ax.scatter(base_rates, parities, marker='o', alpha=0.7, color='dimgray', s=2)

    ax.scatter(base_rates, losses, marker='o', alpha=0.7, color='red', s=2)

    ax.set_ylabel('Demographic Parity or BCE Loss')
    ax.set_xlabel('Base rate gap')
    ax.set_ylim((-0.02, 1.02))
    ax.set_xlim((-1.02, 1.02))
    # plt.legend(loc='best',bbox_to_anchor=(1.05, 1))
    fig.tight_layout()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if path is not None:
        plt.savefig(path, bbox_inches='tight', pad_inches=0.1, dpi=100)
        plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight', pad_inches=0.1, dpi=200)
        print(f'saved figure as {path}')
        plt.close(fig)
        return

# hellinger_distances = np.linspace(0, 1, 50)
# lower_bounds = [0] * 50
# upper_bounds = [1] * 50
# sampled_distances = [random.uniform(0,1) for _ in range(50)]
# sampled_scores = [random.uniform(0,1) for _ in range(50)]
# ori_score = 0.5
# make_plot(hellinger_distances, lower_bounds, upper_bounds, sampled_distances, sampled_scores, ori_score)

