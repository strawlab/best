"""Make plots for displaying results of BEST test.

This module produces plots similar to those in

Kruschke, J. (2012) Bayesian estimation supersedes the t
    test. Journal of Experimental Psychology: General.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import matplotlib.lines as mpllines
import matplotlib.ticker as mticker
import pymc3 as pm
import scipy.stats as st

from .utils import calculate_sample_statistics

PRETTY_BLUE = '#89d1ea'


def plot_posterior(sample_vec, bins=None, ax=None, title=None, stat='mode',
                   label='', draw_zero=False):
    stats = calculate_sample_statistics(sample_vec)
    stat_val = stats[stat]
    hdi_min = stats['hdi_min']
    hdi_max = stats['hdi_max']

    if ax is not None:
        if bins is not None:
            kwargs = {'bins': bins}
        else:
            kwargs = {}
        ax.hist(sample_vec, rwidth=0.8,
                facecolor=PRETTY_BLUE, edgecolor='none', **kwargs)
        if title is not None:
            ax.set_title(title)

        trans = blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(stat_val, 0.99, '%s = %.3g' % (stat, stat_val),
                transform=trans,
                horizontalalignment='center',
                verticalalignment='top',
                )
        if draw_zero:
            ax.axvline(0, linestyle=':')

        # plot HDI
        hdi_line, = ax.plot([hdi_min, hdi_max], [0, 0],
                            lw=5.0, color='k')
        hdi_line.set_clip_on(False)
        ax.text(hdi_min, 0.04, '%.3g' % hdi_min,
                transform=trans,
                horizontalalignment='center',
                verticalalignment='bottom',
                )
        ax.text(hdi_max, 0.04, '%.3g' % hdi_max,
                transform=trans,
                horizontalalignment='center',
                verticalalignment='bottom',
                )

        ax.text((hdi_min + hdi_max) / 2, 0.14, '95% HDI',
                transform=trans,
                horizontalalignment='center',
                verticalalignment='bottom',
                )

        # make it pretty
        ax.spines['bottom'].set_position(('outward', 2))
        for loc in ['left', 'top', 'right']:
            ax.spines[loc].set_color('none')  # don't draw
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks([])  # don't draw
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
        for line in ax.get_xticklines():
            line.set_marker(mpllines.TICKDOWN)
        ax.set_xlabel(label)


def plot_data_and_prediction(data, means, stds, numos, ax=None, bins=None,
                             n_curves=50, group='x'):
    assert ax is not None

    ax.hist(data, bins=bins, rwidth=0.5,
            facecolor='r', edgecolor='none', density=True)

    try:
        xmin = bins[0]
        xmax = bins[-1]
    except TypeError:
        xmin = np.min(data)
        xmax = np.max(data)

    n_samps = len(means)
    idxs = np.random.choice(np.arange(n_samps), n_curves, replace=False)

    x = np.linspace(xmin, xmax, 1000)
    ax.set_xlabel('y')
    ax.set_ylabel('p(y)')

    for i in idxs:
        loc = means[i]
        scale = stds[i]
        numo = numos[i]
        nu = numo + 1
        v = st.t.pdf(x, nu, loc, scale)
        ax.plot(x, v, color=PRETTY_BLUE, zorder=-10, alpha=0.5)

    ax.text(0.95, 0.95, r'$\mathrm{N}=%d$' % len(data),
            transform=ax.transAxes,
            horizontalalignment='right',
            verticalalignment='top'
            )
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
    ax.set_title('%s Data with Post. Pred.' % (group,))


def plot_all(trace, y1, y2, n_bins=30, group1_name='Group 1', group2_name='Group 2'):
    # plotting stuff

    posterior_mean1 = trace.get_values('Group 1 mean')
    posterior_mean2 = trace.get_values('Group 2 mean')
    diff_means = posterior_mean1 - posterior_mean2

    posterior_means = np.concatenate((posterior_mean1, posterior_mean2))
    _, bin_edges_means = np.histogram(posterior_means, bins=n_bins)

    posterior_std1 = trace.get_values('Group 1 SD')
    posterior_std2 = trace.get_values('Group 2 SD')
    diff_stds = posterior_std1 - posterior_std2

    posterior_stds = np.concatenate((posterior_std1, posterior_std2))
    _, bin_edges_stds = np.histogram(posterior_stds, bins=n_bins)

    effect_size = diff_means / np.sqrt((posterior_std1 ** 2
                                        + posterior_std2 ** 2) / 2)

    post_nu_minus_one = trace.get_values('nu - 1')
    posterior_normality = trace.get_values('Normality')

    f = plt.figure(figsize=(8.2, 11), facecolor='white')
    ax1 = f.add_subplot(5, 2, 1)
    plot_posterior(posterior_mean1, bins=bin_edges_means, ax=ax1,
                   title='%s Mean' % group1_name, stat='mean',
                   label=r'$\mu_1$')

    ax3 = f.add_subplot(5, 2, 3)
    plot_posterior(posterior_mean2, bins=bin_edges_means, ax=ax3,
                   title='%s Mean' % group2_name, stat='mean',
                   label=r'$\mu_2$')

    ax5 = f.add_subplot(5, 2, 5)
    plot_posterior(posterior_std1, bins=bin_edges_stds, ax=ax5,
                   title='%s Std. Dev.' % group1_name,
                   label=r'$\sigma_1$')

    ax7 = f.add_subplot(5, 2, 7)
    plot_posterior(posterior_std2, bins=bin_edges_stds, ax=ax7,
                   title='%s Std. Dev.' % group2_name,
                   label=r'$\sigma_2$')

    ax9 = f.add_subplot(5, 2, 9)
    norm_bins = np.logspace(
            0,
            np.log10(pm.hpd(posterior_normality, alpha=0.01)[-1] * 3),
            num = n_bins + 1
    )
    plot_posterior(posterior_normality, bins=norm_bins, ax=ax9,
                   title='Normality',
                   label=r'$\nu$')
    ax9.semilogx()
    ax9.set_xlim(0.5, norm_bins[-1]*3)

    ax6 = f.add_subplot(5, 2, 6)
    plot_posterior(diff_means, bins=n_bins, ax=ax6,
                   title='Difference of Means',
                   stat='mean',
                   draw_zero=True,
                   label=r'$\mu_1 - \mu_2$')

    ax8 = f.add_subplot(5, 2, 8)
    plot_posterior(diff_stds, bins=n_bins, ax=ax8,
                   title='Difference of Std. Dev.s',
                   draw_zero=True,
                   label=r'$\sigma_1 - \sigma_2$')

    ax10 = f.add_subplot(5, 2, 10)
    plot_posterior(effect_size, bins=n_bins, ax=ax10,
                   title='Effect Size',
                   draw_zero=True,
                   label=r'$(\mu_1 - \mu_2) /'
                         r' \sqrt{(\sigma_1^2 + \sigma_2^2)/2}$')

    orig_vals = np.concatenate((y1, y2))
    bin_edges = np.linspace(np.min(orig_vals), np.max(orig_vals), 30)

    ax2 = f.add_subplot(5, 2, 2)
    plot_data_and_prediction(y1, posterior_mean1, posterior_std1,
                             post_nu_minus_one, ax=ax2,
                             bins=bin_edges, group=group1_name)

    ax4 = f.add_subplot(5, 2, 4, sharex=ax2, sharey=ax2)
    plot_data_and_prediction(y2, posterior_mean2, posterior_std2,
                             post_nu_minus_one, ax=ax4,
                             bins=bin_edges, group=group2_name)

    f.subplots_adjust(hspace=0.82, top=0.97, bottom=0.06,
                      left=0.09, right=0.95, wspace=0.45)

    return f
