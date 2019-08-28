"""Make plots for displaying results of BEST test.

This module produces plots similar to those in

Kruschke, J. (2012) Bayesian estimation supersedes the t
    test. Journal of Experimental Psychology: General.
"""
from typing import Optional, Union

try:
    import matplotlib.pyplot as plt
except ImportError:
    import matplotlib as mpl
    mpl.rcParams["backend"] = "TkAgg"
    import matplotlib.pyplot as plt

import numpy as np
import matplotlib.lines as mpllines
from matplotlib.transforms import blended_transform_factory
from matplotlib.ticker import FuncFormatter
import pymc3 as pm
from pymc3.backends.base import MultiTrace
import scipy.stats as st

from .utils import posterior_mode

PRETTY_BLUE = '#89d1ea'


def plot_posterior(best_out: MultiTrace,
                   variable_name: str,
                   ax: Optional[plt.Axes] = None,
                   bins: Union[int, list, np.ndarray] = 30,
                   stat: str = 'mode',
                   title: Optional[str] = None,
                   label: Optional[str] = None,
                   draw_zero: bool = False,
                   **kwargs):
    """Plot a histogram of posterior samples of a variable

    Parameters
    ----------
    best_out : PyMC3 MultiTrace
        The result of the analysis.
    variable_name : string
        The name of the variable to be plotted.
    ax : Matplotlib Axes, optional
        If not None, the Matplotlib Axes instance to be used.
        Default: create a new axes.
    bins : int or list or NumPy array
        The number or edges of the bins used for the histogram of the data.
        If an integer, the number of bins to use.
        If a sequence, then the edges of the bins, including left edge
        of the first bin and right edge of the last bin.
        Default: 30 bins.
    stat : {'mean', 'mode'}
        Whether to print the mean or the mode of the variable on the plot.
        Default: 'mode'.
    title : string, optional
        Title of the plot. Default: don’t print a title.
    label : string, optional
        Label of the *x* axis. Default: don’t print a label.
    draw_zero : bool
        Whether to print a vertical line for the zero value.
    **kwargs : dict
        All other keyword arguments are passed to `plt.hist`.

    Returns
    -------
    Matplotlib Axes
        The Axes object containing the plot. Using this return value, the
        plot can be customized afterwards – for details, see the documentation
        of the `Matplotlib Axes API <https://matplotlib.org/api/axes_api.html>`_.

    Examples
    --------
    To plot a histogram of the samples of the mean of the first group in green
    (the hist_kwargs)

        >>> import matplotlib as plt
        >>> ax = best.plot_posterior(best_out, 'Group 1 mean', color='green')
        >>> plt.show()
    """
    samples = best_out.get_values(variable_name)
    if stat == 'mode':
        stat_val = posterior_mode(best_out, variable_name)
    elif stat == 'mean':
        stat_val = np.mean(samples)
    hdi_min, hdi_max = pm.hpd(samples, alpha=0.05)

    if ax is None:
        _, ax = plt.subplots()

    hist_kwargs = {'bins': bins}
    hist_kwargs.update(kwargs)
    ax.hist(samples, rwidth=0.8,
            facecolor=PRETTY_BLUE, edgecolor='none', **hist_kwargs)

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
    for line in ax.get_xticklines():
        line.set_marker(mpllines.TICKDOWN)
    if label:
        ax.set_xlabel(label)
    if title is not None:
        ax.set_title(title)

    return ax


def plot_data_and_prediction(best_out: MultiTrace,
                             group_data,
                             group_id: int = 1,
                             ax: plt.Axes = None,
                             bins: Union[int, list, np.ndarray] = 30,
                             title: Optional[str] = None,
                             hist_kwargs: dict = {},
                             prediction_kwargs: dict = {}):
    """Plot samples of predictive distributions and a histogram of the data.

    This plot can be used as a *posterior predictive check*, to examine
    how well the model predictions fit the observed data.

    Parameters
    ----------
    best_out
        The result of the analysis.
    group_data : list of numbers, NumPy array, or Pandas Series.
        Data of the group to be plotted.
    group_id : {1, 2}
        Which group to plot (1 or 2).
    ax : Matplotlib Axes, optional
        If not None, the Matplotlib Axes instance to be used.
        Default: create a new plot.
    title : string, optional.
        Title of the plot. Default: don’t print a title.
    bins : int or list or NumPy array.
        The number or edges of the bins used for the histogram of the data.
        If an integer, the number of bins to use.
        If a sequence, then the edges of the bins, including left edge
        of the first bin and right edge of the last bin.
        Default: 30 bins.
    hist_kwargs : dict
        The keyword arguments to be passed to `plt.hist` for the group data.
    prediction_kwargs : dict
        The keyword arguments to be passed to `plt.plot` for the posterior
        predictive curves.

    Returns
    -------
    Matplotlib Axes
        The Axes object containing the plot. Using this return value, the
        plot can be customized afterwards – for details, see the documentation
        of the `Matplotlib Axes API <https://matplotlib.org/api/axes_api.html>`_.

    Examples
    --------
    To print the data of the second group, add a hatch to the histogram, and
    set the limits of the *x* axis to 85 and 115:

        >>> import matplotlib as plt
        >>> ax = best.plot_data_and_prediction(best_out, placebo, 2,
        ...                                    hist_kwargs={'hatch':'...'})
        >>> ax.set_xlim(85, 115)
        >>> plt.show()

    Notes
    -----
    You can move the histogram in front of the predictive curves by passing
    ``hist_kwargs={'zorder': 10}`` as an argument.

    If the plot is large enough, it is suggested to put a legend on it, by
    calling ``ax.legend()`` afterwards.
    """

    if ax is None:
        _, ax = plt.subplots()

    if group_id not in [1, 2]:
        raise ValueError("group_id argument must be either 1 or 2")

    means = best_out.get_values('Group {} mean'.format(group_id))
    stds = best_out.get_values('Group {} SD'.format(group_id))
    nus = best_out.get_values('Normality')

    n_curves = 50
    n_samps = len(means)
    idxs = np.random.choice(np.arange(n_samps), n_curves, replace=False)

    try:
        xmin = bins[0]
        xmax = bins[-1]
    except TypeError:
        xmin = np.min(group_data)
        xmax = np.max(group_data)

    dx = xmax - xmin
    xmin -= dx * 0.05
    xmax += dx * 0.05

    x = np.linspace(xmin, xmax, 1000)

    kwargs = dict(color=PRETTY_BLUE, zorder=1, alpha=0.3)
    kwargs.update(prediction_kwargs)

    for i in idxs:
        v = st.t.pdf(x, nus[i], means[i], stds[i])
        line, = ax.plot(x, v, **kwargs)

    line.set_label('Prediction')

    kwargs = dict(edgecolor='w',
                  facecolor='xkcd:salmon',
                  density=True,
                  bins=bins,
                  label='Observation')
    kwargs.update(hist_kwargs)
    ax.hist(group_data, **kwargs)

    # draw a translucent histogram in front of the curves
    if 'zorder' not in hist_kwargs and 'alpha' not in hist_kwargs:
        kwargs.update(dict(zorder=3, label=None, alpha=0.3))
        ax.hist(group_data, **kwargs)

    ax.text(0.95, 0.95, r'$\mathrm{N}=%d$' % len(group_data),
            transform=ax.transAxes,
            horizontalalignment='right',
            verticalalignment='top'
            )

    for loc in ['top', 'right']:
        ax.spines[loc].set_color('none')  # don't draw
    ax.spines['left'].set_color('gray')
    ax.set_xlabel('Observation')
    ax.set_xlim(xmin, xmax)
    ax.set_ylabel('Probability of observation')
    ax.set_yticklabels([])
    ax.set_ylim(0)
    if title:
        ax.set_title(title)

    return ax


def plot_all(best_out: MultiTrace,
             group1_data,
             group2_data,
             bins: int = 30,
             group1_name: str = 'Group 1',
             group2_name: str = 'Group 2'):
    """Plot posteriors of every parameter and observation.

    TODO: Currently only works with two-group analysis.

    Parameters
    ----------
    best_out : PyMC3 MultiTrace
        The result of the analysis.
    group1_data : list of numbers, NumPy array, or Pandas Series.
        Data of the first group analyzed and to be plotted.
    group2_data : list of numbers, NumPy array, or Pandas Series.
        Data of the second group analyzed and to be plotted.
    bins : int
        The number of bins to be used for the histograms.
        Default: 30.
    group1_name : string
        Group name of first group to be used in the title. Default: "Group 1".
    group2_name : string
        Group name of second group to be used in the title. Default: "Group 2".

    Returns
    -------
    plt.Figure
        The created figure. (The separate plots can be accessed via
        ``fig.axes``, where ``fig`` is the return value of this function.)
    """
    assert type(bins) is int, "bins argument must be an integer."

    posterior_mean1 = best_out.get_values('Group 1 mean')
    posterior_mean2 = best_out.get_values('Group 2 mean')

    posterior_means = np.concatenate((posterior_mean1, posterior_mean2))
    _, bin_edges_means = np.histogram(posterior_means, bins=bins)

    posterior_std1 = best_out.get_values('Group 1 SD')
    posterior_std2 = best_out.get_values('Group 2 SD')

    posterior_stds = np.concatenate((posterior_std1, posterior_std2))
    _, bin_edges_stds = np.histogram(posterior_stds, bins=bins)

    posterior_normality = best_out.get_values('Normality')

    fig, axes = plt.subplots(5, 2, figsize=(8.2, 11))

    axes[0, 0].get_shared_x_axes().join(axes[0, 0], axes[1, 0])

    plot_posterior(best_out,
                   'Group 1 mean',
                   ax=axes[0, 0],
                   bins=bin_edges_means,
                   stat='mean',
                   title='%s Mean' % group1_name,
                   label=r'$\mu_1$')

    plot_posterior(best_out,
                   'Group 2 mean',
                   ax=axes[1, 0],
                   bins=bin_edges_means,
                   stat='mean',
                   title='%s Mean' % group2_name,
                   label=r'$\mu_2$')

    axes[2, 0].get_shared_x_axes().join(axes[2, 0], axes[3, 0])

    plot_posterior(best_out,
                   'Group 1 SD',
                   ax=axes[2, 0],
                   bins=bin_edges_stds,
                   title='%s Std. Dev.' % group1_name,
                   label=r'$\sigma_1$')

    plot_posterior(best_out,
                   'Group 2 SD',
                   ax=axes[3, 0],
                   bins=bin_edges_stds,
                   title='%s Std. Dev.' % group2_name,
                   label=r'$\sigma_2$')

    norm_bins = np.logspace(
            0,
            np.log10(pm.hpd(posterior_normality, alpha=0.005)[-1] * 1.2),
            num = bins + 1
    )
    plot_posterior(best_out,
                   'Normality',
                   ax=axes[4, 0],
                   bins=norm_bins,
                   title='Normality',
                   label=r'$\nu$')
    axes[4, 0].set_xlim(0.8, norm_bins[-1] * 1.2)
    axes[4, 0].semilogx()
    tick_fmt = FuncFormatter(lambda x, _: str(int(x)) if x >= 1 else None)
    axes[4, 0].xaxis.set_major_formatter(tick_fmt)
    axes[4, 0].xaxis.set_minor_formatter(tick_fmt)

    plot_posterior(best_out,
                   'Difference of means',
                   ax=axes[2, 1],
                   bins=bins,
                   title='Difference of Means',
                   stat='mean',
                   draw_zero=True,
                   label=r'$\mu_1 - \mu_2$')

    plot_posterior(best_out,
                   'Difference of SDs',
                   ax=axes[3, 1],
                   bins=bins,
                   title='Difference of Std. Dev.s',
                   draw_zero=True,
                   label=r'$\sigma_1 - \sigma_2$')

    plot_posterior(best_out,
                   'Effect size',
                   ax=axes[4, 1],
                   bins=bins,
                   title='Effect Size',
                   draw_zero=True,
                   label=r'$(\mu_1 - \mu_2) /'
                          r' \sqrt{(\sigma_1^2 + \sigma_2^2)/2}$')

    orig_vals = np.concatenate((group1_data, group2_data))
    bin_edges = np.linspace(np.min(orig_vals), np.max(orig_vals), bins + 1)

    axes[0, 1].get_shared_x_axes().join(axes[0, 1], axes[1, 1])
    axes[0, 1].get_shared_y_axes().join(axes[0, 1], axes[1, 1])

    plot_data_and_prediction(best_out, group1_data, 1,
                             ax=axes[0, 1], bins=bin_edges,
                             title='%s Data with Post. Pred.' % group1_name)

    plot_data_and_prediction(best_out, group2_data, 2,
                             ax=axes[1, 1], bins=bin_edges,
                             title = '%s Data with Post. Pred.' % group2_name)

    fig.subplots_adjust(hspace=0.82, top=0.97, bottom=0.06,
                        left=0.09, right=0.95, wspace=0.45)

    return fig
