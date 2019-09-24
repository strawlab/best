.. _ch-model-history:

Model Version History
=====================

v1
--

This is the model described in the `original publication <http://www.indiana.edu/~kruschke/BEST/>`_.

The model for two-group analysis is described by the following sampling statements:

.. math::

    \mu_1 &\sim \text{Normal}(\hat\mu, 1000\, \hat\sigma) \\
    \mu_2 &\sim \text{Normal}(\hat\mu, 1000\, \hat\sigma) \\
    \sigma_1 &\sim \text{Uniform}(\hat\sigma \,/\, 1000, 1000\, \hat\sigma) \\
    \sigma_2 &\sim \text{Uniform}(\hat\sigma \,/\, 1000, 1000\, \hat\sigma) \\
    \nu &\sim \text{Exponential}(1\,/\,29) + 1 \\
    y_1 &\sim t_\nu(\mu_1, \sigma_1) \\
    y_2 &\sim t_\nu(\mu_2, \sigma_2)

Where :math:`\hat\mu` and :math:`\hat\sigma` are the sample mean and
sample standard deviation of all the data from the two groups.
The effect size is calculated as :math:`(\mu_1 - \mu_2) \big/ \sqrt{(\sigma_1^2 + \sigma_2^2) \,/\, 2}`.

.. _sec-model-latest:

v2
--

Version 2 of the model fixes issues about the standard deviation and normality.

The standard deviation of a *t* distribution :math:`t_\nu(\mu, \sigma)`
is not :math:`\sigma`, but :math:`\sigma \sqrt{\nu / (\nu - 2)}` if :math:`2 < \nu`,
and infinite if :math:`1 < \nu \le 2`. Distributions with infinite standard deviation (SD)
rarely occur in reality (and never when it comes to humans),
so the lower bound of :math:`\nu` is changed from 1 to 2.5.
The plots now display SD instead of :math:`\sigma`,
and the formula for effect size also uses :math:`\mathrm{sd}_i` instead of :math:`\sigma_i`.

    *Why is the lower bound of* :math:`\nu` *2.5 and not 2?*

    The probability density function of :math:`t_2` is
    quite close to that of :math:`t_{2.5}` in the :math:`\mu \pm 5 \sigma` region,
    but for :math:`\nu` close to 2, the SD is arbitrarily large because of the strong outliers.
    Setting a bound of 2.5 prevents strong outliers and extremely large standard deviations.

Another change concerns the sampling of :math:`\sigma_i`.
In the original model :math:`\sigma_i` was uniformly distributed between
:math:`\hat\sigma\, / \,1000` and :math:`1000\,\hat\sigma`,
meaning the *prior* probability of :math:`\sigma > \hat\sigma` was 1000 times that of :math:`\sigma < \hat\sigma`,
which caused an overestimation of :math:`\sigma` with low sample sizes (around :math:`N = 5`).
To make these probabilities equal, now :math:`\log(\sigma_i)` is distributed uniformly between
:math:`\log(\hat\sigma\, / \,1000)` and :math:`\log(1000\, \hat\sigma)`.
At :math:`N=25` this change in the prior does not cause a perceptible change in the posterior.

*Summary of changes*:
 - Lower bound of :math:`\nu` is 2.5.
 - SD is calculated as :math:`\sigma \sqrt{ \nu / (\nu - 2)}`.
 - Effect size is calculated as :math:`(\mu_1 - \mu_2) \big/ \sqrt{(\mathrm{sd}_1^2 + \mathrm{sd}_2^2) \,/\, 2}`.
 - :math:`\log(\sigma_i)` is uniformly distributed.

The model for two-group analysis is described by the following sampling statements:

.. math::

    \mu_1 &\sim \text{Normal}(\hat\mu, 1000 \, \hat\sigma) \\
    \mu_2 &\sim \text{Normal}(\hat\mu, 1000 \, \hat\sigma) \\
    \log(\sigma_1) &\sim \text{Uniform}(\log(\hat\sigma \, / \, 1000), \log(1000\, \hat\sigma)) \\
    \log(\sigma_2) &\sim \text{Uniform}(\log(\hat\sigma \, / \, 1000), \log(1000\, \hat\sigma)) \\
    \nu &\sim \text{Exponential}(1\, / \, 27.5) + 2.5 \\
    y_1 &\sim t_\nu(\mu_1, \sigma_1) \\
    y_2 &\sim t_\nu(\mu_2, \sigma_2)


..
    Note: if there is a new model, move the _sec-model-latest label to here.
