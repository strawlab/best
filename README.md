# Bayesian estimation for two groups

This Python package implements the software described in

> Kruschke, John. (2012) Bayesian estimation supersedes the t
> test. Journal of Experimental Psychology: General.

It implements Bayesian estimation for two groups, providing complete
distributions for effect size, group means and their difference,
standard deviations and their difference, and the normality of the
data. See John Kruschke's [website on
BEST](http://www.indiana.edu/~kruschke/BEST/) for more information.

## Requirements ##

 * tested with Python 2.7 (not Python 3)
 * [PyMC](https://github.com/pymc-devs/pymc) for MCMC sampling
 * [matplotlib](http://matplotlib.org) for plotting

## Example ##

Here is the plot created by `examples/smart_drug.py`:

![smart_drug.png](http://strawlab.org/assets/smart_drug.png)
