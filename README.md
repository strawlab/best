# Bayesian estimation for two groups

**Update: This package has now been incorporated into PyMC. The PyMC developers maintain it. Please use it https://docs.pymc.io/en/v3/pymc-examples/examples/case_studies/BEST.html. This repository is an archive of the original Python version which was used as the basis for incorporation into PyMC.**

This Python package implements the software described in

> Kruschke, John. (2012) Bayesian estimation supersedes the t
> test. Journal of Experimental Psychology: General.

It implements Bayesian estimation for two groups, providing complete
distributions for effect size, group means and their difference,
standard deviations and their difference, and the normality of the
data. See John Kruschke's [website on
BEST](http://www.indiana.edu/~kruschke/BEST/) for more information.

## Requirements ##

 - Python ≥ 3.5.4
 - SciPy
 - [matplotlib](http://matplotlib.org) for plotting
 - [PyMC](https://github.com/pymc-devs/pymc) for sampling from the posterior

## Example ##

Here is the plot created by `examples/smart_drug.py`:

![smart_drug.png](http://strawlab.org/assets/smart_drug.png)
