Developer Interface
===================

.. module:: best

Functions for analysis
----------------------

.. autofunction:: analyze_one
.. autofunction:: analyze_two

These functions return a :class:`BestResultsOne` or :class:`BestResultsTwo`, respectively.

Plotting functions
------------------

.. autofunction:: plot_posterior
.. autofunction:: plot_data_and_prediction
.. autofunction:: plot_all_one
.. autofunction:: plot_all_two
.. autofunction:: plot_all


Classes for analysis results
----------------------------

.. autoclass:: BestResultsOne
.. autoclass:: BestResultsTwo

Both are subclasses of :class:`BestResults`, meaning the following methods are available
for :class:`BestResultsOne` or :class:`BestResultsTwo` objects.

.. autoclass:: BestResults

Lower-level classes
-------------------

.. autoclass:: BestModel
.. autoclass:: BestModelOne
    :exclude-members: model, observed_data, version

.. autoclass:: BestModelTwo
    :exclude-members: model, observed_data, version
