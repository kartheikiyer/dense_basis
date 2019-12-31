.. dense_basis documentation master file, created by
   sphinx-quickstart on Wed Dec 25 22:03:18 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

dense_basis: SED fitting with smooth nonparametric star formation histories
===========================================================================

dense_basis is an implementation of the `Dense Basis <https://iopscience.iop.org/article/10.3847/1538-4357/ab2052/meta>`_ method tailored to SED fitting - in particular, the task of recovering accurate star formation history (SFH) information from galaxy spectral energy distributions (SEDs). The current code is being adapted from its original use-case (simultaneously fitting specific large catalogs of galaxies) to being a general purpose SED fitting code, and acting as a module to compress and decompress SFHs and other time-series.


.. toctree::
   :maxdepth: 2
   :caption: General Usage:
   
   usage/installation
   usage/dependencies
   usage/features
   
   
.. toctree::
   :maxdepth: 2
   :caption: Tutorials:
   
   tutorials/getting_started
   tutorials/fitting_different_SFH_shapes
   
   
The code is designed to be intuitive to use, and and consists of three steps to get you started doing SED fitting:

- defining your priors
- generating a model atlas (params <-> SEDs) to use while fitting
- actually fitting your data and visualizing the posteriors

More detailed descriptions of these modules can be found in the tutorials. If you are interested in going off the beaten track and trying different things, please let me know so that I can help you run the code as you'd like!


Contribute
----------

- Issue Tracker: https://github.com/kartheikiyer/dense_basis/issues
- Source Code: https://github.com/kartheikiyer/dense_basis

Support
-------

If you are having issues, please let me know at: kartheik.iyer@dunlap.utoronto.ca

License & Attribution
---------------------

Copyright 2017-2019 Kartheik Iyer and contributors.

`dense_basis` is being developed by `Kartheik Iyer <http://kartheikiyer.github.io>`_ in a
`public GitHub repository <https://github.com/kartheikiyer/dense_basis>`_.
The source code is made available under the terms of the MIT license.

If you make use of this code, please cite `the recent Dense Basis paper <https://iopscience.iop.org/article/10.3847/1538-4357/ab2052/meta>`_.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`






