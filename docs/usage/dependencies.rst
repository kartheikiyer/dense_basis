Dependencies
============

- FSPS and python-FSPS (v.0.3.0+): The current implementation of the `dense_basis` method uses a backend based on Flexible Stellar Population Synthesis (`FSPS <https://github.com/cconroy20/fsps>`_; Conroy, Gunn, & White 2009, ApJ, 699, 486; Conroy & Gunn 2010, ApJ, 712, 833) to generate spectra corresponding to a set of stellar population parameters. Since this is originally a Fortran package, we use the `python-FSPS <http://dfm.io/python-fsps/current/>`_ (Foreman-Mackey, Sick and Johnson, 2014)  set of bindings to call FSPS from within python. Installation instructions for these packages can be found at their respective homepages: `FSPS <https://github.com/cconroy20/fsps>`_ and `python-FSPS <http://dfm.io/python-fsps/current/>`_.

- Astropy (v.3.2.1+): For redshift and distance calculations based on different cosmologies.

- George (v.0.3.1+): We use the `George <https://george.readthedocs.io/en/latest/>`_ package (`Ambikasaran et al. 2014 <http://arxiv.org/abs/1403.6015>`_) to implement Gaussian processes.

- Scikit-Learn (v.0.21.2+): can be used as an alternative to George, although it doesn't perform as well.

- Corner (v.2.0.1+): `Foreman-Mackey (2016) <https://corner.readthedocs.io/en/latest/>`_ is used to plot prior and posterior distributions.

- Numpy, Scipy, Matplotlib

- This code is written in Python 3.8.
