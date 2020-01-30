    .. module:: dense_basis

Installation
============

The current version of the `dense_basis` module has a few dependencies (see :ref:`this <dependencies>`) that need to be set up before running this package. Once these are met, the package can be installed as follows::

    git clone https://github.com/kartheikiyer/dense_basis.git
    cd dense_basis
    python setup.py install
    
The code will default to looking for filter lists in a `filters/` directory, and will build and store atlases in a `pregrids/` directory within the current working directory. If you would like to supply your own paths, please provide either the relative or absolute paths as inputs to the relevant functions using the `filt_dir` or `path` arguments.
