# Dense Basis SED fitting

An implementation of the Dense Basis method tailored to SED fitting, in particular, to the task of recovering accurate star formation history (SFH) information from galaxy spectral energy distributions (SEDs). The current code is being adapted from its original use-case (simultaneously fitting specific large catalogs of galaxies) to being a general purpose SED fitting code and acting as a module to compress and decompress SFHs. 

As such, it is currently in an `beta` phase, where existing modules are being improved upon and crash-tested and thorough documentation is being written. If you are interested in using, testing or extending the repository, please shoot me an email. 

### Installation and usage:

To use the package, clone the repository and run `python setup.py install` within the dense_basis folder. More detailed intstructions can be found at [dense-basis.readthedocs.io](https://dense-basis.readthedocs.io).

Documentation on usage and basic tutorials can also be found at [dense-basis.readthedocs.io](https://dense-basis.readthedocs.io). 

A good place to get started is [here](https://github.com/kartheikiyer/dense_basis/blob/master/docs/tutorials/getting_started.ipynb).

References:
- [Iyer & Gawiser (2017)](https://iopscience.iop.org/article/10.3847/1538-4357/aa63f0/meta)
- [Iyer et al. (2019)](https://iopscience.iop.org/article/10.3847/1538-4357/ab2052/meta) 

Contact:
- kartheik.iyer@dunlap.utoronto.ca
