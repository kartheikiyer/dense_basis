# Dense Basis SED fitting

An implementation of the Dense Basis method tailored to SED fitting, in particular, to the task of recovering accurate star formation history (SFH) information from galaxy spectral energy distributions (SEDs). The current code is being adapted from its original use-case (simultaneously fitting specific large catalogs of galaxies) to being a general purpose SED fitting code and acting as a module to compress and decompress SFHs. 

As such, it is currently in an `alpha` phase, where modules are being built and crash-tested and documenation is being written. If you are interested in testing or contributing to the repository, please shoot me an email. 

### Installation and usage:

This is not finalized, for now to use the package please:

- go to the folder where you'd like to run the package from
- git clone https://github.com/kartheikiyer/dense_basis.git
- cd dense_basis
- run your code here, where dense_basis can be imported.

My apologies for the inconvenience of not being able to globally import the package yet. Currently I haven't decided on a good format to store filter curves and atlases in a way that's easily accessible to users and yet contains some self-contained filter-sets and pre-built atlases, so I haven't yet put the code on PyPI with a self-contained setup. If you have ideas regarding this, please let me know. It could be as simple as setting a global package path at the beginning of a script that is passed in as input to the functions. 

Documentation on usage and basic tutorials can now be found at [dense-basis.readthedocs.io](https://dense-basis.readthedocs.io). 

A good place to get started is [here](https://github.com/kartheikiyer/dense_basis/blob/master/docs/tutorials/getting_started.ipynb).

References:
- [Iyer & Gawiser (2017)](https://iopscience.iop.org/article/10.3847/1538-4357/aa63f0/meta)
- [Iyer et al. (2019)](https://iopscience.iop.org/article/10.3847/1538-4357/ab2052/meta) 

Contact:
- kartheik.iyer@dunlap.utoronto.ca
