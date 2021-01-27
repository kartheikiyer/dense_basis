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


### Changelog

v.0.1.8 
- added plotting functions in the SedFit() class for posterior spectra and SFHs
- added the dynamic_decoupling flag for the sSFR priors, which automatically sets the timescale on which SFR decoupling occurs (if invoked) to scale with redshift.
- fixed a bug in the 'sSFRlognormal' sampling of the priors.
- added parallelization using MultiPool from [schwimmbad](https://schwimmbad.readthedocs.io/en/latest/_modules/schwimmbad/multiprocessing.html). 
- pregrid generation can now be parallelized using the `generate_atlas_in_parallel_zgrid()` command.
- changed the default value of decouple_sfr_time in tuple_to_sfh() to avoid a low-z error

v.0.1.7
- added arguments in makespec() to return spectra splined to an input wavelength array
- added dynamic_norm argument to sed fitter (calculates free norm during fitting for better accuracy, but slower)

v.0.1.6
- added a class for the SED fitter
- added the plot_atlas_priors() function
- overhauled atlas generation with the makespec() function for self-consistency
- added some bugfixes to the gp_sfh module for high sSFR values

v.0.1.5
- added basic MCMC support with emcee in the main repo instead of dense_basis_toolbelt

v.0.1.4
- The FSPS/python-FSPS requirement is no longer necessary, if a user requires only the GP-SFH module.
- added more options to SFR sampling - flat in SFR, sSFR or lognormal in sSFR. removed the separate sample_sSFR_prior option
- added option for tx_alpha sampling from IllustrisTNG (0<z<6, Nparam<10)
- removed the squeeze_tx option - this can be effectively implemented with a larger value for the concentration parameter
- implemented rough mass-metallicity prior
- implemented flat and exponential dust priors for the Calzetti law, and a rough implementaion of the CF00 law using priors from Pacifici+16
- removed sample_all_params_safesSFR, and the safedraw=True in make_N_prior_draws
- removed the min SFR in the sample_sfh_tuple function
- updated the GP tuple_to_sfh module to decouple SFR if necessary.
- overhauled the generate_atlas() and load_atlas() functions,
- shifted storage of precomputed pregrids/atlas(es) from scipy.io to hickle
