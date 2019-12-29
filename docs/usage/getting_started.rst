    .. module:: dense_basis
    
dense_basis: Getting Started
============================

0. Before you start, if you're fitting photometry, put your photometric filter transmission curves in a folder somewhere and make a list of filter curves inside the filter_curves folder with the paths to each filter. You'll need to pass this to the code to generate SEDs corresponding to a given parameter set. 
Also check all the numbers within `priors.py` to see if they're appropriate for your use-case.

Import the module, (the initial import takes a few minutes because it's initializing its FSPS backend. Don't worry about it) and visualize the set of filter curves used to make SEDs. Let's load a filter list corresponding to the CANDELS GOODS-South photometric catalog for now:

    import dense_basis as db
    
    import numpy as np
    
    filter_list = 'dense_basis/filters/filter_list_goodss.dat'
    
    db.plot_filterset(filter_list)
    
1. The next step is to generate a template atlas that you will use for fitting. The advantage of doing this is that we trade time-complexity for space, which is usually much more abundant, and it only needs to be done once to fit a large dataset. Before doing this, however, we need a set of priors that we can draw from to generate this atlas. This is done using the db.Priors() class:

    priors = db.Priors()
    
    priors.Nparam = 3
    
    priors.tx_alpha = 3.0

    priors.squeeze_tx = True
    
    priors.plot_prior_distributions()

2. Now we can use the priors object to generate the atlas. The important arguments here are the size of the atlas (`N_pregrid`), which samples from the overall multidimensional prior distributions, and the number of SFH parameters (`priors.Nparam`). The generated atlas is then stored in the /pregrids folder with the user-specified fname. The first few SEDs take time to compute because every time FSPS encounters a new metallicity value it needs to load a grid into memory. This happens only once, and when it is done, the atlas generation process speeds up significantly.

    fname = 'test_atlas'
    
    N_pregrid = 1000
    
    db.generate_atlas(N_pregrid = N_pregrid, priors = priors, fname = fname, store=True, filter_list = filter_list)
    
3. To illustrate the SED fitting procedure, let's generate a mock SFH and its corresponding SED. 

    # sample from the prior space to get parameters
    
    rand_sfh_tuple, rand_Z, rand_Av, rand_z = priors.sample_all_params_safesSFR(random_seed = 5)
    
    # generate an SFH corresponding to the SFH-tuple and see how it looks:
    
    rand_sfh, rand_time = db.tuple_to_sfh(rand_sfh_tuple, zval = rand_z) 

    fig = db.plot_sfh(rand_time, rand_sfh, lookback=True) 
    
    sfh_truths = [rand_time, rand_sfh]
    
    # generate a corresponding spectrum and multiply by filter curves to get the SED:
    
    _, sfr_true, mstar_true = db.make_spec(rand_sfh_tuple, rand_Z, rand_Av, rand_z, return_ms = True)
    
    rand_spec, rand_lam = db.make_spec(rand_sfh_tuple, rand_Z, rand_Av, rand_z, return_lam = True)
    
    obs_sed = db.calc_fnu_sed(rand_spec, rand_z, rand_lam, fkit_name = filter_list) 
    
    obs_err = obs_sed * 0.1 # S/N of 10
    
    sed_truths = (mstar_true, sfr_true, rand_sfh_tuple[3:], rand_Z, rand_Av, rand_z)
    
    sed_truths = np.hstack(sed_truths)

4. Step 2 is extremely beneficial in fitting large datasets, since the atlas needs to be generated only once and can be used for fitting as many SEDs as needed using the brute-force Bayesian approach. Having generated this dataset, now an arbitrary SED (`obs_sed`, and its errors `obs_err`) can be fit using the previously generated atlas. If you are interested in seeing how the fits change with a much more finely sampled atlas, try using the precomuted atlas with N_pregrid = 100000.

    # load the atlas 
    
    pg_sfhs, pg_Z, pg_Av, pg_z, pg_seds, norm_method = db.load_atlas(fname, N_pregrid, priors.Nparam) 
    
    pg_theta = [pg_sfhs, pg_Z, pg_Av, pg_z, pg_seds]

    # pass the atlas and the observed SED into the fitter, fit params returns the median and 1-sigma values for the parameters being fit 
    
    fit_params = db.fit_sed_pregrid(obs_sed, obs_err, pg_theta, norm_method=norm_method)
                            
5. If we are intersted in the full posteriors of the fit, this can be visualized by making the fitter return the chi2 array and then computing the full posteriors as prior*likelihood. Let's see how it compares to the truth: 

    chi2_array = db.fit_sed_pregrid(obs_sed, obs_err, pg_theta, return_val = 'chi2', norm_method=norm_method)

    # plot parameter posteriors: 
    
    db.plot_posteriors(chi2_array, obs_sed, pg_theta, truths = sed_truths)
    
    
6. Finally, we can also plot the posterior SFH and see how it compares to the true SFH.
    
    # plot posterior SFH 
    
    db.plot_SFH_posterior(chi2_array, obs_sed, pg_theta, truths = sfh_truths, sfh_threshold = 0.95)