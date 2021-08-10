from .priors import *
from .pre_grid import *
from .gp_sfh import *
from .plotter import *
from .sed_fitter import *


def sample_fesc_prior():
    # uniform between 0 and 1
    return np.random.uniform()


def get_k(f_esc = 0.0, f_dust = 0.0):    
    # factor for downscaling nebular line spectrum, from Inoue (2011)
    # taken from Boquien+20: https://www.aanda.org/articles/aa/full_html/2019/02/aa34156-18/aa34156-18.html
    
    # alphaB_Te: case B recombination rate, in m^3/s
    # alpha1_Te: alpha_A-alpha_B - recombination rate to the ground level
    # Te: electron temp in kelvin
    # fesc: lyman continuum escape fraction
    # fdust: partial absorption by dust before ionization
    
    Te = 1e4 # kelvin
    alphaB_Te = 2.58e-19 #m^3/s
    alpha1_Te = 1.54e-19 #m^3/s
    
    alpharatio = alpha1_Te/alphaB_Te
    
    k = (1-f_esc - f_dust) / (1+(alpharatio)*(f_esc+f_dust))
    
    return k


def makespec_fesc(specdetails, fesc, priors, sp, cosmo, filter_list = [], filt_dir = [], return_spec = False, peraa = False, input_sfh = False):
    """
    makespec() works in two ways to create spectra or SEDs from an input list of physical paramters. 
    with input_sfh = False, specdetails = [sfh_tuple, dust, met, zval]
    with input_sfh = True, specdetails = [sfh, timeax, dust, met, zval]
    
    return_spec can be true, false, or an array of wavelengths. in case 
    
    it uses the db.mocksp object for the underlying SPS calculations, so make sure it's set accordingly.
    it also uses the priors object for things like decouple_sfr. 
    
    """
    
    
    # hardcoded parameters - offload this to a seprarate function
    sp.params['sfh'] = 3
    sp.params['cloudy_dust'] = True
    sp.params['gas_logu'] = -2
    sp.params['add_igm_absorption'] = True
    sp.params['add_neb_emission'] = True
    sp.params['add_neb_continuum'] = True
    sp.params['imf_type'] = 1 # Chabrier
    
    # variable parameters
    if input_sfh == True:
        [sfh, timeax, dust, met, zval] = specdetails
    else:
        [sfh_tuple, dust, met, zval] = specdetails
        sfh, timeax = tuple_to_sfh(sfh_tuple, zval, decouple_sfr = priors.decouple_sfr, decouple_sfr_time = priors.decouple_sfr_time)
    sp.set_tabular_sfh(timeax, sfh)
    # sp.params['dust_type'] = 2
    # sp.params['dust1'] = dust1_rand
    sp.params['dust2'] = dust
    sp.params['logzsol'] = met
    sp.params['gas_logz'] = met # matching stellar to gas-phase metallicity
    sp.params['zred'] = zval
    
    #lam, spec = sp.get_spectrum(tage = cosmo.age(zval).value, peraa = peraa)
    # adding 0.1 Myr here to get the last couple of FSPS SSPs
    
    sp.params['add_neb_emission'] = False
    sp.params['add_neb_continuum'] = False
    lam, spec_noneb_dusty = sp.get_spectrum(tage = cosmo.age(zval).value+1e-4, peraa = peraa)
    
    sp.params['add_neb_emission'] = True
    sp.params['add_neb_continuum'] = True
    lam, spec_dusty = sp.get_spectrum(tage = cosmo.age(zval).value+1e-4, peraa = peraa)

    neb_emission_dusty = spec_dusty - spec_noneb_dusty
    k_factor = get_k(f_esc = fesc)
    neb_emission_dusty = neb_emission_dusty * k_factor
    spec = spec_noneb_dusty + neb_emission_dusty
    
#     lam, spec = sp.get_spectrum(tage = cosmo.age(zval).value+1e-4, peraa = peraa)
    spec_ujy = convert_to_microjansky(spec, zval, cosmo)
    
    if type(return_spec) == type(True):
        
        if return_spec == True:
            return lam, spec_ujy
    
        elif return_spec == False:
            filcurves, _, _ = make_filvalkit_simple(lam, zval, fkit_name = filter_list, filt_dir = filt_dir)
            sed = calc_fnu_sed_fast(spec_ujy, filcurves)
            return sed
    
    elif len(return_spec) > 10:
        return convert_to_splined_spec(spec, lam, return_spec, zval)
    
    else:
        raise('Unknown argument for return_spec. Use True or False, or pass a wavelength grid.')
        
    return 0


def generate_atlas_fesc(N_pregrid = 10, priors=priors, initial_seed = 42, store = True, filter_list = 'filter_list.dat', filt_dir = 'filters/', norm_method = 'median', z_step = 0.01, sp = mocksp, cosmology = cosmo, fname = None, path = 'pregrids/', lam_array_spline = [], rseed = None):

    """Generate a pregrid of galaxy properties and their corresponding SEDs
        drawn from the prior distributions defined in priors.py
        Args:
            N_pregrid[int]: Number of SEDs in the pre-grid.
            Nparam[int]: Number of SFH parameters in each tuple
            Initial_seed[int]: Initial seed for random number generation.
            store[Boolean, optional]: Flag whether to store results
                or return as output
            filter_list[filename]: File that contains a list of filter curves.
            z_step[float]: Step size in redshift for filter curve grid.
                Default is 0.01.
            norm_method[string, default = 'max']: normalization for SEDs and SFHs.
                Currently supported arguments are 'none', 'max', 'median', 'area'.
            sp[stellar population object]: FSPS stellar population object. Initialized previously for speed.
            cosmo[astropy cosmology object]: cosmology. Default is FlatLambdaCDM
        Returns:
            [if store == False]
            rand_sfh_tuples[2d numpy array]: N_samples prior-sampled SFH tuples
            rand_Z: prior-sampled metallicity values
            rand_Av: prior-sampled dust attenuation values
            rand_z: prior-sampled redshift values
            rand_seds: Corresponding SEDs in F_\nu (\muJy)
            norm_method: Argument for how SEDs are normalized, pass into fitter
        """

    print('generating atlas with: ')
    print(priors.Nparam, ' tx parameters, ', priors.sfr_prior_type, ' SFR sampling', priors.sfh_treatment,' SFH treatment', priors.met_treatment,' met sampling', priors.dust_model, ' dust attenuation', priors.dust_prior,' dust prior', priors.decouple_sfr,' SFR decoupling.')
    
    if rseed is not None:
        print('setting random seed to :',rseed)
        np.random.seed(rseed)

    zval_all = []
    sfh_tuple_all = []
    sfh_tuple_rec_all = []
    norm_all = []
    dust_all = []
    met_all = []
    sed_all = []
    mstar_all = []
    sfr_all = []
    sim_timeax_all = []
    sim_sfh_all = []
    
    fesc_all = []

    Nparam = priors.Nparam

    for i in tqdm(range(int(N_pregrid))):

        zval = priors.sample_z_prior()

        massval = priors.sample_mass_prior()
        if priors.sfr_prior_type == 'sSFRlognormal':
            sfrval = priors.sample_sfr_prior(zval=zval)
        else:
            sfrval = priors.sample_sfr_prior()
        txparam = priors.sample_tx_prior()
        sfh_tuple = np.hstack((massval, sfrval, Nparam, txparam))
        norm = 1.0
        
        if priors.dynamic_decouple == True:
            priors.decouple_sfr_time = 100*cosmo.age(zval).value/cosmo.age(0.1).value
            #print('decouple time: %.2f myr at z: %.2f' %(priors.decouple_sfr_time,zval))
        sfh, timeax = tuple_to_sfh(sfh_tuple, zval, decouple_sfr = priors.decouple_sfr, decouple_sfr_time = priors.decouple_sfr_time)
        
        temp = calctimes(timeax, sfh, priors.Nparam)
        temptuple = calctimes_to_tuple(temp)
        
        dust = priors.sample_Av_prior()
        met = priors.sample_Z_prior()

        specdetails = [sfh_tuple, dust, met, zval]
        
        try:
            fesc = priors.sample_fesc_prior() # uniform between 0 & 1
        except:
            raise Exception('Fesc prior undefined. Define a priors.sample_fesc_prior()')
    
        if len(lam_array_spline) > 0:
            sed = makespec_fesc(specdetails, fesc, priors, sp, cosmology, filter_list, filt_dir, return_spec = lam_array_spline, peraa = True)
        else:
            lam, spec_ujy = makespec_fesc(specdetails, fesc, priors, sp, cosmology, filter_list, filt_dir, return_spec = True)

            if i == 0:
                # make grid of filter transmission curves for faster computation
                fc_zgrid = np.arange(priors.z_min-z_step, priors.z_max+z_step, z_step)
                temp_fc, temp_lz, temp_lz_lores = make_filvalkit_simple(lam,priors.z_min,fkit_name = filter_list, filt_dir = filt_dir)

                fcs= np.zeros((temp_fc.shape[0], temp_fc.shape[1], len(fc_zgrid)))
                lzs = np.zeros((temp_lz.shape[0], len(fc_zgrid)))
                lzs_lores = np.zeros((temp_lz_lores.shape[0], len(fc_zgrid)))

                for i in (range(len(fc_zgrid))):
                    fcs[0:,0:,i], lzs[0:,i], lzs_lores[0:,i] = make_filvalkit_simple(lam,fc_zgrid[i],fkit_name = filter_list, filt_dir = filt_dir)

            fc_index = np.argmin(np.abs(zval - fc_zgrid))
            sed = calc_fnu_sed_fast(spec_ujy, fcs[0:,0:,fc_index])

        #-------------------------------------------

        if norm_method == 'none':
            # no normalization
            norm_fac = 1
        elif norm_method == 'max':
            # normalize SEDs to 1 - seems to work better than median for small grids
            norm_fac = np.amax(sed)
        elif norm_method == 'median':
            # normalize SEDs to median
            norm_fac = np.nanmedian(sed)
        elif norm_method == 'area':
            # normalize SFH to 10^9 Msun
            norm_fac == 10**(massval - 9)
        else:
            raise ValueError('undefined normalization argument')

        sed = sed/norm_fac
        norm = norm/norm_fac
        mstar = np.log10(sp.stellar_mass / norm_fac)
        sfr = np.log10(sp.sfr / norm_fac)
        sfh_tuple[0] = sfh_tuple[0] - np.log10(norm_fac)
        sfh_tuple[1] = sfh_tuple[1] - np.log10(norm_fac)
        temptuple[0] = temptuple[0] - np.log10(norm_fac)
        temptuple[1] = temptuple[1] - np.log10(norm_fac)

        #-------------------------------------------

        zval_all.append(zval)
        sfh_tuple_all.append(sfh_tuple)
        sfh_tuple_rec_all.append(temptuple)
        norm_all.append(norm)
        dust_all.append(dust)
        met_all.append(met)
        sed_all.append(sed)
        mstar_all.append(mstar)
        sfr_all.append(sfr)
        fesc_all.append(fesc)


    pregrid_dict = {'zval':np.array(zval_all),
                   'sfh_tuple':np.array(sfh_tuple_all),
                   'sfh_tuple_rec':np.array(sfh_tuple_rec_all),
                   'norm':np.array(norm_all), 'norm_method':norm_method,
                   'mstar':np.array(mstar_all), 'sfr':np.array(sfr_all),
                   'dust':np.array(dust_all), 'met':np.array(met_all),
                   'fesc':np.array(fesc_all),
                   'sed':np.array(sed_all)}

    if store == True:

        if fname is None:
            fname = 'sfh_pregrid_size'
        if os.path.exists(path):
            print('Path exists. Saved atlas at : '+path+fname+'_'+str(N_pregrid)+'_Nparam_'+str(Nparam)+'.dbatlas')
        else:
            os.mkdir(path)
            print('Created directory and saved atlas at : '+path+fname+'_'+str(N_pregrid)+'_Nparam_'+str(Nparam)+'.dbatlas')
        try:
            hickle.dump(pregrid_dict,
                        path+fname+'_'+str(N_pregrid)+'_Nparam_'+str(Nparam)+'.dbatlas',
                        compression='gzip', compression_opts = 9)
        except:
            print('storing without compression')
            hickle.dump(pregrid_dict,
                        path+fname+'_'+str(N_pregrid)+'_Nparam_'+str(Nparam)+'.dbatlas')
            
        return

    return pregrid_dict


def plot_posteriors_fesc(sedfit,truths = [], **kwargs):
    
    chi2_array = sedfit.chi2_array
    norm_fac = sedfit.norm_fac
    sed = sedfit.sed
    atlas = sedfit.atlas
    
    set_plot_style()

    if len(truths) > 0:
        corner_truths = truths

    sfrvals = atlas['sfr'].copy()
    sfrvals[sfrvals<-3] = -3
    pg_params = np.vstack([atlas['mstar'],
                           sfrvals,
                           atlas['sfh_tuple'][0:,3:].T,
                           atlas['met'].ravel(),
                           atlas['dust'].ravel(),
                           atlas['fesc'].ravel(),
                           atlas['zval'].ravel()])
    txs = ['t'+'%.0f' %i for i in quantile_names(pg_params.shape[0]-6)]
    pg_labels = ['log M$_*$', 'log SFR', 'Z', 'Av', 'f$_{esc}$', 'z']
    pg_labels[2:2] = txs

    corner_params = pg_params.copy()
    corner_params[0,0:] += np.log10(norm_fac)
    corner_params[1,0:] += np.log10(norm_fac)
    
    if len(truths) > 0:
        figure = corner.corner(corner_params.T, weights = np.exp(-chi2_array/2),
                                labels=pg_labels, truths=corner_truths,
                                plot_datapoints=False, fill_contours=True,
                                bins=20, smooth=1.0,
                                quantiles=(0.16, 0.84), levels=[1 - np.exp(-(1/1)**2/2),1 - np.exp(-(2/1)**2/2)],
                                label_kwargs={"fontsize": 30}, show_titles=True, **kwargs)
    else:
        figure = corner.corner(corner_params.T, weights = np.exp(-chi2_array/2),
                                labels=pg_labels,
                                plot_datapoints=False, fill_contours=True,
                                bins=20, smooth=1.0,
                                quantiles=(0.16, 0.84), levels=[1 - np.exp(-(1/1)**2/2),1 - np.exp(-(2/1)**2/2)],
                                label_kwargs={"fontsize": 30}, show_titles=True, **kwargs)
    figure.subplots_adjust(right=1.5,top=1.5)

    return figure          

def calc_fesc_posteriors(sedfit):
    """
    add f_esc to the calculated posteriors
    """
    fesc_vals = get_quants_key('fesc', 50, sedfit.chi2_array, sedfit.atlas, sedfit.norm_fac)
    sedfit.fesc = fesc_vals

    return
    
