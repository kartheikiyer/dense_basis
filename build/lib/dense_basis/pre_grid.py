import numpy as np
from tqdm import tqdm
import scipy.io as sio
import os
import pkg_resources
import hickle

# cosmology assumption
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

from .priors import *
from .gp_sfh import *

try:
    import fsps
    mocksp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1,sfh=0, imf_type=1, logzsol=0.0, dust_type=2, dust2=0.0, add_neb_emission=True)
    print('Starting dense_basis. please wait ~ a minute for the FSPS backend to initialize.')

except:
    mocksp = None
    print('Starting dense_basis. Failed to load FSPS, only GP-SFH module will be available.')

priors = Priors()

#-----------------------------------------------------------------------
#                     Calculating spectra and SEDs
#-----------------------------------------------------------------------

def get_path(filename):
    return os.path.dirname(os.path.realpath(filename))

def convert_to_microjansky(spec,z,cosmology):
    """Convert a spectrum in L_\nu (Solar luminosity/Hz, default python-fsps
        output) to F_\nu units in microjansky
        Args:
            spec[1d numpy array]: Spectrum output by FSPS (L_\nu)
            z[float]: redshift for computing luminosity_distance
            cosmo[astropy cosmology object]: cosmology
        Returns:
            spec[1d numpy array]: Spectrum in F_\nu (\muJy)
        """

    temp = (1+z)*spec *1e6 * 1e23*3.48e33/(4*np.pi*3.086e+24*3.086e+24*cosmo.luminosity_distance(z).value*cosmo.luminosity_distance(z).value)
    #temp = spec *1e6 * 1e23*3.48e33/(4*np.pi*3.086e+24*3.086e+24*cosmo.luminosity_distance(z).value*cosmo.luminosity_distance(z).value*(1+z))
    return temp

def makespec_atlas(atlas, galid, priors, sp, cosmo, filter_list = [], filt_dir = [], return_spec = False):

    sfh_tuple = atlas['sfh_tuple'][galid,0:]
    zval = atlas['zval'][galid]
    dust = atlas['dust'][galid]
    met = atlas['met'][galid]

    specdetails = [sfh_tuple, dust, met, zval]
    if priors.dynamic_decouple == True:
        priors.decouple_sfr_time = 100*cosmo.age(zval).value/cosmo.age(0.1).value

    output = makespec(specdetails, priors, sp, cosmo, filter_list, filt_dir, return_spec)

    return output

def makespec(specdetails, priors, sp, cosmo, filter_list = [], filt_dir = [], return_spec = False, peraa = False, input_sfh = False):
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
    lam, spec = sp.get_spectrum(tage = cosmo.age(zval).value+1e-4, peraa = peraa)
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

def convert_to_splined_spec(spec_peraa, lam, lam_spline, redshift, cosmology = cosmo):

    spec = spec_peraa
    spec_ergsec = spec*3.839e33*1e17/(1+redshift)
    lum_dist = cosmology.luminosity_distance(redshift).value
    spec_ergsec_cm2 = spec_ergsec/(4*np.pi*3.086e+24*3.086e+24*lum_dist*lum_dist)
    # this should have units of 1e-17 erg/s/Ang
    # need units to be flux in 1e-17 erg/s/cm^2/Ang/spaxel
    spec_spline = np.interp(lam_spline, lam*(1+redshift), spec_ergsec_cm2)

    #return spec_ergsec_cm2
    return spec_spline

def make_sed_fast(sfh_tuple, metval, dustval, zval, filcurves, igmval = True, return_lam = False, sp = mocksp, cosmology = cosmo):

    """Generate and multiply a spectrum with previously generated
        filter transmission curves to get SED.
        ---WARNING: assumes filcurves have been generated at the correct zval---
        Args:
            sfh_tuple[1d numpy array]: Spectrum output by FSPS (L_\nu)
            metval[float]: log metallicity wrt Solar.
            dustval[float]: Calzetti dust attenuation
            zval[float]: redshift
            filcurves[2d array, (len(spec), Nfilters)]: filter transmission
                curves splined to wavelength array. Generated using the
                make_filvalkit_simple function.
            igmval[float, optional]: Include IGM absorption (Default is True)
            return_lam[boolean, optional]: Return a wavelength array
                along with the spectrum (Default is True)
            sp[stellar population object]: FSPS stellar population object.
                Initialized previously for speed.
            cosmo[astropy cosmology object]: cosmology.
                Default is FlatLambdaCDM.
        Returns:
            sed [1d numpy array, len = Nfilters]: SED in F_nu (muJy)
        """

    spec, logsfr, logmstar = make_spec(sfh_tuple, metval, dustval, zval, igmval = True, return_ms = True, return_lam = False, sp = sp, cosmology = cosmo)
    sed = calc_fnu_sed_fast(spec, filcurves)
    return sed, logsfr, logmstar

def make_filvalkit_simple(lam,z, fkit_name = 'filter_list.dat' ,vb=False, filt_dir = 'filters/'):

    # import os
    # print(os.listdir())
    lam_z = (1+z)*lam

    # change this to logspace later to avoid problems
    # when dealing with FIR filters.
    lam_z_lores = np.arange(2000,150000,2000)

    if filt_dir == 'internal':
        resource_package = __name__
        resource_path = '/'.join(('filters', fkit_name))  # Do not use os.path.join()
        template = pkg_resources.resource_string(resource_package, resource_path)
        f = template.split()
        temp = template.split()
    else:
        if filt_dir[-1] == '/':
            f = open(filt_dir+fkit_name,'r')
        else:
            f = open(filt_dir+'/'+fkit_name,'r')
        temp = f.readlines()
        #resource_package = os.path.dirname(os.path.realpath(__file__))
        #resource_package = __file__
        # resource_package = __name__
        # if filt_dir[-1] == '/':
        #     resource_path = '/'.join((filt_dir[0:-1], fkit_name))
        # else:
        #     resource_path = '/'.join((filt_dir, fkit_name))  # Do not use os.path.join()
        #
        # print(resource_package)
        # print(resource_path)
        # template = pkg_resources.resource_string(resource_package, resource_path)
        # f = template.split()
        # temp = template.split()
        #
        # print(temp)

    # read in the file with the filter curves

    if vb == True:
        print('number of filters to be read in: '+str(len(temp)))

    numlines = len(temp)
    if temp[-1] == '\n':
        numlines = len(temp)-1

    filcurves = np.zeros((len(lam_z),numlines))
    filcurves_lores = np.zeros((len(lam_z_lores),numlines))

    if vb == True:
        plt.figure(figsize= (12,6))

    for i in range(numlines):

        if (filt_dir == 'internal') & (fkit_name == 'filter_list_goodss.dat'):
            tempfilt = np.loadtxt(get_file('filters/filter_curves/goods_s', temp[i][22:].decode("utf-8")))
        elif (filt_dir == 'internal') & (fkit_name == 'filter_list_goodsn.dat'):
            tempfilt = np.loadtxt(get_file('filters/filter_curves/goods_n', temp[i][22:].decode("utf-8")))
        elif (filt_dir == 'internal') & (fkit_name == 'filter_list_cosmos.dat'):
            tempfilt = np.loadtxt(get_file('filters/filter_curves/cosmos', temp[i][21:].decode("utf-8")))
        elif (filt_dir == 'internal') & (fkit_name == 'filter_list_egs.dat'):
            tempfilt = np.loadtxt(get_file('filters/filter_curves/egs', temp[i][18:].decode("utf-8")))
        elif (filt_dir == 'internal') & (fkit_name == 'filter_list_uds.dat'):
            tempfilt = np.loadtxt(get_file('filters/filter_curves/uds', temp[i][18:].decode("utf-8")))
        else:
            if filt_dir[-1] == '/':
                filt_name = filt_dir+temp[i]
            else:
                filt_name = filt_dir+'/'+temp[i]
            if i == numlines-1:
                #print(filt_name[0:][0:])
                tempfilt = np.loadtxt(filt_name[0:-1])
            else:
                if os.path.exists(filt_name[0:][0:-1]):
                    tempfilt = np.loadtxt(filt_name[0:][0:-1])
                else:
                    raise Exception('filters not found. are you sure the folder exists at the right relative path?')

        temp_lam_arr = tempfilt[0:,0]
        temp_response_curve = tempfilt[0:,1]

        bot_val = np.amin(np.abs(lam_z - np.amin(temp_lam_arr)))
        bot_in = np.argmin(np.abs(lam_z - np.amin(temp_lam_arr)))
        top_val = np.amin(np.abs(lam_z - np.amax(temp_lam_arr)))
        top_in = np.argmin(np.abs(lam_z - np.amax(temp_lam_arr)))

        curve_small = np.interp(lam_z[bot_in+1:top_in-1],temp_lam_arr,temp_response_curve)
        splinedcurve = np.zeros((lam_z.shape))
        splinedcurve[bot_in+1:top_in-1] = curve_small
        if np.amax(splinedcurve) > 1:
            splinedcurve = splinedcurve / np.amax(splinedcurve)

        filcurves[0:,i] = splinedcurve

        if vb == True:
            plt.plot(np.log10(lam_z),splinedcurve,'k--',label=filt_name[0:][0:-1])

    if (filt_dir != 'internal'):
        f.close()

    if vb == True:

        print('created filcurve array splined to input lambda array at redshift: '+str(z))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel(r'log $\lambda [\AA]$',fontsize=18)
        plt.ylabel('Filter transmission')
        plt.axis([3.5,5,0,1])
        plt.show()

    return filcurves, lam_z, lam_z_lores

def calc_fnu_sed(spec,z,lam, fkit_name = 'filter_list.dat', filt_dir = 'filters/'):

    #kitname = '/home/kiyer/Documents/codex_dense_basis_data/new_stoc_analysis/letter1_fkilt/codex_filter_val_kit_fsps_'+str(z)+'.mat'
    #filt_data = sio.loadmat(kitname)
    #lam_z = np.array(filt_data['lambda_z'])
    #lam_z_lores = np.array(filt_data['lambda_z_lores'])
    #lam_z_input = lam*(1+z)

    filcurves,lam_z, lam_z_lores = make_filvalkit_simple(lam,z,fkit_name = fkit_name, filt_dir = filt_dir)

    #spec_interp = np.interp(lam_z,lam_z_input,spec)

    nu = 3e18/lam_z

    # change this to appropriate normalisation. see documentation.
    #fnuspec = spec*768.3
    fnuspec = spec
    #print(lam_z_lores.shape,lam_z.shape,fnuspec.shape)
    #fnuspec_lores = np.interp(lam_z_lores,lam_z,fnuspec)

    #filcurves = np.array(filt_data['filtercurve'])
    filvals = np.zeros((filcurves.shape[1],))
    for tindex in range(filcurves.shape[1]):
        #print(tindex)
        temp1 = filcurves[np.where(filcurves[0:,tindex]>0),tindex]
        temp2 = fnuspec[np.where(filcurves[0:,tindex]>0)]
        filvals[tindex] = np.sum(temp1.T[0:,0]*temp2)/np.sum(filcurves[0:,tindex])
        #print(filvals[tindex])
        #filvals[tindex] = temp1.T[0:,0]*temp2*np.sum(filcurves[0:,tindex])
        #print(temp1.T.shape,temp2.shape,temp3.shape,np.sum(filcurves[0:,i]))
        #filvals[i] = filcurves[np.where(filcurves[0:,i]>0),i]*fnuspec[np.where(filcurves[0:,i]>0)]/np.sum(filcurves[0:,i])
    return filvals

#def calc_fnu_sed_fast(spec,z,lam,filcurves,lam_z, lam_z_lores):
def calc_fnu_sed_fast(fnuspec,filcurves):
    filvals = np.zeros((filcurves.shape[1],))
    for tindex in range(filcurves.shape[1]):
        temp1 = filcurves[np.where(filcurves[0:,tindex]>0),tindex]
        temp2 = fnuspec[np.where(filcurves[0:,tindex]>0)]
        filvals[tindex] = np.sum(temp1.T[0:,0]*temp2)/np.sum(filcurves[0:,tindex])
    return filvals

def generate_atlas(N_pregrid = 10, priors=priors, initial_seed = 42, store = True, filter_list = 'filter_list.dat', filt_dir = 'filters/', norm_method = 'median', z_step = 0.01, sp = mocksp, cosmology = cosmo, fname = None, path = 'pregrids/', lam_array_spline = [], rseed = None):

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

        #-------------------------------------------

#         sp.params['sfh'] = 3
#         sp.set_tabular_sfh(timeax, sfh)
#         sp.params['dust2'] = dust
#         sp.params['logzsol'] = met
#         sp.params['add_igm_absorption'] = True
#         sp.params['add_neb_emission'] = True
#         sp.params['add_neb_continuum'] = True
#         sp.params['imf_type'] = 1 # Chabrier
#         sp.params['zred'] = zval

#         lam, spec = sp.get_spectrum(tage = cosmology.age(zval).value)
#         spec_ujy = convert_to_microjansky(spec, zval, cosmology)

        specdetails = [sfh_tuple, dust, met, zval]

        if len(lam_array_spline) > 0:
            sed = makespec(specdetails, priors, sp, cosmology, filter_list, filt_dir, return_spec = lam_array_spline, peraa = True)
        else:
            lam, spec_ujy = makespec(specdetails, priors, sp, cosmology, filter_list, filt_dir, return_spec = True)

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


    pregrid_dict = {'zval':np.array(zval_all),
                   'sfh_tuple':np.array(sfh_tuple_all),
                   'sfh_tuple_rec':np.array(sfh_tuple_rec_all),
                   'norm':np.array(norm_all), 'norm_method':norm_method,
                   'mstar':np.array(mstar_all), 'sfr':np.array(sfr_all),
                   'dust':np.array(dust_all), 'met':np.array(met_all),
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

def load_atlas(fname, N_pregrid, N_param, path = 'pregrids/'):

    fname_full = path+fname+'_'+str(N_pregrid)+'_Nparam_'+str(N_param)+'.dbatlas'
    cat = hickle.load(fname_full)
    return cat

def quantile_names(N_params):
    return (np.round(np.linspace(0,100,N_params+2)))[1:-1]


#---------------- deprecated functions -----------------------

# def make_spec_deprecated(sfh_tuple, metval, dustval, zval, igmval = True, return_lam = False, return_ms = False, sp = mocksp, cosmology = cosmo):

#     """Use FSPS to generate a spectrum corresponding to a set of
#         input galaxy properties.
#         Args:
#             sfh_tuple[1d numpy array]: SFH parameters, input to gp_sfh_sklearn
#             metval[float]: log metallicity wrt Solar.
#             dustval[float]: Calzetti dust attenuation
#             zval[float]: redshift
#             igmval[float, optional]: Include IGM absorption (Default is True)
#             return_lam[boolean, optional]: Return a wavelength array along
#                 with the spectrum (Default is True)
#             sp[stellar population object]: FSPS stellar population object.
#                 Initialized previously for speed.
#             cosmo[astropy cosmology object]: cosmology.
#                 Default is FlatLambdaCDM
#         Returns:
#             spec[1d numpy array]: Spectrum in F_\nu (\muJy)
#             lam[1d numpy array]: Wavelength in Angstrom corresponding to spectrum
#         """

#     sp.params['gas_logu'] = -2
#     sp.params['sfh'] = 3
#     sp.params['cloudy_dust'] = True
#     sp.params['dust_type'] = 2

#     sp.params['add_igm_absorption'] = igmval
#     sp.params['zred'] = zval

#     sfh, timeax = tuple_to_sfh(sfh_tuple, zval = zval)
#     timeax = timeax
#     sp.set_tabular_sfh(timeax, sfh)

#     # sp.params['dust1'] = dust1_rand
#     sp.params['dust2'] = dustval
#     sp.params['logzsol'] = metval
#     [lam_arr,spec] = sp.get_spectrum(tage = np.amax(timeax))
#     spec_ujy = convert_to_microjansky(spec,zval,cosmology)

#     # add option to work in energy or F_\lambda as well

#     if return_ms == True:
#         return spec_ujy, np.log10(sfh[-1]), np.log10(sp.stellar_mass)
#     elif return_lam == True:
#         return spec_ujy, lam_arr
#     else:
#         return spec_ujy
