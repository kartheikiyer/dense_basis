import numpy as np
from tqdm import tqdm
import scipy.io as sio

# cosmology assumption
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

from .priors import *
from .gp_sfh import *

import fsps
mocksp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1,sfh=0, imf_type=1, logzsol=0.0, dust_type=2, dust2=0.0, add_neb_emission=True)
print('Initialized stellar population with FSPS.')



#-----------------------------------------------------------------------
#                     Calculating spectra and SEDs
#-----------------------------------------------------------------------

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

def make_spec(sfh_tuple, metval, dustval, zval, igmval = True, return_lam = False, sp = mocksp, cosmology = cosmo):

    """Use FSPS to generate a spectrum corresponding to a set of
        input galaxy properties.
        Args:
            sfh_tuple[1d numpy array]: SFH parameters, input to gp_sfh_sklearn
            metval[float]: log metallicity wrt Solar.
            dustval[float]: Calzetti dust attenuation
            zval[float]: redshift
            igmval[float, optional]: Include IGM absorption (Default is True)
            return_lam[boolean, optional]: Return a wavelength array along
                with the spectrum (Default is True)
            sp[stellar population object]: FSPS stellar population object.
                Initialized previously for speed.
            cosmo[astropy cosmology object]: cosmology.
                Default is FlatLambdaCDM
        Returns:
            spec[1d numpy array]: Spectrum in F_\nu (\muJy)
            lam[1d numpy array]: Wavelength in Angstrom corresponding to spectrum
        """

    sp.params['add_igm_absorption'] = igmval
    sp.params['zred'] = zval
    sfh, timeax = gp_sfh_sklearn(sfh_tuple, zval = zval)
    timeax = timeax/1e9
    sp.params['sfh'] = 3
    sp.set_tabular_sfh(timeax, sfh)
    sp.params['cloudy_dust'] = True
    sp.params['dust_type'] = 2
    # sp.params['dust1'] = dust1_rand
    sp.params['dust2'] = dustval
    sp.params['logzsol'] = metval
    [lam_arr,spec] = sp.get_spectrum(tage = np.amax(timeax))
    spec_ujy = convert_to_microjansky(spec,zval,cosmology)

    # add option to work in energy or F_\lambda as well

    if return_lam == True:
        return spec_ujy, lam_arr
    else:
        return spec_ujy


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
            sed[1d numpy array, len = Nfilters]: SED in F_\nu (\muJy)
        """

    spec = make_spec(sfh_tuple, metval, dustval, zval, igmval = True, return_lam = False, sp = mocksp, cosmology = cosmo)
    sed = calc_fnu_sed_fast(spec, filcurves)
    return sed

def make_filvalkit_simple(lam,z, fkit_name = 'filter_list.dat' ,vb=False, filt_dir = 'dense_basis/filters/'):

    # import os
    # print(os.listdir())
    lam_z = (1+z)*lam

    # change this to logspace later to avoid problems
    # when dealing with FIR filters.
    lam_z_lores = np.arange(2000,150000,2000)

    f = open(fkit_name,'r')

    # read in the file with the filter curves
    temp = f.readlines()
    if vb == True:
        print('number of filters to be read in: '+str(len(temp)))

    numlines = len(temp)

    filcurves = np.zeros((len(lam_z),numlines))
    filcurves_lores = np.zeros((len(lam_z_lores),numlines))

    if vb == True:
        plt.figure(figsize= (12,6))

    for i in range(numlines):

        filt_name = filt_dir+temp[i]

        if i == numlines-1:
            #print(filt_name[0:][0:])
            tempfilt = np.loadtxt(filt_name[0:][0:-1])
        else:
            tempfilt = np.loadtxt(filt_name[0:][0:-1])

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

    f.close()

    if vb == True:

        print('created filcurve array splined to input lambda array at redshift: '+str(z))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel(r'log $\lambda [\AA]$',fontsize=18)
        plt.ylabel('Filter transmission')
        plt.axis([3.5,5,0,1])
        plt.show()

    return filcurves, lam_z, lam_z_lores

def calc_fnu_sed(spec,z,lam, fkit_name = 'filter_list.dat'):

    #kitname = '/home/kiyer/Documents/codex_dense_basis_data/new_stoc_analysis/letter1_fkilt/codex_filter_val_kit_fsps_'+str(z)+'.mat'
    #filt_data = sio.loadmat(kitname)
    #lam_z = np.array(filt_data['lambda_z'])
    #lam_z_lores = np.array(filt_data['lambda_z_lores'])
    #lam_z_input = lam*(1+z)

    filcurves,lam_z, lam_z_lores = make_filvalkit_simple(lam,z,fkit_name = fkit_name)

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



#-----------------------------------------------------------------------
#                         Pre-grid generation
#-----------------------------------------------------------------------

def generate_pregrid(N_pregrid = 10, Nparam = 1, initial_seed = 12, store = False, filter_list = 'filter_list.dat', z_step = 0.01, sp = mocksp, cosmology = cosmo):

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
            filcurves[2d array, (len(spec), Nfilters)]: filter transmission curves splined to wavelength array. Generated using the make_filvalkit_simple function.
            igmval[float, optional]: Include IGM absorption (Default is True)
            return_lam[boolean, optional]: Return a wavelength array along with the spectrum (Default is True)
            sp[stellar population object]: FSPS stellar population object. Initialized previously for speed.
            cosmo[astropy cosmology object]: cosmology. Default is FlatLambdaCDM
        Returns:
            [if store == False]
            rand_sfh_tuples[2d numpy array]: N_samples prior-sampled SFH tuples
            rand_Z: prior-sampled metallicity values
            rand_Av: prior-sampled dust attenuation values
            rand_z: prior-sampled redshift values
            rand_seds: Corresponding SEDs in F_\nu (\muJy)
        """

    rand_sfh_tuple, rand_Z, rand_Av, rand_z = sample_all_params(random_seed = initial_seed, Nparam = Nparam)
    _, lam = make_spec(rand_sfh_tuple, rand_Z, rand_Av, rand_z, igmval = True, return_lam = True, sp = mocksp, cosmology = cosmo)

    fc_zgrid = np.arange(z_min-z_step, z_max+z_step, z_step)

    temp_fc, temp_lz, temp_lz_lores = make_filvalkit_simple(lam,z_min,fkit_name = filter_list)

    fcs= np.zeros((temp_fc.shape[0], temp_fc.shape[1], len(fc_zgrid)))
    lzs = np.zeros((temp_lz.shape[0], len(fc_zgrid)))
    lzs_lores = np.zeros((temp_lz_lores.shape[0], len(fc_zgrid)))

    for i in tqdm(range(len(fc_zgrid))):
        fcs[0:,0:,i], lzs[0:,i], lzs_lores[0:,i] = make_filvalkit_simple(lam,fc_zgrid[i],fkit_name = filter_list)

    rand_sfh_tuples = np.zeros((Nparam+3, N_pregrid))
    rand_Z = np.zeros((N_pregrid,))
    rand_Av = np.zeros((N_pregrid,))
    rand_z = np.zeros((N_pregrid,))
    rand_seds = np.zeros((fcs.shape[1],N_pregrid))

    for i in tqdm(range(N_pregrid)):
        rand_sfh_tuples[0:,i], rand_Z[i], rand_Av[i], rand_z[i] = sample_all_params(random_seed = initial_seed+i*7, Nparam = Nparam)
        fc_index = np.argmin(np.abs(rand_z[i] - fc_zgrid))
        rand_seds[0:,i] = make_sed_fast(rand_sfh_tuples[0:,i], rand_Z[i], rand_Av[i], rand_z[i], fcs[0:,0:,fc_index], sp = mocksp, cosmology = cosmo)

    if store == True:
        pregrid_mdict = {'rand_sfh_tuples':rand_sfh_tuples, 'rand_Z':rand_Z, 'rand_Av':rand_Av, 'rand_z':rand_z, 'rand_seds':rand_seds}
        sio.savemat('dense_basis/pregrids/sfh_pregrid_size_'+str(N_pregrid)+'.mat', mdict = pregrid_mdict)
        return

    return rand_sfh_tuples, rand_Z, rand_Av, rand_z, rand_seds