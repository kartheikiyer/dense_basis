import numpy as np

# cosmology assumption
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

from .priors import *
from .gp_sfh import *

import fsps
mocksp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1,sfh=0, imf_type=1, logzsol=0.0, dust_type=2, dust2=0.0, add_neb_emission=True)
print('Initialized stellar population with FSPS.')

#---------Functions to generate spectra and SEDs----------

def convert_to_microjansky(spec,z,cosmology):
    temp = (1+z)*spec *1e6 * 1e23*3.48e33/(4*np.pi*3.086e+24*3.086e+24*cosmo.luminosity_distance(z).value*cosmo.luminosity_distance(z).value)
    #temp = spec *1e6 * 1e23*3.48e33/(4*np.pi*3.086e+24*3.086e+24*cosmo.luminosity_distance(z).value*cosmo.luminosity_distance(z).value*(1+z))
    return temp

def make_spec(sfh_tuple, metval, dustval, zval, igmval = True, return_lam = False, sp = mocksp, cosmology = cosmo):

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
