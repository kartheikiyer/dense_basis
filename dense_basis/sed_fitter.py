import numpy as np
from tqdm import tqdm
import scipy.io as sio

# cosmology assumption
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

from .priors import *
from .gp_sfh import *
from .plotter import *

def fit_sed_pregrid(sed, sed_err, pg_theta, fit_mask = [True], norm_method = 'none', return_val = 'params', make_posterior_plots = False, make_sed_plot = False, truths = [np.nan], zbest = None):

    # preprocessing:
    if len(fit_mask) == len(sed):
        fit_mask = fit_mask & (sed > 0)
    else:
        fit_mask = (sed > 0)

    if len(sed) != len(sed_err):
        raise ValueError('SED error array does not match SED')

    sed = sed[fit_mask]
    sed_err = sed_err[fit_mask]

    pg_sfhs, pg_Z, pg_Av, pg_z, pg_seds = pg_theta

    if norm_method == 'none':
        # no normalization
        norm_fac = 1
    elif norm_method == 'max':
        # normalize SEDs to 1
        norm_fac = np.amax(sed)
    elif norm_method == 'median':
        # normalize SEDs to median
        norm_fac = np.median(sed)
    elif norm_method == 'area':
        # normalize SFH to 10^9 Msun - currently not implemented
        norm_fac == 1
    else:
        raise ValueError('undefined normalization argument')

    sed_normed = sed.reshape(-1,1)/norm_fac
    sed_err_normed = sed_err.reshape(-1,1)/norm_fac

    # fitting the SED
    chi2 = np.mean((pg_seds[fit_mask,0:] - sed_normed)**2 / (sed_err_normed)**2, 0)
    
    if zbest is not None:
        redshift_mask = (np.abs(pg_z - zbest) > 0.1*zbest)
        chi2[redshift_mask] = np.amax(chi2)+1e3
        

    if make_sed_plot == True:
        plt.plot(sed_normed)
        plt.plot(sed_normed+sed_err_normed,'k--')
        plt.plot(sed_normed-sed_err_normed,'k--')
        plt.plot(pg_seds[0:,np.argmin(chi2)])
        plt.show()

    if make_posterior_plots == True:
        plt.figure(figsize=(27,4))
        plt.subplot(1,5,1);plt.xlabel('log Stellar Mass')
        plt.hist(pg_sfhs[0,0:]+np.log10(norm_fac),10, weights=np.exp(-chi2/2), normed=True)
        plt.subplot(1,5,2);plt.xlabel('log SFR')
        plt.hist(pg_sfhs[1,0:]+np.log10(norm_fac),10, weights=np.exp(-chi2/2), normed=True)
        plt.subplot(1,5,3);plt.xlabel(r't$_{50}$')
        plt.hist(pg_sfhs[3,0:],10, weights=np.exp(-chi2/2), normed=True)
        plt.subplot(1,5,4);plt.xlabel(r'A$_v$')
        plt.hist(pg_Av,10, weights=np.exp(-chi2/2), normed=True)
        plt.subplot(1,5,5);plt.xlabel(r'log Z/Z$_{\odot}$')
        plt.hist(pg_Z,10, weights=np.exp(-chi2/2), normed=True)
        if len(truths) > 1:
            plt.subplot(1,5,1);tempy = plt.ylim();plt.plot([truths[0],truths[0]],tempy,'k--', lw=3)
            plt.subplot(1,5,2);tempy = plt.ylim();plt.plot([truths[1],truths[1]],tempy,'k--', lw=3)
            plt.subplot(1,5,3);tempy = plt.ylim();plt.plot([truths[2],truths[2]],tempy,'k--', lw=3)
            plt.subplot(1,5,4);tempy = plt.ylim();plt.plot([truths[3],truths[3]],tempy,'k--', lw=3)
            plt.subplot(1,5,5);tempy = plt.ylim();plt.plot([truths[4],truths[4]],tempy,'k--', lw=3)
        plt.show()

    if return_val == 'chi2':
        return chi2
    elif return_val == 'params':
        return calculate_50_16_84_params(chi2, len(sed), np.vstack([pg_sfhs, pg_Av, pg_Z, pg_z]), norm_fac)
    elif return_Val == 'posteriors':
        return calculate_50_16_84_params(chi2, len(sed), np.vstack([pg_sfhs, pg_Av, pg_Z, pg_z]), norm_fac, return_posterior = True)
    else:
        raise ValueError('Undefined return request. Use chi2, params or posteriors.')

def calculate_50_16_84_params(chi2, nbands, params, norm_fac, nbins = 30, return_posterior = False):

    relprob = -0.5*chi2
    relprob = relprob - np.amax(relprob)
    relprob = np.exp(relprob/nbands)

    pg_params = params.copy()

    param_arr = pg_params[0,0:] + np.log10(norm_fac)
    param_arr[param_arr < 0] = 0

    [n,bins,] = np.histogram(param_arr,weights=relprob,density=True,bins=nbins)
    n_c = np.cumsum(n)
    n_c = n_c / np.amax(n_c)
    bin_centers = bins[0:-1] + (bins[1]-bins[0])/2

    mass_median = bin_centers[np.argmin(np.abs(n_c - 0.5))]
    mass_16 = bin_centers[np.argmin(np.abs(n_c - 0.16))]
    mass_84 = bin_centers[np.argmin(np.abs(n_c - 0.84))]
    mass_posterior = [n,bins]

    [n,bins,] = np.histogram(pg_params[1,0:] + np.log10(norm_fac),weights=relprob,density=True,bins=nbins)
    n_c = np.cumsum(n)
    n_c = n_c / np.amax(n_c)
    bin_centers = bins[0:-1] + (bins[1]-bins[0])/2

    sfr_median = bin_centers[np.argmin(np.abs(n_c - 0.5))]
    sfr_16 = bin_centers[np.argmin(np.abs(n_c - 0.16))]
    sfr_84 = bin_centers[np.argmin(np.abs(n_c - 0.84))]
    sfr_posterior = [n,bins]

    [n,bins,] = np.histogram(pg_params[-3,0:],weights=relprob,density=True,bins=nbins)
    n_c = np.cumsum(n)
    n_c = n_c / np.amax(n_c)
    bin_centers = bins[0:-1] + (bins[1]-bins[0])/2

    dust_median = bin_centers[np.argmin(np.abs(n_c - 0.5))]
    dust_16 = bin_centers[np.argmin(np.abs(n_c - 0.16))]
    dust_84 = bin_centers[np.argmin(np.abs(n_c - 0.84))]
    dust_posterior = [n,bins]

    [n,bins,] = np.histogram(pg_params[-2,0:],weights=relprob,density=True,bins=nbins)
    n_c = np.cumsum(n)
    n_c = n_c / np.amax(n_c)
    bin_centers = bins[0:-1] + (bins[1]-bins[0])/2

    met_median = bin_centers[np.argmin(np.abs(n_c - 0.5))]
    met_16 = bin_centers[np.argmin(np.abs(n_c - 0.16))]
    met_84 = bin_centers[np.argmin(np.abs(n_c - 0.84))]
    met_posterior = [n,bins]

    times_median = np.zeros(((pg_params.shape[0] - 6),))
    times_16 = np.zeros(((pg_params.shape[0] - 6),))
    times_84 = np.zeros(((pg_params.shape[0] - 6),))
    times_posterior = []
    for i in range(len(times_median)):
        [n,bins,] = np.histogram(pg_params[3+i,0:],weights=relprob,density=True,bins=nbins)
        n_c = np.cumsum(n)
        n_c = n_c / np.amax(n_c)
        bin_centers = bins[0:-1] + (bins[1]-bins[0])/2
        times_median[i] = bin_centers[np.argmin(np.abs(n_c - 0.5))]
        times_16[i] = bin_centers[np.argmin(np.abs(n_c - 0.16))]
        times_84[i] = bin_centers[np.argmin(np.abs(n_c - 0.84))]
        times_posterior.append([n, bins])

    if return_posterior == False:
        return mass_median, mass_16, mass_84, sfr_median, sfr_16, sfr_84, dust_median, dust_16, dust_84, met_median, met_16, met_84, times_median, times_16, times_84
    else:
        return mass_posterior, sfr_posterior, dust_posterior, met_posterior, times_posterior
