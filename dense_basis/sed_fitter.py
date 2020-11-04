import numpy as np
from tqdm import tqdm
import scipy.io as sio

# cosmology assumption
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

from .priors import *
from .gp_sfh import *
from .plotter import *

def normerr(nf, pg_seds, sed, sed_err, fit_mask):
    c2v = np.amin(np.mean((pg_seds[fit_mask,0:] - sed.reshape(-1,1)/nf)**2 / (sed_err.reshape(-1,1)/nf)**2, 0))
    return c2v

def evaluate_sed_likelihood(sed, sed_err, atlas, fit_mask = [], zbest = None, deltaz = None):

    """
    Evaluate the likeihood of model SEDs in an atlas given
    an observed SED with uncertainties.
    """

    # preprocessing:
    if len(fit_mask) == len(sed):
        fit_mask = fit_mask & (sed > 0)
    else:
        fit_mask = (sed > 0)

    if len(sed) != len(sed_err):
        raise ValueError('SED uncertainty array does not match SED')

    sed = sed[fit_mask]
    sed_err = sed_err[fit_mask]
    pg_seds = atlas['sed'].copy().T

    nfmin = minimize(normerr, np.nanmedian(sed), args = (pg_seds, sed, sed_err, fit_mask))
    norm_fac = nfmin['x'][0]

    sed_normed = sed.reshape(-1,1)/norm_fac
    sed_err_normed = sed_err.reshape(-1,1)/norm_fac

    chi2 = np.mean((pg_seds[fit_mask,0:] - sed_normed)**2 / (sed_err_normed)**2, 0)

    if zbest is not None:
        pg_z = atlas['zval']
        #redshift_mask = (np.abs(pg_z - zbest) > 0.1*zbest)
        redshift_mask = (np.abs(pg_z - zbest) > deltaz)
        chi2[redshift_mask] = np.amax(chi2)+1e3

    return chi2, norm_fac

def get_quants(chi2_array, cat, norm_fac, bw_dex = 0.001, return_uncert = True, vb = False):

    """
    remember to check bin limits and widths before using quantities if you're fitting a new sample
    """

    #relprob = np.exp(-(chi2_array*np.sum(obs_sed>0))/2)
    relprob = np.exp(-(chi2_array)/2)
    if vb == True:
        plt.hist(relprob,100)
        plt.yscale('log')
        plt.show()


    # ---------------- stellar mass and SFR -----------------------------------

    mstar_vals = calc_percentiles(cat['mstar'] + np.log10(norm_fac),
                                 weights = relprob,
                                 bins = np.arange(4,14,bw_dex),
                                 percentile_values= [50.,16.,84.], vb=vb)

    sfr_vals = calc_percentiles(cat['sfr'] + np.log10(norm_fac),
                                 weights = relprob,
                                 bins = np.arange(-6,4,bw_dex),
                                 percentile_values= [50.,16.,84.],vb=vb)

     # ---------------- SFH -----------------------------------

    sfh_tuple_vals = np.zeros((3, cat['sfh_tuple'].shape[1]))

    for i in range(cat['sfh_tuple'].shape[1]):

        if i == 0:
            sfh_tuple_vals[0:,i] = calc_percentiles(cat['sfh_tuple'][0:,0] + np.log10(norm_fac),
                                 weights = relprob, bins = np.arange(4,14,bw_dex),
                                 percentile_values= [50.,16.,84.], vb=vb)
        elif i == 1:
            sfh_tuple_vals[0:,i] = calc_percentiles(cat['sfh_tuple'][0:,1] + np.log10(norm_fac),
                                 weights = relprob, bins = np.arange(-6,4,bw_dex),
                                 percentile_values= [50.,16.,84.], vb=vb)
        elif i == 2:
            sfh_tuple_vals[0:,i] = sfh_tuple_vals[0:,i] + np.nanmean(cat['sfh_tuple'][0:,2])
        else:
            sfh_tuple_vals[0:,i] = calc_percentiles(cat['sfh_tuple'][0:,1],
                                 weights = relprob, bins = np.arange(0,1,bw_dex),
                                 percentile_values= [50.,16.,84.], vb=vb)

    # ------------------------ dust, metallicity, redshift ----------------------

    Av_vals = calc_percentiles(cat['dust'].ravel(),
                                 weights = relprob,
                                 bins = np.arange(0,np.amax(cat['dust']),bw_dex),
                                 percentile_values= [50.,16.,84.],vb=vb)

    Z_vals = calc_percentiles(cat['met'].ravel(),
                                 weights = relprob,
                                 bins = np.arange(-1.5, 0.5,bw_dex),
                                 percentile_values= [50.,16.,84.],vb=vb)

    z_vals = calc_percentiles(cat['zval'].ravel(),
                                 weights = relprob,
                                 bins = np.arange(np.amin(cat['zval']), np.amax(cat['zval']),bw_dex),
                                 percentile_values= [50.,16.,84.],vb=vb)

    return [mstar_vals, sfr_vals, Av_vals, Z_vals, z_vals, sfh_tuple_vals]

def get_flat_posterior(qty, weights, bins):

    post, xaxis = np.histogram(qty, weights=weights, bins=bins)
    prior, xaxis = np.histogram(qty, bins=bins)

#     post_weighted = post/prior
    post_weighted = post
    post_weighted[np.isnan(post_weighted)] = 0

    return post_weighted, xaxis

def calc_percentiles(qty, weights, bins, percentile_values, vb = False):

    qty_percentile_values = np.zeros((len(percentile_values),))

    post_weighted, xaxis = get_flat_posterior(qty, weights, bins)
    bw = np.nanmean(np.diff(xaxis))

    normed_cdf = np.cumsum(post_weighted)/np.amax(np.cumsum(post_weighted))

    for i in range(len(percentile_values)):
        qty_percentile_values[i] = xaxis[0:-1][np.argmin(np.abs(normed_cdf - percentile_values[i]/100))] + bw/2
        if (qty_percentile_values[i] == xaxis[0]+bw/2) or (qty_percentile_values[i] == xaxis[-1]+bw/2):
            qty_percentile_values[i] = np.nan

    if vb == True:

        qty_50 = xaxis[0:-1][np.argmin(np.abs(normed_cdf - 0.5))] + bw/2
        qty_16 = xaxis[0:-1][np.argmin(np.abs(normed_cdf - 0.16))] + bw/2
        qty_84 = xaxis[0:-1][np.argmin(np.abs(normed_cdf - 0.84))] + bw/2

        plt.plot(xaxis[0:-1]+bw/2, post_weighted)
        bf_value = qty[np.argmax(weights)]
        tempy = plt.ylim()
        plt.plot([bf_value, bf_value],[0,tempy[1]],'k-',label='best-fit')
        plt.plot([qty_50, qty_50],[0,tempy[1]],'-',label='50th percentile')
        plt.plot([qty_16, qty_16],[0,tempy[1]],'-',label='16th percentile')
        plt.plot([qty_84, qty_84],[0,tempy[1]],'-',label='84th percentile')
        print(bf_value, qty_50, np.argmax(weights), np.amax(weights))
        plt.legend(edgecolor='w',fontsize=14)
        plt.show()

    return qty_percentile_values



def fit_sed_pregrid_old(sed, sed_err, pg_theta, fit_mask = [True], fit_method = 'chi2', norm_method = 'none', return_val = 'params', make_posterior_plots = False, make_sed_plot = False, truths = [np.nan], zbest = None, deltaz = None):
    """
    DEPRECATED
    """
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

    if fit_method == 'chi2':

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
        elif norm_method == 'pg_band':
            # normalize to band that pg_seds peak in
            norm_fac = 1
        else:
            raise ValueError('undefined normalization argument')

        sed_normed = sed.reshape(-1,1)/norm_fac
        sed_err_normed = sed_err.reshape(-1,1)/norm_fac

        # fitting the SED
        chi2 = np.mean((pg_seds[fit_mask,0:] - sed_normed)**2 / (sed_err_normed)**2, 0)

        if norm_method == 'pg_band':
            # which bands do the SEDs peak in?
            chi2 = np.ones_like(chi2)*1e3
            pg_sed_peaks = np.argmax(pg_seds,0)
            for tempi in range(len(sed)):
                if fit_mask[i] == True:
                    temp_norm_fac = sed[i]
                    temp_sed_normed = sed.reshape(-1,1)/temp_norm_fac
                    temp_err_normed = sed_err.reshape(-1,1)/temp_norm_fac
                    chi2[pg_sed_peaks == i] = np.mean((pg_seds[fit_mask,pg_sed_peaks == i] - sed_normed)**2 / (sed_err_normed)**2, 0)
            best_chi2 = np.argmin(chi2)
            norm_fac = sed[pg_sed_peaks[best_chi2]]

        if zbest is not None:
            #redshift_mask = (np.abs(pg_z - zbest) > 0.1*zbest)
            redshift_mask = (np.abs(pg_z - zbest) > deltaz)
            chi2[redshift_mask] = np.amax(chi2)+1e3


        if make_sed_plot == True:
            plt.plot(sed_normed)
            plt.plot(sed_normed+sed_err_normed,'k--')
            plt.plot(sed_normed-sed_err_normed,'k--')
            plt.plot(pg_seds[fit_mask,np.argmin(chi2)])
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

    elif fit_method == 'dot':
        dot_prods = np.linalg.multi_dot((pg_seds[fit_mask,0:].T, sed[fit_mask]/np.amax(sed)))/np.linalg.norm(pg_seds,axis=0)
        best_dotprod = np.argmax(dot_prods)
        return 1 - dot_prods/np.amax(dot_prods)

    else:
        print('unknown fit_method - use chi2 or dot.')
        return 0

def get_sfr_mstar(chi2_array, pg_theta, obs_sed, bw_dex = 0.01, return_uncert = True):
    """
    DEPRECATED
    """


    a,b = np.histogram(pg_theta[0][0,0:] + np.log10(np.amax(obs_sed)),
                       weights = np.exp(-chi2_array/2),
                       bins = np.arange(8,12,bw_dex))
    mstar_50 = b[0:-1][np.argmin(np.abs(np.cumsum(a)/np.amax(np.cumsum(a)) - 0.5))] + bw_dex/2
    mstar_16 = b[0:-1][np.argmin(np.abs(np.cumsum(a)/np.amax(np.cumsum(a)) - 0.16))] + bw_dex/2
    mstar_84 = b[0:-1][np.argmin(np.abs(np.cumsum(a)/np.amax(np.cumsum(a)) - 0.84))] + bw_dex/2

    a,b = np.histogram(pg_theta[0][1,0:] + np.log10(np.amax(obs_sed)),
                       weights = np.exp(-chi2_array/2),
                       bins = np.arange(-3,3,bw_dex))
    sfr_50 = b[0:-1][np.argmin(np.abs(np.cumsum(a)/np.amax(np.cumsum(a)) - 0.5))] + bw_dex/2
    sfr_16 = b[0:-1][np.argmin(np.abs(np.cumsum(a)/np.amax(np.cumsum(a)) - 0.16))] + bw_dex/2
    sfr_84 = b[0:-1][np.argmin(np.abs(np.cumsum(a)/np.amax(np.cumsum(a)) - 0.84))] + bw_dex/2

    if return_uncert == False:
        return mstar_50, sfr_50
    else:
        return [mstar_50, mstar_16, mstar_84], [sfr_50, sfr_16, sfr_84]

def calculate_50_16_84_params(chi2, nbands, params, norm_fac, nbins = 30, return_posterior = False):
    """
    DEPRECATED
    """

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
