import numpy as np
from tqdm import tqdm
import scipy.io as sio

try:
    import emcee
except:
    print('running without emcee')

from .priors import *
from .gp_sfh import *
from .plotter import *
from .sed_fitter import *

def get_mcmc_variables(atlas):

    model_params = np.vstack((atlas['mstar'],atlas['sfr'],atlas['sfh_tuple'][0:,-3:].T,atlas['met'].ravel(),atlas['dust'].ravel(),atlas['zval'].ravel())).T
    model_seds = atlas['sed']
    interp_engine = NearestNDInterpolator(model_params, model_seds)
    param_limits = [np.amin(model_params,0), np.amax(model_params,0)]
    param_limits[0][1] = -3.0
    param_limits[0][2] = 0.0

    return model_params, model_seds, interp_engine, param_limits

# def log_prior(theta):
#
#     # change these to match the priors i'm using, esp. for SFHs
#
#     mass, sfr, t25, t50, t75, met, dust, zval = theta
#     if mass_min < mass < mass_max and sfr_min < sfr < sfr_max \
#     and 0.0 < t25 < t50 and t25 < t50 < t75 and t50 < t75 < t75_max \
#     and dust_min < dust < dust_max and met_min < met < met_max \
#     and zval_min < zval < zval_max:
#         return 0.0
#     return -np.inf

def log_prior(theta, param_limits):

    # change these to match the priors i'm using, esp. for SFHs
    mass, sfr, t25, t50, t75, met, dust, zval = theta

    if (param_limits[0] < theta).all() and (param_limits[1] > theta).all() and (0<t25<t50) and (t25<t50<t75):
        return 0.0
    return -np.inf

def log_likelihood(theta, sed, sed_err, interp_engine):

    pred_sed = interp_engine(theta)[0]
    fit_mask = (sed > 0)

    chi2 = (pred_sed[fit_mask] - sed[fit_mask])**2 / (sed_err[fit_mask])**2

    return -0.5 * np.sum(chi2)

def log_probability(theta, sed, sed_err, interp_engine, param_limits):
    lp = log_prior(theta, param_limits)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, sed, sed_err, interp_engine)

def run_emceesampler(gal_sed, gal_err, atlas, fit_mask=[], zbest=None, deltaz=None, nwalkers = 100, epochs = 1000, plot_posteriors = False):

    model_params, model_seds, interp_engine, param_limits = get_mcmc_variables(atlas)

    pos = model_params[0:nwalkers,0:].copy()
    pos[pos[0:,1]<-2,1] = -2
    _, ndim = pos.shape

    _, norm_fac = db.evaluate_sed_likelihood(gal_sed,gal_err,atlas,fit_mask=[],
                                            zbest=None,deltaz=None)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                    args=(gal_sed/norm_fac, gal_err/norm_fac, interp_engine, param_limits))
    sampler.run_mcmc(pos, epochs, progress=True)

    if plot_posteriors == True:
        plot_emcee_posterior(sampler, norm_fac)

    return sampler, norm_fac

def plot_emcee_posterior(sampler, norm_fac, discard = 100, thin = 1):

    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    flat_samples[0:,0] = flat_samples[0:,0] + np.log10(norm_fac)
    flat_samples[0:,1] = flat_samples[0:,1] + np.log10(norm_fac)

    atlas_labels = ['log M*', 'log SFR', 't25','t50','t75', 'Z', 'Av', 'z']

    fig = corner.corner(
        flat_samples, labels = atlas_labels,
        plot_datapoints=False, fill_contours=True,
        bins=20, smooth=1.0,
        quantiles=(0.16, 0.84), levels=[1 - np.exp(-(1/1)**2/2),1 - np.exp(-(2/1)**2/2)],
        label_kwargs={"fontsize": 30}, show_titles=True
    );
    fig.subplots_adjust(right=1.5,top=1.5)
    plt.show()
