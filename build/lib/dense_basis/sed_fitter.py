import numpy as np
from tqdm import tqdm
import scipy.io as sio

from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import gaussian_kde

# cosmology assumption
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

from .priors import *
from .pre_grid import *
from .gp_sfh import *
from .plotter import *

class SedFit(object):
    """
    Class to incorporate SED likelihood evaluation and resulting posteriors
    
    Attributes
    ----------
    mass : log stellar mass
        upper and lower limits of distribution
    sfr : log instantaneous SFR
        upper and lower limits of distribution
    ssfr : log specific SFR = SFR/M*
        negative lognormal distribution parameters
    tx : lookback times at which a galaxy formed certain fractions of its mass
        concentration (tx_alpha) and number of parameters (Nparam) of distribution
    Z  : log metallicity/solar
        upper and lower limits of distribution
    Av : calzetti dust attenuation
        upper and lower limits of distribution
    z  : redshift
        upper and lower limits of distribution

    Methods
    -------
    method: what it does
    """
    
    def __init__(self, sed, sed_err, atlas, fit_mask = [], zbest = None, deltaz = None):
        
        self.sed = sed
        self.sed_err = sed_err
        self.atlas = atlas
        self.fit_mask = fit_mask
        self.zbest = zbest
        self.deltaz = deltaz
        self.dynamic_norm = True
        
    def evaluate_likelihood(self):
        
        chi2_array, norm_fac = evaluate_sed_likelihood(self.sed, self.sed_err, self.atlas, self.fit_mask, self.zbest, self.deltaz, self.dynamic_norm)
        self.chi2_array = chi2_array
        self.norm_fac = norm_fac
        self.likelihood = np.exp(-(chi2_array)/2)
        
        return
    
    def evaluate_posterior_percentiles(self, bw_dex = 0.001, percentile_values = [50.,16.,84.], vb = False):
        """
        by default, the percentile values are median, lower68, upper68. 
        change this to whatever the desired sampling of the posterior is.
        """
        
        quants = get_quants(self.chi2_array, self.atlas, self.norm_fac, bw_dex = bw_dex, percentile_values = percentile_values, vb = vb)
        
        self.mstar = quants[0]
        self.sfr = quants[1]
        self.Av = quants[2]
        self.Z = quants[3]
        self.z = quants[4]
        self.sfh_tuple = quants[5]
        self.percentile_values = percentile_values
        
        return 
    
    def evaluate_MAP_mstar(self, bw_dex = 0.001, smooth = 'kde', lowess_frac = 0.3, bw_method = 'scott', vb = False):
        
        qty = self.atlas['mstar'] + np.log10(self.norm_fac),
        weights = self.likelihood
        bins = np.arange(4,14,bw_dex)
        self.mstar_MAP = evaluate_MAP(qty, weights, bins, smooth = smooth, lowess_frac=lowess_frac, bw_method=bw_method, vb=vb)
        return self.mstar_MAP
    
    def evaluate_MAP_sfr(self, bw_dex = 0.001, smooth = 'kde', lowess_frac = 0.3, bw_method = 'scott', vb = False):
        
        qty = self.atlas['sfr'] + np.log10(self.norm_fac),
        weights = self.likelihood
        bins = np.arange(-6,4,bw_dex),
        self.sfr_MAP = evaluate_MAP(qty, weights, bins, smooth = smooth, lowess_frac=lowess_frac, bw_method=bw_method, vb=vb)
        return self.sfr_MAP
    
    def plot_posteriors(self,truths = []):
        
        figure = plot_posteriors(self.chi2_array, self.norm_fac, self.sed, self.atlas, truths = truths)
        return figure                     
    
    def plot_posterior_spec(self, filt_centers, priors, ngals = 100, alpha=0.1, fnu=True, yscale='log', speccolor = 'k', sedcolor='b', titlestr = [],figsize=(12,7)):
        
        set_plot_style()
        
        lam_all = []
        spec_all = []
        z_all = []
        
        bestn_gals = np.argsort(self.likelihood)

        for i in range(ngals):
        
            lam_gen, spec_gen =  makespec_atlas(self.atlas, bestn_gals[-(i+1)], priors, mocksp, cosmo, filter_list = [], filt_dir = [], return_spec = True)
        
            lam_all.append(lam_gen)
            spec_all.append(spec_gen)
            z_all.append(self.atlas['zval'][bestn_gals[-(i+1)]])
            
        fig = plt.subplots(1,1,figsize=figsize)
        

        if fnu == True:
            for i in range(ngals):
                plt.plot(lam_all[i]*(1+z_all[i]), spec_all[i]*self.norm_fac, color = speccolor, alpha=alpha)
            plt.errorbar(filt_centers[self.sed>0], self.sed[self.sed>0], yerr=self.sed_err[self.sed>0]*2, color=sedcolor,lw=0, elinewidth=2, marker='o', markersize=12, capsize=5)
            plt.ylabel(r'$F_\nu$ [$\mu$Jy]')
            
        elif fnu == False:
            for i in range(ngals):
                spec_flam = ujy_to_flam(spec_all[i]*self.norm_fac, lam_all[i]*(1+z_all[i]))
                plt.plot(lam_all[i]*(1+z_all[i]), spec_flam, color = speccolor, alpha=alpha)
            sed_flam = ujy_to_flam(self.sed,filt_centers)
            sed_flam_err_up = ujy_to_flam(self.sed+self.sed_err,filt_centers) - sed_flam
            sed_flam_err_dn = sed_flam - ujy_to_flam(self.sed-self.sed_err,filt_centers)
            # make these F_\lam errors, not F_nu errors
            plt.errorbar(filt_centers[self.sed>0], sed_flam[self.sed>0], yerr=(sed_flam_err_up[self.sed>0], sed_flam_err_dn[self.sed>0]), color=sedcolor,lw=0, elinewidth=2, marker='o', markersize=12, capsize=5)
            plt.ylabel(r'$F_\lambda$')
            
        plt.xlabel(r'$\lambda$ [$\AA$]')
        plt.xlim(np.amin(filt_centers)*0.81, np.amax(filt_centers)*1.2)
        plt.ylim(np.amin(self.sed[self.sed>0])*0.8,np.amax(self.sed[self.sed>0]+self.sed_err[self.sed>0])*1.5)
        plt.xscale('log');plt.yscale(yscale);
        #plot_lines(filt_centers, gal_z)
        #plt.title(titlestr,fontsize=18)
        
        return fig
    
    def evaluate_posterior_SFH(self, zval,ngals=100):
        
        bestn_gals = np.argsort(self.likelihood)              
        common_time = np.linspace(0,cosmo.age(zval).value,100)
        if priors.dynamic_decouple == True:
            priors.decouple_sfr_time = 100*cosmo.age(zval).value/cosmo.age(0.1).value
            
        all_sfhs = []
        all_weights = []
        for i in (range(ngals)):
            sfh, timeax =  tuple_to_sfh(self.atlas['sfh_tuple'][bestn_gals[-(i+1)],0:],self.atlas['zval'][bestn_gals[-(i+1)]])
            sfh = sfh*self.norm_fac
            sfh_interp = np.interp(common_time, timeax, sfh)

            all_sfhs.append(sfh_interp)
            all_weights.append(self.likelihood[bestn_gals[-(i+1)]])
            
        all_sfhs = np.array(all_sfhs)
        all_weights = np.array(all_weights)
        
        sfh_50 = np.zeros_like(common_time)
        sfh_16 = np.zeros_like(common_time)
        sfh_84 = np.zeros_like(common_time)
        for ti in range(len(common_time)):
            qty = np.log10(all_sfhs[0:,ti])
            qtymask = (qty > -np.inf) & (~np.isnan(qty))
            if np.sum(qtymask) > 0:
                smallwts = all_weights.copy()[qtymask.ravel()]
                qty = qty[qtymask]
                if len(qty>0):
                    sfh_50[ti], sfh_16[ti], sfh_84[ti] = 10**calc_percentiles(qty, smallwts, bins=50, percentile_values=[50., 16., 84.])
#             else:
#                 print(common_time[ti])

        return sfh_50, sfh_16, sfh_84, common_time
                        
                        
    def plot_posterior_SFH(self, zval,ngals=100,alpha=0.1, speccolor = 'k', sedcolor='b',figsize=(12,7)):
        
                        
        set_plot_style()
        bestn_gals = np.argsort(self.likelihood)              
        
        fig = plt.subplots(1,1,figsize=figsize)
                        
        common_time = np.linspace(0,cosmo.age(zval).value,100)
        if priors.dynamic_decouple == True:
            priors.decouple_sfr_time = 100*cosmo.age(zval).value/cosmo.age(0.1).value
        
        all_sfhs = []
        all_weights = []
        for i in (range(ngals)):
            sfh, timeax =  tuple_to_sfh(self.atlas['sfh_tuple'][bestn_gals[-(i+1)],0:],self.atlas['zval'][bestn_gals[-(i+1)]])
            sfh = sfh*self.norm_fac
            sfh_interp = np.interp(common_time, timeax, sfh)
            all_sfhs.append(sfh_interp)
            all_weights.append(self.likelihood[bestn_gals[-(i+1)]])
            alphawt = 1.0*alpha*self.likelihood[bestn_gals[-(i+1)]]/self.likelihood[bestn_gals[-1]]
            #print(alphawt)
            plt.plot(np.amax(common_time)-common_time, sfh_interp, color = sedcolor, alpha=alphawt)
            #plt.plot(np.amax(timeax) - timeax, sfh*norm_fac, color = speccolor, alpha=alpha)
        all_sfhs = np.array(all_sfhs)
        all_weights = np.array(all_weights)

        sfh_50 = np.zeros_like(common_time)
        sfh_16 = np.zeros_like(common_time)
        sfh_84 = np.zeros_like(common_time)
        for ti in range(len(common_time)):
            qty = np.log10(all_sfhs[0:,ti])
            qtymask = (qty > -np.inf) & (~np.isnan(qty))
            if np.sum(qtymask) > 0:
                smallwts = all_weights.copy()[qtymask.ravel()]
                qty = qty[qtymask]
                if len(qty>0):
                    sfh_50[ti], sfh_16[ti], sfh_84[ti] = 10**calc_percentiles(qty, smallwts, bins=50, percentile_values=[50., 16., 84.])
#             else:
#                 print(common_time[ti])
        plt.plot(np.amax(common_time)-common_time, sfh_50,lw=3,color=speccolor)
        plt.fill_between(np.amax(common_time)-common_time.ravel(), sfh_16.ravel(), sfh_84.ravel(),alpha=0.3,color=speccolor)

        plt.xlabel(r'lookback time [$Gyr$]')
        plt.ylabel(r'$SFR(t)$ [M$_\odot /$yr]')
        try:
            plt.ylim(0,np.amax(sfh_84)*1.2)
        except:
            print('couldnt set axis limits')
            
        return fig
                        
                        
                           
    
    
#-------------------------------------------------------------


def normerr(nf, pg_seds, sed, sed_err, fit_mask):
    c2v = np.amin(np.mean((pg_seds[fit_mask,0:] - sed.reshape(-1,1)/nf)**2 / (sed_err.reshape(-1,1)/nf)**2, 0))
    return c2v

def evaluate_sed_likelihood(sed, sed_err, atlas, fit_mask = [], zbest = None, deltaz = None, dynamic_norm = True):

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
    
    if dynamic_norm == True:
        nfmin = minimize(normerr, np.nanmedian(sed), args = (pg_seds, sed, sed_err, fit_mask))
        norm_fac = nfmin['x'][0]
    elif dynamic_norm == False:
        norm_fac = np.nanmedian(sed)
    else:
        norm_fac = 1.0
        print('undefined norm method. using norm_fac = 1')

    sed_normed = sed.reshape(-1,1)/norm_fac
    sed_err_normed = sed_err.reshape(-1,1)/norm_fac

    chi2 = np.mean((pg_seds[fit_mask,0:] - sed_normed)**2 / (sed_err_normed)**2, 0)

    if zbest is not None:
        pg_z = atlas['zval'].ravel()
        #redshift_mask = (np.abs(pg_z - zbest) > 0.1*zbest)
        redshift_mask = (np.abs(pg_z - zbest) > deltaz)
        chi2[redshift_mask] = np.amax(chi2)+1e3

    return chi2, norm_fac

def get_quants_key(key, bins, chi2_array, cat, norm_fac, percentile_values = [50.,16.,84.], return_uncert = True, vb = False):
    """
    Get posterior percentiles for an input key
    """
    relprob = np.exp(-(chi2_array)/2)    
    key_vals = calc_percentiles(cat[key], weights = relprob, bins = bins, percentile_values = percentile_values, vb = vb)
        
    return key_vals
    

def get_quants(chi2_array, cat, norm_fac, bw_dex = 0.001, percentile_values = [50.,16.,84.], vb = False):

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
                                 percentile_values = percentile_values, vb=vb)

    sfr_vals = calc_percentiles(cat['sfr'] + np.log10(norm_fac),
                                 weights = relprob,
                                 bins = np.arange(-6,4,bw_dex),
                                 percentile_values = percentile_values,vb=vb)

     # ---------------- SFH -----------------------------------

    sfh_tuple_vals = np.zeros((3, cat['sfh_tuple_rec'].shape[1]))

    for i in range(cat['sfh_tuple_rec'].shape[1]):

        if i == 0:
            sfh_tuple_vals[0:,i] = calc_percentiles(cat['sfh_tuple_rec'][0:,0] + np.log10(norm_fac),
                                 weights = relprob, bins = np.arange(4,14,bw_dex),
                                 percentile_values = percentile_values, vb=vb)
        elif i == 1:
            sfh_tuple_vals[0:,i] = calc_percentiles(cat['sfh_tuple_rec'][0:,1] + np.log10(norm_fac),
                                 weights = relprob, bins = np.arange(-6,4,bw_dex),
                                 percentile_values = percentile_values, vb=vb)
        elif i == 2:
            sfh_tuple_vals[0:,i] = sfh_tuple_vals[0:,i] + np.nanmean(cat['sfh_tuple_rec'][0:,2])
        else:
            sfh_tuple_vals[0:,i] = calc_percentiles(cat['sfh_tuple_rec'][0:,i],
                                 weights = relprob, bins = np.arange(0,1,bw_dex),
                                 percentile_values = percentile_values, vb=vb)

    # ------------------------ dust, metallicity, redshift ----------------------

    Av_vals = calc_percentiles(cat['dust'].ravel(),
                                 weights = relprob,
                                 bins = np.arange(0,np.amax(cat['dust']),bw_dex),
                                 percentile_values = percentile_values,vb=vb)

    Z_vals = calc_percentiles(cat['met'].ravel(),
                                 weights = relprob,
                                 bins = np.arange(-1.5, 0.5,bw_dex),
                                 percentile_values = percentile_values,vb=vb)

    z_vals = calc_percentiles(cat['zval'].ravel(),
                                 weights = relprob,
                                 bins = np.arange(np.amin(cat['zval']), np.amax(cat['zval']),bw_dex),
                                 percentile_values = percentile_values,vb=vb)

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


def evaluate_MAP(qty, weights, bins, smooth = 'kde', lowess_frac = 0.3, bw_method = 'scott', vb = False):
    
    post, xaxis = np.histogram(qty, weights=weights, bins=bins)
    xaxis_centers = xaxis[0:-1] + np.mean(np.diff(xaxis))
    
    if smooth == 'lowess':
        a = lowess(post, xaxis_centers,frac=lowess_frac)
        MAP = a[np.argmax(a[0:,1]),0]
    elif smooth == 'kde':
        a = gaussian_kde(qty,bw_method=bw_method, weights=weights)
        MAP = xaxis[np.argmax(a.evaluate(xaxis))]
    else:
        MAP = xaxis[np.argmax(post)+1]
    
    if vb == True:
        areapost = np.trapz(x=xaxis_centers, y=post)
        plt.plot(xaxis_centers, post/areapost)
        if smooth == 'lowess':
            plt.plot(a[0:,0],a[0:,1]/areapost)
        elif smooth == 'kde':
            plt.plot(xaxis, a.pdf(xaxis))
        plt.plot([MAP,MAP],plt.ylim())
        plt.show()
        
    return MAP


def get_lines(centers, zval):
    
    # line list from Table 4 of https://iopscience.iop.org/article/10.3847/1538-4357/aa6c66 (Byler+17)
    
    lam_min = np.amin(centers/(1+zval))*0.81
    lam_max = np.amax(centers/(1+zval))*1.1
    
    #line_lam = [1215.6701, 6564.6, 4862.71, 4341.692, 4102.892, 18756.4]
    #line_name = [r'Ly$\alpha$',r'H$\alpha$',r'H$\beta$',r'H$\gamma$',r'H$\delta$',r'Pa$\alpha$']
    
    #line_lam = [1215.6701, 6564.6, 4862.71, 4102.892, 18756.4,4472.735,1640.42, 9852.9, 8729.53,4622.864,5201.705,6585.27,5756.19, 5008.24, 3727.1,3869.86,40522.79]
    #line_name = [r'Ly$\alpha$',r'H$\alpha$',r'H$\beta$',r'H$\delta$',r'Pa$\alpha$','HeI','HeII','[CI]','[CI]', '[CI]','[NI]','[NII]','[NII]','[OIII]','[OII]','[NeIII]',r'Br$\alpha$']
    #line_offset = [0.0,0.0,0.0,0.0,0.0,0.007,0.007,0.014,0.014,0.014,0.007,0.007,0.007,0.021,0.021,0.028,0.0]
    
    line_lam = [  1215.67,   1906.68,   3727.1 ,  3869.86 ,  4102.89,
   4862.71 ,  4960.3 ,  6564.6 ,   9071.1,    9533.2,
  10832.06,  10941.17 , 12821.58,  18756.4,   40522.79, 105105.,
 155551.,   187130.,   334800.,   518145.,   883564.  ]
    line_name = [r'Ly$\alpha$','[CIII]','[OII]$_{x2}$','[NeIII]',r'H$\delta$',
                r'H$\beta$','[OIII]$_{x2}$',r'H$\alpha$','[SIII]','[SIII]',
                'He I',r'Pa$\gamma$',r'Pa$\beta$', r'Pa$\alpha$',r'Br$\alpha$','[S IV]',
                '[Ne III]','[S III]','[S III]','[O III]','[O III]']
    line_offset = [0.0,0.007,0.007,0.014,0.000,
                   0.0,0.007,0.0,0.007,0.014,
                  0.007,0.0,0.0,0.0,0.0,0.0,0.0,
                  0.0,0.007,0.014,0.0,0.007]
    
    good_line_lams, good_line_names, good_line_offsets = [], [], []
    for i in range(len(line_lam)):
        if (line_lam[i] > lam_min) & (line_lam[i] < lam_max):
            good_line_lams.append(line_lam[i])
            good_line_names.append(line_name[i])
            good_line_offsets.append(line_offset[i])
    return good_line_lams, good_line_names, good_line_offsets

    
def plot_lines(filt_centers, zval,color = 'forestgreen',alpha=0.3,fontsize=14):
    
    lls, lns, los = get_lines(filt_centers, zval)
    tempy = plt.ylim()
    for i in range(len(lls)):
        plt.plot([lls[i]*(1+zval),lls[i]*(1+zval)],tempy,':',color=color,alpha=alpha)
        plt.text(lls[i]*(1+zval)*1.03, (0.003+los[i])*(tempy[1] - tempy[0]) + tempy[0],lns[i],fontsize=fontsize,color=color)
    plt.ylim(tempy)
    return

def ujy_to_flam(data,lam):
    flam = ((3e-5)*data)/((lam**2.)*(1e6))
    return flam/1e-19

#---------------------------------------------
#----------- deprecated functions ------------
#---------------------------------------------

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
