try:
    from schwimmbad import SerialPool, MultiPool
    from functools import partial
except:
    print('running without parallelization.')
import numpy as np
import os
import time

from astropy.table import Table
import time
import pylab as pl
from IPython import display

from .gp_sfh import *
from .sed_fitter import *
from .pre_grid import *
from .priors import *


def gen_pg_parallel(data_i, atlas_vals):
    
    fname, zval, priors, pg_folder, filter_list, filt_dir, N_pregrid = atlas_vals
    fname_full = fname + '_zval_%.0f_' %(zval*10000) + '_chunk_%.0f' %(data_i)

    generate_atlas(N_pregrid = N_pregrid,
                          priors = priors,
                          fname = fname_full, store=True, path=pg_folder,
                          filter_list = filter_list, filt_dir = filt_dir,
                          rseed = (N_pregrid * data_i + 1))
    return

def generate_atlas_in_parallel_chunking(zval, chunksize, nchunks, fname = 'temp_parallel_atlas', filter_list = 'filter_list_goodss.dat', filt_dir = 'internal', priors = [], z_bw = 0.05, pg_folder = 'parallel_atlases/'):
    
    N_pregrid = chunksize
    
    atlas_vals = [fname, zval, priors, pg_folder, filter_list, filt_dir, N_pregrid]

    time_start = time.time()
    try:
        with MultiPool() as pool:
            values = list(pool.map(partial(gen_pg_parallel,atlas_vals), data))
    finally:
        print('Generated pregrid (%.0f chunks, %.0f sedsperchunk) at zval',zval)
        print('time taken: %.2f mins.' %((time.time()-time_start)/60))
        
       # need to add code here to then concatenate chunks into a single file and delete the individual ones
          
    return


def make_atlas_parallel(zval, atlas_params):
    """
    Make a single atlas given a redshift value and a list of parameters (including a priors object).
    Atlas Params: [N_pregrid, priors, fname, store, path, filter_list, filt_dir, z_bw]
    """
    
    # currently only works for photometry, change to include a variable list of atlas_kwargs
    N_pregrid, priors, fname, store, path, filter_list, filt_dir, z_bw = atlas_params
    
    priors.z_max = zval + z_bw/2
    priors.z_min = zval - z_bw/2
    
    fname = fname+'_zval_%.0f_' %(zval*10000)
    
    generate_atlas(N_pregrid = N_pregrid,
                      priors = priors,
                      fname = fname, store=True, path=path,
                      filter_list = filter_list, filt_dir = filt_dir,
                      rseed = int(zval*100000))
    
    return


def generate_atlas_in_parallel_zgrid(zgrid, atlas_params, dynamic_decouple = True):
    """
    Make a set of atlases given a redshift grid and a list of parameters (including a priors object).
    Atlas Params: [N_pregrid, priors, fname, store, path, filter_list, filt_dir, z_bw]
    """
    
    time_start = time.time()

    try:
        with MultiPool() as pool:
            values = list(pool.map(partial(make_atlas_parallel, atlas_params = atlas_params), zgrid))
    finally:
        time_end = time.time()
        print('time taken [parallel]: %.2f min.' %((time_end-time_start)/60))

        
#------------------------------------------------------------------------------------

def fit_gals(gal_id, catvals):


    #if not fit_mask:
    if len(catvals) == 3:
        cat_seds, cat_errs, atlas = catvals
        fit_mask = []

    elif len(catvals) == 4:
        cat_seds, cat_errs, fit_mask, atlas = catvals
        
    else:
        print('wrong number of arguments supplied to fitter')
        
    gal_sed = cat_seds[gal_id, 0:].copy()
    gal_err = cat_errs[gal_id, 0:].copy()
    #gal_err = cat_errs[gal_id, 0:].copy() + gal_sed*0.03
    #gal_err = cat_errs[gal_id, 0:].copy() + gal_sed*0.1
    #gal_err = cat_errs[gal_id, 0:].copy() + gal_sed*0.5
    fit_likelihood, fit_norm_fac = evaluate_sed_likelihood(gal_sed,gal_err,atlas,fit_mask=fit_mask,
                                            zbest=None,deltaz=None)

    quants = get_quants(fit_likelihood, atlas, fit_norm_fac)
    
    return quants, fit_likelihood
    
#     try:
#         map_mstar = evaluate_MAP(atlas['mstar']+np.log10(fit_norm_fac), 
#                                  np.exp(-fit_likelihood/2), 
#                                  bins = np.arange(4,14,0.001), 
#                                  smooth = 'kde', lowess_frac = 0.3, vb = False)

#         map_sfr = evaluate_MAP(atlas['sfr']+np.log10(fit_norm_fac), 
#                                  np.exp(-fit_likelihood/2), 
#                                  bins = np.arange(-6,4,0.001), 
#                                  smooth = 'kde', lowess_frac = 0.3, vb = False)
#         return quants, fit_likelihood, map_mstar, map_sfr
#     except:
#         print('couldnt calculate MAP for galid: ',gal_id)
#         return quants, fit_likelihood, np.nan, np.nan
    
    

def fit_catalog(fit_cat, atlas_path, atlas_fname, output_fname, N_pregrid = 10000, N_param = 3, z_bw = 0.05, f160_cut = 100, fit_mask = [], zgrid = [], sfr_uncert_cutoff = 2.0):
    
    cat_id, cat_zbest, cat_seds, cat_errs, cat_f160, cat_class_star = fit_cat
    
    #if not zgrid:
    if isinstance(zgrid, (np.ndarray)) == False:
        zgrid = np.arange(np.amin(cat_zbest),np.amax(cat_zbest),z_bw)
    
    fit_id = cat_id.copy()
    fit_logM_50 = np.zeros_like(cat_zbest)
    fit_logM_MAP = np.zeros_like(cat_zbest)
    fit_logM_16 = np.zeros_like(cat_zbest)
    fit_logM_84 = np.zeros_like(cat_zbest)
    fit_logSFRinst_50 = np.zeros_like(cat_zbest)
    fit_logSFRinst_MAP = np.zeros_like(cat_zbest)
    fit_logSFRinst_16 = np.zeros_like(cat_zbest)
    fit_logSFRinst_84 = np.zeros_like(cat_zbest)

    fit_logZsol_50 = np.zeros_like(cat_zbest)
    fit_logZsol_16 = np.zeros_like(cat_zbest)
    fit_logZsol_84 = np.zeros_like(cat_zbest)
    fit_Av_50 = np.zeros_like(cat_zbest)
    fit_Av_16 = np.zeros_like(cat_zbest)
    fit_Av_84 = np.zeros_like(cat_zbest)

    fit_zfit_50 = np.zeros_like(cat_zbest)
    fit_zfit_16 = np.zeros_like(cat_zbest)
    fit_zfit_84 = np.zeros_like(cat_zbest)
    fit_logMt_50 = np.zeros_like(cat_zbest)
    fit_logMt_16 = np.zeros_like(cat_zbest)
    fit_logMt_84 = np.zeros_like(cat_zbest)
    fit_logSFR100_50 = np.zeros_like(cat_zbest)
    fit_logSFR100_16 = np.zeros_like(cat_zbest)
    fit_logSFR100_84 = np.zeros_like(cat_zbest)
    fit_nparam = np.zeros_like(cat_zbest)
    fit_t25_50 = np.zeros_like(cat_zbest)
    fit_t25_16 = np.zeros_like(cat_zbest)
    fit_t25_84 = np.zeros_like(cat_zbest)
    fit_t50_50 = np.zeros_like(cat_zbest)
    fit_t50_16 = np.zeros_like(cat_zbest)
    fit_t50_84 = np.zeros_like(cat_zbest)
    fit_t75_50 = np.zeros_like(cat_zbest)
    fit_t75_16 = np.zeros_like(cat_zbest)
    fit_t75_84 = np.zeros_like(cat_zbest)

    fit_nbands = np.zeros_like(cat_zbest)
    fit_f160w = np.zeros_like(cat_zbest)
    fit_stellarity = np.zeros_like(cat_zbest)
    fit_chi2 = np.zeros_like(cat_zbest)
    fit_flags = np.zeros_like(cat_zbest)

    for i in (range(len(zgrid))):
        
        print('loading atlas at', zgrid[i])
    
        # for a given redshift slice,
        zval = zgrid[i]
        
        # select the galaxies to be fit
        z_mask = (cat_zbest < (zval + z_bw/2)) & (cat_zbest > (zval - z_bw/2)) & (cat_f160 < f160_cut)
        fit_ids = np.arange(len(cat_zbest))[z_mask]            
        
        
#         for gal_id in fit_ids:
        
#             gal_sed = cat_seds[gal_id, 0:]
#             gal_err = cat_errs[gal_id, 0:]

#             fit_likelihood, fit_norm_fac = evaluate_sed_likelihood(gal_sed,gal_err,atlas,fit_mask=[],
#                                                 zbest=None,deltaz=None)

#             quants = get_quants(fit_likelihood, atlas, fit_norm_fac)

        print('starting parallel fitting for Ngals = ',len(fit_ids),' at redshift ', str(zval))

        try:
#             load the atlas
            fname = atlas_fname+'_zval_%.0f_' %(zgrid[i]*10000)
            atlas = load_atlas(fname, N_pregrid, N_param = N_param, path = atlas_path)
            print('loaded atlas')
            with MultiPool() as pool:
                # note: Parallel doesn't work in Python2.6

#                 if not fit_mask:
                if isinstance(fit_mask, np.ndarray) == False:
                    all_quants = list(pool.map(partial(fit_gals, catvals=(cat_seds, cat_errs, atlas)), fit_ids))
                else:
                    all_quants = list(pool.map(partial(fit_gals, catvals=(cat_seds, cat_errs, fit_mask, atlas)), fit_ids))
            print('finished fitting parallel zbest chunk at z=%.3f' %zval)

            print('starting to put values in arrays')
            for ii, gal_id in enumerate(fit_ids):

                gal_sed = cat_seds[gal_id, 0:]
                gal_err = cat_errs[gal_id, 0:]

                quants = all_quants[ii][0]
                fit_likelihood = all_quants[ii][1]
    #             fit_logM_MAP[gal_id] = all_quants[ii][2]
    #             fit_logSFRinst_MAP[gal_id] = all_quants[ii][3]

                fit_logM_50[gal_id] = quants[0][0]
                fit_logM_16[gal_id] = quants[0][1]
                fit_logM_84[gal_id] = quants[0][2]
                fit_logSFRinst_50[gal_id] = quants[1][0]
                fit_logSFRinst_16[gal_id] = quants[1][1]
                fit_logSFRinst_84[gal_id] = quants[1][2]

                fit_Av_50[gal_id] = quants[2][0]
                fit_Av_16[gal_id] = quants[2][1]
                fit_Av_84[gal_id] = quants[2][2]

                fit_logZsol_50[gal_id] = quants[3][0]
                fit_logZsol_16[gal_id] = quants[3][1]
                fit_logZsol_84[gal_id] = quants[3][2]

                fit_zfit_50[gal_id] = quants[4][0]
                fit_zfit_16[gal_id] = quants[4][1]
                fit_zfit_84[gal_id] = quants[4][2]

                fit_logMt_50[gal_id] = quants[5][0][0]
                fit_logMt_16[gal_id] = quants[5][1][0]
                fit_logMt_84[gal_id] = quants[5][2][0]
                fit_logSFR100_50[gal_id] = quants[5][0][1]
                fit_logSFR100_16[gal_id] = quants[5][1][1]
                fit_logSFR100_84[gal_id] = quants[5][2][1]
                fit_nparam[gal_id] = quants[5][0][2]
                fit_t25_50[gal_id] = quants[5][0][3]
                fit_t25_16[gal_id] = quants[5][1][3]
                fit_t25_84[gal_id] = quants[5][2][3]
                fit_t50_50[gal_id] = quants[5][0][4]
                fit_t50_16[gal_id] = quants[5][1][4]
                fit_t50_84[gal_id] = quants[5][2][4]
                fit_t75_50[gal_id] = quants[5][0][5]
                fit_t75_16[gal_id] = quants[5][1][5]
                fit_t75_84[gal_id] = quants[5][2][5]

                fit_nbands[gal_id] = np.sum(gal_sed>0)
                fit_f160w[gal_id] = cat_f160[gal_id]
                fit_stellarity[gal_id] = cat_class_star[gal_id]
                fit_chi2[gal_id] = np.amin(fit_likelihood)

                # flagging galaxies that either
                # 1. have nan values for mass
                # 2. have SFR uncertainties > sfr_uncert_cutoff
                # 3. are flagged as a star
                # 4. have extremely large chi2
                if np.isnan(quants[0][0]):
                    fit_flags[gal_id] = 1.0
                elif (np.abs(fit_logSFRinst_84[gal_id] - fit_logSFRinst_16[gal_id]) > sfr_uncert_cutoff): 
                    fit_flags[gal_id] = 2.0
                elif (cat_class_star[gal_id] > 0.5):
                    fit_flags[gal_id] = 3.0
                elif (fit_chi2[gal_id] > 1000):
                    fit_flags[gal_id] = 4.0
                else:
                    fit_flags[gal_id] = 0.0

        except:
            print('couldn\'t fit with pool at z=',zval)
        
        print('finishing that')
        pl.clf()
        pl.figure(figsize=(12,6))
        pl.hist(cat_zbest[cat_zbest>0],np.arange(0,6,z_bw),color='black',alpha=0.3)
        #pl.hist(fit_zfit_50[fit_zfit_50>0],np.arange(0,6,z_bw),color='royalblue')
        pl.hist(cat_zbest[fit_zfit_50>0],np.arange(0,6,z_bw),color='royalblue')
        pl.title('fit %.0f/%.0f galaxies' %(np.sum(fit_zfit_50>0), len(cat_zbest)))
        pl.xlabel('redshift');pl.ylabel('# galaxies')
        
        display.clear_output(wait=True)
        display.display(pl.gcf())
        
    pl.show()
    
    #'logSFRinst_MAP':fit_logSFRinst_MAP,
    #'logM_MAP':fit_logM_MAP,
    
    fit_mdict = {'ID':fit_id, 
                 'logM_50':fit_logM_50, 'logM_16':fit_logM_16,'logM_84':fit_logM_84,
                 'logSFRinst_50':fit_logSFRinst_50, 'logSFRinst_16':fit_logSFRinst_16, 'logSFRinst_84':fit_logSFRinst_84,
                 'logZsol_50':fit_logZsol_50, 'logZsol_16':fit_logZsol_16, 'logZsol_84':fit_logZsol_84, 
                 'Av_50':fit_Av_50, 'Av_16':fit_Av_16, 'Av_84':fit_Av_84, 
                 'zfit_50':fit_zfit_50, 'zfit_16':fit_zfit_16, 'zfit_84':fit_zfit_84, 
                 'logMt_50':fit_logMt_50, 'logMt_16':fit_logMt_16, 'logMt_84':fit_logMt_84,  
                 'logSFR100_50':fit_logSFR100_50, 'logSFR100_16':fit_logSFR100_16, 'logSFR100_84':fit_logSFR100_84, 
                 't25_50':fit_t25_50, 't25_16':fit_t25_16, 't25_84':fit_t25_84, 
                 't50_50':fit_t50_50, 't50_16':fit_t50_16, 't50_84':fit_t50_84, 
                 't75_50':fit_t75_50, 't75_16':fit_t75_16, 't75_84':fit_t75_84, 
                 'nparam':fit_nparam,
                 'nbands':fit_nbands, 
                 'F160w':fit_f160w, 
                 'stellarity':fit_stellarity,
                 'chi2': fit_chi2, 
                 'fit_flags':fit_flags}

    fit_cat = Table(fit_mdict)

    fit_cat.write(output_fname, format='ascii.commented_header')
    
    return

    
    
    
    
    
