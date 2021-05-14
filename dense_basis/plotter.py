import numpy as np
import matplotlib.pyplot as plt
import corner

import seaborn as sns

from pylab import *

from .pre_grid import make_filvalkit_simple, load_atlas
from .gp_sfh import *

def set_plot_style():
    sns.set(font_scale=2)
    sns.set_style('ticks')

    rc('axes', linewidth=3)
    rcParams['xtick.major.size'] = 12
    rcParams['ytick.major.size'] = 12
    rcParams['xtick.minor.size'] = 9
    rcParams['ytick.minor.size'] = 9
    rcParams['xtick.major.width'] = 3
    rcParams['ytick.major.width'] = 3

def plot_sfh(timeax, sfh, lookback = False, logx = False, logy = False, fig = None, label=None, **kwargs):
    set_plot_style()

    if fig is None:
        fig = plt.figure(figsize=(12,4))
    if lookback == True:
        plt.plot(np.amax(timeax) - timeax, sfh, label=label, **kwargs)
        plt.xlabel('lookback time [Gyr]');
    else:
        plt.plot(timeax, sfh, label=label, **kwargs)
        plt.xlabel('cosmic time [Gyr]');
    if label != None:
        plt.legend(edgecolor='w')
    plt.ylabel(r'SFR(t) [$M_\odot yr^{-1}$]')
    if logx == True:
        plt.xscale('log')
    if logy == True:
        plt.yscale('log')
    plt.xlim(0,np.amax(timeax))
    tempy = plt.ylim()
    plt.ylim(0,tempy[1])
    # plt.show()
    return fig

def plot_spec(lam, spec, logx = True, logy = True,
xlim = (1e2,1e8),
clip_bottom = True):
    set_plot_style()

    plt.figure(figsize=(12,4))
    plt.plot(lam, spec)
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'$F_\nu$ [$\mu$Jy]')
    if logx == True:
        plt.xscale('log')
    if logy == True:
        plt.yscale('log')
    plt.xlim(xlim)
    if clip_bottom == True:
        plt.ylim(1e-3,np.amax(spec)*3)
    plt.show()

def plot_filterset(filter_list = 'filter_list_goodss.dat', filt_dir = 'filters/', zval = 1.0, lam_arr = 10**np.linspace(2,8,10000), rest_frame = True):
    set_plot_style()

    filcurves, lam_z, lam_z_lores = make_filvalkit_simple(lam_arr, zval, fkit_name = filter_list, filt_dir = filt_dir)

    plt.figure(figsize=(12,4))
    if rest_frame == True:
        plt.plot(lam_arr*(1+zval), filcurves,'k:');
        plt.xlabel(r'$\lambda$ [$\AA$]');
    else:
        plt.plot(lam_arr, filcurves,'k:');
        plt.xlabel(r'$\lambda$ [$\AA$; obs. frame]');
    plt.xscale('log'); plt.xlim(1e3,2e5)
    plt.ylabel('Filter transmission')
    plt.show()

def quantile_names(N_params):
    return (np.round(np.linspace(0,100,N_params+2)))[1:-1]

def plot_atlas_priors(atlas):
    
    mass_unnormed = np.log10(10**atlas['mstar'] / atlas['norm'])
    sfr_unnormed = np.log10(10**atlas['sfr'] / atlas['norm'])
    ssfr = sfr_unnormed - mass_unnormed
    txs = atlas['sfh_tuple_rec'][0:,3:]

    dust = atlas['dust'].ravel()
    met = atlas['met'].ravel()
    zval = atlas['zval'].ravel()
    quants = np.vstack((mass_unnormed, sfr_unnormed, ssfr, txs.T, met, dust, zval)).T

    txs = ['t'+'%.0f' %i for i in quantile_names(txs.shape[1])]
    pg_labels = ['log M*', 'log SFR', 'log sSFR', 'Z', 'Av', 'z']
    pg_labels[3:3] = txs

    figure = corner.corner(quants,plot_datapoints=False, fill_contours=True,labels=pg_labels,
                                    bins=20, smooth=1.0,
                                    quantiles=(0.16, 0.84), 
                                    levels=[1 - np.exp(-(1/1)**2/2),1 - np.exp(-(2/1)**2/2)],
                                    label_kwargs={"fontsize": 30})
    figure.subplots_adjust(right=1.5,top=1.5)

    plt.show()
    
    return



def plot_posteriors(chi2_array, norm_fac, sed, atlas, truths = [], **kwargs):
    set_plot_style()

    if len(truths) > 0:
        corner_truths = truths
#         corner_truths[3:(3+int(sed_truths[2]))] = corner_truths[3:(3+int(sed_truths[2]))]/cosmo.age(corner_truths[-1]).value
    #pg_params = np.vstack([pg_theta[0][0,0:], pg_theta[0][1,0:], pg_theta[0][3:,0:], pg_theta[1], pg_theta[2], pg_theta[3]])
    sfrvals = atlas['sfr'].copy()
    sfrvals[sfrvals<-3] = -3
    pg_params = np.vstack([atlas['mstar'],sfrvals,atlas['sfh_tuple'][0:,3:].T,atlas['met'].ravel(),atlas['dust'].ravel(),atlas['zval'].ravel()])
    txs = ['t'+'%.0f' %i for i in quantile_names(pg_params.shape[0]-5)]
    pg_labels = ['log M*', 'log SFR', 'Z', 'Av', 'z']
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


def plot_priors(fname, N_pregrid, N_param, dir = 'pregrids/'):
    set_plot_style()

    cat = load_atlas(fname, N_pregrid, N_param, path = dir)
    sfh_tuples = cat['sfh_tuple']
    Av = cat['dust'].ravel()
    Z = cat['met'].ravel()
    z = cat['zval'].ravel()
    seds = cat['sed']
    norm_method = cat['norm_method']
    norm_facs = cat['norm'].ravel()

    pg_theta = [sfh_tuples, Z, Av, z, seds]
    pg_params = np.vstack([pg_theta[0][0,0:], pg_theta[0][1,0:], pg_theta[0][3:,0:], pg_theta[1], pg_theta[2], pg_theta[3]])

    txs = ['t'+'%.0f' %i for i in quantile_names(pg_params.shape[0]-5)]
    pg_labels = ['log M*', 'log SFR', 'Z', 'Av', 'z']
    pg_labels[2:2] = txs

    pg_priors = pg_params.copy()
    pg_priors[0,0:] += np.log10(norm_facs)
    pg_priors[1,0:] += np.log10(norm_facs)
    figure = corner.corner(pg_priors.T,labels=pg_labels, plot_datapoints=False, fill_contours=True,
            bins=50, smooth=1.0, levels=[1 - np.exp(-(1/1)**2/2),1 - np.exp(-(2/1)**2/2)], label_kwargs={"fontsize": 30})
    figure.subplots_adjust(right=1.5,top=1.5)
    plt.show()


def plot_SFH_posterior(chi2_array, norm_fac, sed, atlas, truths = [], plot_ci = True, sfh_threshold = 0.9, **kwargs):
    # to be phased out with a newer function
    set_plot_style()

    #pg_sfhs, pg_Z, pg_Av, pg_z, pg_seds = pg_theta
    pg_sfhs = atlas['sfh_tuple'].T
    pg_z = atlas['zval'].ravel()

    weighted_chi2_indices = np.argsort(np.exp(-chi2_array/2))
    num_sfhs = np.sum(np.exp(-chi2_array[weighted_chi2_indices]/2) > sfh_threshold*(np.exp(-np.amin(chi2_array)/2)))

    temp_sfhs = np.zeros((1000, num_sfhs))
    temp_times = np.zeros((1000, num_sfhs))
    rel_likelihoods = np.zeros((num_sfhs,))
    for i in range(num_sfhs):
        temp_sfh_tuple = pg_sfhs[0:, weighted_chi2_indices[-(i+1)]].copy()
        temp_sfh_tuple[0] = temp_sfh_tuple[0] + np.log10(norm_fac)
        temp_sfh_tuple[1] = temp_sfh_tuple[1] + np.log10(norm_fac)
        temp_sfhs[0:,i], temp_times[0:,i] = tuple_to_sfh(temp_sfh_tuple, zval = pg_z[weighted_chi2_indices[-(i+1)]])
        temp_sfhs[0:,i] = np.flip(correct_for_mass_loss(np.flip(temp_sfhs[0:,i],0), temp_times[0:,i], fsps_time, fsps_massloss),0)

        rel_likelihoods[i] = np.exp(-chi2_array[weighted_chi2_indices[-(i+1)]]/2)

    if plot_ci == False:
        for i in range(num_sfhs):
            if i == 0:
                fig = plot_sfh(temp_times[0:,i], temp_sfhs[0:,i], lookback=True, color='k', alpha=rel_likelihoods[i]**3, **kwargs)
            else:
                plot_sfh(temp_times[0:,i], temp_sfhs[0:,i], lookback=True, fig = fig, color='k', alpha=rel_likelihoods[i]**3, **kwargs)

    if plot_ci == True:
        _, temp_common_time = tuple_to_sfh(temp_sfh_tuple, zval = pg_z[np.argmin(chi2_array)])
        temp_sfhs_splined = np.zeros_like(temp_sfhs)
        fig = plt.figure(figsize=(12,4))
        for i in range(num_sfhs):
            temp_sfhs_splined[0:,i] = np.interp(temp_common_time, temp_times[0:,i], np.flip(temp_sfhs[0:,i],0))
#         plt.fill_between(temp_common_time, np.nanpercentile(temp_sfhs_splined,18,axis=1),
#                          np.nanpercentile(temp_sfhs_splined,84,axis=1), color='k', alpha=0.1)
        qt_array = np.arange(25,76,5)
        for i in range(5):
            if i == 0:
                plt.fill_between(temp_common_time, np.nanpercentile(temp_sfhs_splined,qt_array[i],axis=1),
                                 np.nanpercentile(temp_sfhs_splined,qt_array[-(i+1)],axis=1), color='k', alpha=0.1,
                                 label='25-75 CI')
            else:
                plt.fill_between(temp_common_time, np.nanpercentile(temp_sfhs_splined,qt_array[i],axis=1),
                                 np.nanpercentile(temp_sfhs_splined,qt_array[-(i+1)],axis=1), color='k', alpha=0.1)

        plot_sfh(temp_common_time, np.flip(np.nanmedian(temp_sfhs_splined,axis=1),0),
                       lookback=True, color='k', lw=3, label='median DB-SFH', fig = fig)
#         print(np.nanpercentile(temp_sfhs_splined,49,axis=1))
    if len(truths) == 2:
        plot_sfh(truths[0], truths[1], lookback=True, fig = fig, lw=3,label='true SFH')
        plt.ylim(0,np.amax(truths[1])*1.5)
    plt.legend(edgecolor='w', fontsize=18)
    plt.show()
    return

def plot_SFH_posterior_v2(chi2_array, sed, pg_theta, truths = [], plot_ci = True, sfh_threshold = 0.9, Nbins = 30, max_num = 100, npow = 3, **kwargs):
    set_plot_style()

    pg_sfhs, pg_Z, pg_Av, pg_z, pg_seds = pg_theta
    weighted_chi2_indices = np.argsort(np.exp(-chi2_array/(2*np.amin(chi2_array))))

    num_sfhs = np.sum(np.exp(-chi2_array/2) > sfh_threshold*(np.exp(-np.amin(chi2_array)/2)))
    if num_sfhs > max_num:
        num_sfhs = max_num
        print('truncated to %.0f SFHs to reduce computation time. increase max_num if desired.' %max_num)

    # to-do: add other norm_facs
    norm_fac = np.amax(sed)

    temp_sfh_tuple = pg_sfhs[0:, weighted_chi2_indices[-1]].copy()
    _, temp_common_time = tuple_to_sfh(temp_sfh_tuple, zval = pg_z[np.argmin(chi2_array)])

    temp_sfhs = np.zeros((1000, num_sfhs))
    temp_sfhs_splined = np.zeros_like(temp_sfhs)
    temp_times = np.zeros((1000, num_sfhs))
    rel_likelihoods = np.zeros((num_sfhs,))
    for i in range(num_sfhs):
        temp_sfh_tuple = pg_sfhs[0:, weighted_chi2_indices[-(i+1)]].copy()
        temp_sfh_tuple[0] = temp_sfh_tuple[0] + np.log10(norm_fac)
        temp_sfh_tuple[1] = temp_sfh_tuple[1] + np.log10(norm_fac)
        temp_sfhs[0:,i], temp_times[0:,i] = tuple_to_sfh(temp_sfh_tuple, zval = pg_z[weighted_chi2_indices[-(i+1)]])
        temp_sfhs[0:,i] = np.flip(correct_for_mass_loss(np.flip(temp_sfhs[0:,i],0), temp_times[0:,i], fsps_time, fsps_massloss),0)
        temp_sfhs_splined[0:,i] = np.interp(temp_common_time, temp_times[0:,i], np.flip(temp_sfhs[0:,i],0))
        rel_likelihoods[i] = np.exp(-chi2_array[weighted_chi2_indices[-(i+1)]]/(np.amin(chi2_array)*2))

    sfr_range = np.linspace(0, np.nanpercentile(temp_sfhs_splined,99), Nbins+1)
    sfh_median = np.zeros_like(temp_common_time)
    sfh_posterior = np.zeros((Nbins, len(temp_common_time) ))
    for i in range(len(temp_common_time)):

        a,b = np.histogram(temp_sfhs_splined[i,0:], weights= rel_likelihoods**npow, bins = sfr_range, density=True)
        sfh_posterior[0:,i] = a
        n_c = np.cumsum(a)
        n_c = n_c / np.amax(n_c)
        bin_centers = b[0:-1] + (b[1]-b[0])/2
        sfh_median[i] = bin_centers[np.argmin(np.abs(n_c - 0.5))]

    fig = plt.figure(figsize=(12,4))
    plt.pcolor(temp_common_time, sfr_range[0:-1], sfh_posterior,cmap='magma')
#     clbr = plt.colorbar()
#     clbr.set_label('P(SFR(t))')
    plot_sfh(temp_common_time, sfh_median, lookback=False, fig = fig, lw=3,label='median SFH',color='b')
    if len(truths) == 2:
        plot_sfh(truths[0], truths[1], lookback=True, fig = fig, lw=3,label='true SFH',color='w')
        plt.ylim(0,np.amax(truths[1])*1.5)
    plt.xlim(0,np.amax(temp_common_time))
    plt.xlabel('t [lookback time; Gyr]')
    plt.ylabel('SFR(t)')
    plt.ylim(0,np.amax(sfr_range[0:-1]))
    l = plt.legend(framealpha=0.0)
    for text in l.get_texts():
        text.set_color("white")
    plt.show()

    return


def plot_SFH_posterior_v3(chi2_array, sed, pg_theta, truths = [], plot_ci = True, sfh_threshold = 0.9, Nbins = 30, max_num = 1000, npow = 3, **kwargs):
    set_plot_style()

    pg_sfhs, pg_Z, pg_Av, pg_z, pg_seds = pg_theta
    weighted_chi2_indices = np.argsort(np.exp(-chi2_array/(2*np.amin(chi2_array))))
    Nparam = pg_sfhs.shape[0]-3

    num_sfhs = np.sum(np.exp(-chi2_array/2) > sfh_threshold*(np.exp(-np.amin(chi2_array)/2)))
    if num_sfhs > max_num:
        num_sfhs = max_num
        print('truncated to %.0f SFHs to reduce computation time. increase max_num if desired.' %max_num)

    # to-do: add other norm_facs
    norm_fac = np.amax(sed)

    temp_sfh_tuple = pg_sfhs[0:, weighted_chi2_indices[-1]].copy()
    _, temp_common_time = tuple_to_sfh(temp_sfh_tuple, zval = pg_z[np.argmin(chi2_array)])

    temp_sfhs = np.zeros((1000, num_sfhs))
    temp_sfhs_splined = np.zeros_like(temp_sfhs)
    temp_times = np.zeros((1000, num_sfhs))
    rel_likelihoods = np.zeros((num_sfhs,))
    for i in range(num_sfhs):
        temp_sfh_tuple = pg_sfhs[0:, weighted_chi2_indices[-(i+1)]].copy()
        temp_sfh_tuple[0] = temp_sfh_tuple[0] + np.log10(norm_fac)
        temp_sfh_tuple[1] = temp_sfh_tuple[1] + np.log10(norm_fac)
        temp_sfhs[0:,i], temp_times[0:,i] = tuple_to_sfh(temp_sfh_tuple, zval = pg_z[weighted_chi2_indices[-(i+1)]])
        temp_sfhs[0:,i] = np.flip(correct_for_mass_loss(np.flip(temp_sfhs[0:,i],0), temp_times[0:,i], fsps_time, fsps_massloss),0)
        temp_sfhs_splined[0:,i] = np.interp(temp_common_time, temp_times[0:,i], np.flip(temp_sfhs[0:,i],0))
        rel_likelihoods[i] = np.exp(-chi2_array[weighted_chi2_indices[-(i+1)]]/(np.amin(chi2_array)*2))

    sfr_range = np.linspace(0, np.nanpercentile(temp_sfhs_splined,99), Nbins+1)
    sfh_median = np.zeros_like(temp_common_time)
    sfh_up = np.zeros_like(temp_common_time)
    sfh_dn = np.zeros_like(temp_common_time)
    sfh_posterior = np.zeros((Nbins, len(temp_common_time) ))
    for i in range(len(temp_common_time)):
        a,b = np.histogram(temp_sfhs_splined[i,0:], weights= rel_likelihoods**npow, bins = sfr_range, density=True)
        sfh_posterior[0:,i] = a
        n_c = np.cumsum(a)
        n_c = n_c / np.amax(n_c)
        bin_centers = b[0:-1] + (b[1]-b[0])/2
        sfh_median[i] = bin_centers[np.argmin(np.abs(n_c - 0.5))]
        sfh_up[i] = bin_centers[np.argmin(np.abs(n_c - 0.16))]
        sfh_dn[i] = bin_centers[np.argmin(np.abs(n_c - 0.84))]

    a,b,c = calctimes(temp_common_time, np.flip(sfh_median,0), Nparam+3)
    sfh_median_smooth, time_smooth = tuple_to_sfh(np.hstack([a,b,Nparam+3,c.ravel()]), zval = pg_z[np.argmin(chi2_array)])
    a,b,c = calctimes(temp_common_time, np.flip(sfh_up,0), Nparam+3)
    sfh_up_smooth, time_smooth = tuple_to_sfh(np.hstack([a,b,Nparam+3,c.ravel()]), zval = pg_z[np.argmin(chi2_array)])
    a,b,c = calctimes(temp_common_time, np.flip(sfh_dn,0), Nparam+3)
    sfh_dn_smooth, time_smooth = tuple_to_sfh(np.hstack([a,b,Nparam+3,c.ravel()]), zval = pg_z[np.argmin(chi2_array)])

    fig = plt.figure(figsize=(12,4))
#     plt.pcolor(temp_common_time, sfr_range[0:-1], sfh_posterior,cmap='magma')
#     clbr = plt.colorbar()
#     clbr.set_label('P(SFR(t))')
#     db.plot_sfh(temp_common_time, sfh_median, lookback=False, fig = fig, lw=3,label='median SFH')
    plot_sfh(time_smooth, sfh_median_smooth, lookback=True, color='k', fig = fig, lw=3,label='median SFH')
    plt.fill_between(np.amax(time_smooth) - time_smooth, sfh_dn_smooth, sfh_up_smooth, color='k',alpha=0.1)
    #plt.fill_between(temp_common_time, sfh_dn, sfh_up, color='k',alpha=0.1)

    if len(truths) == 2:
        plot_sfh(truths[0], truths[1], lookback=True, fig = fig, lw=3,label='true SFH')
        plt.ylim(0,np.amax(truths[1])*1.5)
    plt.xlim(0,np.amax(temp_common_time))
    plt.xlabel('t [lookback time; Gyr]')
    plt.ylabel('SFR(t)')
    plt.ylim(0,np.amax(sfr_range[0:-1]))
    l = plt.legend(framealpha=0.0)
#     for text in l.get_texts():
#         text.set_color("white")
    plt.show()

    return
