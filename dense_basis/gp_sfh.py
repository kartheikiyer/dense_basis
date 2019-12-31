import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')

import george
from george import kernels

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,WhiteKernel, ConstantKernel as C, DotProduct, RationalQuadratic, Matern

from scipy.optimize import minimize
from scipy.interpolate import PchipInterpolator, interp1d
import scipy.io as sio

from .priors import *

import pkg_resources

def get_file(folder, filename):
    resource_package = __name__
    resource_path = '/'.join((folder, filename))  # Do not use os.path.join()
    template = pkg_resources.resource_stream(resource_package, resource_path)
    return template

fsps_mlc = sio.loadmat(get_file('train_data','fsps_mass_loss_curve.mat'))
#fsps_mlc = sio.loadmat('dense_basis/train_data/fsps_mass_loss_curve.mat')
fsps_time = fsps_mlc['timeax_fsps'].ravel()
fsps_massloss = fsps_mlc['mass_loss_fsps'].ravel()

# basic SFH tuples
rising_sfh = np.array([10.0,1.0,3,0.5,0.7,0.9])
regular_sfg_sfh = np.array([10.0,0.3,3,0.25,0.5,0.75])
young_quenched_sfh = np.array([10.0,-1.0,3,0.3,0.6,0.8])
old_quenched_sfh = np.array([10.0,-1.0,3,0.1,0.2,0.4])
old_very_quenched_sfh = np.array([10.0,-10.0,3,0.1,0.2,0.4])
double_peaked_SF_sfh = np.array([10.0,0.5,3,0.25,0.4,0.7])
double_peaked_Q_sfh = np.array([10.0,-1.0,3,0.2,0.4,0.8])

# functions:
def neg_ln_like(p, gp, y):
    gp.set_parameter_vector(p)
    return -gp.log_likelihood(y)

def grad_neg_ln_like(p, gp, y):
    gp.set_parameter_vector(p)
    return -gp.grad_log_likelihood(y)

def correct_for_mass_loss(sfh, time, mass_loss_curve_time, mass_loss_curve):
    correction_factors = np.interp(time, mass_loss_curve_time, mass_loss_curve)
    return sfh * correction_factors

def gp_interpolator(x,y,res = 1000, Nparam = 3):
    
    yerr = np.zeros_like(y)
    yerr[2:(2+Nparam)] = 0.001/np.sqrt(Nparam)
    if len(yerr) > 26:
        yerr[2:(2+Nparam)] = 0.1/np.sqrt(Nparam)

    #kernel = np.var(yax) * kernels.ExpSquaredKernel(np.median(yax)+np.std(yax))
    #k2 = np.var(yax) * kernels.LinearKernel(np.median(yax),order=1)
    #kernel = np.var(y) * kernels.Matern32Kernel(np.median(y)) #+ k2
    kernel = np.var(y) * (kernels.Matern32Kernel(np.median(y)) + kernels.LinearKernel(np.median(y), order=2))
    gp = george.GP(kernel)

    #print(xax.shape, yerr.shape)

    gp.compute(x.ravel(), yerr.ravel())

    x_pred = np.linspace(np.amin(x), np.amax(x), res)
    y_pred, pred_var = gp.predict(y.ravel(), x_pred, return_var=True)
    return x_pred, y_pred

def gp_sklearn_interpolator(x,y,res = 1000):
    
    kernel = DotProduct(10.0, (1e-2,1e2)) *RationalQuadratic(0.1)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gp.fit(x.reshape(-1,1),(y-x).reshape(-1,1))

    x_pred = np.linspace(0,1,1000)
    y_pred, sigma = gp.predict(x_pred[:,np.newaxis], return_std=True)
    y_pred = y_pred.ravel() + x_pred

    return x_pred, y_pred

def linear_interpolator(x,y,res = 1000):
    
    interpolator = interp1d(x,y)
    x_pred = np.linspace(np.amin(x), np.amax(x), res)
    y_pred = interpolator(x_pred)
    
    return x_pred, y_pred

def Pchip_interpolator(x,y,res = 1000):
    
    interpolator = PchipInterpolator(x,y)
    x_pred = np.linspace(np.amin(x), np.amax(x), res)
    y_pred = interpolator(x_pred)
    
    return x_pred, y_pred
    
def tuple_to_sfh(sfh_tuple, zval, interpolator = 'gp_george', set_sfr_100Myr = False, vb = False):
    # generate an SFH from an input tuple (Mass, SFR, {tx}) at a specified redshift
    
    
    Nparam = sfh_tuple[2]
    mass_quantiles = np.linspace(0,1,Nparam+2)
    time_quantiles = np.zeros_like(mass_quantiles)
    time_quantiles[-1] = 1
    time_quantiles[1:-1] = sfh_tuple[3:]

    # now add SFR constraints
    
    # SFR smoothly increasing from 0 at the big bang
    mass_quantiles = np.insert(mass_quantiles,1,[0.00])
    time_quantiles = np.insert(time_quantiles,1,[0.01])
    
    # SFR constrained to SFR_inst at the time of observation
    SFH_constraint_percentiles = np.array([0.96,0.97,0.98,0.99])
    for const_vals in SFH_constraint_percentiles:
        
        delta_mstar = 10**(sfh_tuple[0]) *(1-const_vals)
        delta_t = 1 - delta_mstar/(10**sfh_tuple[1])/(cosmo.age(zval).value*1e9)
        
        if (delta_t > time_quantiles[-2]) & (delta_t > 0.9):
            mass_quantiles = np.insert(mass_quantiles, -1, [const_vals], )
            time_quantiles = np.insert(time_quantiles, -1, [delta_t],)
        else:
            delta_m = 1 - ((cosmo.age(zval).value*1e9)*(1-const_vals)*(10**sfh_tuple[1]))/(10**sfh_tuple[0])
            time_quantiles = np.insert(time_quantiles, -1, [const_vals]) 
            mass_quantiles=  np.insert(mass_quantiles, -1, [delta_m])
    
    if interpolator == 'gp_george':
        time_arr_interp, mass_arr_interp = gp_interpolator(time_quantiles, mass_quantiles, Nparam = int(Nparam))
    elif interpolator == 'gp_sklearn':
        time_arr_interp, mass_arr_interp = gp_sklearn_interpolator(time_quantiles, mass_quantiles)
    elif interpolator == 'linear':
        time_arr_interp, mass_arr_interp = linear_interpolator(time_quantiles, mass_quantiles)
    elif interpolator == 'pchip':
        time_arr_interp, mass_arr_interp = Pchip_interpolator(time_quantiles, mass_quantiles)
    else:
        raise Exception('specified interpolator does not exist: {}. \n use one of the following: gp_george, gp_sklearn, linear, and pchip '.format(interpolator))
    
    sfh_scale = 10**(sfh_tuple[0])/(cosmo.age(zval).value*1e9/1000)
    sfh = np.diff(mass_arr_interp)*sfh_scale
    sfh[sfh<0] = 0
    sfh = np.insert(sfh,0,[0])
    if set_sfr_100Myr == True:
        time_100Myr = np.argmin(np.abs(time_arr_interp*cosmo.age(zval).value - 0.1))
        sfh[-time_100Myr:] = 10**sfh_tuple[1] 
    timeax = time_arr_interp * cosmo.age(zval).value
    
    if vb == True:
        print('time and mass quantiles:')
        print(time_quantiles, mass_quantiles)
        plt.plot(time_quantiles, mass_quantiles,'--o')
        plt.plot(time_arr_interp, mass_arr_interp)
        plt.axis([0,1,0,1])
        #plt.axis([0.9,1.05,0.9,1.05])
        plt.show()

        print('instantaneous SFR: %.1f' %sfh[-1])
        plt.plot(np.amax(time_arr_interp) - time_arr_interp, sfh)
        #plt.xscale('log')
        plt.show()
        
    return sfh, timeax



def calctimes(timeax,sfh,nparams):
    
    massint = np.cumsum(sfh)
    massint_normed = massint/np.amax(massint)
    tx = np.zeros((nparams,))
    for i in range(nparams):
        tx[i] = timeax[np.argmin(np.abs(massint_normed - 1*(i+1)/(nparams+1)))]
        #tx[i] = (np.argmin(np.abs(massint_normed - 1*(i+1)/(nparams+1))))
        #print(1*(i+1)/(nparams+1))
        
    #mass = np.log10(np.sum(sfh)*1e9)
    mass = np.log10(np.trapz(sfh,timeax*1e9))
    sfr = np.log10(sfh[-1])
        
    return mass, sfr, tx/np.amax(timeax)
