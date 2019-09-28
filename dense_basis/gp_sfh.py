import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,WhiteKernel, ConstantKernel as C, DotProduct, RationalQuadratic, Matern

import warnings
warnings.filterwarnings('ignore')

from .priors import *

def gp_sfh_sklearn(sfh_tuple,zval=5.606,vb = False,std=False):

    """
    Function to generate a (possibly random?) SFH given the following input parameters:
    mass = stellar mass
    sfr = current sfr, averaged over some timescale the SED is sensitive to, 10-100Myr
    Nparam = number of parameters. 1=t50, 2=t33,t67, 3=t25,t50,75 etc
    param_arr = the numpy array of actual parameters

    """

    mass = sfh_tuple[0]
    sfr = 10**sfh_tuple[1]
    Nparam = int(sfh_tuple[2])
    param_arr = sfh_tuple[3:]

    # fix SFH(t=0) = 0
    # fix SFH(t=tobs) = SFR
    # fit \int SFH(t)dt = mass


    #zval = 1.0

    # this is the times at which the galaxy formed x percentiles of its observed stellar mass...
    input_Nparams = Nparam
    input_params = param_arr

    # these are time quantities (tx)
    input_params_full = np.zeros((input_Nparams+5,))
    input_params_full[0] = 0
    input_params_full[1] = 0.01
    input_params_full[2:-3] = input_params
    input_params_full[-3] = 0.98
    input_params_full[-2] = 0.99
    input_params_full[-1] = 1

    # these are galaxy mass quantities M(t)
    temp_mass = np.linspace(0,1,input_Nparams+2)
    input_mass = np.zeros((input_params_full.shape))

    input_mass[1:-3] = temp_mass[0:-1]
    input_mass[1] = 0.0
    input_mass[-3] = 1.0 - sfr*(1.0-0.98)*(cosmo.age(zval).value*1e9)/np.power(10,mass)
    input_mass[-2] = 1.0 - sfr*(1.0-0.99)*(cosmo.age(zval).value*1e9)/np.power(10,mass)
    input_mass[-1] = 1.0
    # the last two statements help to fix the SFR at t_obs

    xax = input_params_full.reshape(-1,1)
    yax = (input_mass-input_params_full).reshape(-1,1)

    #------------------------------------

    kernel = DotProduct(10.0, (1e-2,1e2)) *RationalQuadratic(0.1)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gp.fit(xax,yax)

    x = np.linspace(0,1,1000)
    y_pred, sigma = gp.predict(x[:,np.newaxis], return_std=True)
    y_pred = y_pred.ravel() + x

    mass_unnormed = y_pred*np.power(10,mass)
    time_unnormed = x*cosmo.age(zval).value*1e9

    sfr_unnormed = np.diff(mass_unnormed)/np.diff(time_unnormed)
    gen_sfh = np.zeros((mass_unnormed.shape))
    gen_sfh[1:] = sfr_unnormed

    mask = (gen_sfh<0)
    gen_sfh[mask] = 0

    #y_mean, y_cov = gp.predict(X_[:,np.newaxis], return_cov=True)
    if vb == True:

        plt.plot(input_params_full,input_mass,'k.',markersize=15)
        plt.plot(x,y_pred,'g--')
        plt.fill_between(x,y_pred.ravel()- sigma,y_pred.ravel()+sigma,alpha=0.1,color='g')
        plt.axis([0,1,0,1])
        plt.xlabel('normalized time')
        plt.ylabel('normalized mass')
        plt.show()

        plt.plot(np.amax(time_unnormed)/1e9-time_unnormed/1e9,gen_sfh,'k-',lw=3)
        plt.xlabel('t [lookback time; Gyr]',fontsize=14)
        plt.ylabel('SFR(t) [solar masses/yr]',fontsize=14)
        plt.show()

    if std == False:
        return gen_sfh, time_unnormed
    else:
        mass_sigmaup_unnormed = (y_pred.ravel()+sigma)*np.power(10,mass)
        mass_sigmadn_unnormed = (y_pred.ravel()-sigma)*np.power(10,mass)
        sfr_sigmaup_unnormed = np.diff(mass_sigmaup_unnormed)/np.diff(time_unnormed)
        sfr_sigmadn_unnormed = np.diff(mass_sigmadn_unnormed)/np.diff(time_unnormed)
        gen_sfh_up = np.zeros((mass_unnormed.shape))
        gen_sfh_dn = np.zeros((mass_unnormed.shape))
        gen_sfh_up[1:] = sfr_sigmaup_unnormed
        gen_sfh_dn[1:] = sfr_sigmadn_unnormed

        maskup = (gen_sfh_up < 0)
        maskdn = (gen_sfh_dn < 0)

        gen_sfh_up[maskup] = 0
        gen_sfh_dn[maskdn] = 0
        return gen_sfh, gen_sfh_up, gen_sfh_dn, time_unnormed
