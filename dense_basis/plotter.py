import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(font_scale=2)
sns.set_style('ticks')

from pylab import *
rc('axes', linewidth=3)
rcParams['xtick.major.size'] = 12
rcParams['ytick.major.size'] = 12
rcParams['xtick.minor.size'] = 9
rcParams['ytick.minor.size'] = 9
rcParams['xtick.major.width'] = 3
rcParams['ytick.major.width'] = 3

def plot_sfh(timeax, sfh, lookback = False,
logx = False, logy = False):
    plt.figure(figsize=(12,4))
    if lookback == True:
        plt.plot(np.amax(timeax)/1e9 - timeax/1e9, sfh)
        plt.xlabel('lookback time [Gyr]');
    else:
        plt.plot(timeax/1e9, sfh)
        plt.xlabel('cosmic time [Gyr]');
    plt.ylabel(r'SFR(t) [$M_\odot yr^{-1}$]')
    if logx == True:
        plt.xscale('log')
    if logy == True:
        plt.yscale('log')
    plt.xlim(0,np.amax(timeax/1e9))
    plt.show()
    return

def plot_spec(lam, spec, logx = True, logy = True,
xlim = (1e2,1e8),
clip_bottom = True):
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
