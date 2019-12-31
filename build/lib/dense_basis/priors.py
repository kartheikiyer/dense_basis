# SED fitting priors and modeling assumptions, all in one place

import numpy as np
import matplotlib.pyplot as plt
import corner

# cosmology assumption
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

from .gp_sfh import *

def quantile_names(N_params):
    return (np.round(np.linspace(0,100,N_params+2)))[1:-1]

class Priors(object):
    """ A class the holds prior information for the various parameters 
    during SED fitting - their distributions and bounds. Gets passed into
    generate_atlas methods.
    
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
    sample_mass_prior : sample from mass prior
    sample_sfr_prior : sample from SFR prior
    sample_sSFR_prior : sample from sSFR prior
    sample_tx_prior : sample from tx prior
    sample_Av_prior : sample from dust attenuation prior
    sample_Z_prior : sample from metallicity prior
    sample_z_prior : sample from redshift prior
    print_priors : print active priors and their distributions
    sample_sfh_tuple : sample SFH priors
    sample_all_params : sample all priors 
    sample_all_params_safesSFR : sample all priors with an sSFR limit to ensure gp_sfh doesn't have negative derivatives
    make_N_prior_draws : draw N parameter tuples from prior space
    plot_prior_distributions : plot corner plot of parameters and their priors
    plot_sfh_prlor : compute and plot the effective prior in SFH space
    
    """

    def __init__(self):
        self.mass_max = 12.0
        self.mass_min = 9.0
        self.sfr_max = -1.0
        self.sfr_min = 2.0
        self.z_min = 0.9
        self.z_max = 1.1
        self.Z_min = -1.5
        self.Z_max = 0.5
        self.dust_model = 'Calzetti'
        self.Av_min = 0.0
        self.Av_max = 1.0
        self.tx_alpha = 5.0
        self.squeeze_tx = True
        self.Nparam = 3
        self.ssfr_mean = 0
        self.ssfr_sigma = 0.5
        self.ssfr_shift = 9.0
        self.use_ssfr_prior = False
        
    def sample_mass_prior(self, size = 1):
        return np.random.uniform(size=size)*(self.mass_max-self.mass_min) + self.mass_min
    
    def sample_sfr_prior(self, size=1):
        return np.random.uniform(size=size)*(self.sfr_max-self.sfr_min) + self.sfr_min
        
    def sample_tx_prior(self, size=1):
        if size == 1:
            temp_tx = np.cumsum(np.random.dirichlet(np.ones((self.Nparam+1,))*self.tx_alpha, size=size))[0:-1]
            if self.squeeze_tx == True:
                #temp_tx = temp_tx*0.85 + 0.1 # bounding tx to [0.1,0.95] instead of [0,1]
                randfac1 = np.random.random()
                randfac2 = np.random.random()
                temp_tx = temp_tx*(1-randfac1*0.5) + (randfac1*0.5)*randfac2
            return temp_tx
        else:
            temp_tx = np.cumsum(np.random.dirichlet(np.ones((self.Nparam+1,))*self.tx_alpha, size=size),axis=1)[0:,0:-1]
            if self.squeeze_tx == True:
                #temp_tx = temp_tx*0.85 + 0.1
                randfac1 = np.random.random()
                randfac2 = np.random.random()
                temp_tx = temp_tx*(1-randfac1*0.5) + (randfac1*0.5)*randfac2
            return temp_tx
        
    def sample_z_prior(self, size=1):
        return np.random.uniform(size=size)*(self.z_max-self.z_min) + self.z_min
    
    def sample_Z_prior(self, size=1):
        return np.random.uniform(size=size)*(self.Z_max-self.Z_min) + self.Z_min
    
    def sample_Av_prior(self, size=1):
        if self.dust_model == 'Calzetti':
            return np.random.uniform(size=size)*(self.Av_max-self.Av_min) + self.Av_min
        else:
            print('not currently coded up, please email me regarding this functionality.')
            return np.zeros(size=size)*np.nan
        
    def sample_sSFR_prior(self, size=1):
        return -(np.random.lognormal(mean=self.ssfr_mean, sigma = self.ssfr_sigma, size = size)+self.ssfr_shift)
    
    def print_priors(self):
        print('--------------Priors:--------------')
        print('The prior on log mass is uniform from %.1f to %.1f.' %(self.mass_min, self.mass_max))
        if self.use_ssfr_prior == True:
            print('The prior on log sSFR_inst is -ve lognormal with mean %.1f and sigma %.1f.' %(self.ssfr_shift, self.ssfr_sigma))
        else:
            print('The prior on log SFR_inst is uniform from %.1f to %.1f.' %(self.sfr_min, self.sfr_max))
        print('The prior on tx is dirichlet with alpha = %.1f.' %(self.tx_alpha))
        print('The prior on redshift is uniform from %.1f to %.1f.' %(self.z_min, self.z_max))
        print('The prior on log metallicity/Zsolar is uniform from %.1f to %.1f.' %(self.Z_min, self.Z_max))
        print('The prior on dust (model: '+self.dust_model+') is uniform with Av: %.1f to %.1f.' %(self.Av_min, self.Av_max))
        print('-----------------------------------')
        
    def sample_sfh_tuple(self, random_seed = np.random.randint(1)):
        np.random.seed(random_seed)
        sfh_tuple = np.zeros((self.Nparam + 3,))
        sfh_tuple[0] = self.sample_mass_prior()
        if self.use_ssfr_prior == True:
            sfh_tuple[1] = self.sample_sSFR_prior() + sfh_tuple[0]
            if (sfh_tuple[1] < -3):
                sfh_tuple[1] = -3
        else:
            sfh_tuple[1] = self.sample_sfr_prior()
        sfh_tuple[2] = self.Nparam
        sfh_tuple[3:] = self.sample_tx_prior()
        return sfh_tuple
    
    def sample_all_params(self, random_seed = np.random.randint(1)):
        np.random.seed(random_seed)
        sfh_tuple = self.sample_sfh_tuple(random_seed = random_seed)
        temp_z = self.sample_z_prior()
        temp_Z = self.sample_Z_prior()
        temp_Av = self.sample_Av_prior()
        return sfh_tuple, temp_Z, temp_Av, temp_z

    def sample_all_params_safesSFR(self, random_seed = np.random.randint(1), safe_ssfr_value = 8.3):
        sfh_tuple, temp_Z, temp_Av, temp_z = self.sample_all_params(random_seed = random_seed)
        ctr = 0
        rseed = random_seed
        while (sfh_tuple[0] - sfh_tuple[1] < safe_ssfr_value) & (ctr < 10):
            sfh_tuple, temp_Z, temp_Av, temp_z = self.sample_all_params(random_seed = rseed)
            ctr = ctr + 1
            rseed = rseed + np.random.choice(100000)
        if ctr == 10:
            sfh_tuple[1] = sfh_tuple[0] - safe_ssfr_value
        return sfh_tuple, temp_Z, temp_Av, temp_z
    
    def make_N_prior_draws(self, size=10, safedraw = True, random_seed = np.random.randint(1), safe_ssfr_value = 8.3):
        sfh_tuples = np.zeros((self.Nparam+3, size))
        Avs = np.zeros((size,))
        Zs = np.zeros((size,))
        zs = np.zeros((size,))
        for i in range(size):
            if safedraw == True:
                sfh_tuples[0:,i], Zs[i], Avs[i], zs[i] = self.sample_all_params_safesSFR(random_seed = random_seed+i*7, safe_ssfr_value = safe_ssfr_value)
            else:
                sfh_tuples[0:,i], Zs[i], Avs[i], zs[i] = self.sample_all_params(random_seed = random_seed+i*7)
        return sfh_tuples, Zs, Avs, zs
    
    def plot_prior_distributions(self):
        
        a,b,c,d = self.make_N_prior_draws(size=100000, random_seed=10)
        theta_arr = np.vstack((a[0,0:], a[1,0:], a[3:,0:], b,c,d))
        
        txs = ['t'+'%.0f' %i for i in quantile_names(self.Nparam)]
        prior_labels = ['log M*', 'log SFR', 'Z', 'Av', 'z']
        prior_labels[2:2] = txs

        figure = corner.corner(theta_arr.T,labels=prior_labels, 
                plot_datapoints=False, fill_contours=True, 
                bins=50, smooth=1.0, levels=[1 - np.exp(-(1/1)**2/2),1 - np.exp(-(2/1)**2/2)], 
                label_kwargs={"fontsize": 30})
        figure.subplots_adjust(right=1.5,top=1.5)
        plt.show()
        
    def plot_sfh_prior(self, numdraws = 100, ref_mstar = 10.0, ssfr_max = -8.1, zval = 5.606):
    
        sfhs = np.zeros((1000, numdraws))
        sfh_tuples = np.zeros((self.Nparam+3, numdraws))
        sfr_error = np.zeros((numdraws,))
        mass_error = np.zeros((numdraws,))
        
        for i in (range(numdraws)):
            sfh_tuples[0:,i], _,_,_ = self.sample_all_params_safesSFR(random_seed = i*7, safe_ssfr_value = -ssfr_max)
            ssfr = sfh_tuples[1,i] - sfh_tuples[0,i]
            sfh_tuples[0,i] = ref_mstar
            sfh_tuples[1,i] = ssfr + ref_mstar
            sfhs[0:,i], time = tuple_to_sfh(sfh_tuples[0:,i], zval = zval)
            sfr_error[i] = np.log10(sfhs[-1,i]) - sfh_tuples[1,i]
            mass_error[i] = np.log10(np.trapz(x=time*1e9, y=sfhs[0:,i])) - sfh_tuples[0,i]

        sfr_error[sfr_error<-2] = -2
        sfr_error[sfr_error>2] = 2
        plt.plot(sfh_tuples[1,0:], sfr_error,'.')
        plt.show()
        plt.hist(sfr_error,30)
        plt.show()
        
        plt.plot(sfh_tuples[0,0:], mass_error,'.')
        plt.show()
        plt.hist(mass_error,30)
        plt.show()
            
        plt.figure(figsize=(12,6))
        plt.plot((np.amax(time)-time), np.nanmedian(sfhs,1),'k',lw=3)
        plt.fill_between((np.amax(time)-time), np.nanpercentile(sfhs,40,axis=1), 
                         np.nanpercentile(sfhs,60,axis=1),color='k',alpha=0.1)
        plt.fill_between((np.amax(time)-time), np.nanpercentile(sfhs,25,axis=1), 
                         np.nanpercentile(sfhs,75,axis=1),color='k',alpha=0.1)
        plt.fill_between((np.amax(time)-time), np.nanpercentile(sfhs,16,axis=1), 
                         np.nanpercentile(sfhs,84,axis=1),color='k',alpha=0.1)
        plt.plot((np.amax(time)-time), sfhs[0:,0:6],'k',lw=1, alpha=0.3)
        plt.ylabel('normalized SFR')
        plt.xlabel('lookback time [Gyr]')
        plt.show()

        txs = ['t'+'%.0f' %i for i in quantile_names(self.Nparam)]
        pg_labels = ['log sSFR']
        pg_labels[1:1] = txs

        arr = np.vstack((sfh_tuples[1,0:]-10, sfh_tuples[3:,0:]))
        corner.corner(arr.T, labels=pg_labels, plot_datapoints=False, fill_contours=True, 
                bins=50, smooth=1.0, levels=[1 - np.exp(-(1/1)**2/2),1 - np.exp(-(2/1)**2/2)], label_kwargs={"fontsize": 30})
        plt.subplots_adjust(right=1.5,top=1.5)
        plt.show()
        