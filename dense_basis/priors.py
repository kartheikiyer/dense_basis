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

        self.sfr_prior_type = 'sSFRflat' # options are SFRflat, sSFRflat, sSFRlognormal
        self.sfr_max = -1.0
        self.sfr_min = 2.0
        self.ssfr_min = -11.0
        self.ssfr_max = -7.0
        self.ssfr_mean = 0.6
        self.ssfr_sigma = 0.4 # roughly ~2.5 dex width, use 0.1 for a tighter correlation
        self.ssfr_shift = 9.0

        self.z_min = 0.9
        self.z_max = 1.1

        self.met_treatment = 'flat' # options are flat and massmet
        self.Z_min = -1.5
        self.Z_max = 0.25
        self.massmet_width = 0.3

        self.dust_model = 'Calzetti'
        self.dust_prior = 'exp'
        self.Av_min = 0.0
        self.Av_max = 1.0
        self.Av_exp_scale = 1.0/3.0

        self.sfh_treatment = 'custom' # options are custom and TNGlike
        self.tx_alpha = 5.0
        self.Nparam = 3
        self.decouple_sfr = False
        self.decouple_sfr_time = 100 # in Myr


    def sample_mass_prior(self, size = 1):
        massval = np.random.uniform(size=size)*(self.mass_max-self.mass_min) + self.mass_min
        self.massval = massval
        return massval

    def sample_sfr_prior(self, size=1):
        if self.sfr_prior_type == 'SFRflat':
            return np.random.uniform(size=size)*(self.sfr_max-self.sfr_min) + self.sfr_min
        elif self.sfr_prior_type == 'sSFRflat':
            return np.random.uniform(size=size)*(self.ssfr_max-self.ssfr_min) + self.ssfr_min + self.massval
        elif self.sfr_prior_type == 'sSFRlognormal':
            temp = np.random.lognormal(mean = self.ssfr_mean, sigma = self.ssfr_shift, size=size) # about a ~2.5 dex width.
            temp = temp - np.nanmedian(temp)
            temp = np.log10(10.0/(db.cosmo.age(zval).value*1e9))-temp
            sfrval = temp + self.massval
            return sfrval
        else:
            print('unknown SFR prior type. choose from SFRflat, sSFRflat, or sSFRlognormal.')
            return np.nan

    def sample_tx_prior(self, size=1):
        if self.sfh_treatment == 'TNGlike':
            #tng_zvals = np.load('train_data/alpha_lookup_tables/tng_alpha_zvals.npy')
            #tng_alphas = np.load('train_data/alpha_lookup_tables/tng_alpha_Nparam_%.0f.npy' %self.Nparam)
            tng_zvals = np.load(get_file('train_data/alpha_lookup_tables','tng_alpha_zvals.npy'))
            tng_alphas = np.load(get_file('train_data/alpha_lookup_tables','tng_alpha_Nparam_%.0f.npy' %self.Nparam))
            tng_best_z_index = np.argmin(np.abs(tng_zvals - self.zval))
            self.tx_alpha = tng_alphas[0:,tng_best_z_index]

        if size == 1:
            temp_tx = np.cumsum(np.random.dirichlet(np.ones((self.Nparam+1,))*self.tx_alpha, size=size))[0:-1]
            return temp_tx
        else:
            temp_tx = np.cumsum(np.random.dirichlet(np.ones((self.Nparam+1,))*self.tx_alpha, size=size),axis=1)[0:,0:-1]
            return temp_tx

    def sample_z_prior(self, size=1):
        zval = np.random.uniform(size=size)*(self.z_max-self.z_min) + self.z_min
        self.zval = zval
        return zval

    def sample_Z_prior(self, size=1):
        if self.met_treatment == 'flat':
            return np.random.uniform(size=size)*(self.Z_max-self.Z_min) + self.Z_min
        elif self.met_treatment == 'massmet':
            met = np.random.normal(scale = self.massmet_width, size=size) + (self.massval - 7)/(10.8-7) - 1.0 # roughly solar met at MW-mass
            return met

    def sample_Av_prior(self, size=1):
        if self.dust_model == 'Calzetti':
            if self.dust_prior == 'flat':
                return np.random.uniform(size=size)*(self.Av_max-self.Av_min) + self.Av_min
            elif self.dust_prior == 'exp':
                return np.random.exponential(size=size)*(self.Av_exp_scale)
            else:
                print('unknown dust_prior. options are flat and exp')
        if self.dust_model == 'CF00':
            # using parameters from Pacifici+16 for now
            slope_ism = np.random.uniform(size=size)*(0.6) - 1.0 # ism slope in range [-1.0,-0.4]
            slope_bc = -1.3*np.ones((size,))
            mu_ISM_v = np.random.uniform(size=size)*0.6 + 0.1 #\mu in range [0.1,0.7]
            if self.dust_prior == 'flat':
                tau_v = (np.random.uniform(size=size)*(self.Av_max-self.Av_min) + self.Av_min)
            elif self.dust_prior == 'exp':
                tau_v = np.random.exponential(size=size)*(self.Av_exp_scale)
            else:
                print('unknown dust_prior. options are flat and exp')
            return slope_ism, slope_bc, mu_ISM_v, tau_v
        else:
            print('not currently coded up, please email me regarding this functionality.')
            return np.zeros(size=size)*np.nan

    def print_priors(self):
        print('--------------Priors:--------------')
        print('The prior on log mass is uniform from %.1f to %.1f.' %(self.mass_min, self.mass_max))
        if self.use_ssfr_prior == True:
            print('The prior on log sSFR_inst is -ve lognormal with mean %.1f and sigma %.1f.' %(self.ssfr_shift, self.ssfr_sigma))
        else:
            print('The prior on log SFR_inst is uniform from %.1f to %.1f.' %(self.sfr_min, self.sfr_max))
        print('The prior on tx is dirichlet with alpha = ',(self.tx_alpha))
        print('The prior on redshift is uniform from %.1f to %.1f.' %(self.z_min, self.z_max))
        print('The prior on log metallicity/Zsolar is uniform from %.1f to %.1f.' %(self.Z_min, self.Z_max))
        print('The prior on dust (model: '+self.dust_model+') is uniform with Av: %.1f to %.1f.' %(self.Av_min, self.Av_max))
        print('-----------------------------------')

    def sample_sfh_tuple(self):
        sfh_tuple = np.zeros((self.Nparam + 3,))
        sfh_tuple[0] = self.sample_mass_prior()
        sfh_tuple[1] = self.sample_sfr_prior()
        sfh_tuple[2] = self.Nparam
        sfh_tuple[3:] = self.sample_tx_prior()
        return sfh_tuple

    def sample_all_params(self, random_seed = np.random.randint(1)):
        np.random.seed(random_seed)
        temp_z = self.sample_z_prior()
        sfh_tuple = self.sample_sfh_tuple()
        temp_Z = self.sample_Z_prior()
        temp_Av = self.sample_Av_prior()
        return sfh_tuple, temp_Z, temp_Av, temp_z

    def make_N_prior_draws(self, size=10, random_seed = np.random.randint(1)):
        sfh_tuples = np.zeros((self.Nparam+3, size))
        Avs = np.zeros((size,))
        Zs = np.zeros((size,))
        zs = np.zeros((size,))
        for i in range(size):
            sfh_tuples[0:,i], Zs[i], Avs[i], zs[i] = self.sample_all_params(random_seed = random_seed+i*7)
        return sfh_tuples, Zs, Avs, zs

    def plot_prior_distributions(self, num_draws = 100000):

        a,b,c,d = self.make_N_prior_draws(size = num_draws, random_seed=10)
        theta_arr = np.vstack((a[0,0:], a[1,0:], a[3:,0:], b,c,d))

        txs = ['t'+'%.0f' %i for i in quantile_names(self.Nparam)]
        if self.dust_model == 'Calzetti':
            prior_labels = ['log M*', 'log SFR', 'Z', 'Av', 'z']
        elif self.dust_model == 'CF00':
            print('currently not implemented.')
            return
        prior_labels[2:2] = txs

        figure = corner.corner(theta_arr.T,labels=prior_labels,
                plot_datapoints=False, fill_contours=True,
                bins=50, smooth=1.0, levels=[1 - np.exp(-(1/1)**2/2),1 - np.exp(-(2/1)**2/2)],
                label_kwargs={"fontsize": 30})
        figure.subplots_adjust(right=1.5,top=1.5)
        plt.show()

    def plot_sfh_prior(self, numdraws = 100, ref_mstar = 10.0, zval = 5.606):

        sfhs = np.zeros((1000, numdraws))
        sfh_tuples = np.zeros((self.Nparam+3, numdraws))
        sfr_error = np.zeros((numdraws,))
        mass_error = np.zeros((numdraws,))

        for i in (range(numdraws)):
            sfh_tuples[0:,i], _,_,_ = self.sample_all_params(random_seed = i*7)
            ssfr = sfh_tuples[1,i] - sfh_tuples[0,i]
            sfh_tuples[0,i] = ref_mstar
            sfh_tuples[1,i] = ssfr + ref_mstar
            sfhs[0:,i], time = tuple_to_sfh(sfh_tuples[0:,i], zval = zval)
            sfr_error[i] = np.log10(sfhs[-1,i]) - sfh_tuples[1,i]
            mass_error[i] = np.log10(np.trapz(x=time*1e9, y=sfhs[0:,i])) - sfh_tuples[0,i]

        # sfr_error[sfr_error<-2] = -2
        # sfr_error[sfr_error>2] = 2
        # plt.plot(sfh_tuples[1,0:], sfr_error,'.')
        # plt.show()
        # plt.hist(sfr_error,30)
        # plt.show()
        #
        # plt.plot(sfh_tuples[0,0:], mass_error,'.')
        # plt.show()
        # plt.hist(mass_error,30)
        # plt.show()

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
