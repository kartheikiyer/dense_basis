# SED fitting priors and modeling assumptions, all in one place

import numpy as np

# cosmology assumption
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# number of SFH parameters
Nparam = 3

#-------------------PRIORS-------------------------

# stellar mass - uniform prior for now
log_mass_min = 9.0
log_mass_max = 12.0
mass_prior = lambda : np.random.uniform()*(log_mass_max-log_mass_min) + log_mass_min
mass_prior_statement = 'The prior on log mass is uniform from %.1f to %.1f' %(log_mass_min, log_mass_max)

# sfr - uniform prior for now
log_sfr_min = -1.0
log_sfr_max = 2.0
sfr_prior = lambda : np.random.uniform()*(log_sfr_max-log_sfr_min) + log_sfr_min
sfr_prior_statement = 'The prior on log SFR_inst is uniform from %.1f to %.1f' %(log_sfr_min, log_sfr_max)

# tx - dirichlet priors
alpha = 3.0 # dirichlet concentration parameter
tx_prior = lambda Nparam, alpha: np.cumsum(np.sort(np.random.dirichlet(np.ones((Nparam+1,))*alpha)))[0:-1]

# redshift - uniform prior for now - testing around z~1 for now.
z_min = 0.9
z_max = 1.1
z_prior = lambda : np.random.uniform()*(z_max-z_min) + z_min
z_prior_statement = 'The prior on redshift is uniform from %.1f to %.1f' %(z_min, z_max)

# metallicity - uniform prior for now
Z_min = -1.5
Z_max = 0.5
Z_prior = lambda : np.random.uniform()*(Z_max-Z_min) + Z_min
Z_prior_statement = 'The prior on log metallicity/Zsolar is uniform from %.1f to %.1f' %(Z_min, Z_max)

# dust - uniform prior for now
Av_model = 'Calzetti'
Av_min = 0.0
Av_max = 1.0
Av_prior = lambda : np.random.uniform()*(Av_max-Av_min) + Av_min
Av_prior_statement = 'The prior on dust (model: '+Av_model+') is uniform from %.1f to %.1f' %(Av_min, Av_max)

#------------------functions for easy sampling--------------------

def print_priors():
    print(mass_prior_statement)
    print(sfr_prior_statement)
    print(z_prior_statement)
    print(Z_prior_statement)
    print(Av_prior_statement)
    return

def sample_sfh_tuple(random_seed = np.random.randint(1), Nparam = 3, alpha = 5.0):
    np.random.seed(random_seed)
    sfh_tuple = np.zeros((Nparam + 3,))
    sfh_tuple[0] = mass_prior()
    sfh_tuple[1] = sfr_prior()
    sfh_tuple[2] = Nparam
    sfh_tuple[3:] = tx_prior(Nparam, alpha)
    return sfh_tuple

def sample_all_params(random_seed = np.random.randint(1), Nparam = 3, alpha = 5.0):
    np.random.seed(random_seed)
    sfh_tuple = np.zeros((Nparam + 3,))
    sfh_tuple[0] = mass_prior()
    sfh_tuple[1] = sfr_prior()
    sfh_tuple[2] = Nparam
    sfh_tuple[3:] = tx_prior(Nparam, alpha)
    temp_z = z_prior()
    temp_Z = Z_prior()
    temp_Av = Av_prior()
    return sfh_tuple, temp_Z, temp_Av, temp_z