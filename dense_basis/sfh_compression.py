import numpy as np
from tqdm import tqdm
import scipy.io as sio
from scipy.integrate import cumtrapz
from scipy.signal import savgol_filter

# cosmology assumption
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

from .priors import *
from .pre_grid import *
from .gp_sfh import *
from .plotter import *

from unyt import yr, Myr, Angstrom, Msun
from synthesizer.filters import FilterCollection
from synthesizer.grid import Grid
from synthesizer.parametric import SFH, Stars, ZDist
from synthesizer.parametric.galaxy import Galaxy

_SOLAR_MET = 0.02

def tofloatarr(line):
    temp = line.split()
    return np.array([float(x) for x in temp])

def ingest_sfhist_file(work_dir, sfhfile):

    id1 = []
    id2 = []
    galz = []
    sfharrs = []
    issfh = False

    file_path = work_dir + sfhfile

    ctr = 0
    try:
        with open(file_path, 'r') as file:
            for line in file:

                if ctr == 0:
                    temp = line.split()
                    H_0 = float(temp[0])
                    Omega_m = float(temp[1])

                if ctr == 1:
                    ntbins = int(line)
                    print(ntbins)
                    # Zbins = tofloatarr(line)

                if ctr == 2:
                    tbins = tofloatarr(line)

                if ctr > 2:
                    temp = tofloatarr(line)
                    if len(temp) == 3:
                        id1.append(temp[0])
                        id2.append(temp[1])
                        galz.append(temp[2])
                        if issfh == True:
                            sfharrs.append(np.array(sfharr))
                        sfharr = []
                        issfh = True
                    else:
                        sfharr.append(temp)

                ctr = ctr+1
            sfharrs.append(np.array(sfharr))
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except IOError:
        print(f"Error reading file '{file_path}'.")
        
    return id1, id2, galz, sfharrs, tbins

def ingest_sfhist_file_14march24(work_dir, sfhfile):
    """
    Updated function to read the sfhist files for the CAMELS-SAM (sam ver 14th march 2024)
    this reads the SFHs stored as 2d arrays omitting zeros.
    """

    id1 = []
    id2 = []
    galz = []
    sfharrs = []

    file_path = work_dir + sfhfile

    try:
        with open(file_path, 'r') as file:
            for lid, line in tqdm(enumerate(file)):

                if lid == 0:
                    temp = line.split()
                    H_0 = float(temp[0])
                    Omega_m = float(temp[1])

                if lid == 1:
                    ntbins = int(line)
                    print('# t bins:', ntbins)
                    newgalsfh= np.zeros((ntbins,2))

                if lid == 2:
                    tbins = tofloatarr(line)

                if lid > 2:
                    if line[0] == '#':
                        temp = tofloatarr(line[2:])
                        id1.append(temp[0])
                        id2.append(temp[1])
                        galz.append(temp[2])
                        if lid > 3:
                            sfharrs.append(newgalsfh)
                            
                        newgalsfh= np.zeros((ntbins,2))
                    else:
                        temp = tofloatarr(line)
                        newgalsfh[int(temp[0]),:] = temp[1:]                

    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except IOError:
        print(f"Error reading file '{file_path}'.")
        
    return id1, id2, galz, sfharrs, tbins



def calctimes_log(
    timeax, sfh, nparams, 
    sfr_Myr = 10, 
    scale = 'log', log_stretch = -2
):
    """
    Function to convert an input time-series into a tuple.
    In case of an SFH, this is (M*, <SFR>|_Myr, tx/tuniv)
    
    Inputs:
        timeax: (np array) times
        sfh: (np array) SFR(t) at those times
        nparams: (int) number of tx params 
        sfr_Myr: (optional, float [10]) time in Myr over which to average recent SFR
        scale: (optional, 'lin' or ['log']) scale for dividing the tx params
        log_stretch: (optional [-2]) how finely to divide the last few percentiles. only for 'log' scale
        
    Outputs:
        mass: (float) 
        sfr: (float) SFR averaged over sfr_Myr
        tx: (array of floats) tx parameters normalized to 1
    
    """
    

    sfr_index = np.argmin(np.abs(timeax-sfr_Myr/1e3))
    mass = np.log10(np.trapz(sfh,timeax*1e9))
    sfr = np.log10(np.nanmedian(sfh[-sfr_index:]))
    
    massint = np.cumsum(sfh)
    massint_normed = massint/np.amax(massint)
    tx = np.zeros((nparams,))
    
    if sfr != -np.inf:
        mass_formed_sfr_Myr = sfr_Myr * 1e6 * (10**sfr)
        mass_frac_rem = 1 - (mass_formed_sfr_Myr / (10**mass))
    else:
        mass_frac_rem = 1.0
    
    if scale == 'log':
        txints = (1-np.flip(np.logspace(log_stretch,0,nparams+2),0))/(1-10**log_stretch)*mass_frac_rem
        txints = txints[1:-1]
    elif scale == 'lin':
        txints = np.linspace(0,mass_frac_rem,nparams+2)[1:-1]
        
    # calculating the tXs from the mass quantiles
    for i in range(nparams):
        tx[i] = timeax[np.argmin(np.abs(massint_normed - txints[i]))]
    return mass, sfr, tx/np.amax(timeax)

def gp_george_interpolator(x,y,res = 1000, Nparam = 3, decouple_sfr = False):

    yerr = np.zeros_like(y)
    yerr[2:(2+Nparam)] = 0.001/np.sqrt(Nparam)
    if len(yerr) > 26:
        yerr[2:(2+Nparam)] = 0.1/np.sqrt(Nparam)
    if decouple_sfr == True:
        yerr[(2+Nparam):] = 0.1

#     kernel = np.var(y) * (kernels.Matern32Kernel(np.median(y)) *( kernels.LinearKernel(np.median(y), order=2)))
#     kernel = np.var(y) * (kernels.Matern32Kernel(np.median(y)) *( kernels.DotProductKernel(np.median(y))))
    kernel = np.var(y) * (kernels.Matern32Kernel(np.median(y)) *( kernels.PolynomialKernel(np.median(y), order=2)))
    gp = george.GP(kernel, solver=george.HODLRSolver)

    gp.compute(x.ravel(), yerr.ravel())
    
#     try:
    # optimize kernel parameters
    p0 = gp.get_parameter_vector()
    results = minimize(nll, p0, jac=grad_nll, method="L-BFGS-B", args = (gp, y))
    gp.set_parameter_vector(results.x)
#     print(results)
#     except:
#         print('couldn\'t optimize GP parameters.')

    x_pred = np.linspace(np.amin(x), np.amax(x), res)
    y_pred, pred_var = gp.predict(y.ravel(), x_pred, return_var=True)

    return x_pred, y_pred

def tuple_to_sfh_log(sfh_tuple, zval, 
                     interpolator = 'gp-george', 
                     scale='log', log_stretch = -2, 
                     cosmo = cosmo, 
                     sfr_Myr = 100, 
                     Nsfr_const_perc = 3, 
                     zerosfr_at_tbb = True, 
                     smooth_savgol = 9,
                     vb = False):
    
    Nparam = sfh_tuple.shape[0]-3
    mass = 10**sfh_tuple[0]
    sfr = 10**sfh_tuple[1]
    
    if sfr != -np.inf:
        mass_formed_sfr_Myr = sfr_Myr * 1e6 * (10**sfr)
        mass_frac_rem = 1 - (mass_formed_sfr_Myr / (10**mass))
    else:
        mass_frac_rem = 1.0
    
    if scale == 'log':
        txints = (1-np.flip(np.logspace(log_stretch,0,Nparam+2),0))/(1-10**log_stretch)*mass_frac_rem
        txints = txints[0:-1]
    elif scale == 'lin':
        txints = np.linspace(0,mass_frac_rem,nparams+2)[0:-1]
    
    mass_quantiles = txints
    time_quantiles = np.zeros_like(mass_quantiles)
    time_quantiles[1:] = sfh_tuple[3:]
    
    mass_quantiles = np.append(mass_quantiles,[1.0])
    time_quantiles = np.append(time_quantiles,[1.0])
    
    # ================== SFR at t=0 ========================
    
    if zerosfr_at_tbb == True:
        # SFR smoothly increasing from 0 at the big bang
        mass_quantiles = np.insert(mass_quantiles,1,[0.0000])
        time_quantiles = np.insert(time_quantiles,1,[1e-3])
    
    # ================ sSFR at t = t_obs ====================
    
    #correct for mass loss? (small at small sfr_Myr so ignore for now)
    mass_formed_sfr_Myr = sfr_Myr * 1e6 * (10**sfh_tuple[1])
    mass_remaining = (10**sfh_tuple[0]) - mass_formed_sfr_Myr
    
    mass_frac = 1 - mass_formed_sfr_Myr / (10**sfh_tuple[0])
    sfr_tuniv = 1 - (sfr_Myr / (cosmo.age(zval).value*1e3))
    
    mass_quantiles = np.insert(mass_quantiles, -1, [mass_frac])
    time_quantiles = np.insert(time_quantiles, -1, [sfr_tuniv])
    
    # ========= removing duplicate values ==============
    
    tqmask = (np.diff(time_quantiles)>0)
    tqmask = np.insert(tqmask,0,[True])
    time_quantiles = time_quantiles[tqmask]
    mass_quantiles = mass_quantiles[tqmask]
   
    if not np.all(np.isfinite(mass_quantiles)):
        print('Error in interpolation...')
        return np.nan, np.nan
    
    # ========= splining the time-cumulative SFR array ==============
    
    if interpolator == 'linear':
        time_arr_interp, mass_arr_interp = linear_interpolator(time_quantiles, mass_quantiles)
    elif interpolator == 'pchip':
        try:
            time_arr_interp, mass_arr_interp = Pchip_interpolator(time_quantiles, mass_quantiles)
        except:
            print((np.diff(time_quantiles)<=0))
            print(time_quantiles, np.diff(time_quantiles))
            time_arr_interp, mass_arr_interp = Pchip_interpolator(time_quantiles, mass_quantiles)
    elif interpolator == 'gp-george':
        
        time_arr_interp, mass_arr_interp = gp_george_interpolator(time_quantiles, mass_quantiles)
        time_arr_interp2, mass_arr_interp2 = Pchip_interpolator(time_quantiles, mass_quantiles)
        
        # dealing with -ve SFR in a bin
        for k in range(1,len(mass_quantiles)-1):
            trange = (time_arr_interp >= time_quantiles[k]) & (time_arr_interp <= time_quantiles[k+1])
            if np.nanmedian(np.diff(mass_arr_interp[trange])) < 1e-4:
                mass_arr_interp[trange] = mass_arr_interp2[trange]
    else:
        raise Exception('specified interpolator does not exist: {}. \n use one of the following: gp_george, gp_sklearn, linear, and pchip '.format(interpolator))
    
    # ========= derivative to compute SFR array ==============
    
    dx = np.mean(np.diff(time_arr_interp))
    sfh = np.gradient(mass_arr_interp, dx, edge_order=2)
    if smooth_savgol > 0:
        sfh = savgol_filter(sfh, smooth_savgol, 1)
    sfh[sfh<0] = 0
    timeax = time_arr_interp * cosmo.age(zval).value
    
    
    # ======== scale SFH to match input mass & SFR =============
    
    massformed = np.trapz(x=timeax*1e9, y=sfh)
    sfh = sfh / massformed * (mass)
    temp = np.argmin(np.abs(timeax-sfr_Myr/1e3))
    sfh[-temp:] = sfr

    if vb == True:

        plt.figure(figsize=(7,7))
        plt.plot([0,1],[0,1],'k--',alpha=0.3)
        plt.plot(time_quantiles, mass_quantiles,'o')
        plt.plot(time_arr_interp, mass_arr_interp)
        plt.xlabel('time / t$_{univ}$'); plt.ylabel('fraction of mass formed')
        plt.show()

        plt.plot(timeax, sfh)
        plt.xlabel('time'); plt.ylabel('SFR')
        plt.show()
    
    return sfh, timeax

def compress_sfh(galid, 
                 nparam = 30, 
                 sfhist_vars = [],
                 sfr_Myr = 10, dustval = 0.0, 
                 scale='log', log_stretch = -2,
                 interpolator = 'gp-george'):
    
    galz, sfharrs, tbins = sfhist_vars
    zval = galz[galid]    
    z_index = np.argmin(np.abs(tbins - cosmo.age(zval).value))+5
    sfhZt, tvals = (sfharrs)[galid].copy()[0:z_index,0:], tbins.copy()[0:z_index]    

    # plt.plot(tvals, sfhZt[:, 0])
    # sfht = (sfhZt[:, 1] / (0.01 * 1e9))
    sfht = np.log10(sfhZt[:, 1])
    sfht[sfht < -3] = -3 
    # plt.plot(tvals, sfht)
    
    # sfht = np.log10(np.sum(sfhZt,1))
    # Zt = np.log10(sfhZt[:, 0]) # / 10**sfht
    # Zt[Zt < -2.5] = -2.5
    # Zt[Zt > 0.5] = 0.5
    if galid > 0:
        pmhist = np.log10(sfharrs[galid-1].T[0])
        methist = np.log10(np.log10(sfhZt.T[0]) - pmhist[0:z_index])
    else:    
        methist = np.log10(np.log10(sfhZt.T[0]))
    methist[methist < -2.5] = -2.5
    methist[methist > 0.5] = 0.5
    Zt = methist
    
    # Zt = np.sum(sfhZt.T*Zvals.reshape(-1,1),0) / 10**sfht
    
    spline_time = np.linspace(0,cosmo.age(zval).value,1000)
    sfht = np.interp(spline_time, tvals, sfht)
    Zt = np.interp(spline_time, tvals, Zt)
    tvals = spline_time
    
    metval = np.nanmedian(Zt)
    Zt[np.isnan(Zt)] = -2.1
    m, sfr, tx = calctimes_log(tvals, 10**sfht, nparam, sfr_Myr = sfr_Myr, scale=scale, log_stretch=log_stretch)
    sfhtuple = calctimes_to_tuple([m, sfr, tx])
    m, sfr, tx = calctimes_log(tvals, 10**Zt, nparam, sfr_Myr = sfr_Myr, scale=scale, log_stretch=log_stretch)
    Ztuple = calctimes_to_tuple([m, sfr, tx])
    return sfhtuple, Ztuple, metval, tvals, sfht, Zt

def interp_sfh(sfh, tax, newtax):

    sfh_cdf = cumtrapz(sfh,x=tax, initial=0)
    # cdf_interp = np.interp(newtax, tax, np.flip(sfh_cdf,0))
    cdf_interp = np.interp(newtax, tax, np.flip(sfh_cdf,0))
    sfh_interp = np.zeros_like(cdf_interp)
    sfh_interp[0:-1] = - np.diff(cdf_interp)
    
    return sfh_interp


def synthesizer_nonparam_sfh(timeax, sfhist, methist, grid=grid):
    """
    convert a star formation and metallicity history
    to a format that can be ingested by synthsizer
    assuming inputs are log SFR and log Z/Zsolar
    outputs are mass formed per bin in a 2d age-met grid
    """
    
    tbw = np.mean(np.diff(grid.log10age))
    twidths = 10**(grid.log10age+tbw/2) - 10**(grid.log10age-tbw/2)
    
    # conversion to absolute units for synthesizer
    Zsolar = _SOLAR_MET
    methist = methist + np.log10(Zsolar)
    Zhist_interp = np.interp(grid.log10age, 
                             timeax+10**grid.log10age[0]/1e9, 
                             np.flip(methist,0))
    met_indices = np.array([np.argmin(np.abs(np.log10(grid.metallicity) - Zhist_interp[i])) for i in range(len(Zhist_interp))])

    # SFH
    finegrid = np.linspace(np.amin(grid.log10age), np.amax(grid.log10age),1000)
    tbw = np.mean(np.diff(finegrid))
    finewidths = 10**(finegrid+tbw/2) - 10**(finegrid-tbw/2)
    
    intsfh = interp_sfh(sfhist, timeax, 10**finegrid/1e9)/(finewidths/1e9)
    intsfh2 = np.interp(grid.log10age, finegrid, intsfh)
    
    sfhZt = np.zeros((len(intsfh2), len(grid.metallicity)))
    for i in range(len(intsfh2)):
        sfhZt[i,met_indices[i]] = intsfh2[i] * twidths[i]
    
    sfh_grid = Stars(
        log10ages = grid.log10age,
        metallicities = grid.metallicity,
        sfzh = sfhZt,
    )
    
    return sfh_grid

def get_synth_spec(stars, zval, cosmo, grid):
    
    galaxy = Galaxy(stars)
    galaxy.stars.get_spectra_incident(grid)
    lam = galaxy.stars.spectra['incident'].lam
    spec_ujy = galaxy.stars.spectra['incident'].get_fnu(cosmo, zval) / 1e3
    return lam, spec_ujy

def process_single_gal(gal_id, sfhist_vars, grid, nparam = 10, interpolator = 'gp-george', filter_list = [], filt_dir = []):
    """
    Function to compress the SFHt and Zt for a single galaxy from the SC-SAM
    or any object packaged in the sfhist_vars format
    where sfhist_vars = galz, sfharrs, tbins
    galz is the redshift
    tbins is the time array in gyr
    sfharrs is a 2d array for each galaxy storing (Zt, SFHt)
    grid is the synthesizer grid
    nparam and interpolator are the compression settings
    filter_list and filt_dir are the filter transmission curves
    """
    
    galz, sfharrs, tbins = sfhist_vars
    zval = sfhist_vars[0][gal_id]
    tindex = np.argmin(np.abs(tbins - cosmo.age(zval).value))
    timeax = tbins[0:tindex]
    sfhist = sfharrs[gal_id][0:tindex,1]

    if gal_id > 0:
        pmhist = np.log10(sfharrs[gal_id-1].T[0])
        methist = np.log10(np.log10((sfharrs)[gal_id][0:tindex].T[0]) - pmhist[0:tindex])
    else:    
        methist = np.log10(np.log10((sfharrs)[gal_id][0:tindex].T[0]))
    methist[methist < -2.5] = -2.5
    methist[methist > 0.5] = 0.5
    methist[np.isnan(methist)] = -2.1

    sfr_Myr = cosmo.age(zval).value * 20
    if sfr_Myr > 100:
        sfr_Myr = 100

    sfh_comp = compress_sfh(gal_id, nparam=nparam, sfhist_vars=sfhist_vars)
    tempsfh, temptime = tuple_to_sfh_log(np.array(sfh_comp[0]), zval, interpolator = interpolator, 
                                                         sfr_Myr=sfr_Myr, vb=False)
    tempmet, temptime = tuple_to_sfh_log(np.array(sfh_comp[1]), zval, interpolator = interpolator, 
                                                         sfr_Myr=sfr_Myr, vb=False)
    tempmet = np.log10(tempmet)
    tempmet[tempmet<-2.1] = -2.1
    temp = [np.hstack(sfh_comp[0:3])]

    # stars_true = synthesizer_nonparam_sfh(timeax, sfhist, methist)
    stars_true = synthesizer_nonparam_sfh(temptime, np.interp(temptime, timeax, sfhist), np.interp(temptime, timeax, methist), grid=grid)
    lam_true, spec_true = get_synth_spec(stars_true, zval, cosmo, grid)
    # fig, ax = stars_true.plot_sfzh(show=True);plt.show()
    # print(stars_true)

    stars_comp = synthesizer_nonparam_sfh(temptime, tempsfh, tempmet, grid=grid)
    lam_comp, spec_comp = get_synth_spec(stars_comp, zval, cosmo, grid)
    # fig, ax = stars_comp.psamlot_sfzh(show=True);plt.show()
    # print(stars_comp)

    lam = lam_true.value

    lam_mask = (lam*(1+zval) > 3.5e3)  & (lam*(1+zval) < 1e5)
    norm_fac = np.nanmedian(spec_true[lam_mask]) / np.nanmedian(spec_comp[lam_mask])
    # norm_fac = 1.0
    sfh_comp = (*sfh_comp, norm_fac)
    filcurves, _, _ = make_filvalkit_simple(lam, zval, fkit_name = filter_list, filt_dir = filt_dir)
    sed_true = np.array(calc_fnu_sed_fast(spec_true, filcurves))
    sed_comp = np.array(calc_fnu_sed_fast(spec_comp * norm_fac, filcurves))

    return sfh_comp, sed_true, sed_comp, norm_fac, temp, tempsfh, temptime, lam, lam_mask, spec_true, spec_comp, zval

def get_lam_centers_widths(filter_list, filt_dir):
    """
    Function to calculate the centers and widths of individual filters
    """
    
    lam_centers = []
    lam_widths = []

    lam = np.linspace(1e3,1e6,int(1e5))
    z = 0.0
    vb = False

    fcurves = make_filvalkit_simple(lam,z, fkit_name = filter_list ,vb=False, filt_dir = filt_dir)

    for i in range(fcurves[0].shape[1]):

        normed_fc = np.cumsum(fcurves[0].T[i])/np.amax(np.cumsum(fcurves[0].T[i]))
        lam_eff = fcurves[1][np.argmin(np.abs(normed_fc-0.5))]
        lam_width_lo = fcurves[1][np.argmin(np.abs(normed_fc-0.025))]
        lam_width_hi = fcurves[1][np.argmin(np.abs(normed_fc-0.975))]
        lam_width = lam_width_hi - lam_width_lo

        lam_centers.append(lam_eff)
        lam_widths.append(lam_width)
        if vb == True:
            print(lam_eff, lam_width)
            print(filter_names[i])
            plt.plot(fcurves[1], normed_fc)
            plt.xscale('log')
            plt.show()

    lam_centers = np.array(lam_centers)
    lam_widths = np.array(lam_widths)
    return lam_centers, lam_widths