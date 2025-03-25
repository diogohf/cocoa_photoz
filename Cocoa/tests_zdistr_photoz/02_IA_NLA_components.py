import sys, platform, os
os.environ['OMP_NUM_THREADS'] = '8'
import matplotlib
import math
from matplotlib import pyplot as plt
import numpy as np
import euclidemu2
import scipy
import cosmolike_lsst_y1_interface as ci
from getdist import IniFile
import itertools
import iminuit
import functools
print(sys.version)
print(os.getcwd())

# GENERAL PLOT OPTIONS
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['xtick.bottom'] = True
matplotlib.rcParams['xtick.top'] = False
matplotlib.rcParams['ytick.right'] = False
matplotlib.rcParams['axes.edgecolor'] = 'black'
matplotlib.rcParams['axes.linewidth'] = '1.0'
matplotlib.rcParams['axes.labelsize'] = 'medium'
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['grid.linewidth'] = '0.0'
matplotlib.rcParams['grid.alpha'] = '0.18'
matplotlib.rcParams['grid.color'] = 'lightgray'
matplotlib.rcParams['legend.labelspacing'] = 0.77
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.format'] = 'pdf'
matplotlib.rcParams['text.usetex'] = True

# Jupyter Notebook Display options
import IPython
IPython.display.display(IPython.display.HTML("<style>:root { --jp-notebook-max-width: 85% !important; }</style>"))
IPython.display.display(IPython.display.HTML("<style>div.output_scroll { height: 54em; }</style>"))

# IMPORT CAMB
sys.path.insert(0, os.environ['ROOTDIR']+'/external_modules/code/CAMB/build/lib.linux-x86_64-'+os.environ['PYTHON_VERSION'])
import camb
from camb import model
print('Using CAMB %s installed at %s'%(camb.__version__,os.path.dirname(camb.__file__)))

CAMBAccuracyBoost = 1.1
non_linear_emul = 2
CLprobe="xi"

path= "../external_modules/data/lsst_y1"
data_file="lsst_y1_M1_GGL0.05.dataset"

IA_model = 0
IA_redshift_evolution = 3

ntheta = 26 
theta_min_arcmin = 2.5 
theta_max_arcmin = 900

### PARAMETERS
As_1e9 = 2.1
ns = 0.96605
H0 = 67.32
omegab = 0.04
omegam = 0.3
mnu = 0.06
LSST_DZ_S1 = 0.0414632
LSST_DZ_S2 = 0.00147332
LSST_DZ_S3 = 0.0237035
LSST_DZ_S4 = -0.0773436
LSST_DZ_S5 = -8.67127e-05
LSST_M1 = 0.0191832
LSST_M2 = -0.0431752
LSST_M3 = -0.034961
LSST_M4 = -0.0158096
LSST_M5 = -0.0158096
LSST_A1_1 = 0.606102
LSST_A1_2 = -1.51541
w0pwa = -0.9
w = -0.9


### CAMB FUNCTION
def get_camb_cosmology(omegam = omegam, omegab = omegab, H0 = H0, ns = ns, 
                       As_1e9 = As_1e9, w = w, w0pwa = w0pwa, AccuracyBoost = 1.0, 
                       kmax = 10, k_per_logint = 20, CAMBAccuracyBoost=1.1):

    As = lambda As_1e9: 1e-9 * As_1e9
    wa = lambda w0pwa, w: w0pwa - w
    omegabh2 = lambda omegab, H0: omegab*(H0/100)**2
    omegach2 = lambda omegam, omegab, mnu, H0: (omegam-omegab)*(H0/100)**2-(mnu*(3.046/3)**0.75)/94.0708
    omegamh2 = lambda omegam, H0: omegam*(H0/100)**2

    CAMBAccuracyBoost = CAMBAccuracyBoost*AccuracyBoost
    kmax = max(kmax/2.0, kmax*(1.0 + 3*(AccuracyBoost-1)))
    k_per_logint = max(k_per_logint/2.0, int(k_per_logint) + int(3*(AccuracyBoost-1)))
    extrap_kmax = max(max(2.5e2, 3*kmax), max(2.5e2, 3*kmax) * AccuracyBoost)

    z_interp_1D = np.concatenate( (np.concatenate( (np.linspace(0,2.0,1000),
                                                    np.linspace(2.0,10.1,200)),
                                                    axis=0
                                                 ),
                                   np.linspace(1080,2000,20)),
                                   axis=0)
    
    z_interp_2D = np.concatenate((np.linspace(0, 2.0, 95), np.linspace(2.25, 10, 5)),  axis=0)

    log10k_interp_2D = np.linspace(-4.2, 2.0, 1200)

    pars = camb.set_params(H0=H0, 
                           ombh2=omegabh2(omegab, H0), 
                           omch2=omegach2(omegam, omegab, mnu, H0), 
                           mnu=mnu, 
                           omk=0, 
                           tau=0.06,  
                           As=As(As_1e9), 
                           ns=ns, 
                           halofit_version='takahashi', 
                           lmax=10,
                           AccuracyBoost=CAMBAccuracyBoost,
                           lens_potential_accuracy=1.0,
                           num_massive_neutrinos=1,
                           nnu=3.046,
                           accurate_massive_neutrino_transfers=False,
                           k_per_logint=k_per_logint,
                           kmax = kmax);
    
    pars.set_dark_energy(w=w, wa=wa(w0pwa, w), dark_energy_model='ppf');    
    
    pars.NonLinear = model.NonLinear_both
    
    pars.set_matter_power(redshifts = z_interp_2D, kmax = kmax, silent = True);
    results = camb.get_results(pars)
    
    PKL  = results.get_matter_power_interpolator(var1="delta_tot", var2="delta_tot", nonlinear = False, 
                                                 extrap_kmax = extrap_kmax, hubble_units = False, k_hunit = False);
    
    PKNL = results.get_matter_power_interpolator(var1="delta_tot", var2="delta_tot",  nonlinear = True, 
                                                 extrap_kmax = extrap_kmax, hubble_units = False, k_hunit = False);
    
    lnPL = np.empty(len(log10k_interp_2D)*len(z_interp_2D))
    for i in range(len(z_interp_2D)):
        lnPL[i::len(z_interp_2D)] = np.log(PKL.P(z_interp_2D[i], np.power(10.0,log10k_interp_2D)))
    lnPL  += np.log(((H0/100.)**3)) 
    
    lnPNL  = np.empty(len(log10k_interp_2D)*len(z_interp_2D))
    if non_linear_emul == 1:
        params = { 'Omm'  : omegam, 
                   'As'   : As(As_1e9), 
                   'Omb'  : omegab,
                   'ns'   : ns, 
                   'h'    : H0/100., 
                   'mnu'  : mnu,  
                   'w'    : w, 
                   'wa'   : wa(w0pwa, w)
                 }
        kbt, bt = euclidemu2.get_boost( params, 
                                        z_interp_2D, 
                                        np.power(10.0, np.linspace( -2.0589, 0.973, len(log10k_interp_2D)))
                                      )
        log10k_interp_2D = log10k_interp_2D - np.log10(H0/100.)
        
        for i in range(len(z_interp_2D)):    
            lnbt = scipy.interpolate.interp1d(np.log10(kbt), np.log(bt[i]), kind = 'linear', 
                                              fill_value = 'extrapolate', 
                                              assume_sorted = True)(log10k_interp_2D)
            lnbt[np.power(10,log10k_interp_2D) < 8.73e-3] = 0.0
            lnPNL[i::len(z_interp_2D)]  = lnPL[i::len(z_interp_2D)] + lnbt
    elif non_linear_emul == 2:
        for i in range(len(z_interp_2D)):
            lnPNL[i::len(z_interp_2D)] = np.log(PKNL.P(z_interp_2D[i], np.power(10.0, log10k_interp_2D)))            
        log10k_interp_2D = log10k_interp_2D - np.log10(H0/100.)
        lnPNL += np.log(((H0/100.)**3))

    G_growth = np.sqrt(PKL.P(z_interp_2D,0.0005)/PKL.P(0,0.0005))
    G_growth = G_growth*(1 + z_interp_2D)
    G_growth = G_growth/G_growth[len(G_growth)-1]

    chi = results.comoving_radial_distance(z_interp_1D, tol=1e-4) * (H0/100.)

    return (log10k_interp_2D, z_interp_2D, lnPL, lnPNL, G_growth, z_interp_1D, chi)


### COSMIC SHEAR FUNCTION
def C_ss_tomo_limber(ell, 
                     omegam = omegam, 
                     omegab = omegab, 
                     H0 = H0, 
                     ns = ns, 
                     As_1e9 = As_1e9, 
                     w = w, 
                     w0pwa = w0pwa,
                     A1  = [LSST_A1_1, LSST_A1_2, 0, 0, 0], 
                     A2  = [0, 0, 0, 0, 0],
                     BTA = [0, 0, 0, 0, 0],
                     shear_photoz_bias = [LSST_DZ_S1, LSST_DZ_S2, LSST_DZ_S3, LSST_DZ_S4, LSST_DZ_S5],
                     M = [LSST_M1, LSST_M2, LSST_M3, LSST_M4, LSST_M5],
                     baryon_sims = None,
                     AccuracyBoost = 1.0, 
                     kmax = 10, 
                     k_per_logint = 20, 
                     CAMBAccuracyBoost=1.1,
                     CLAccuracyBoost = 1.0, 
                     CLIntegrationAccuracy = 1,
                     ia_nla_term=-1):

    (log10k_interp_2D, z_interp_2D, lnPL, lnPNL, G_growth, z_interp_1D, chi) = get_camb_cosmology(omegam=omegam, 
        omegab=omegab, H0=H0, ns=ns, As_1e9=As_1e9, w=w, w0pwa=w0pwa, AccuracyBoost=AccuracyBoost, kmax=kmax,
        k_per_logint=k_per_logint, CAMBAccuracyBoost=CAMBAccuracyBoost)

    CLAccuracyBoost = CLAccuracyBoost * AccuracyBoost
    CLSamplingBoost = CLAccuracyBoost * AccuracyBoost
    CLIntegrationAccuracy = max(0, CLIntegrationAccuracy + 3*(AccuracyBoost-1.0))
    ci.init_accuracy_boost(1.0, CLSamplingBoost, int(CLIntegrationAccuracy))

    ci.set_cosmology(omegam = omegam, 
                     H0 = H0, 
                     log10k_2D = log10k_interp_2D, 
                     z_2D = z_interp_2D, 
                     lnP_linear = lnPL,
                     lnP_nonlinear = lnPNL,
                     G = G_growth,
                     z_1D = z_interp_1D,
                     chi = chi)
    ci.set_nuisance_shear_calib(M = M)
    ci.set_nuisance_shear_photoz(bias = shear_photoz_bias)
    ci.set_nuisance_ia(A1 = A1, A2 = A2, B_TA = BTA)
    ci.set_nuisance_pz_model(pz_model=0)
    ci.set_nuisance_ia_nla_term(ia_nla_term=ia_nla_term)

    if baryon_sims is None:
        ci.reset_bary_struct()
    else:
        ci.init_baryons_contamination(sim = baryon_sims)
        
    return ci.C_ss_tomo_limber(l = ell), ci.C_ss_tomo_limber_WK1WK2PK(l = ell)


### PLOT FUNCTION
def plot_C_ss_tomo_limber(ell, C_sss, ia_nla_terms, ylabel, C_ss_ref = None, param = None, colorbarlabel = None, lmin = 30, lmax = 1500, 
                          cmap = 'gist_rainbow', ylim = [0.75,1.25], linestyle = None, linewidth = None,
                          legend = None, legendloc = (0.65,0.60), yaxislabelsize = 16, yaxisticklabelsize = 10, 
                          xaxisticklabelsize = 20, bintextpos = [0.2, 0.85], bintextsize = 15, figsize = (12, 12), 
                          show = 1):

    nell, ntomo, ntomo2 = C_sss[0][0].shape
    if ntomo != ntomo2:
        print("Bad Input (ntomo)")
        return 0
      
    if nell != len(ell):
        print("Bad Input (number of ell)")
        return 0
    if not (C_ss_ref is None):
        nell2, ntomo3, ntomo4 = C_ss_ref.shape
        if (ntomo3 != ntomo4) or (nell != nell2):
            print(f"notomo = {ntomo}, ntomo_REF = {ntomo3}")
            print(f"Nell = {nell}, Nell_REF = {nell2}")
            return 0   
        
    if C_ss_ref is None:
        fig, axes = plt.subplots(
            nrows = ntomo, 
            ncols = ntomo, 
            figsize = figsize, 
            sharex = True, 
            sharey = False, 
            gridspec_kw = {'wspace': 0.25, 'hspace': 0.05})
    else:
        fig, axes = plt.subplots(
            nrows = ntomo, 
            ncols = ntomo, 
            figsize = figsize, 
            sharex = True, 
            sharey = True, 
            gridspec_kw = {'wspace': 0, 'hspace': 0})
    
    cm = plt.get_cmap(cmap)
    
    if not (param is None):
        cb = fig.colorbar(
            matplotlib.cm.ScalarMappable(norm = matplotlib.colors.Normalize(param[0], param[-1]), cmap = 'gist_rainbow'), 
            ax = axes.ravel().tolist(), 
            orientation = 'vertical', 
            aspect = 50, 
            pad = -0.16, 
            shrink = 0.5)
        if not (colorbarlabel is None):
            cb.set_label(label = colorbarlabel, size = 20, weight = 'bold', labelpad = 2)
        if len(param) != len(C_sss[0]):
            print("Bad Input")
            return 0
    
    colors=['C0','red','darkgreen','orange','k']
    ls=['-','-','-','-','--']
    
    for i in range(ntomo):
        for j in range(ntomo):
            if i>j:                
                axes[j,i].axis('off')
            else:
                clmin = []
                clmax = []
                for ia_idx,C_ss in enumerate(C_sss):
                    for Cl in C_ss:  
                        tmp = Cl[:,i,j]
                        clmin.append(np.min(tmp))
                        clmax.append(np.max(tmp))
        
                    axes[j,i].set_xlim([lmin, lmax])
                    
                    if C_ss_ref is None:
                        axes[j,i].set_yscale('log')
                        # pass
                        # axes[j,i].set_ylim([np.min(ylim[0]*np.array(clmin)), np.max(ylim[1]*np.array(clmax))])
                    else:
                        tmp = np.array(ylim) - 1
                        # axes[j,i].set_ylim(tmp.tolist())
                        # axes[j,i].set_yscale('linear')
                        
                    axes[j,i].set_xscale('log')
                    
                    if i == 0:
                        if C_ss_ref is None:
                            axes[j,i].set_ylabel(ylabel, fontsize=yaxislabelsize)
                        else:
                            axes[j,i].set_ylabel("frac. diff.", fontsize=yaxislabelsize)
                    for item in (axes[j,i].get_yticklabels()):
                        item.set_fontsize(yaxisticklabelsize)
                    for item in (axes[j,i].get_xticklabels()):
                        item.set_fontsize(xaxisticklabelsize)
                    
                    if j == 4:
                        axes[j,i].set_xlabel(r"$\ell$", fontsize=16)
                    
                    axes[j,i].text(bintextpos[0], bintextpos[1], 
                        "$(" +  str(i) + "," +  str(j) + ")$", 
                        horizontalalignment = 'center', 
                        verticalalignment = 'center',
                        fontsize = bintextsize,
                        usetex = True,
                        transform = axes[j,i].transAxes)
                    
                    for x, Cl in enumerate(C_ss):
                        if C_ss_ref is None:
                            tmp = Cl[:,i,j]
                        else:
                            tmp = Cl[:,i,j] / C_ss_ref[:,i,j] - 1 
                        lines = axes[j,i].plot(ell, tmp, 
                                            color=colors[ia_idx], 
                                            linewidth=2, 
                                            linestyle=ls[ia_idx])
    
    fig.legend(
        legend, 
        loc=legendloc,
        borderpad=0.1,
        handletextpad=0.4,
        handlelength=1.5,
        columnspacing=0.35,
        scatteryoffsets=[0],
        frameon=False,
        fontsize=20)

    if not (show is None):
        fig.show()
    else:
        return (fig, axes)
    

### Init Cosmolike
ini = IniFile(os.path.normpath(os.path.join(path, data_file)))

lens_file = ini.relativeFileName('nz_lens_file')

source_file = ini.relativeFileName('nz_source_file')

lens_ntomo = ini.int("lens_ntomo")

source_ntomo = ini.int("source_ntomo")

ci.initial_setup()

ci.init_accuracy_boost(1.0, 1.0, int(1))

ci.init_cosmo_runmode(is_linear = False)

ci.init_redshift_distributions_from_files(
      lens_multihisto_file=lens_file,
      lens_ntomo=int(lens_ntomo), 
      source_multihisto_file=source_file,
      source_ntomo=int(source_ntomo))

ci.init_IA( ia_model = int(IA_model), 
            ia_redshift_evolution = int(IA_redshift_evolution))

### PLOT
ell = np.arange(25., 1500., 20.) # Make sure np.arange are set w/ float numbers (otherwise there are aliasing problems)

ia_nla_terms = [0,1,2,3,-1]

C_ss_0,C_ss_1,C_ss_2,C_ss_3,C_ss = [],[],[],[],[]
C_ss_f0,C_ss_f1,C_ss_f2,C_ss_f3,C_ssf = [],[],[],[],[]

C_sss = [C_ss_0,C_ss_1,C_ss_2,C_ss_3,C_ss]
C_sssf = [C_ss_f0,C_ss_f1,C_ss_f2,C_ss_f3,C_ssf]

(Cl_tot, tmp_tot) = C_ss_tomo_limber(ell=ell)[0]

for i,idx in enumerate(ia_nla_terms):
    if idx == -1:
        (Cl_i, tmp_i) = C_ss_tomo_limber(ell=ell)[0]
    else:    
        (Cl_i, tmp_i) = C_ss_tomo_limber(ell=ell, ia_nla_term=i)[1]
    C_sss[i].append(Cl_i)
    frac =  Cl_i/Cl_tot
    C_sssf[i].append(frac)

plt.figure()
plot_C_ss_tomo_limber(ell=ell, C_sss=abs(np.array(C_sss)),ylabel="$|C^{NLA}_{AB}|$",
    ia_nla_terms=ia_nla_terms,legend=[r'$C^{ij}_{\kappa\kappa}$',r'$C^{ij}_{\kappa I_E}$',r'$C^{ji}_{\kappa I_E}$',r'$C^{ij}_{I_E I_E}$','$C_{EE}$'])
plt.savefig(f'./02_IA_NLA_components.pdf')

plt.figure()
plot_C_ss_tomo_limber(ell=ell, C_sss=abs(np.array(C_sssf)),ylabel="$|C^{NLA}_{AB}/C_{EE}|$",
    ia_nla_terms=ia_nla_terms,legend=[r'$C^{ij}_{\kappa\kappa}/C^{ij}_{EE}$',r'$C^{ij}_{\kappa I_E}/C^{ij}_{EE}$',r'$C^{ji}_{\kappa I_E}/C^{ij}_{EE}$',r'$C^{ij}_{I_E I_E}/C^{ij}_{EE}$','$C_{EE}/C_{EE}$'])
plt.savefig(f'./02_IA_NLA_fracs.pdf')