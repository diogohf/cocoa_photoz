"""
Diogo H. F. de Souza - Fri Mar 28 2025
Get the mean of redshift distribution for 9 millions of Roman scenarios
"""

## LIBS ##
# import fitsio as fio
import numpy as np
from numpy import linalg as la
from numpy.lib.recfunctions import stack_arrays
import matplotlib
matplotlib.use ('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import pylab
from scipy.special import softmax
import h5py
import os

###################################
###### REDSHIFT DISTRIBUTION ######
###################################

path = 'path / to / file.h5' # open simulation: this file contains z and nzs

nzs = h5py.File(path, 'r')
z = nzs["zbinsc"][:]
# print(nzs.keys())

def nz_at(tomo,sim,zidx):
    """
    Return n(z) at tomographic bin `tomo`, 
    for a given simulation with index `sim`
    and at specific redshift with index `zidx`.
    """
    return nzs[f"bin{tomo}"][sim][zidx]

def mean_nz_at(tomo,nsim,zidx):
    """
    Return the mean of n(z) for n simulations `nsim`
    for a given tomographic bin with index `tomo`
    and a given redshift with index `zidx`.
    """
    x = [nz_at(tomo,i,zidx) for i in np.arange(nsim)]
    return np.mean(x)

def mean_nz(n):
    """
    Return the mean of n(z) for n simulations `n`
    through all 9 tomographic bins and
    through all 46 redshift bins.
    """
    n0={'bin0':[], 'bin1':[], 'bin2':[],
        'bin3':[], 'bin4':[], 'bin5':[],
        'bin6':[], 'bin7':[], 'bin8':[]}
    for key,tomo in zip(n0.keys(),np.arange(9)):
        print(key,tomo)
        for zidx in np.arange(len(z)):
            n0[key].append(mean_nz_at(tomo=tomo,nsim=n,zidx=zidx))
    return n0

def validate_function_mean_nz():
    """
    When the number of simulations is 1, the mean_nz 
    MUST equals n(z) for the SAME simulation index.
    This test works just of the fist simulation.
    To Do: generalize for arbitrary simulation.
    """
    plt.figure()
    for i in np.arange(9):
        plt.plot(z,nzs[f"bin{i}"][0][:],c="C0",label= "nzs" if i==0 else None)
        plt.plot(z,mean_nz(1)[f'bin{i}'],c='k',ls='--',label= "mean_nz" if i==0 else None)
    plt.xlabel("z",fontsize=14)
    plt.ylabel("n(z)",fontsize=14)
    plt.legend(loc="best")
    plt.title("Simulation 0")
    plt.tight_layout()
    plt.savefig("./test.pdf")    

# validate_function_mean_nz()

########################################
###### MEAN REDSHIFT DISTRIBUTION ######
########################################
def mean_plot(Nsims=1):
    """
    Return a plot of the mean 
    tomographic n(z) for N simulations.
    """
    Ntomos = 9
    plt.figure()
    for i in np.arange(Ntomos):
        plt.plot(z,mean_nz(Nsims)[f'bin{i}'])
        print(i)
    plt.xlabel(r"$z$",fontsize=14)
    plt.ylabel(r"$\bar{n}(z)$",fontsize=14)
    plt.title(f"Mean n(z) over {Nsims} simulations")
    plt.tight_layout()
    plt.savefig("./test.pdf")

# mean_plot(Nsims=1000)


########################################
########################################

num_bins = 9
nsims = 1000000

n0 = mean_nz(nsims) ## TIME CONSUMING

data = np.column_stack(([n0[f"bin{i}"] for i in np.arange(9)]))

np.savetxt("05_mean_roman_nz.txt",data)

zbins = np.array(nzs['zbinsc'])
nzs = np.stack([nzs[f'bin{i}'][0:nsims] for i in range(9)], axis=1)

fig, ax = plt.subplots(num_bins,1,figsize=(10,20), sharex=True)

for i in range(num_bins):
    print(i)
    parts2 = ax[i].violinplot(
            nzs[:,i],positions=zbins, widths=0.05, showmeans=False, showmedians=False, showextrema=False)
    ax[i].plot(zbins,n0[f"bin{i}"],c="k")
ax[0].set_title(f"Violin plots for {nsims} simulations")
plt.savefig('./04_mean_roman_nz.pdf')