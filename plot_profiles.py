# Plots initial profiles as used in the report

import swiftsimio as sw
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl
from multiprocessing import Pool, current_process
import time
from tqdm import tqdm
plt.rcParams["font.family"] = "Times New Roman"

R_earth = 6.371e6   # m
M_earth = 5.9724e24  # kg
G = 6.67408e-11  # m^3 kg^-1 s^-2


if __name__=='__main__':

    # For 1.5 uranus
    loc_uranus = '/data/cluster4/oo21461/Planets/1.5_uranus/relax_sim/output/snapshot_0120.hdf5'    # 1.5
    loc_imp1 = '/data/cluster4/oo21461/Planets/1.5_uranus/impactors/0.5_1/chuck_in_swift_1/output/1_M_0.5_for_1.5_uranus_0001.hdf5' # 0.5
    loc_imp2 = '/data/cluster4/oo21461/Planets/1.5_uranus/impactors/0.5_1/chuck_in_swift_2/output/2_M_1_for_1.5_uranus_0001.hdf5'  # 1


    try:
        # Open uranus
        data_uranus = sw.load(loc_uranus)
    except Exception as err:
        print(err, '.\nCould not open ', loc_uranus)
        sys.exit()
    try:
        # Open imp1
        data_imp1 = sw.load(loc_imp1)
    except Exception as err:
        print(err, '.\nCould not open ', loc_imp1)
        sys.exit()
    try:
        # Open imp2
        data_imp2 = sw.load(loc_imp2)
    except Exception as err:
        print(err, '.\nCould not open ', loc_imp2)
        sys.exit()

    data_uranus.gas.coordinates.convert_to_mks()
    data_uranus.gas.densities.convert_to_mks()
    data_uranus.gas.pressures.convert_to_mks()
    data_uranus.gas.masses.convert_to_mks()
    coords_uranus = data_uranus.gas.coordinates.to_ndarray()
    rho_uranus = data_uranus.gas.densities.to_ndarray()
    p_uranus = data_uranus.gas.pressures.to_ndarray()
    m_uranus = data_uranus.gas.masses.to_ndarray()

    data_imp1.gas.coordinates.convert_to_mks()
    data_imp1.gas.densities.convert_to_mks()
    data_imp1.gas.pressures.convert_to_mks()
    data_imp1.gas.masses.convert_to_mks()
    coords_imp1 = data_imp1.gas.coordinates.to_ndarray()
    rho_imp1 = data_imp1.gas.densities.to_ndarray()
    p_imp1 = data_imp1.gas.pressures.to_ndarray()
    m_imp1 = data_imp1.gas.masses.to_ndarray()

    data_imp2.gas.coordinates.convert_to_mks()
    data_imp2.gas.densities.convert_to_mks()
    data_imp2.gas.pressures.convert_to_mks()
    data_imp2.gas.masses.convert_to_mks()
    coords_imp2 = data_imp2.gas.coordinates.to_ndarray()
    rho_imp2 = data_imp2.gas.densities.to_ndarray()
    p_imp2 = data_imp2.gas.pressures.to_ndarray()
    m_imp2 = data_imp2.gas.masses.to_ndarray()


    pos_centerM = np.sum(coords_uranus * m_uranus[:,np.newaxis], axis=0) / np.sum(m_uranus)
    coords_uranus -= pos_centerM
    pos_centerM = np.sum(coords_imp1 * m_imp1[:,np.newaxis], axis=0) / np.sum(m_imp1)
    coords_imp1 -= pos_centerM
    pos_centerM = np.sum(coords_imp2 * m_imp2[:,np.newaxis], axis=0) / np.sum(m_imp2)
    coords_imp2 -= pos_centerM

    r_uranus  = np.hypot(np.hypot(coords_uranus[:,0],coords_uranus[:,1]),coords_uranus[:,2])
    r_imp1  = np.hypot(np.hypot(coords_imp1[:,0],coords_imp1[:,1]),coords_imp1[:,2])
    r_imp2  = np.hypot(np.hypot(coords_imp2[:,0],coords_imp2[:,1]),coords_imp2[:,2])


    # Plotting

    # Sort by r
    # sorted_indices_r = np.argsort(r_uranus)
    # r_sorted  = r_uranus[sorted_indices_r]
    # rho_sorted_r = rho_uranus[sorted_indices_r]

    max_x_extent = 4 # R_earth
    fig, ax = plt.subplots(2, 1, figsize=(4,7),sharex=True)
    ax[0].scatter(r_uranus/R_earth,rho_uranus,s=0.6,edgecolors='none',c='#3286C9',alpha=0.5,label='Uranus')
    # ax[0].plot(r_sorted/R_earth,rho_sorted_r,c='#3286C9',linewidth=0.5,linestyle='--')
    ax[0].scatter(r_imp1/R_earth,rho_imp1,s=0.6,edgecolors='none',c='#C93286',alpha=0.5,label='Imp1')
    ax[0].scatter(r_imp2/R_earth,rho_imp2,s=0.6,edgecolors='none',c='#86C932',alpha=0.5,label='Imp2')
    # ax.title.set_text(r'r vs $\rho$ for final snapshot of '+collision_str)
    ax[0].set_xlim(0, max_x_extent)
    #ax[0].set_ylim(0, None)
    #ax[0].set_xlabel(r"Radius $[R_\oplus]$")
    ax[0].set_ylabel(r"Density [kg $\text{m}^{-3}$]")
    ax[0].set_yscale('log')

    # Create custom legend handles using Line2D with '-' as the marker
    custom_legend = [
        Line2D([0], [0], color='#3286C9', linestyle='-', label="Uranus"),
        Line2D([0], [0], color='#C93286', linestyle='-', label="Imp1"),
        Line2D([0], [0], color='#86C932', linestyle='-', label="Imp2")
    ]
    lgnd = ax[0].legend(loc='upper right',handles=custom_legend)

    #lgnd.legendHandles[0]._legmarker.set_markersize(MY_SIZE)
    lgnd.legend_handles[0]._sizes = [30]
    lgnd.legend_handles[1]._sizes = [30]
    lgnd.legend_handles[2]._sizes = [30]
    # lgnd.legend_handles[0].set_alpha(1)
    # lgnd.legend_handles[1].set_alpha(1)
    # lgnd.legend_handles[2].set_alpha(1)
    




    ax[1].scatter(r_uranus/R_earth,p_uranus,s=0.6,edgecolors='none',c='#3286C9',alpha=0.5,label='Uranus')
    ax[1].scatter(r_imp1/R_earth,p_imp1,s=0.6,edgecolors='none',c='#C93286',alpha=0.5,label='Imp1')
    ax[1].scatter(r_imp2/R_earth,p_imp2,s=0.6,edgecolors='none',c='#86C932',alpha=0.5,label='Imp2')
    # ax.title.set_text(r'r vs $\rho$ for final snapshot of '+collision_str)
    #ax[1].set_xlim(0, max_x_extent)
    #ax[1].set_ylim(0, None)
    ax[1].set_xlabel(r"Radius [$R_\oplus$]")
    ax[1].set_ylabel(r"Pressure [Pa]")
    ax[1].set_yscale('log')

    # plt.legend()
    plt.subplots_adjust(wspace=1)
    fig.tight_layout()
    figname = f'/home/oo21461/Documents/tools/initial_profiles.png'
    fig.savefig(figname,dpi=500)
    print(figname + ' saved.\n')