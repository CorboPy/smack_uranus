import swiftsimio as sw
import sys
import h5py
import woma
import unyt
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from multiprocessing import Pool, current_process
import time
from tqdm import tqdm
plt.rcParams["font.family"] = "Times New Roman"
#plt.rcParams["mathtext.fontset"] = "Times New Roman"
plt.rcParams.update({'font.size': 15})


R_earth = 6.371e6   # m
M_earth = 5.9724e24  # kg

def plot_with_IDs(coords_u,coords_i1,coords_i2,IDs_u,IDs_i1,IDs_i2,R_u,R_i1,R_i2,folder):
    """ Plots (x,y) gas particles with material ID legend """

    x_u, y_u, z_u = coords_u.T
    x_u -= np.mean(x_u) - R_u
    y_u -= np.mean(y_u)
    z_u -= np.mean(z_u)
    IDs_array_u = np.array(IDs_u)
    atmosphere_u = np.where(IDs_array_u == 200)
    mantle_u = np.where(IDs_array_u == 900)
    core_u = np.where(IDs_array_u == 400)

    x_i1, y_i1, z_i1 = coords_i1.T
    x_i1 -= np.mean(x_i1) - 2*R_u - R_i1
    y_i1 -= np.mean(y_i1)
    z_i1 -= np.mean(z_i1)
    IDs_array_i1 = np.array(IDs_i1)
    atmosphere_i1 = np.where(IDs_array_i1 == 200)
    mantle_i1 = np.where(IDs_array_i1 == 900)
    core_i1 = np.where(IDs_array_i1 == 400)

    x_i2, y_i2, z_i2 = coords_i2.T
    x_i2 -= np.mean(x_i2) - 2*R_i1 - 2*R_u - R_i2 
    y_i2 -= np.mean(y_i2)
    z_i2 -= np.mean(z_i2)
    IDs_array_i2 = np.array(IDs_i2)
    atmosphere_i2 = np.where(IDs_array_i2 == 200)
    mantle_i2 = np.where(IDs_array_i2 == 900)
    core_i2 = np.where(IDs_array_i2 == 400)

    #plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(7.5,5))

    #ax.title.set_text('Snapshot '+num+': Time '+str(round(snap_time.round()/3600,3))+' h')
    ax.scatter(x_u[atmosphere_u],y_u[atmosphere_u],s=5,c='lightcyan',marker='.', label='HM80_HHe',zorder=1,edgecolors='none',alpha=0.3)
    ax.scatter(x_u[mantle_u],y_u[mantle_u],s=5,c='#3286C9',marker='.', label='AQUA',zorder=2,edgecolors='none',alpha=0.3)
    ax.scatter(x_u[core_u],y_u[core_u],s=5,c='#225B89',marker='.', label='ANEOS_Forsterite',zorder=3, edgecolors='none',alpha=0.3)

    ax.scatter(x_i1[atmosphere_i1],y_i1[atmosphere_i1],s=5,c='lightcyan',marker='.', label='HM80_HHe',zorder=1,edgecolors='none',alpha=0.3)
    ax.scatter(x_i1[mantle_i1],y_i1[mantle_i1],s=5,c='#86C932',marker='.', label='AQUA',zorder=2,edgecolors='none',alpha=0.3)
    ax.scatter(x_i1[core_i1],y_i1[core_i1],s=5,c='#5B8922',marker='.', label='ANEOS_Forsterite',zorder=3, edgecolors='none',alpha=0.3)

    ax.scatter(x_i2[atmosphere_i2],y_i2[atmosphere_i2],s=5,c='lightcyan',marker='.', label='HM80_HHe',zorder=1,edgecolors='none',alpha=0.3)
    ax.scatter(x_i2[mantle_i2],y_i2[mantle_i2],s=5,c='#C93286',marker='.', label='AQUA',zorder=2,edgecolors='none',alpha=0.3)
    ax.scatter(x_i2[core_i2],y_i2[core_i2],s=5,c='#89225B',marker='.', label='ANEOS_Forsterite',zorder=3, edgecolors='none',alpha=0.3)

    ax.hlines(R_u, R_u,2*R_u,color='white',zorder=100)
    #ax.text(R_u,R_u+0.1,f'{round(R_u,2)}'+r'R $\mathrm{_{\oplus}}$',color='white',fontsize=20)
    ax.hlines(R_i1, 2*R_u+R_i1,2*R_u+2*R_i1,color='white',zorder=100)
    #ax.text(2*R_u+R_i1,R_i1+0.1,f'{round(R_i1,2)}'+r'$R_{\oplus}$',color='white',fontsize=30)    
    ax.hlines(R_i2, 2*R_u+2*R_i1+R_i2,2*R_u+2*R_i1+2*R_i2,color='white',zorder=100)
    #ax.text(2*R_u+2*R_i1+R_i2,R_i2+0.1,f'{round(R_i2,2)}'+r'$R_{\oplus}$',color='white',fontsize=20)    

    #plt.legend(loc='lower right')

    # LIMITS
    scope = 2*R_u +  2*R_i1 + 2*R_i2 # width of the plot
    ax.set_xlim([0,(scope)])
    ax.set_ylim([0.6 -((7.5/11) * scope)/2, 0.6 + ((7.5/11) * scope)/2])
    #ax.set_aspect('equal')#, 'box')
    ax.set_xlabel(r'x [$R_{\oplus}$]')
    ax.set_ylabel(r'y [$R_{\oplus}$]')
    #ax.tick_params(axis='both', which='minor',length= 10, width = 2,labelsize=86)

    ax.set_facecolor('black')
    plt.subplots_adjust(hspace=1)
    #fig.tight_layout()
    name = f'{folder}/uranus_and_impactor.png'
    try:
        fig.savefig(name,dpi=1000)
    except Exception as err:
        print('ERROR: PNG not saved. Has a PNGs/ directory been created?')
    plt.close()

    print(f'Uranus radius: {R_u}, imp1 radius: {R_i1}, imp2 radius: {R_i2}')

def load_to_woma(snapshot):
    # Load
    data    = sw.load(snapshot)
    
    # Convert to metre, kilogram, second
    data.gas.coordinates.convert_to_mks()
    data.gas.velocities.convert_to_mks()
    data.gas.smoothing_lengths.convert_to_mks()
    data.gas.masses.convert_to_mks()
    data.gas.densities.convert_to_mks()
    data.gas.pressures.convert_to_mks()
    data.gas.internal_energies.convert_to_mks()
    data.gas.potentials.convert_to_mks()      
    box_mid = 0.5*data.metadata.boxsize[0].to(unyt.m)

    pos     = np.array(data.gas.coordinates - box_mid)
    vel     = np.array(data.gas.velocities)
    h       = np.array(data.gas.smoothing_lengths)
    m       = np.array(data.gas.masses)
    rho     = np.array(data.gas.densities)
    p       = np.array(data.gas.pressures)
    u       = np.array(data.gas.internal_energies)
    matid   = np.array(data.gas.material_ids)
    parids = data.gas.particle_ids.to_ndarray()
    pots = data.gas.potentials.to_ndarray()
    
    pos_centerM = np.sum(pos * m[:,np.newaxis], axis=0) / np.sum(m)
    vel_centerM = np.sum(vel * m[:,np.newaxis], axis=0) / np.sum(m)
    
    pos -= pos_centerM
    vel -= vel_centerM
    
    xy = np.hypot(pos[:,0],pos[:,1])
    r  = np.hypot(xy,pos[:,2])
    r  = np.sort(r)
    R  = np.mean(r[-100:])
    
    return pos, vel, h, m, rho, p, u, matid, parids, R, pots


def load_hdf5(file,verbose=True):
    f = h5py.File(file, "r")
    for key in f.keys():
        print("f.key: ",key,". Type: ",type(f[key])) #Names of the root level object names in HDF5 file - can be groups or datasets.
        #Get the HDF5 group; key needs to be a group name from above
        group = f[key]
        print("Group: ",group)

        #Checkout what keys are inside that group.
        for key in group.keys():
            print("Group key: ",key)

    # Get data
    positions =f['PartType0']['Coordinates'][()]
    ids = f['PartType0']['MaterialIDs'][()]
    internal_energies = f['PartType0']['InternalEnergies'][()]
    velocities = f['PartType0']['Velocities'][()]
    masses = f['PartType0']['Masses'][()]
    rhos = f['PartType0']['Densities'][()]
    

    # Convert to mks
    positions *= R_earth
    velocities *= R_earth
    masses *= M_earth

    if verbose:
        print(f'Sim pos zero point (R_earth) = ({np.mean(positions[:,0])/R_earth},{np.mean(positions[:,1])/R_earth},{np.mean(positions[:,2])/R_earth})')
        print(f'Sim vel zero point (m/s) = ({np.mean(velocities[:,0])},{np.mean(velocities[:,1])},{np.mean(velocities[:,2])})')
        print('Centering on (0,0,0) and centre of momentum... \n')

    pos_centerM = np.sum(positions * masses[:,np.newaxis], axis=0) / np.sum(masses)
    vel_centerM = np.sum(velocities * masses[:,np.newaxis], axis=0) / np.sum(masses)
    
    positions -= pos_centerM
    velocities -= vel_centerM

    xy = np.hypot(positions[:,0],positions[:,1])
    r  = np.hypot(xy,positions[:,2])
    r  = np.sort(r)
    R  = np.mean(r[-100:])  # defining the radius of the planet by taking the average of the 100 outer most particles??

    return positions, velocities, masses, ids, rhos, internal_energies, R

file_uranus = f"/data/cluster4/oo21461/Planets/1_uranus/relax_sim/output/snapshot_0120.hdf5"  
file_impactor_1 = f"/data/cluster4/oo21461/Planets/1.5_uranus/impactors/0.5_1/2_M_1_for_1.5_uranus.hdf5"
file_impactor_2 = f"/data/cluster4/oo21461/Planets/1.5_uranus/impactors/0.5_1/1_M_0.5_for_1.5_uranus.hdf5"

###########################
coords_u, v_u,h_u, m_u, rho_u, p_u, u_u, IDs_u,parids_u,R_u,pots_u =  load_to_woma(file_uranus)
#coords_u /= R_earth
print('Uranus has ',len(m_u),' particles, a mass of ',np.sum(m_u)/M_earth,', and a radius of ',R_u/R_earth,' R_earth')

# Uranus thermal energy calc
e_th = np.sum(m_u*u_u)
L = 4 * np.pi * R_u**2  * 70**4  * 5.67 * 10**(-8)
t_th = e_th/L
print(t_th/3600, 'h')
print(t_th/3.154e+7, 'yr')
print('{:.5e}'.format(t_th/3.154e+7))
print('URANUS ',np.mean(m_u)/M_earth)


###########################

coords_i1, v_i1, m_i1, IDs_i1,_,_,R_i1 =  load_hdf5(file_impactor_1)
coords_i1 /= R_earth
print('Impactor 1 has ',len(m_i1),' particles with radius of ',R_i1/R_earth,' R_earth')
print('IMP1', np.mean(m_i1)/M_earth)

###########################
coords_i2, v_i2, m_i2, IDs_i2,_,_,R_i2 =  load_hdf5(file_impactor_2)
coords_i2 /= R_earth
print('Impactor 2 has ',len(m_i2),' particles with radius of ',R_i2/R_earth,' R_earth')
print('IMP2',np.mean(m_i2)/M_earth)
sys.exit()
plot_with_IDs(coords_u,coords_i1,coords_i2,IDs_u,IDs_i1,IDs_i2,R_u/R_earth,R_i1/R_earth,R_i2/R_earth,'/home/oo21461/Documents/tools')