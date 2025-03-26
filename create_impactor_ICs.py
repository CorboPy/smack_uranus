# This script creates the impactors for a proto-Uranus with mass (mass_of_impactors_mearth)

import woma
import swiftsimio as sw
import h5py
import unyt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
import sys
import os
import numpy as np
import datetime as dt
import math
from itertools import combinations


R_earth = 6.371e6   # m
M_earth = 5.9724e24  # kg
G = 6.67408e-11  # m^3 kg^-1 s^-2


def load_to_woma(snapshot,verbose=True):
    # Load
    data    = sw.load(snapshot)
    meta = data.metadata # Gets metadata of the file (not used in plotting anything here but might be useful in future for organising simulations)
    snap_time = float(meta.t)
    
    # Convert to metre, kilogram, second
    data.gas.coordinates.convert_to_mks()
    data.gas.velocities.convert_to_mks()
    data.gas.smoothing_lengths.convert_to_mks()
    data.gas.masses.convert_to_mks()
    data.gas.densities.convert_to_mks()
    data.gas.pressures.convert_to_mks()
    data.gas.internal_energies.convert_to_mks()
    box_mid = 0.5*data.metadata.boxsize[0].to(unyt.m)

    pos     = np.array(data.gas.coordinates - box_mid)
    vel     = np.array(data.gas.velocities)
    h       = np.array(data.gas.smoothing_lengths)
    m       = np.array(data.gas.masses)
    rho     = np.array(data.gas.densities)
    p       = np.array(data.gas.pressures)
    u       = np.array(data.gas.internal_energies)
    matid   = np.array(data.gas.material_ids)

    if verbose:
        print(f'Sim pos zero point (R_earth) = ({np.mean(np.array(data.gas.coordinates[:,0]))/R_earth},{np.mean(np.array(data.gas.coordinates[:,1]))/R_earth},{np.mean(np.array(data.gas.coordinates[:,2]))/R_earth})')
        print(f'Sim vel zero point (m/s) = ({np.mean(np.array(data.gas.velocities[:,0]))},{np.mean(np.array(data.gas.velocities[:,1]))},{np.mean(np.array(data.gas.velocities[:,2]))})')
        print('Centering on (0,0,0) and centre of momentum... \n')

    pos_centerM = np.sum(pos * m[:,np.newaxis], axis=0) / np.sum(m)
    vel_centerM = np.sum(vel * m[:,np.newaxis], axis=0) / np.sum(m)
    
    pos -= pos_centerM
    vel -= vel_centerM
    
    xy = np.hypot(pos[:,0],pos[:,1])
    r  = np.hypot(xy,pos[:,2])
    r  = np.sort(r)
    R  = np.mean(r[-100:])  # defining the radius of the planet by taking the average of the 100 outer most particles??
    
    return pos, vel, h, m, rho, p, u, matid, R, snap_time

def plot_spinning_profiles(sp,path): 
    plt.style.use('default')
   
    fig, ax = plt.subplots(1, 2, figsize=(12,6))
        
    ax[0].plot(sp.planet.A1_r / R_earth, sp.planet.A1_rho, label="original spherical")
    ax[0].plot(sp.A1_R / R_earth, sp.A1_rho, label="equatorial")
    ax[0].plot(sp.A1_Z / R_earth, sp.A1_rho, label="polar")
    ax[0].set_xlabel(r"Radius, $r$ $[R_\oplus]$")
    ax[0].set_ylabel(r"Density, $\rho$ [kg m$^{-3}$]")
    ax[0].set_yscale("log")
    ax[0].set_xlim(0, 1.1 * sp.R_eq / R_earth)
    ax[0].legend()
    
    for i, e in enumerate([
        Ellipse(
            xy=[0, 0],
            width=2 * sp.A1_R[i] / R_earth, 
            height=2 * sp.A1_Z[i] / R_earth,
            zorder=-i,
        )
        for i in range(len(sp.A1_R))
    ]):
        ax[1].add_artist(e)
        e.set_clip_box(ax[1].bbox)
        e.set_facecolor(plt.get_cmap("viridis")(
            (sp.A1_rho[i] - sp.rho_s) / (sp.rho_0 - sp.rho_s)
        ))
    
    ax[1].set_xlabel(r"Equatorial Radius, $r_{xy}$ $[R_\oplus]$")
    ax[1].set_ylabel(r"Polar Radius, $z$ $[R_\oplus]$")    
    ax[1].set_xlim(0, 1.1 * sp.R_eq / R_earth)
    ax[1].set_ylim(0, 1.1 * sp.R_po / R_earth)
    ax[1].set_aspect("equal")
    ax[1].set_title(r"Density [kg m$^{-3}$]")
    
    plt.tight_layout()
    figname = path+str(sp.planet.name)+'_spin.png'
    fig.savefig(figname)


def plot_spherical_profiles(planet,path):    
    plt.style.use('default')

    fig, ax = plt.subplots(2, 2, figsize=(8,8))
    
    ax[0, 0].plot(planet.A1_r / R_earth, planet.A1_rho)
    ax[0, 0].set_xlabel(r"Radius, $r$ $[R_\oplus]$")
    ax[0, 0].set_ylabel(r"Density, $\rho$ [kg m$^{-3}$]")
    ax[0, 0].set_yscale("log")
    ax[0, 0].set_xlim(0, None)
    
    ax[1, 0].plot(planet.A1_r / R_earth, planet.A1_m_enc / M_earth)
    ax[1, 0].set_xlabel(r"Radius, $r$ $[R_\oplus]$")
    ax[1, 0].set_ylabel(r"Enclosed Mass, $M_{<r}$ $[M_\oplus]$")
    ax[1, 0].set_xlim(0, None)
    ax[1, 0].set_ylim(0, None)
    
    ax[0, 1].plot(planet.A1_r / R_earth, planet.A1_P)
    ax[0, 1].set_xlabel(r"Radius, $r$ $[R_\oplus]$")
    ax[0, 1].set_ylabel(r"Pressure, $P$ [Pa]")
    ax[0, 1].set_yscale("log")
    ax[0, 1].set_xlim(0, None)
    
    ax[1, 1].plot(planet.A1_r / R_earth, planet.A1_T)
    ax[1, 1].set_xlabel(r"Radius, $r$ $[R_\oplus]$")
    ax[1, 1].set_ylabel(r"Temperature, $T$ [K]")
    ax[1, 1].set_xlim(0, None)
    ax[1, 1].set_ylim(0, None)
    
    plt.tight_layout()
    figname = path+str(planet.name)+'_profile.png'
    fig.savefig(figname)

def plot_vels(pos,vel,planet,path):
    plt.style.use('default')
    fig = plt.figure(figsize=(8,4))
    gs = fig.add_gridspec(1,2)

    pos = pos/R_earth

    r = np.sqrt(pos[:,0]**2 + pos[:,1]**2)  # sqrt ( x^2 + y^2 )
    v_norm = np.sqrt(vel[:,0]**2 + vel[:,1]**2)  # sqrt ( x^2 + y^2 )
    ax1 = fig.add_subplot(gs[0,0])
    ax1.scatter(r,v_norm,marker='.',s=1)
    ax1.set_xlabel(r"Radius, $r$ $[R_{Earth}]$")
    ax1.set_ylabel(r"Velocity, $v$ [m/s]")
    ax1.set_xlim(0, None)
    ax1.set_ylim(0, None)
    ax1.set_title('Velocity vs radius from centre of planet')

    #v_xy = vel[:,0:2]
    norm = np.power(np.add(np.power(vel[:,0:1],2), np.power(vel[:,1:2],2)),0.5)
    ax2 = fig.add_subplot(gs[0,1])
    ax2.quiver(pos[:,0:1],pos[:,1:2], vel[:,0:1]/norm, vel[:,1:2]/norm)
    ax2.set_xlabel(r"x, $x$ $[R_{Earth}]$")
    ax2.set_ylabel(r"y, $y$ $[R_{Earth}]$")
    ax2.set_xlim( - (planet.R/R_earth+1) , (planet.R/R_earth+1) )
    ax2.set_ylim( - (planet.R/R_earth+1) , (planet.R/R_earth+1) )
    ax2.set_title('Velocity Vectors in x, y plane')

    plt.tight_layout()
    figname = path+str(planet.name)+'_vels.png'
    fig.savefig(figname,dpi=400)


def plot_ICs(planet,particleplanet,pos,ids,boxsize,path):
    plt.style.use('default')
    vx, vy, vz = particleplanet.A2_vel.T
    x, y, z, = pos.T

    now = dt.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    printout = f"""Min, max values (file units):
pos = {round(np.min(x),3), round(np.max(x),3)},    {round(np.min(y),3), round(np.max(y),3)},    {round(np.min(z),3), round(np.max(z),3)}
vel = {round(np.min(vx),3), round(np.max(vx),3)},    {round(np.min(vy),3), round(np.max(vy),3)},    {round(np.min(vz),3), round(np.max(vz),3)}
m = {np.min(particleplanet.A1_m)},  {np.max(particleplanet.A1_m)}
rho = {np.min(particleplanet.A1_rho)},  {np.max(particleplanet.A1_rho)}
P = {np.min(particleplanet.A1_P)},  {np.max(particleplanet.A1_P)}
u = {np.min(particleplanet.A1_u)},  {np.max(particleplanet.A1_u)}
h = {np.min(particleplanet.A1_h)},  {np.max(particleplanet.A1_h)}
s = {np.min(particleplanet.A1_s)},  {np.max(particleplanet.A1_s)}
N = {particleplanet.N_particles}
boxsize = {boxsize/R_earth}
mat_ids = {np.unique(particleplanet.A1_mat_id)}
datetime = {now}
"""

    origin_x = round(np.mean(x),1)
    origin_y = round(np.mean(y),1)
    origin_z = round(np.mean(z),1)

    IDs_array = np.array(ids)
    atmosphere = np.where(IDs_array == 200)
    mantle = np.where(IDs_array == 900)
    core = np.where(IDs_array == 400)

    plt.style.use('dark_background')
    fig, ax = plt.subplots()
    ax.title.set_text('Planet ICs: '+planet.name+'.hdf5')
    ax.scatter(x[atmosphere],y[atmosphere],s=1,c='lightcyan',marker='.', label='HM80_HHe',zorder=1,edgecolors='none')
    ax.scatter(x[mantle],y[mantle],s=1,c='dodgerblue',marker='.', label='AQUA',zorder=2,edgecolors='none')
    ax.scatter(x[core],y[core],s=1,c='slategray',marker='.', label='ANEOS_forsterite',zorder=3,edgecolors='none')
    plt.legend(loc='lower right')

    # Printout
    plt.subplots_adjust(right=1.5)
    plt.figtext(0.8, 0.5, printout, wrap=True, horizontalalignment='left', fontsize=5)

    # LIMITS
    ax.set_xlim([origin_x-5, origin_x+5])
    ax.set_ylim([origin_y-5, origin_y+5])
    ax.set_aspect('equal', 'box')
    ax.set_xlabel(r'x ($R_{\bigoplus}$)')
    ax.set_ylabel(r'y ($R_{\bigoplus}$)')

    fig.tight_layout()
    figname = path+str(planet.name)+'_plot.png'
    fig.savefig(figname,dpi=600)

def find_combinations(list1,list2,target):
    return([(x, y) for x in list1 for y in list2 if x + y == target])

def relax_test(M_u,R_u,vels,core,mantle,atmosphere,verbose=True):
    # Relaxation test via v_rms / v_esc
    v_e = np.sqrt( (2 * G * M_u) / (R_u))   # Escape velocity at surface of the proto Uranus
    mag_vels = np.linalg.norm(vels, axis=1)
    sum_square = 0.0
    sum_square_atmos = 0.0
    sum_square_mantle = 0.0
    sum_square_core = 0.0
    for vel in mag_vels:
        sum_square += (vel**2)
    for vel in mag_vels[atmosphere]:
        sum_square_atmos += (vel**2)
    for vel in mag_vels[mantle]:
        sum_square_mantle += (vel**2)
    for vel in mag_vels[core]:
        sum_square_core += (vel**2)
    v_rms = math.sqrt(sum_square / N_u) 
    v_rms_atmos = math.sqrt(sum_square_atmos / N_u) 
    v_rms_mantle = math.sqrt(sum_square_mantle / N_u) 
    v_rms_core = math.sqrt(sum_square_core / N_u)

    max = np.max(mag_vels)
    max_core = np.max(mag_vels[core])
    max_mantle = np.max(mag_vels[mantle])
    max_atmos = np.max(mag_vels[atmosphere])

    message = f'''Relaxation test:
    All: v_rms = {v_rms}, v_e = {v_e}, v_rms/v_e = {(v_rms/v_e)}. max_v = {max}
    Atmos: v_rms = {v_rms_atmos}, v_e = {v_e}, v_rms/v_e = {(v_rms_atmos/v_e)}. max_v = {max_atmos}
    Mantle: v_rms = {v_rms_mantle}, v_e = {v_e}, v_rms/v_e = {(v_rms_mantle/v_e)}. max_v = {max_mantle}
    Core: v_rms = {v_rms_core}, v_e = {v_e}, v_rms/v_e = {(v_rms_core/v_e)}. max_v = {max_core}'''
    if verbose:
        print(message)
    assert v_rms/v_e < 0.01, 'Relaxation test not passed!'
    assert v_rms_atmos/v_e < 0.01, 'Relaxation test not passed!'
    assert v_rms_mantle/v_e < 0.01, 'Relaxation test not passed!'
    assert v_rms_core/v_e < 0.01, 'Relaxation test not passed!'
    if verbose:
        print('PASSED\n')
    return(message)

if __name__=='__main__':

    # Load in the Uranus from output of relaxation sim
    mass_of_impactors_mearth = 1
    spin_period = 10 # hours
    loc_uranus=f'/data/cluster4/oo21461/Planets/{mass_of_impactors_mearth}_uranus/relax_sim/output/snapshot_0120.hdf5'
    possible_impactor_masses = [0.5,0.75,1]
    combinations = find_combinations(possible_impactor_masses, possible_impactor_masses, mass_of_impactors_mearth)
    mass_of_impactors = mass_of_impactors_mearth * M_earth
    N_s = int(1e6) # number of particles in whole system
    coords, vels, smoothinglengths , masses, rhos, pressures , energies, ids, R_u, relax_time = load_to_woma(loc_uranus)
    M_u = np.sum(masses)
    N_u = len(masses)

    print('Uranus selected: ')
    print(f'M = {M_u/M_earth} M_earth for {mass_of_impactors_mearth} M_earth impactors.')
    print(f'N = {N_u:.5e}')
    print(f'Relaxed for {relax_time} seconds ({relax_time/3600:.4f} hours)')
    print(f'Pos zero point = ({np.mean(coords[:,0])},{np.mean(coords[:,1])},{np.mean(coords[:,2])})')
    print(f'Vel zero point = ({np.mean(vels[:,0])},{np.mean(vels[:,1])},{np.mean(vels[:,2])})')
    print(f'IDs: {np.unique(ids)}')
    print('Possible impactor mass combinations (M_earth): ',combinations) 

    atmosphere = np.where(ids == 200)
    mantle = np.where(ids == 900)
    core = np.where(ids == 400)

    message = relax_test(M_u,R_u,vels,core,mantle,atmosphere)

    # Compare with the first snapshot
    _, vels, _ , masses_, _, _ , _, ids_, R_u_, relax_time_ = load_to_woma(loc_uranus.replace('120','000'),verbose=False)
    message_ = relax_test(np.sum(masses_),R_u_,vels,core,mantle,atmosphere,verbose=False)
    message_ = '\n\nComparison with ' + loc_uranus.replace('120','000') +':\n'+message_

    # Mass per particle
    M_uranus = np.sum(masses) * M_earth     # the masses are in units of M_earth
    N_uranus = len(masses)
    mass_per_particle = M_u/N_u
    print('M_uranus = ',M_u,'kg, N_uranus = ', N_u)
    print('Mass per particle = ',mass_per_particle,'kg, or ',mass_per_particle/M_earth,'M_earth')

    # Getting rock_ice_mass_ratio
    rock_mass = np.sum(masses[core])
    ice_mass = np.sum(masses[mantle])
    rock_ice_mass_ratio = rock_mass/ ice_mass
    print('Rock to ice mass ratio for selected Uranus =',rock_ice_mass_ratio)


    # Saving a .txt info on relax stats
    relax_info_str = loc_uranus+'\n\n'+f'Relaxed for {relax_time} seconds ({relax_time/3600} hours)'+'\n\n'+f'rock/ice mass ratio = {rock_ice_mass_ratio}, mass_per_particle = {mass_per_particle} kg'+'\n'+f'Estimated radius (averaging over outermost 100 particles) = {R_u}'+'\n\n'+message+message_
    txt_file = open(loc_uranus.split('relax_sim')[0]+'relax_info.txt', "w")
    txt_file.write(relax_info_str)
    txt_file.close()

    # Make impactors for each mass combination 
    # Impactors do not depend on phi!
    radii_to_try = {1:[1.092*R_earth,1.415*R_earth],1.5:[1.45*R_earth,1.57*R_earth],2:[1.6*R_earth,1.7*R_earth],
                    0.5:[0.992*R_earth,1.165*R_earth], 0.75:[1.092*R_earth,1.315*R_earth]}   
    for combination in combinations:
        print('\n\nMaking '+str(combination)+' impactors...\n')
        M_i1_mearth, M_i2_mearth = combination 
        print(M_i1_mearth,M_i2_mearth)
        M_i1 = M_i1_mearth* M_earth
        M_i2 = M_i2_mearth* M_earth
        N_i1 = M_i1/mass_per_particle
        N_i2 = M_i2/mass_per_particle

        # Make folders to save impactors to
        path = loc_uranus.split('relax_sim')[0]+'impactors/'+f'{M_i1_mearth}_{M_i2_mearth}/'
        if not os.path.exists(path):
            os.makedirs(path)

        mass_rock_1 = rock_ice_mass_ratio*M_i1
        mass_ice_1 = (1-rock_ice_mass_ratio)*M_i1

        mass_rock_2 = rock_ice_mass_ratio*M_i2
        mass_ice_2 = (1-rock_ice_mass_ratio)*M_i2

        # Create impactor 1
        i_1 = woma.Planet(
            name            = f'1_M_{M_i1_mearth}_for_{mass_of_impactors_mearth}_uranus',
            A1_mat_layer    = ["ANEOS_forsterite", "AQUA"],
            A1_T_rho_type   = ["adiabatic", "power=0.9"],
            P_s             = 1e9,
            T_s             = 450,
            A1_M_layer      = [mass_rock_1,mass_ice_1],
        )

        # Create impactor 2
        i_2 = woma.Planet(
            name            = f'2_M_{M_i2_mearth}_for_{mass_of_impactors_mearth}_uranus',
            A1_mat_layer    = ["ANEOS_forsterite", "AQUA"],
            A1_T_rho_type   = ["adiabatic", "power=0.9"],
            P_s             = 1e9,
            T_s             = 450,
            A1_M_layer      = [mass_rock_2,mass_ice_2],
        )

        # Generate the profiles - NEEDS DOING FOR 0.5, 0.75, 1 IMPACTORS
        i_1.gen_prof_L2_find_R_R1_given_M1_M2(R_min=radii_to_try[M_i1_mearth][0],R_max=radii_to_try[M_i1_mearth][1])
        i_2.gen_prof_L2_find_R_R1_given_M1_M2(R_min=radii_to_try[M_i2_mearth][0],R_max=radii_to_try[M_i2_mearth][1])

        print('\n\nSpinning up impactors to '+str(spin_period)+'h... \n')
        # Spin planet
        spin_i_1 = woma.SpinPlanet(
            planet = i_1,
            period = spin_period, # in hours
        )
        spin_i_2 = woma.SpinPlanet(
            planet = i_2,
            period = spin_period, # in hours
        )
        plot_spinning_profiles(spin_i_1,path)
        plot_spinning_profiles(spin_i_2,path)
        plot_spherical_profiles(spin_i_1.planet,path)
        plot_spherical_profiles(spin_i_2.planet,path)

        # SPH particle planet
        boxsize_i = 20*R_earth # for the impactor hdf5s
        particle_i_1 = woma.ParticlePlanet(spin_i_1, N_i1, N_ngb=48, verbosity=0) # N_ngb is number of neighbours (can use this for a quick and crude estimate of each particle's smoothing length from its density)
        particle_i_1.A1_mat_id[particle_i_1.A1_mat_id==304] = 900   # Cus swift wants 900
        particle_i_2 = woma.ParticlePlanet(spin_i_2, N_i2, N_ngb=48, verbosity=0) # N_ngb is number of neighbours (can use this for a quick and crude estimate of each particle's smoothing length from its density)
        particle_i_2.A1_mat_id[particle_i_2.A1_mat_id==304] = 900   # Cus swift wants 900

        # Plot spin vels
        plot_vels(np.array(particle_i_1.A2_pos)[::100,:],np.array(particle_i_1.A2_vel)[::100,:],spin_i_1.planet,path)
        plot_vels(np.array(particle_i_2.A2_pos)[::100,:],np.array(particle_i_2.A2_vel)[::100,:],spin_i_2.planet,path)

        # Save to .hdf5
        particle_i_1.save(path+str(spin_i_1.planet.name)+".hdf5",boxsize=boxsize_i,file_to_SI=woma.Conversions(M_earth, R_earth, 1),do_entropies=True) # read particle.save() docstring
        print(f'N = {particle_i_1.N_particles}')
        print(f'Mass per particle = {spin_i_1.planet.M / particle_i_1.N_particles} kg, or {(spin_i_1.planet.M / particle_i_1.N_particles) / M_earth} M_earth')
        particle_i_2.save(path+str(spin_i_2.planet.name)+".hdf5",boxsize=boxsize_i,file_to_SI=woma.Conversions(M_earth, R_earth, 1),do_entropies=True) # read particle.save() docstring
        print(f'N = {particle_i_2.N_particles}')
        print(f'Mass per particle = {spin_i_2.planet.M / particle_i_2.N_particles} kg, or {(spin_i_2.planet.M / particle_i_2.N_particles) / M_earth} M_earth')


        # Plot impactors
        plot_ICs(spin_i_1.planet, particle_i_1, particle_i_1.A2_pos, particle_i_1.A1_mat_id, boxsize_i,path)
        plot_ICs(spin_i_2.planet, particle_i_2, particle_i_2.A2_pos, particle_i_2.A1_mat_id, boxsize_i,path)



# Then, spin-up Uranus at phi (use Louis' method)
