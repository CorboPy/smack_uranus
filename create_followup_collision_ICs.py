# This script creates the initial conditions for the second impact 

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
from tqdm import tqdm
from scipy.optimize import root_scalar
from analysis import separate_particles
import numba



R_earth = 6.371e6   # m
M_earth = 5.9724e24  # kg
G = 6.67408e-11  # m^3 kg^-1 s^-2

# For swift outputs
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

def find_combinations(list1,list2,target):
    return([(x, y) for x in list1 for y in list2 if x + y == target])

def starting_distance(pos_tar, pos_imp,m_tar,m_imp, R_tar, R_imp):
    # This function will assess the gravitational force on the closest impactor particle to the uranus
    # and return the distance when the grav force of all impactor particles is ten times greater than the grav force of all uranus particles  

    x_i, y_i, z_i = pos_imp.T
    x_t, y_t, z_t = pos_tar.T
    edge_impactor_idx = np.argmin(x_i)
    edge_impactor_particle = pos_imp[edge_impactor_idx]
    x_particle, y_particle, z_particle = x_i[edge_impactor_idx], y_i[edge_impactor_idx], z_i[edge_impactor_idx]
    m_particle = m_imp[edge_impactor_idx]
    print(f'Edge of impactor identified as particle {edge_impactor_idx}')
    print(f'Coordinates: (',x_particle/R_earth, y_particle/R_earth, z_particle/R_earth,') R_earth')
    print(f'Mass: ', m_particle)

    center_tar = np.mean(pos_tar, axis=0)
    center_imp = np.mean(pos_imp, axis=0)

    # Mass and distance contributions
    M_tar_total = np.sum(m_tar)
    M_imp_total = np.sum(m_imp)

    def force_ratio_condition(d):
        # Distance from edge particle to the center of the target
        r_tar = np.linalg.norm(edge_impactor_particle - (center_tar - np.array([d, 0, 0])))
        # Distance from edge particle to the center of the impactor
        r_imp = np.linalg.norm(edge_impactor_particle - center_imp)
        
        # Forces
        F_target =  M_tar_total / r_tar**2  # m_particle and G gets cancelled 
        F_impactor = M_imp_total / r_imp**2

        print_str = '\nd = '+str(d/R_earth)+'\nForce from impactor: '+str(F_impactor)+'\nForce from uranus: '+ str(F_target)+'\nRoot: '+str((F_impactor / F_target ) - 10)
        print(print_str)
        # Ratio of forces
        return (F_impactor / F_target ) - 10



    result = root_scalar(force_ratio_condition, bracket=[R_tar, 50*R_earth], method='brentq')
    if result.converged:
        separation_distance = result.root
        print('\nFinal distance converged: ',separation_distance/R_earth, ' R_earth')
        return separation_distance
    else:
        raise ValueError("\nFailed to find a valid separation distance")

# NEEDS TO BE EDITED IF WANT A PERIOD ESTIMATION
def period_at_equator(pos,vel,R,angle):
    theta = np.radians(angle)  # Convert angle to radians


    # Rotation matrix about x-axis
    R_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

    # Transform positions and velocities
    pos_inclined = pos @ R_matrix.T  # Rotate positions
    vel_inclined = vel @ R_matrix.T  # Rotate velocities

    r = np.sqrt(pos_inclined[:,0]**2 + pos_inclined[:,1]**2)  # sqrt ( x^2 + y^2 )
    v_norm = np.sqrt(vel_inclined[:,0]**2 + vel_inclined[:,1]**2)  # sqrt ( x^2 + y^2 )

    # Get indices of the 100 largest values in r
    largest_indices = np.argsort(r)[-100:]  # Indices of the largest 100 r values

    # Extract the corresponding r and v values
    r_100 = r[largest_indices]
    v_100 = v_norm[largest_indices]

    # print(R)
    # print(np.mean(r_100))
    # print(np.mean(v_100))
    return((((2*np.pi*R)/np.mean(v_100)) )/ 3600)

def angle_to_z(vec,return_norm=False):
    """Calculates the angle of a vector to the +Z axis"""  
    vec_mag = np.sqrt(np.sum(vec**2))
    vec_norm = vec / vec_mag
    angle = (360 / (2 * np.pi)) * np.arccos(np.dot([0, 0, 1], vec) / vec_mag) # Angle of remnant rotational angular momentum to +Z  

    if return_norm:
        return(angle,vec_norm)
    return(angle)


# Very similar to process_snap_ang_mom in analysis.py
def ang_mom(pos,vels,masses,matids,R):
   
    # Center of mass correction
    pos_centerM = np.sum(pos * masses[:, np.newaxis], axis=0) / np.sum(masses)
    vel_centerM = np.sum(vels * masses[:, np.newaxis], axis=0) / np.sum(masses)
    pos -= pos_centerM
    vels -= vel_centerM


    # Angular momentum calculation
    L = np.sum(masses[:, np.newaxis] * np.cross(pos, vels), axis=0)
    angle, L_norm = angle_to_z(L,return_norm=True)
    # L_mag = np.sqrt(np.sum(L**2))
    # L_norm = L / L_mag
    # angle = (360 / (2 * np.pi)) * np.arccos(np.dot([0, 0, 1], L) / L_mag) # Angle of remnant rotational angular momentum to +Z  
    return L, L_norm, angle

def plot_ang_moms(l_init,l_1,l,l_2,l_uranus_today,path,scenario_str):
    """This function plots the change in Uranus' rotaional angular momentum after the first impact and what it still needs to get to Uranus today""" 
    fig,ax = plt.subplots()

    ax.arrow(0,0, l_init[2], l_init[1],color='g',label=r'$S_{\text{init}}$',head_width=0.05e36,length_includes_head=True)
    ax.arrow(l_init[2],l_init[1], l_1[2], l_1[1],color='purple',label=r'$\Delta S_1$',head_width=0.05e36,length_includes_head=True,linestyle='--') # what we need for second
    ax.arrow(0,0, l[2], l[1],color='b',label=r'$S_{\text{rem}}$',head_width=0.05e36,length_includes_head=True) # after first impact
    ax.arrow(l[2],l[1], l_2[2], l_2[1],color='orange',label=r'$\Delta S_2$ needed',head_width=0.05e36,length_includes_head=True,linestyle='--') # what we need for second
    ax.arrow(0,0, l_uranus_today[2], l_uranus_today[1],color='r',label=r'$S_{\text{today}}$',head_width=0.05e36,length_includes_head=True)
    plt.legend()

    lim = np.abs(l_uranus_today[1]) +0.1*np.abs(l_uranus_today[1])

    ax.set_xlabel('z')
    ax.set_ylabel('y')
    ax.set_xlim([lim, -lim])    # reversed horizontal axis
    ax.set_ylim([-lim, lim])
    ax.set_aspect('equal', 'box')

    ax.text(-l_uranus_today[1],0.7e36,f'init: {l_init}\nl1: {l_1}\nrem: {l}\nl2: {l_2}\ntoday: {l_uranus_today}',fontsize='x-small')
    fig.suptitle(f'Scenario: {scenario_str}\n'+r"$S_{\text{uranus}}$ after first impact ($z$ axis parallel to $L_{\text{uranus}}$)")

    fig.tight_layout()
    figname = path + 'ang_moms_after_first_impact.png'
    plt.savefig(figname)
    print(figname + ' saved.\n')

def radius_by_densities(sim_folder, pos, rho, threshold_rho, collision_str, plot=True):    # pos, rhos, etc of the BOUND material
    
    xy = np.hypot(pos[:,0],pos[:,1])
    r  = np.hypot(xy,pos[:,2])

    # Sort by r
    sorted_indices_r = np.argsort(r)
    r_sorted  = r[sorted_indices_r]
    rho_sorted_r = rho[sorted_indices_r]

    # Sort by rho
    sorted_indices_rho = np.argsort(rho)
    r_sorted_rho  = r[sorted_indices_rho]
    rho_sorted = rho[sorted_indices_rho]

    idx = np.searchsorted(rho_sorted, threshold_rho, side='left')
    rho_of_the_particle = rho_sorted[idx]
    R = r_sorted_rho[idx]            # Use this as the radius
    print(f'Using radius {R/R_earth} (R_earth)\n')

    if plot:
        max_x_extent = 6 # R_earth
        fig, ax = plt.subplots()
        ax.scatter(r_sorted/R_earth,rho_sorted_r,s=0.5,edgecolors='none',c='black',alpha=0.4)
        ax.title.set_text(r'r vs $\rho$ for final snapshot of '+collision_str)
        ax.set_xlim(0, max_x_extent)
        #ax.set_ylim(0, None)
        ax.set_xlabel(r"Radius, $r$ $[R_\oplus]$")
        ax.set_ylabel(r"Density, $\rho$ [kg $\text{m}^{-3}$]")
        ax.set_yscale('log')
        _, max_y_extent = ax.get_ylim()

        # ax.axvline(R/R_earth,ymin=0,ymax=np.log(threshold_rho)/np.log(max_y_extent),color='r',linestyle='dashed',linewidth=0.8)       
        ax.plot((R/R_earth, R/R_earth), (0, rho_of_the_particle),color='r',linestyle='dashed',linewidth=0.8)  # Put a line where radius is 
        ax.scatter(R/R_earth,rho_of_the_particle,marker='x',color='r',s=50)

        fig.tight_layout()
        figname = f'{sim_folder}radius_by_densities.png'
        fig.savefig(figname,dpi=500)
        print(figname + ' saved.\n')
    
    return(R)

def radius_by_averaging(pos):    # pos of BOUND material
    xy = np.hypot(pos[:,0],pos[:,1])
    r  = np.hypot(xy,pos[:,2])
    r  = np.sort(r)
    R  = np.mean(r[-100:])
    return(R)    

def bound_particles(sim_folder, pos, vel, pots, m, parids):
    N = len(m)
    x, y, z = pos.T

    # Compute kinetic energy: KE = 0.5 * m * v^2
    ke = 0.5 * m * np.sum(vel**2, axis=1)

    # Compute total energy
    e_tot = ke + (pots * m)

    print('Max KE: ',np.max(ke),'J\nMin PE:',(np.min(pots[0]*m[0])),'J')

    # Identify bound and unbound particles
    bound_mask = e_tot < 0
    num_bound = bound_mask.sum()

    print(f'Bound: {num_bound}, unbound: {N-num_bound}\n')

    return(bound_mask)
    # print(bound_mask)
    # bound_ids = parids[bound_mask]
    # unbound_ids = parids[~bound_mask]

    # print(bound_ids)
    # print(parids)
    # sys.exit()

def generate_second_ICs(mass_of_impactors_mearth, M_i1_mearth, M_i2_mearth, v_first_impact, v, angle_first_impact, angle, sim_time, time_between_snaps, threshold_rho, obliquity_after_first_impact,num_unbound,
                                                    uranus_ids, impactor_1_ids, pos_tar, vel_tar, h_tar,m_tar, rho_tar, p_tar, u_tar, matid_tar, parids_tar, R_tar):

    """Big function that creates and saves second-impact ICs."""

    # Cannot write parids to HDF5 via woma so have to do this to keep track of uranus / imp 1 particles:
    # Split arrays based on uranus_ids and impactor_1_ids
    # Put uranus particles first, then impactor 1.
    # Write npy files that give parids for each planet 

    # Find indices of particles belonging to uranus and impactor 1
    indicies_uranus = np.where(np.isin(parids_tar, uranus_ids))[0]
    indicies_impactor_1 = np.where(np.isin(parids_tar, impactor_1_ids))[0]

    #print(pos_tar[indicies_uranus][0], pos_tar[indicies_impactor_1][0])
    # Position the uranus particles at the start of the arrays, followed by impactor 1 
    pos_tar = np.concatenate((pos_tar[indicies_uranus], pos_tar[indicies_impactor_1])),
    vel_tar = np.concatenate((vel_tar[indicies_uranus], vel_tar[indicies_impactor_1])),
    m_tar = np.concatenate((m_tar[indicies_uranus], m_tar[indicies_impactor_1])),
    h_tar = np.concatenate((h_tar[indicies_uranus], h_tar[indicies_impactor_1])),
    rho_tar = np.concatenate((rho_tar[indicies_uranus], rho_tar[indicies_impactor_1])),
    p_tar = np.concatenate((p_tar[indicies_uranus], p_tar[indicies_impactor_1])),
    u_tar = np.concatenate((u_tar[indicies_uranus], u_tar[indicies_impactor_1])),
    matid_tar = np.concatenate((matid_tar[indicies_uranus], matid_tar[indicies_impactor_1])),
    
    # I have no idea why a new axis is created with concatenate above, but this will remove it:
    pos_tar = np.squeeze(pos_tar,axis=0)
    vel_tar = np.squeeze(vel_tar,axis=0)
    m_tar = np.squeeze(m_tar,axis=0)
    h_tar = np.squeeze(h_tar,axis=0)
    rho_tar = np.squeeze(rho_tar,axis=0)
    p_tar = np.squeeze(p_tar,axis=0)
    u_tar = np.squeeze(u_tar,axis=0)
    matid_tar = np.squeeze(matid_tar,axis=0)
    #print(pos_tar.shape,vel_tar.shape,m_tar.shape,h_tar.shape,rho_tar.shape,p_tar.shape,u_tar.shape,matid_tar.shape)
    #print(pos_tar[0], pos_tar[len(indicies_uranus)],'\n')
    # Impactor particles start at index [len(indicies_uranus)]

    # Fix silly decimal point formatting (totally my fault)
    if (str(M_i1_mearth).count('.')==1) and (str(M_i1_mearth).split('.')[1]=='0'):
        M_i1_mearth = int(M_i1_mearth)
    if (str(M_i2_mearth).count('.')==1) and (str(M_i2_mearth).split('.')[1]=='0'):
        M_i2_mearth = int(M_i2_mearth)

    #imp_1_path = path + f'impactors/{M_i1_mearth}_{M_i2_mearth}/chuck_in_swift_1/output/1_M_{M_i1_mearth}_for_{mass_of_impactors_mearth}_uranus_0001.hdf5'
    imp_2_path = f'/data/cluster4/oo21461/Planets/{mass_of_impactors_mearth}_uranus/' + f'impactors/{M_i1_mearth}_{M_i2_mearth}/chuck_in_swift_2/output/2_M_{M_i2_mearth}_for_{mass_of_impactors_mearth}_uranus_0001.hdf5'
    
    # Ensure angle is float
    if type(angle)!=float:
        float(angle)

    # Need to rotate Uranus by phi - clockwise rotation about the x axis, so negative angle
    R_matrix = np.array([
    [1, 0, 0],
    [0, np.cos(np.deg2rad(-angle)), -np.sin(np.deg2rad(-angle))],
    [0, np.sin(np.deg2rad(-angle)), np.cos(np.deg2rad(-angle))]])

    pos_tar = pos_tar @ R_matrix.T  # Rotate positions
    vel_tar = vel_tar @ R_matrix.T  # Rotate velocities

    # Load in impactor (probably quicker to do this outside of the function as it is the same for all phi but that would mean looaadsss of arguments)
    pos_imp, vel_imp, h_imp,m_imp, rho_imp, p_imp, u_imp, matid_imp, parids_imp, R_imp, pots_imp = load_to_woma(imp_2_path)

    print('\n'+f'~~~ Impactor info ~~~')
    print(f'Mass: {np.sum(m_imp)/M_earth} M_earth')
    print(f'Radius: {R_imp/R_earth} R_earth')

    print('\n'+f'~~~ Uranus info ~~~')
    print(f'Mass: {np.sum(m_tar)/M_earth} M_earth')
    print(f'Radius: {R_tar/R_earth} R_earth')
    print(f'Rotated by phi: {angle}')

    name = f'2_phi_{round(angle,2)}_M_{M_i2_mearth}_C_{mass_of_impactors_mearth}'       # collision 1: phi_ : M_ mass of impactor : C_ collective mass of impactors

    print('\nCollision scenario: ',name)

    # Masses and radii of the target and impactor
    M_t = np.sum(m_tar)
    M_i = np.sum(m_imp)
    R_t = R_tar
    R_i = R_imp

    #print(f'Mass of target = {M_t}')
    #print(f'Mass of impactor = {M_i}')

    # Mutual escape speed
    v_esc = np.sqrt(2 * G * (M_t + M_i) / (R_t + R_i))

    # Starting distance
    d = starting_distance(pos_tar, pos_imp,m_tar,m_imp,R_tar, R_imp)
    d += R_i    # add on radius of impactor (since d is from the edge of planet)
    print('Starting distance d = ',d/R_earth,' R_earth')

    # Initial position and velocity of the target
    ### IS THIS RIGHT??? ###
    A1_pos_t = np.array([0., 0., 0.])
    A1_vel_t = np.array([0., 0., 0.])

    A1_pos_i, A1_vel_i, t = woma.impact_pos_vel_b_v_c_r(
        b       = 0.3,  # Impact parameter 
        v_c     = v * v_esc, # Impactors speed at contact but in weird units?
        r       = d,        # Initial distance between target and impactor
        R_t     = R_t, 
        R_i     = R_i, 
        M_t     = M_t, 
        M_i     = M_i,
        return_t=True,  # need this for the parameter yml file
    )



    print("Time until collision = ",t,'s (',t/3600,' hours )')

    # Centre of mass?
    A1_pos_com = (M_t * A1_pos_t + M_i * A1_pos_i) / (M_t + M_i)
    A1_pos_t -= A1_pos_com
    A1_pos_i -= A1_pos_com

    # Centre of momentum
    A1_vel_com = (M_t * A1_vel_t + M_i * A1_vel_i) / (M_t + M_i)
    A1_vel_t -= A1_vel_com
    A1_vel_i -= A1_vel_com

    pos_tar += A1_pos_t
    vel_tar[:] += A1_vel_t

    pos_imp += A1_pos_i
    vel_imp[:] += A1_vel_i

    # Save hdf5
    if v_first_impact==1:
        save_path = f'/data/cluster4/oo21461/Simulations/{mass_of_impactors_mearth}_uranus/{M_i1_mearth}-{M_i2_mearth}/1_{angle_first_impact}/2_{round(angle,2)}/'
    else:
        save_path = f'/data/cluster4/oo21461/Simulations/{mass_of_impactors_mearth}_uranus/{M_i1_mearth}-{M_i2_mearth}/1_{angle_first_impact}_{v_first_impact}/2_{round(angle,2)}/'

    # Create directories if they don't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    assert (len(indicies_uranus) + len(indicies_impactor_1)) == len(pos_tar)
    # Export parids for each body
    new_ids_uranus = [*range(0,len(indicies_uranus))]    # Up to number of uranus particles
    new_ids_impactor_1 = [*range( len(indicies_uranus) , len(indicies_uranus) + len(indicies_impactor_1) )]      # From uranus particles, up to impactor 1
    new_ids_impactor_2 = [*range( len(indicies_uranus) + len(indicies_impactor_1), len(indicies_uranus) + len(indicies_impactor_1) + len(pos_imp) )] # From imp1 particles, up to impactor 2
    # print('Ur + imp1: ', len(pos_tar), '. imp2: ',len(pos_imp),'. tot: ' ,len(pos_tar) + len(pos_imp))
    # print('Ur + imp1: ',len(new_ids_uranus) + len(new_ids_impactor_1), '. imp2: ', len(new_ids_impactor_2),'. tot: ' , len(new_ids_uranus) + len(new_ids_impactor_1) + len(new_ids_impactor_2))
    assert (len(pos_tar) + len(pos_imp)) == (len(new_ids_uranus) + len(new_ids_impactor_1) + len(new_ids_impactor_2))

    np.save(save_path+'particles_uranus.npy',np.array(new_ids_uranus))
    np.save(save_path+'particles_impactor_1.npy',np.array(new_ids_impactor_1))
    np.save(save_path+'particles_impactor_2.npy',np.array(new_ids_impactor_2))

    print(np.max(pos_tar)/R_earth,np.max(pos_imp)/R_earth)
    print(np.min(pos_tar)/R_earth,np.min(pos_imp)/R_earth)

    with h5py.File(save_path + name + '.hdf5', "w") as f:
        woma.save_particle_data(
            f,
            np.append(pos_tar, pos_imp, axis=0),
            np.append(vel_tar, vel_imp, axis=0),
            np.append(m_tar, m_imp),
            np.append(h_tar, h_imp),
            np.append(rho_tar, rho_imp),
            np.append(p_tar, p_imp),
            np.append(u_tar, u_imp),
            np.append(matid_tar, matid_imp),
            boxsize=200 * R_earth,
            file_to_SI=woma.Conversions(M_earth, R_earth, 1),
        )

        # Save info.txt
    info_str = f'''Collision scenario: {name}
IC saved to : {save_path}{name}.hdf5
Mass of impactor 1: {M_i1_mearth} M_earth
Mass of impactor 2: {M_i2_mearth} M_earth
Initial distance between target and impactor: {d/R_earth} R_earth
Collision velocity: {v}*v_esc = {v*v_esc} m/s (?)
Time to collision: t = {t} s ({t/3600} h)
Sim end time (calculated): {sim_time+t}s
Sim end time (actual): {math.ceil((sim_time+t) / time_between_snaps) * time_between_snaps}s (rounded to nearest {time_between_snaps})
Expected number of snapshots = {(math.ceil((sim_time+t) / time_between_snaps) * time_between_snaps) / time_between_snaps}

Obliquity after first impact: {obliquity_after_first_impact} deg
Phi_2 = {angle} deg 
Threshold rho = {threshold_rho} kg/m^3
R_tar (calculated using threshold rho) = {R_tar/R_earth} R_earth
Number of unbound particles after impact 1 = {num_unbound}
        ''' # will need to edit analysis.py to catch this extra line (phi_2)
    
    info = open(save_path + f"info_{name}.txt", "w")
    info.write(info_str)
    info.close()
    print(f'Saved "info_{name}.txt"')

    # Save yml file for the collision
    yml_str = f'''
# Define the system of units to use internally.
InternalUnitSystem:
    UnitMass_in_cgs:        5.97240e27        # Sets Earth mass = 5.972
    UnitLength_in_cgs:      6.371e8        # Sets Earth radius = 6.371
    UnitVelocity_in_cgs:    6.371e8         # Sets time in seconds
    UnitCurrent_in_cgs:     1.0           # Amperes
    UnitTemp_in_cgs:        1.0           # Kelvin

# Parameters related to the initial conditions
InitialConditions:      
    file_name:  ./{name}.hdf5      # The initial conditions file to read
    periodic:   0                       # Are we running with periodic ICs?

# Parameters governing the time integration
TimeIntegration:
    time_begin:     0                   # The starting time of the simulation (in internal units).
    time_end:       {math.ceil((sim_time+t) / time_between_snaps) * time_between_snaps}               # The end time of the simulation (in internal units). ROUNDED UP TO MULTIPLE OF {time_between_snaps}
    dt_min:         0.00001              # The minimal time-step size of the simulation (in internal units).
    dt_max:         1000                # The maximal time-step size of the simulation (in internal units).

# Parameters governing the snapshots
Snapshots:
    basename:           snapshot    # Common part of the name of output files
    time_first:         0               # Time of the first output (in internal units)
    delta_time:         {int(time_between_snaps)}            # Time difference between consecutive outputs (in internal units)
    subdir:             ./output

# Parameters governing the conserved quantities statistics
Statistics:
    time_first: 0                       # Time of the first output (in internal units)
    delta_time: 1000                    # Time between statistics output

# Parameters controlling restarts
Restarts:
    enable:         1                   # Whether to enable dumping restarts at fixed intervals.
    save:           1                   # Whether to save copies of the previous set of restart files (named .prev)
    onexit:         1                   # Whether to dump restarts on exit (*needs enable*)
    subdir:         ./RESTART          # Name of subdirectory for restart files.
    basename:       Rfile              # Prefix used in naming restart files.
    delta_hours:    0.5                   # Decimal hours between dumps of restart files.

# Parameters for the hydrodynamics scheme
SPH:
    resolution_eta:     1.2348          # Target smoothing length in units of the mean inter-particle separation (1.2348 == 48Ngbs with the cubic spline kernel).
    delta_neighbours:   0.1             # The tolerance for the targetted number of neighbours.
    CFL_condition:      0.2             # Courant-Friedrich-Levy condition for time integration.
    h_max:              0.08            # Maximal allowed smoothing length (in internal units).
    viscosity_alpha:    1.5             # Override for the initial value of the artificial viscosity.

# Parameters for the self-gravity scheme
Gravity:
    eta:                            0.025       # Constant dimensionless multiplier for time integration.
    MAC:                            adaptive    # Choice of mulitpole acceptance criterion: 'adaptive' OR 'geometric'.
    epsilon_fmm:                    0.001       # Tolerance parameter for the adaptive multipole acceptance criterion.
    theta_cr:                       0.5         # Opening angle for the purely gemoetric criterion.
    max_physical_baryon_softening:  0.05        # Physical softening length (in internal units).
    use_tree_below_softening:       0

DomainDecomposition:
    trigger:        0.1                 # Fractional (<1) CPU time difference between MPI ranks required to trigger a new decomposition, or number of steps (>1) between decompositions
    adaptive:         0

# Parameters for the task scheduling
Scheduler:
    max_top_level_cells:    16          # Maximal number of top-level cells in any dimension. The nu
    cell_split_size:        400         # Maximal number of particles per cell (400 is the default value).
    tasks_per_cell:         10         # The average number of tasks per cell. If not large enough the simulation will fail (means guess...)
    links_per_tasks:        20 
    mpi_message_limit:      4096        
    nr_queues:              28

# Parameters related to the equation of state
EoS:
    # Select which planetary EoS material(s) to enable for use.
    planetary_use_HM80_HHe:       1     
    planetary_use_HM80_ice:       0   
    planetary_use_HM80_rock:      0  
    planetary_use_ANEOS_forsterite: 1
    planetary_use_custom_0:     1       # AQUA

    planetary_HM80_HHe_table_file:            ../HM80_HHe.txt
    planetary_custom_0_table_file:            ../AQUA_H20.txt
    planetary_ANEOS_forsterite_table_file:    ../ANEOS_forsterite_S19.txt

        '''

    yml = open(save_path + f"parameters_impact.yml", "w")
    yml.write(yml_str)
    yml.close()
    print('Saved "parameters_impact.yml"')
    print(f'Sim end time: {sim_time+t}s. Rounded: {math.ceil((sim_time+t) / 600) * 600}s')

    # if copy_across_eos:
    #     print('\ncopy_across_eos = True')
    #     items = os.listdir(save_path)
    #     change = False
    #     if 'ANEOS_forsterite_S19.txt' not in items:
    #         print('ANEOS_forsterite_S19 not found. Copying...')
    #         os.system(f'cp /data/cluster4/oo21461/EOS/ANEOS_forsterite_S19.txt {save_path}ANEOS_forsterite_S19.txt')
    #         change = True
    #     if 'AQUA_H20.txt' not in items:
    #         print('AQUA_H20 not found. Copying...')
    #         os.system(f'cp /data/cluster4/oo21461/EOS/AQUA_H20.txt {save_path}AQUA_H20.txt')
    #         change = True
    #     if 'HM80_HHe.txt' not in items:
    #         print('HM80_HHe not found. Copying...')
    #         os.system(f'cp /data/cluster4/oo21461/EOS/HM80_HHe.txt {save_path}HM80_HHe.txt')       
    #         change = True

    #     if not change:
    #         print(f'All EoS already in {save_path}')
    #     else:
    #         # Check it
    #         items = os.listdir(save_path)
    #         #print(items)
    #         for eos in ['ANEOS_forsterite_S19.txt','AQUA_H20.txt','HM80_HHe.txt']:
    #             if eos in items:
    #                 success=True
    #                 continue
    #             else:
    #                 success=False
    #                 print(f'Error: one or more EoS did not copy correctly - {eos}')
    #                 break
    #         if success:
    #             print(f'EoS copying to {save_path} successful!')

def custom_hdf5_save(f,ids,pos,vel,m,h,rho,p,u,matid,pots):
    """Custom HDF5 writer for saving unbound particle info only - adapted from WoMa source code (woma/misc/io)""" 


    Di_hdf5_particle_label = {  # Type
        "pos": "Coordinates",  # d
        "vel": "Velocities",  # f
        "m": "Masses",  # f
        "h": "SmoothingLengths",  # f
        "u": "InternalEnergies",  # f
        "rho": "Densities",  # f
        "P": "Pressures",  # f
        "s": "Entropies",  # f
        "id": "ParticleIDs",  # L
        "mat_id": "MaterialIDs",  # i
        "phi": "Potentials",  # f
        "T": "Temperatures",  # f
        "pot": "Potentials",
    }

    # Particles
    grp = f.create_group("/PartType0")
    grp.create_dataset(Di_hdf5_particle_label["pos"], data=pos, dtype="d")
    grp.create_dataset(Di_hdf5_particle_label["vel"], data=vel, dtype="f")
    grp.create_dataset(Di_hdf5_particle_label["m"], data=m, dtype="f")
    dset_h = grp.create_dataset(Di_hdf5_particle_label["h"], data=h, dtype="f")
    dset_u = grp.create_dataset(Di_hdf5_particle_label["u"], data=u, dtype="f")
    dset_rho = grp.create_dataset(Di_hdf5_particle_label["rho"], data=rho, dtype="f")
    grp.create_dataset(Di_hdf5_particle_label["P"], data=p, dtype="f")
    grp.create_dataset(Di_hdf5_particle_label["id"], data=ids, dtype="L")
    grp.create_dataset(Di_hdf5_particle_label["mat_id"], data=matids, dtype="i")
    grp.create_dataset(Di_hdf5_particle_label["pot"], data=pots)
    
    print('Saved "%s"' % f.filename[-64:])


def main():
    ### PARAMETERS ###

    # Collision parameters (written into yml file)
    # mass_of_impactors_mearth = 1.5
    sim_time = 40 * 3600 # 40 Hours AFTER the collision occurs
    time_between_snaps = 10 * 60    # 10 minutes between each hdf5 snapshot
    delta_phi = 10  # degrees. The three phis will be +/- this value in addition to the estimate directly 
    #v = 1 # * v_c (collision velocity) # nevermind - set this equal to phi_1
    l_uranus_today = np.array([0,0,1.27e36])    # Calculated using M_u = 14.54 M_earth, R_u = 4 R_earth, T = 17 h (61200 s), I/MR^2 = 0.225
                                                # So, L_rot = I (2 pi) / (61200) = 0.225 * (M_u) * (R_u)**2 * (2 pi) '/ (61200) = 1.2878 * 10**36
                                                # Might want to check this with Phil/Zoe
    obliquity_today = 97.77
    R_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(np.deg2rad(obliquity_today)), -np.sin(np.deg2rad(obliquity_today))],
        [0, np.sin(np.deg2rad(obliquity_today)), np.cos(np.deg2rad(obliquity_today))]
    ])
    l_uranus_today = l_uranus_today @ R_matrix.T    # Rotate to 97.77 deg (in frame of Uranus where +Z is the direction of uranus' ORBITAL ang mom)
    sanity_check = angle_to_z(l_uranus_today)
    print(f'Confirmed: {sanity_check}\n')

    # Get sim folder from cmd line arg
    try:
        first_impact_path = sys.argv[1]
    except Exception as err:
        print(err,"\nPlease enter the path to the simulation folder as the first argument to plot_snapshot.py")
        sys.exit()
    
    if first_impact_path[-1]!='/':
        first_impact_path+='/'

    # first_impact_path = '/data/cluster4/oo21461/SimulationsBC4/1.5_uranus/0.5-1/1_155_1.5/'
    phi_1 = first_impact_path.split('_uranus')[1].split('/1_')[1][:3]
    scenario_str = f'1_phi_{ phi_1 }_M_{ first_impact_path.split('_uranus/')[1].split('/1_')[0].split('-')[0] }_C_{ first_impact_path.split('_uranus')[0].split('/')[-1] }'

    # Determine if it is 1.5 * v_c or just 1 * v_c:
    if first_impact_path.split('/')[-2].count('_') !=1:
        v_1 = float(first_impact_path.split('/')[-2].split('_')[-1])
    else:
        v_1 = 1

    v = v_1     # Set v_2 equal to v_1 to keep it constant across impact scenario

    # Open info.txt
    try:
        info_file = open(first_impact_path + 'info_' + scenario_str + '.txt', "r")
    except Exception as err:
        print(err,'\nCannot open info.txt\nscenario_str attempted: ',scenario_str)
        sys.exit()
    info = info_file.read()
    info_file.close()
    print('\n~~ Impact 1 IC info ~~ \n'+info)
    M_i1_mearth = float(info.split('Mass of impactor 1: ')[1].split('M_earth')[0])
    M_i2_mearth = float(info.split('Mass of impactor 2: ')[1].split('M_earth')[0])
    d_rearth = float(info.split('Initial distance between target and impactor: ')[1].split('R_earth')[0])
    v_c = float(info.split('Collision velocity: 1*v_esc = ')[1].split(' m/s')[0])
    if v_1 != 1:    # COLLISION VELOCITY GIVEN IN INFO FILE IS WRONG FOR v_1 = 1.5
        v_c *= v_1
    t = float(info.split('Time to collision: t = ')[1].split(' s ')[0])
    no_of_snaps = str(round(float(info.split('Expected number of snapshots = ')[1]))).zfill(4)

    # Silly formating stuff that I should've thought about when designing my file system but oh well
    if len(str(M_i1_mearth).split('.'))>1 and str(M_i1_mearth).split('.')[1]=='0':
        M_i1_mearth = int(M_i1_mearth)
    if len(str(M_i2_mearth).split('.'))>1 and str(M_i2_mearth).split('.')[1]=='0':
        M_i2_mearth = int(M_i2_mearth)

    mass_of_impactors_mearth = M_i1_mearth + M_i2_mearth
    if len(str(mass_of_impactors_mearth).split('.'))>1 and str(mass_of_impactors_mearth).split('.')[1]=='0':
        mass_of_impactors_mearth = int(mass_of_impactors_mearth)    

    # Get path to initial uranus (for angular mom and averaging outermost densities later)
    path_to_initial_uranus = f'/data/cluster4/oo21461/Planets/{mass_of_impactors_mearth}_uranus/chuck_in_swift/spunup_{phi_1}/output/swift_spunup_{phi_1}_0001.hdf5'

    # Get radius of the proto-Uranus
    uranus_relax_info_file = open(f'/data/cluster4/oo21461/Planets/{mass_of_impactors_mearth}_uranus/relax_info.txt', "r")
    uranus_relax_info = uranus_relax_info_file.read()
    uranus_relax_info_file.close()
    R_uranus = float(uranus_relax_info.split('Estimated radius (averaging over outermost 100 particles) = ')[1].split('\n')[0]) / R_earth   # in units of R_earth

    uranus_ids, impactor_1_ids = separate_particles(first_impact_path,d_rearth,R_uranus,True) 

    # Open remnant 
    pos_rem, vel_rem, h_rem, m_rem, rho_rem, p_rem, u_rem, matid_rem, parids_rem, R_rem, pots_rem = load_to_woma(first_impact_path + f'output/snapshot_{no_of_snaps}.hdf5')
    # Note here R_rem is not reliable as it could include unbound particles in its average

    # Need to remove unbound particles
    bound_mask = bound_particles(first_impact_path, pos_rem, vel_rem, pots_rem, m_rem, parids_rem)
    unbound_mask = ~bound_mask  # These are True/False arrays (masks) which can be used to filter out as below 

    # Keep these particles
    bound_ids = parids_rem[bound_mask]
    bound_pos = pos_rem[bound_mask]
    bound_vel = vel_rem[bound_mask]
    bound_m = m_rem[bound_mask]
    bound_h = h_rem[bound_mask]
    bound_rho = rho_rem[bound_mask]
    bound_p = p_rem[bound_mask]
    bound_u = u_rem[bound_mask]
    bound_matid = matid_rem[bound_mask]
    bound_pots = pots_rem[bound_mask]

    # Discarded particles - Do analysis on this later (is it even worth it?)
    unbound_ids = parids_rem[unbound_mask]
    unbound_pos = pos_rem[unbound_mask]
    unbound_vel = vel_rem[unbound_mask]
    unbound_m = m_rem[unbound_mask]
    unbound_h = h_rem[unbound_mask]
    unbound_rho = rho_rem[unbound_mask]
    unbound_p = p_rem[unbound_mask]
    unbound_u = u_rem[unbound_mask]
    unbound_matid = matid_rem[unbound_mask]
    unbound_pots = pots_rem[unbound_mask]
    # IF NOT EQUAL TO ZERO, SHOULD SAVE THESE TO A HDF5
    if len(unbound_m)>0:
        with h5py.File(f"1_{phi_1}_unbound.hdf5", "w") as f:
            custom_hdf5_save(f,unbound_ids,unbound_pos,unbound_vel,unbound_m,unbound_h,unbound_rho,unbound_p,unbound_u,unbound_matid,unbound_pots)
    else:
        print('No unbound particles!\n')

    # Do ang mom before de-rotate (this is the only way I could get it to work)
    l, l_norm, L_angle_to_z = ang_mom(bound_pos,bound_vel,bound_m,bound_matid,0)    # In the impactors frame (angle to z here is the angle to the impactor's L_orbital)
    print(f'Obliquity after first impact: {float(phi_1)-L_angle_to_z} deg')

    continue_full = True
    if np.abs(float(phi_1)-L_angle_to_z) > obliquity_today:
        print('\nWarning: obliquity after first impact > obliquity today.')
        result = ''
        while True:
            result = input("To continue making ICs, input 'c'\nTo only plot ang mom, input 'l'\nTo quit, input 'q'\n")
            if result.lower() == 'q':
                print('\nQuitting...')
                sys.exit()
            elif result.lower() == 'l':
                continue_full = False
                break
            elif result.lower() == 'c':
                continue_full = True
                break
            else:
                print('Invalid input.')

    ## Derotate such that +Z is in the direction of Uranus' orbital angular momentum
    # Rotation matrix. This is a counter-clockwise rotation so angle is +ve

    R_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(np.deg2rad(float(phi_1))), -np.sin(np.deg2rad(float(phi_1)))],
        [0, np.sin(np.deg2rad(float(phi_1))), np.cos(np.deg2rad(float(phi_1)))]
    ])
    #print('Before: ',l_norm)
    # Transform positions and velocities oppositely (derotate) 
    bound_pos = bound_pos @ R_matrix.T  # Rotate positions
    bound_vel = bound_vel @ R_matrix.T  # Rotate velocities
    l = l @ R_matrix.T                  # Rotate L
    l_norm = l_norm @ R_matrix.T
    # print('After: ',l_norm)
    # test_to_z = angle_to_z(l_norm)  
    # print(test_to_z)

    # Need to figure out what density to cut-off at for radiu.
    # A reasonable estimate might be found by averaging the densities of the outermost 100 particles of the initial uranus (before first impact)
    pos_init, vel_init, h_init, m_init, rho_init, p_init, u_init, matid_init, parids_init, R_init, pots_init = load_to_woma(path_to_initial_uranus) # Open initial uranus
    xy = np.hypot(pos_init[:,0],pos_init[:,1])
    r_init  = np.hypot(xy,pos_init[:,2])
    sorted_indices_r = np.argsort(r_init)   # Sort by r
    r_sorted  = r_init[sorted_indices_r]
    rho_sorted = rho_init[sorted_indices_r]
    rho_edge = np.mean(rho_sorted[-100:])   # Get mean density of outermost particles
    print(f'Threshold density at edge of planet: {rho_edge} kg / m^3\n')

    # Get initial ang mom of uranus
    l_init, l_init_norm, L_init_angle_to_z = ang_mom(pos_init,vel_init,m_init,matid_init,0)
    print(f'This should be close to {phi_1}: ',L_init_angle_to_z)
    l_init = l_init @ R_matrix.T    # This should now be pointing in the +Z direction
    l_1 = l - l_init

    # Then need to define a cut off radius using bound particles
    R_tar = radius_by_densities(first_impact_path, bound_pos, bound_rho, rho_edge ,scenario_str)
    # If I get simulations that chuck out enough material into orbit, I will analyse this here 

    # Next, we need to find the phis we want to try. First find the ang mom needed to get to uranus today
    #print(f'Rot ang mom after first impact: {l}\nRot ang mom of Uranus today: {l_uranus_today}')
    l_2 = l_uranus_today-l

    # Estimate the phi needed for second impact based on the angle l_2 makes with the orbital angular momentum of Uranus
    phi_2_estimate = angle_to_z(l_2)
    print(f'Ang mom required in second impact: {l_2}\nPhi estimate for second impact: {phi_2_estimate}\n')

    # Phi_2s to try
    phi_2s = [phi_2_estimate-delta_phi, phi_2_estimate, phi_2_estimate+delta_phi]  

    # Plot ang moms
    plot_ang_moms(l_init,l_1,l,l_2,l_uranus_today,first_impact_path,scenario_str)

    if continue_full==False:
        print('\nQuitting...')
        sys.exit()

    # Create ICs for each phi
    for phi_2 in phi_2s:
        generate_second_ICs(mass_of_impactors_mearth, M_i1_mearth, M_i2_mearth, v_1, v, phi_1, phi_2, sim_time, time_between_snaps, rho_edge, float(phi_1)-L_angle_to_z, len(unbound_m),
                                                        uranus_ids, impactor_1_ids, bound_pos, bound_vel, bound_h, bound_m, bound_rho, bound_p, bound_u, bound_matid, bound_ids, R_tar)
    print(f'\nICs created for phi = {phi_2s}. Saved to cluster4/oo21461/Simulations/ \nDone')

if __name__=='__main__':
    main()

