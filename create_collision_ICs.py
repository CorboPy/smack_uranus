# This script creates the first impact initial conditions

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
from scipy.optimize import root_scalar

R_earth = 6.371e6   # m
M_earth = 5.9724e24  # kg
G = 6.67408e-11  # m^3 kg^-1 s^-2

# For rotated uranus
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
    box_mid = 0.5*data.metadata.boxsize[0].to(unyt.m)

    pos     = np.array(data.gas.coordinates - box_mid)
    vel     = np.array(data.gas.velocities)
    h       = np.array(data.gas.smoothing_lengths)
    m       = np.array(data.gas.masses)
    rho     = np.array(data.gas.densities)
    p       = np.array(data.gas.pressures)
    u       = np.array(data.gas.internal_energies)
    matid   = np.array(data.gas.material_ids)
    
    pos_centerM = np.sum(pos * m[:,np.newaxis], axis=0) / np.sum(m)
    vel_centerM = np.sum(vel * m[:,np.newaxis], axis=0) / np.sum(m)
    
    pos -= pos_centerM
    vel -= vel_centerM
    
    xy = np.hypot(pos[:,0],pos[:,1])
    r  = np.hypot(xy,pos[:,2])
    r  = np.sort(r)
    R  = np.mean(r[-100:])
    
    return pos, vel, h, m, rho, p, u, matid, R

# For impactors (straight out of woma)
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
    h = f['PartType0']['SmoothingLengths'][()]
    p = f['PartType0']['Pressures'][()]

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

    return positions, velocities, h, masses, rhos, internal_energies, ids, R
    # pos_tar, vel_tar, h_tar,m_tar, rho_tar, p_tar, u_tar, matid_tar, R_tar

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

### PARAMETERS ###

# Collision parameters (written into yml file)
mass_of_impactors_mearth = 4    # CHANGE possible_impactor_masses too
sim_time = 40 * 3600 # 40 Hours AFTER the collision occurs
time_between_snaps = 10 * 60    # 10 minutes between each hdf5 snapshot
copy_across_eos = True      # Set to False if you don't want to check that the EoS are in the simulation directory
collision_velocity = 1.5    # (escape_velocity)

angles = ['110','135','145','155']
#]possible_impactor_masses = [0.5,0.75,1,1.5,2]   
possible_impactor_masses = [1,1.5,2]   
combinations = find_combinations(possible_impactor_masses, possible_impactor_masses, mass_of_impactors_mearth)
path = f'/data/cluster4/oo21461/Planets/{mass_of_impactors_mearth}_uranus/'
paths_of_uranuses = [path + f'chuck_in_swift/spunup_{angle}/output/swift_spunup_{angle}_0001.hdf5' for angle in angles]
paths_of_impactors = []
for combination in combinations:
    M_i1_mearth, M_i2_mearth = combination 
    imp_1_path = path + f'impactors/{M_i1_mearth}_{M_i2_mearth}/1_M_{M_i1_mearth}_for_{mass_of_impactors_mearth}_uranus.hdf5'
    imp_2_path = path + f'impactors/{M_i1_mearth}_{M_i2_mearth}/2_M_{M_i2_mearth}_for_{mass_of_impactors_mearth}_uranus.hdf5'
    paths_of_impactors.append([imp_1_path,imp_2_path])
print('\n'+f'Uranus selected: {mass_of_impactors_mearth}_uranus')
print('Paths of uranuses: \n',paths_of_uranuses,'\n')
print('Paths of impactors: \n',paths_of_impactors)

path_to_ICs = '/data/cluster4/oo21461/Simulations'

for i, loc_tar in enumerate(paths_of_uranuses):
    # Looping through every Uranus: i.e. 2_uranus_135, 2_uranus_145, etc (we are looping through uranus angles here too)
    angle = loc_tar.split('/output/swift_spunup')[0].split('spunup_')[1]
    # if angle =='135':
    #     continue

    for combination in combinations:
        M_i1_mearth, M_i2_mearth = combination 
        #/data/cluster4/oo21461/Planets/2_uranus/impactors/1_1/chuck_in_swift_1
        imp_1_path = path + f'impactors/{M_i1_mearth}_{M_i2_mearth}/chuck_in_swift_1/output/1_M_{M_i1_mearth}_for_{mass_of_impactors_mearth}_uranus_0001.hdf5'
        #imp_2_path = path + f'impactors/{M_i1_mearth}_{M_i2_mearth}/2_M_{M_i2_mearth}_for_{mass_of_impactors_mearth}_uranus.hdf5'
        
        pos_tar, vel_tar, h_tar,m_tar, rho_tar, p_tar, u_tar, matid_tar, R_tar = load_to_woma(loc_tar)
        pos_imp, vel_imp, h_imp,m_imp, rho_imp, p_imp, u_imp, matid_imp, R_imp = load_to_woma(imp_1_path)

        print('\n'+f'~~~ Impactor info ~~~')
        print(f'Mass: {np.sum(m_imp)/M_earth} M_earth')
        print(f'Radius: {R_imp/R_earth} R_earth')

        print('\n'+f'~~~ Uranus info ~~~')
        print(f'Mass: {np.sum(m_tar)/M_earth} M_earth')
        print(f'Radius: {R_tar/R_earth} R_earth')
        print(f'Rotated by phi: {angle}')

        name = f'1_phi_{angle}_M_{M_i1_mearth}_C_{mass_of_impactors_mearth}'       # collision 1: phi_ : M_ mass of impactor : C_ collective mass of impactors

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
            v_c     = collision_velocity * v_esc, # Impactors speed at contact but in weird units?
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
        save_path = f'/data/cluster4/oo21461/Simulations/{mass_of_impactors_mearth}_uranus/{M_i1_mearth}-{M_i2_mearth}/1_{angle}_{collision_velocity}/'
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
                boxsize=100 * R_earth,
                file_to_SI=woma.Conversions(M_earth, R_earth, 1),
            )

            # Save info.txt
        info_str = f'''Collision scenario: {name}
IC saved to : {save_path}{name}.hdf5
Mass of impactor 1: {M_i1_mearth} M_earth
Mass of impactor 2: {M_i2_mearth} M_earth
Initial distance between target and impactor: {d/R_earth} R_earth
Collision velocity: 1*v_esc = {v_esc} m/s (?)
Time to collision: t = {t} s ({t/3600} h)
Sim end time (calculated): {sim_time+t}s
Sim end time (actual): {math.ceil((sim_time+t) / time_between_snaps) * time_between_snaps}s (rounded to nearest {time_between_snaps})
Expected number of snapshots = {(math.ceil((sim_time+t) / time_between_snaps) * time_between_snaps) / time_between_snaps}
        '''
        
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

    planetary_HM80_HHe_table_file:            ./HM80_HHe.txt
    planetary_custom_0_table_file:            ./AQUA_H20.txt
    planetary_ANEOS_forsterite_table_file:    ./ANEOS_forsterite_S19.txt

        '''

        yml = open(save_path + f"parameters_impact.yml", "w")
        yml.write(yml_str)
        yml.close()
        print('Saved "parameters_impact.yml"')
        print(f'Sim end time: {sim_time+t}s. Rounded: {math.ceil((sim_time+t) / 600) * 600}s')

        if copy_across_eos:
            print('\ncopy_across_eos = True')
            items = os.listdir(save_path)
            change = False
            if 'ANEOS_forsterite_S19.txt' not in items:
                print('ANEOS_forsterite_S19 not found. Copying...')
                os.system(f'cp /data/cluster4/oo21461/EOS/ANEOS_forsterite_S19.txt {save_path}ANEOS_forsterite_S19.txt')
                change = True
            if 'AQUA_H20.txt' not in items:
                print('AQUA_H20 not found. Copying...')
                os.system(f'cp /data/cluster4/oo21461/EOS/AQUA_H20.txt {save_path}AQUA_H20.txt')
                change = True
            if 'HM80_HHe.txt' not in items:
                print('HM80_HHe not found. Copying...')
                os.system(f'cp /data/cluster4/oo21461/EOS/HM80_HHe.txt {save_path}HM80_HHe.txt')       
                change = True

            if not change:
                print(f'All EoS already in {save_path}')
            else:
                # Check it
                items = os.listdir(save_path)
                #print(items)
                for eos in ['ANEOS_forsterite_S19.txt','AQUA_H20.txt','HM80_HHe.txt']:
                    if eos in items:
                        success=True
                        continue
                    else:
                        success=False
                        print(f'Error: one or more EoS did not copy correctly - {eos}')
                        break
                if success:
                    print(f'EoS copying to {save_path} successful!')
