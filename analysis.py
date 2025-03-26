# This script does all the major analysis for each simulation. It began life nice and readable but decended into coding hell as I added more and more to it and my dissertation deadline got closer and closer


# arg1 is path to output folder
# arg2 is the specfic snapshot number or range (indicated by a colon, i.e. 10, 11, ..., 20 would be 10:20) you want to plot
# leave arg2 empty if you want to plot all snapshots

import swiftsimio as sw
import sys
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import matplotlib.patches as mpatches
import matplotlib.font_manager
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import multiprocessing
from multiprocessing import Pool, current_process
from scipy.signal import savgol_filter
import pickle
import h5py
import time
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams["font.family"] = "Times New Roman"

R_earth = 6.371e6   # m
M_earth = 5.9724e24  # kg
G = 6.67408e-11  # m^3 kg^-1 s^-2

def custom_hdf5_save(f,ids,pos,vel,m,h,rho,p,u,matids,pots):
    """Custom HDF5 writer for saving unbound particle info only - adapted from WoMa source code (woma/misc/io)""" 

    # Save
    # Header
    # grp = f.create_group("/Header")
    # grp.attrs["BoxSize"] = [fake_boxsize] * 3
    # grp.attrs["NumPart_Total"] = [num_particle, 0, 0, 0, 0, 0]
    # grp.attrs["NumPart_Total_HighWord"] = [0, 0, 0, 0, 0, 0]
    # grp.attrs["NumPart_ThisFile"] = [num_particle, 0, 0, 0, 0, 0]
    # grp.attrs["Time"] = [0.0]
    # grp.attrs["NumFilesPerSnapshot"] = [1]
    # grp.attrs["MassTable"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # grp.attrs["Flag_Entropy_ICs"] = [0]
    # grp.attrs["Dimension"] = [3]

    # Runtime parameters
    # grp = f.create_group("/RuntimePars")
    # grp.attrs["PeriodicBoundariesOn"] = [0]

    # Units
    # grp = f.create_group("/Units")
    # grp.attrs["Unit mass in cgs (U_M)"] = [file_to_SI.m * SI_to_cgs.m]
    # grp.attrs["Unit length in cgs (U_L)"] = [file_to_SI.l * SI_to_cgs.l]
    # grp.attrs["Unit time in cgs (U_t)"] = [file_to_SI.t]
    # grp.attrs["Unit current in cgs (U_I)"] = [1.0]
    # grp.attrs["Unit temperature in cgs (U_T)"] = [1.0]

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

def get_hdf5_basename(directory):
    files = os.listdir(directory)
    hdf5_files = [f for f in files if f.endswith('.hdf5')]  #hdf5 filter
    for file in hdf5_files:
        if file[2:5]!='phi':
            hdf5_files.remove(file)

    if len(hdf5_files) == 1:
        # return basename
        return hdf5_files[0]
    elif len(hdf5_files) == 0:
        raise FileNotFoundError(f"No HDF5 files found in {directory}")
    else:
        raise ValueError(f"Multiple HDF5 files found in {directory}")

def separate_particles(sim_folder,d_rearth,R_uranus,verbose):
    # CHANGE SIM FOLDER TO FILE!!
    # Open HDF5
    #file = sim_folder + 'output/snapshot_0000.hdf5'
    try:
        data = sw.load(sim_folder)
    except Exception as err:
        print(err, '.\nCould not open ',sim_folder)
        sys.exit()

    # Metadata
    meta = data.metadata
    snap_time = meta.t
    num_particles = meta.n_gas
    boxsize = meta.boxsize
    np_boxsize = boxsize.to_ndarray()

    # Particle data
    parids = data.gas.particle_ids
    matids = data.gas.material_ids
    coords = data.gas.coordinates
    np_coords = coords.to_ndarray()
    np_coords*=R_earth
    #print(np_coords)

    assert np_coords.shape[0] == parids.shape[0], "Mismatch between coords and parids lengths!"

    # Filtering particles
    threshold =  (d_rearth/2 + np_boxsize[0]/2) * R_earth 
    impactor_ids = parids[np_coords[:, 0] > threshold]
    uranus_ids = parids[np_coords[:, 0] < threshold]
    if verbose:
        print(f'Impactor has: {len(impactor_ids)} particles')
        print(f'Proto_uranus has: {len(uranus_ids)} particles')
    assert len(impactor_ids) + len(uranus_ids) == num_particles

    # Save as .npy
    #np.save(sim_folder + 'particles_uranus.npy',uranus_ids.to_ndarray())
    #np.save(sim_folder + 'particles_impactor.npy',impactor_ids.to_ndarray())
    if verbose:
        print(sim_folder + 'particles_uranus.npy saved.')
        print(sim_folder + 'particles_impactor.npy saved.\n')

    return(uranus_ids.to_ndarray(),impactor_ids.to_ndarray())

def bound_particles(pos, vel, pots, m, parids):
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

def radius_by_densities(sim_folder, pos, rho, threshold_rho, collision_str):    # pos, rhos, etc of the BOUND material
    
    xy = np.hypot(pos[:,0],pos[:,1])
    r  = np.hypot(xy,pos[:,2])

    # Sort by r
    sorted_indices_r = np.argsort(r)
    r_sorted  = r[sorted_indices_r]
    rho_sorted_r = rho[sorted_indices_r]

    window = round(0.01*len(r),0)
    if window % 2 ==0:
        window+=1
    #print(window)

    # Apply Savitzky-Golay filter to smooth rho_sorted_r
    rho_smoothed = savgol_filter(rho_sorted_r, int(window), 3)

    # Sort by rho
    sorted_indices_rho = np.argsort(rho_smoothed)
    r_sorted_rho  = r_sorted[sorted_indices_rho]
    rho_sorted = rho_smoothed[sorted_indices_rho]

    # Find the radius corresponding to the threshold density
    idx = np.searchsorted(rho_sorted, threshold_rho, side='left')
    rho_of_the_particle = rho_sorted[idx]
    R = r_sorted_rho[idx]  # Use the corresponding radius
    print(f'Using radius {R/R_earth} (R_earth)\n')


    # # Sort by rho
    # sorted_indices_rho = np.argsort(rho)
    # r_sorted_rho  = r[sorted_indices_rho]
    # rho_sorted = rho[sorted_indices_rho]

    # idx = np.searchsorted(rho_sorted, threshold_rho, side='left')
    # rho_of_the_particle = rho_sorted[idx]
    # R = r_sorted_rho[idx]            # Use this as the radius
    # print(f'Using radius {R/R_earth} (R_earth)\n')
    
    return(R,r_sorted,rho_sorted_r,rho_of_the_particle,rho_smoothed)

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

def process_snap_ang_mom(args):
    file, uranus_ids, impactor_1_ids, impactor_2_ids, t, angle_to_z = args
    result = {}
    try:
        data = sw.load(file)
    except Exception as err:
        print(err, '.\nCould not open ', file)
        return None

    # Metadata
    meta = data.metadata
    snap_time = meta.t

    # Unit conversions
    data.gas.velocities.convert_to_mks()
    data.gas.coordinates.convert_to_mks()
    data.gas.masses.convert_to_mks()
    data.gas.internal_energies.convert_to_mks()
    data.gas.densities.convert_to_mks()

    # Particle data
    parids = data.gas.particle_ids.to_ndarray()
    matids = data.gas.material_ids.to_ndarray()
    coords = data.gas.coordinates.to_ndarray()
    vels = data.gas.velocities.to_ndarray()
    masses = data.gas.masses.to_ndarray()
    int_energies = data.gas.internal_energies.to_ndarray()
    densities = data.gas.densities.to_ndarray()

    # Center of mass correction
    pos_centerM = np.sum(coords * masses[:, np.newaxis], axis=0) / np.sum(masses)
    vel_centerM = np.sum(vels * masses[:, np.newaxis], axis=0) / np.sum(masses)
    coords -= pos_centerM
    vels -= vel_centerM

    # Min/max properties
    result['min_u'] = np.min(int_energies)
    result['max_u'] = np.max(int_energies)
    result['min_rho'] = np.min(densities)
    result['max_rho'] = np.max(densities)

    # Find indices of particles belonging to each body
    indicies_uranus = np.where(np.isin(parids, uranus_ids))[0]
    indicies_impactor_1 = np.where(np.isin(parids, impactor_1_ids))[0]
    if impactor_2_ids.size!=0:
        indicies_impactor_2 = np.where(np.isin(parids, impactor_2_ids))[0]
        # Create masks for Impactor 2 particles
        mask_impactor_2 = np.zeros_like(matids, dtype=bool)
        mask_impactor_2[indicies_impactor_2] = True
        mantle_impactor_2 = (matids == 900) & mask_impactor_2

    # Create masks for Uranus particles
    mask_uranus = np.zeros_like(matids, dtype=bool)
    mask_uranus[indicies_uranus] = True
    mantle_uranus = (matids == 900) & mask_uranus 

    # Create masks for Impactor 1 particles
    mask_impactor_1 = np.zeros_like(matids, dtype=bool)
    mask_impactor_1[indicies_impactor_1] = True
    mantle_impactor_1 = (matids == 900) & mask_impactor_1 

    if impactor_2_ids.size==0: # impact 1
        # i.e. collision num = 1
        if snap_time < t:
            # Uranus mantle only
            coords_filtered = coords[mantle_uranus]
            vels_filtered = vels[mantle_uranus]
            masses_filtered = masses[mantle_uranus]
        else:
            # (impact has occured)
            # uranus and impactor mantle
            coords_filtered = np.concatenate((coords[mantle_uranus],coords[mantle_impactor_1]),axis = 0)
            vels_filtered = np.concatenate((vels[mantle_uranus],vels[mantle_impactor_1]),axis = 0)
            masses_filtered = np.concatenate((masses[mantle_uranus],masses[mantle_impactor_1]),axis = 0)
    else: # impact 2
        # collision num =2
        if snap_time < t:
            # Uranus mantle and impactor 1 mantle only
            coords_filtered = np.concatenate((coords[mantle_uranus],coords[mantle_impactor_1]),axis = 0)
            vels_filtered = np.concatenate((vels[mantle_uranus],vels[mantle_impactor_1]),axis = 0)
            masses_filtered = np.concatenate((masses[mantle_uranus],masses[mantle_impactor_1]),axis = 0)
        else:
            # (impact has occured)
            # uranus, impactor1 and impactor2 mantle
            coords_filtered = np.concatenate((np.concatenate((coords[mantle_uranus],coords[mantle_impactor_1]),axis=0),coords[mantle_impactor_2]),axis=0) 
            vels_filtered = np.concatenate((np.concatenate((vels[mantle_uranus],vels[mantle_impactor_1]),axis=0),vels[mantle_impactor_2]),axis=0) 
            masses_filtered = np.concatenate((np.concatenate((masses[mantle_uranus],masses[mantle_impactor_1]),axis=0),masses[mantle_impactor_2]),axis=0) 

    # Angular momentum of mantle (filtered) 
    L_filtered = np.sum(masses_filtered[:, np.newaxis] * np.cross(coords_filtered, vels_filtered), axis=0)
    L_mag_filtered = np.sqrt(np.sum(L_filtered**2))
    L_norm_filtered = L_filtered / L_mag_filtered
    angle_filtered = (360 / (2 * np.pi)) * np.arccos(np.dot([0, 0, 1], L_filtered) / L_mag_filtered)
    obliquity_filtered = angle_filtered - angle_to_z

    # Angular momentum of everything 
    L = np.sum(masses[:, np.newaxis] * np.cross(coords, vels), axis=0)
    L_mag = np.sqrt(np.sum(L**2))
    L_norm = L / L_mag
    angle = (360 / (2 * np.pi)) * np.arccos(np.dot([0, 0, 1], L) / L_mag)
    obliquity = angle - angle_to_z

    # Estimate planetary radius
    xy = np.hypot(coords[:, 0], coords[:, 1])
    r = np.hypot(xy, coords[:, 2])
    R = np.mean(np.sort(r)[-100:])

    # Period estimation
    P = period_at_equator(coords, vels, R, angle)

    result['L_norm'] = L_norm
    result['L'] = L
    result['angle'] = angle
    result['P'] = P
    result['obliquity'] = obliquity

    result['L_norm_filtered'] = L_norm_filtered
    result['L_filtered'] = L_filtered
    result['angle_filtered'] = angle_filtered
    result['obliquity_filtered'] = obliquity_filtered
    return result

# Needs changing to Uranus mantle only
def ang_mom_tilt(sim_folder, t, uranus_ids, impactor_1_ids, impactor_2_ids, angle_to_z, verbose):
    files = [os.path.join(sim_folder, 'output', f) for f in os.listdir(sim_folder + 'output/') if f.endswith('hdf5')]
    files.sort()
    if verbose:
        print(f'Number of .hdf5 snapshots found: {len(files)}')

    # Prepare arguments for multiprocessing
    args = [(file, uranus_ids, impactor_1_ids, impactor_2_ids, t, angle_to_z) for file in files]

    # Process files in parallel
    with Pool(processes=8) as pool:
        results = list(tqdm(pool.imap(process_snap_ang_mom, args), total=len(files)))
    # Series for debugging
    # for arg in tqdm(args[7:]):
    #     process_snap_ang_mom(arg)


    # Combine results
    ang_mom_arr = np.zeros((len(files), 17))
    min_u, max_u, min_rho, max_rho = np.inf, -np.inf, np.inf, -np.inf
    for i, res in enumerate(results):
        if res is not None:
            min_u = min(min_u, res['min_u'])
            max_u = max(max_u, res['max_u'])
            min_rho = min(min_rho, res['min_rho'])
            max_rho = max(max_rho, res['max_rho'])
            ang_mom_arr[i, 0:3] = res['L_norm']
            ang_mom_arr[i, 3:6] = res['L']
            ang_mom_arr[i, 6] = res['angle']
            ang_mom_arr[i, 7] = res['P']
            ang_mom_arr[i, 8] = res['obliquity']
            ang_mom_arr[i, 9:12] = res['L_norm_filtered']   # of mantle only
            ang_mom_arr[i, 12:15] = res['L_filtered']   # of mantle only
            ang_mom_arr[i, 15] = res['angle_filtered']  # of mantle only
            ang_mom_arr[i, 16] = res['obliquity_filtered']  # of mantle only
        else:
            print(f"Houston we have a problem. Entry {i} in results is {res}.")
            sys.exit()

    if verbose:
        print(f'Final angle to +Z = {res["angle"]} deg')
        print(f'Final period at equator = {res["P"]} hours')
        print(f'Final obliquity = {res["obliquity"]} deg')

    # Save min/max info and angular momentum array
    with open(sim_folder + 'min_max_info.txt', "w") as f:
        f.write(f'min_u={min_u}\nmax_u={max_u}\nmin_rho={min_rho}\nmax_rho={max_rho}')
    np.save(sim_folder + 'ang_mom_tilt.npy', ang_mom_arr)

    return min_u, max_u, min_rho, max_rho, ang_mom_arr

def plot_densities(sim_folder,i,coords,time,matids,indicies_uranus,indicies_impactor,snap_time, ang_mom_arr, boxsize):
    # Needs doing
    pass

def plot_energies(sim_folder, i , coords, time, u, indicies_uranus, indicies_impactor_1, indicies_impactor_2, min_energy, max_energy, snap_time, ang_mom_arr, boxsize, obliquity_after_first_impact):
    # Needs doing and needs to be log scale
    num = str(i).zfill(4)
    x, y, z, = coords.T

    # Min/max energy rounding
    min_energy = 10 ** round(math.log10(float(min_energy))) - (5*10 ** (round(math.log10(float(min_energy)))-1))
    max_energy =  2 * 10 ** round(math.log10(float(max_energy))) #- (2*10 ** (round(math.log10(float(max_energy)))-1))

    # Add condition to exclude particles with z > 0
    z_condition = z <= (boxsize[2]/2)

    x_all = x[z_condition]
    y_all = y[z_condition]
    z_all = z[z_condition]
    u_all = u[z_condition]

    # Sort by z-coordinate
    sorted_indices_z = np.argsort(z_all)
    x_sorted_z = x_all[sorted_indices_z]
    y_sorted_z = y_all[sorted_indices_z]
    u_sorted_z = u_all[sorted_indices_z]

    # Move zero point to (0,0,0)
    x_sorted_z -= (boxsize[0]/2)
    y_sorted_z -= (boxsize[1]/2)

    # Plotting

    #plt.style.use('dark_background')
    fig, ax_init = plt.subplots(1, 2, figsize=(8,4),sharey=True) #figsize=(8,4.5))#figsize=(10, 5))        # set to 6, 3 for poster figures
    ax = []
    ax.append(ax_init[1])
    ax.append(ax_init[0])
    ax[0].set_facecolor('black')
    ax[1].set_facecolor('black')

    #fig.suptitle(f'Snapshot {num}: Time {snap_time:.3f} h')

    ax0_scatter_object = ax[0].scatter(x_sorted_z, y_sorted_z,c=u_sorted_z, cmap='plasma', s=5, marker='.', edgecolors='none',alpha=0.3, vmin=min_energy, vmax=0.1*max_energy, norm='log')

    # LIMITS
    midpoint_x = 0 #boxsize[0]/2
    midpoint_y = 0 #boxsize[1]/2
    midpoint_z = 0 #boxsize[2]/2
    scope = 20 # width of the plot
    ax[0].set_xlim([midpoint_x-(scope/2), midpoint_x+(scope/2)])
    ax[0].set_ylim([midpoint_y-(scope/2), midpoint_y+(scope/2)])
    ax[0].set_aspect('equal', 'box')
    ax[0].set_xlabel(r'x [R$_{\oplus}$]', fontsize = 20)
    ax[1].set_ylabel(r'y [R$_{\oplus}$]', fontsize = 20)
    ax[0].set_xticks([ midpoint_x+(scope/2.5), midpoint_x+(scope/2.5)/2, 0, midpoint_x-(scope/2.5)/2, midpoint_x-(scope/2.5)] )
    ax[0].set_yticks([ midpoint_y+(scope/2.5), midpoint_y+(scope/2.5)/2, 0, midpoint_y-(scope/2.5)/2, midpoint_y-(scope/2.5)] )
    if snap_time <0:
        time_str = f'{snap_time:.1f} h'
    else:
        time_str = f'+{snap_time:.1f} h'
    ax[0].text(4,7.5,time_str, fontsize = 24,color='white')
    ax[0].tick_params(axis='both', which='major', labelsize=20)
   
    # Plot 2

    # Sort by x coordinate
    sorted_indices_x = np.argsort(x)
    y_sorted_x = y[sorted_indices_x]
    z_sorted_x = z[sorted_indices_x]
    u_sorted_x = u[sorted_indices_x]

    # Move zero point to (0,0,0)
    y_sorted_x -= (boxsize[0]/2)
    z_sorted_x -= (boxsize[1]/2)

    ax1_scatter_object = ax[1].scatter(z_sorted_x,y_sorted_x,s=5,c=u_sorted_x, cmap='plasma',marker='.',edgecolors='none',alpha=0.2, vmin=min_energy, vmax=0.1*max_energy, norm='log')
    #ax[1].set_facecolor('white') # FOR REPORT FIGURE
    #ax[1].invert_xaxis() # Reverse z 

    # Ang mom arrows
    if indicies_impactor_2.size!=0:

        dy_init2 = ang_mom_arr[0,10]*scope/3
        dz_init2 = ang_mom_arr[0,11]*scope/3
        ax[1].arrow(midpoint_z, midpoint_y, dz_init2, dy_init2, head_width=0.2, head_length=0.2,linewidth=2, fc='white', ec='white',zorder=6, linestyle=':')
    
        # Rotate back by obliquity_after_first_impact to get obliquity before any impacts
        obliquity_after_first_impact_rad = np.radians(obliquity_after_first_impact)
        R_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(-obliquity_after_first_impact_rad), -np.sin(-obliquity_after_first_impact_rad)],
            [0, np.sin(-obliquity_after_first_impact_rad), np.cos(-obliquity_after_first_impact_rad)]
        ])
        init1 = np.array([0, dy_init2, dz_init2]) @ R_matrix.T
        ax[1].arrow(midpoint_z, midpoint_y, init1[2], init1[1], head_width=0.2, head_length=0.2,linewidth=2, fc='white', ec='white',zorder=6) # change to black for white plot

        # Ang mom now
        dy = ang_mom_arr[i,10]*scope/3
        dz = ang_mom_arr[i,11]*scope/3
        ax[1].arrow(midpoint_z, midpoint_y, dz, dy, head_width=0.2, head_length=0.2,linewidth=2, fc='lime', ec='lime',zorder=7)
        #ax.arrow(x, y, dx, dy, head_width, head_length, fc, ec)
    else:
        dy_init = ang_mom_arr[0,10]*scope/3
        dz_init = ang_mom_arr[0,11]*scope/3
        ax[1].arrow(midpoint_z, midpoint_y, dz_init, dy_init, head_width=0.2, head_length=0.2,linewidth=2, fc='white', ec='white',zorder=6)   # change to black for white plot
    
        dy = ang_mom_arr[i,10]*scope/3
        dz = ang_mom_arr[i,11]*scope/3
        ax[1].arrow(midpoint_z, midpoint_y, dz, dy, head_width=0.2, head_length=0.2,linewidth=2, fc='lime', ec='lime',zorder=7)

    ax[1].set_xlim([midpoint_z+(scope/2), midpoint_z-(scope/2)])
    ax[1].set_ylim([midpoint_y-(scope/2), midpoint_y+(scope/2)])
    ax[1].set_aspect('equal', 'box')
    ax[1].set_xlabel(r'z [R$_{\oplus}$]', fontsize = 20)
    #ax[1].set_ylabel(r'y ($R_{\bigoplus}$)') 
    #ax[1].sharey(ax[0]) # Share the y axis with ax[0]
    ax[1].set_xticks([ midpoint_z+(scope/2.5), midpoint_z+(scope/2.5)/2, 0, midpoint_z-(scope/2.5)/2, midpoint_z-(scope/2.5)] )
    ax[1].set_yticks([ midpoint_y+(scope/2.5), midpoint_y+(scope/2.5)/2, 0, midpoint_y-(scope/2.5)/2, midpoint_y-(scope/2.5)] )

    ax[1].text(-4,7.5,time_str, fontsize = 24,color='white')
    ax[1].tick_params(axis='both', which='major', labelsize=20)

    # REMOVE LATER
    # ax[0].axes.xaxis.set_ticklabels([])
    #ax[1].axes.yaxis.set_ticklabels([])

    #fig.tight_layout()

    # Apply tight layout but leave space on the right for the colorbar
    fig.tight_layout(rect=[0, 0, 0.9, 1])  # Reserve space on the right (0.85 controls space for colorbar)

    # Adjust space between subplots and right edge for the colorbar
    fig.subplots_adjust(right=0.9)

    # Create a separate ScalarMappable for the colorbar without alpha
    norm = mcolors.LogNorm(vmin=min_energy, vmax=0.1*max_energy)
    sm = cm.ScalarMappable(cmap='plasma', norm=norm)
    sm.set_array([])  # Required for colorbar

    # COLOURBAR
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.0225, pad=0.01)
    cbar.set_label(r'Specific Internal Energy [J kg$^{-1}$]', fontsize=15)
    cbar.ax.tick_params(labelsize=20) 

    
    name = f'{sim_folder}PNGs/snapshot_energies_{num}.png'
    try:
        fig.savefig(name,dpi=800)
    except Exception as err:
        sys.exit()
    plt.close()
    return()


def plot_ids(sim_folder,i,coords,time,matids,indicies_uranus, indicies_impactor_1, indicies_impactor_2,snap_time, ang_mom_arr, boxsize, obliquity_after_first_impact):
    #coords /= R_earth

    num = str(i).zfill(4)
    x, y, z, = coords.T

    # Add condition to exclude particles with z > 0
    z_condition = z <= (boxsize[2]/2)
   
    # Create masks for Uranus particles
    mask_uranus = np.zeros_like(matids, dtype=bool)
    mask_uranus[indicies_uranus] = True

    atmosphere_uranus = (matids == 200) & mask_uranus & z_condition
    mantle_uranus = (matids == 900) & mask_uranus & z_condition
    core_uranus = (matids == 400) & mask_uranus & z_condition

    # Create masks for Impactor 1 particles
    mask_impactor_1 = np.zeros_like(matids, dtype=bool)
    mask_impactor_1[indicies_impactor_1] = True
    
    mantle_impactor_1 = (matids == 900) & mask_impactor_1 & z_condition
    core_impactor_1 = (matids == 400) & mask_impactor_1 & z_condition

    if indicies_impactor_2.size!=0:
        # Create masks for Impactor 2 particles
        mask_impactor_2 = np.zeros_like(matids, dtype=bool)
        mask_impactor_2[indicies_impactor_2] = True
        mantle_impactor_2 = (matids == 900) & mask_impactor_2 & z_condition
        core_impactor_2 = (matids == 400) & mask_impactor_2 & z_condition
    else:
        # Empty bool array so that arr[mantle_impactor_2] and arr[core_impactor_2] return empty arrays
        mantle_impactor_2 = np.array([],dtype=bool)
        core_impactor_2 = np.array([],dtype=bool)

    # Combine all layers
    x_all = np.concatenate([x[atmosphere_uranus], x[mantle_uranus], x[core_uranus], x[core_impactor_1], x[mantle_impactor_1], x[core_impactor_2], x[mantle_impactor_2]])
    y_all = np.concatenate([y[atmosphere_uranus], y[mantle_uranus], y[core_uranus], y[core_impactor_1], y[mantle_impactor_1], y[core_impactor_2], y[mantle_impactor_2]])
    z_all = np.concatenate([z[atmosphere_uranus], z[mantle_uranus], z[core_uranus], z[core_impactor_1], z[mantle_impactor_1], z[core_impactor_2], z[mantle_impactor_2]])
    #print(x_all.shape)

    # Assign colors to each layer
    # colors = np.concatenate([
    #     np.full(x[atmosphere_uranus].shape[0], 'lightcyan'),    # Uranus atmosphere
    #     np.full(x[mantle_uranus].shape[0], '#6CBDFF'),       # Uranus mantle
    #     np.full(x[core_uranus].shape[0], 'slategray'),        # Uranus core
    #     np.full(x[core_impactor_1].shape[0], 'purple'),        # Impactor 1 core
    #     np.full(x[mantle_impactor_1].shape[0], 'lightgreen'),    # Impactor 1 mantle
    #     np.full(x[core_impactor_2].shape[0], 'olive'),        # Impactor 2 core
    #     np.full(x[mantle_impactor_2].shape[0], 'lemonchiffon')    # Impactor 2 mantle
    # ])

    # Assign colors to each layer
    colors = np.concatenate([
        np.full(x[atmosphere_uranus].shape[0], 'lightcyan'),    # Uranus atmosphere
        np.full(x[mantle_uranus].shape[0], '#3286C9'),       # Uranus mantle
        np.full(x[core_uranus].shape[0], '#225B89'),        # Uranus core
        np.full(x[core_impactor_1].shape[0], '#89225B'),        # Impactor 1 core
        np.full(x[mantle_impactor_1].shape[0], '#C93286'),    # Impactor 1 mantle
        np.full(x[core_impactor_2].shape[0], '#5B8922'),        # Impactor 2 core
        np.full(x[mantle_impactor_2].shape[0], '#86C932')    # Impactor 2 mantle
    ])

    # Sort by z-coordinate
    sorted_indices_z = np.argsort(z_all)
    x_sorted_z = x_all[sorted_indices_z]
    y_sorted_z = y_all[sorted_indices_z]
    colors_sorted_z = colors[sorted_indices_z]

    # Move zero point to (0,0,0)
    x_sorted_z -= (boxsize[0]/2)
    y_sorted_z -= (boxsize[1]/2)

    # Plotting

    #plt.style.use('dark_background')
    fig, ax = plt.subplots(1, 2, figsize = (8,4),sharey=True)#figsize=(8,4.5))#figsize=(10, 5))        # set to 6, 3 for poster figures

    #fig.suptitle(f'Snapshot {num}: Time {snap_time:.3f} h')

    ax[0].scatter(x_sorted_z, y_sorted_z, s=5, c=colors_sorted_z, marker='.', edgecolors='none',alpha=0.3)
    ax[0].set_facecolor('black')
    ax[1].set_facecolor('black')

    # LIMITS
    midpoint_x = 0 #boxsize[0]/2
    midpoint_y = 0 #boxsize[1]/2
    midpoint_z = 0 #boxsize[2]/2
    scope = 20 # width of the plot
    ax[0].set_xlim([midpoint_x-(scope/2), midpoint_x+(scope/2)])
    ax[0].set_ylim([midpoint_y-(scope/2), midpoint_y+(scope/2)])
    ax[0].set_aspect('equal', 'box')
    ax[0].set_xlabel(r'x [R$_{\oplus}$]', fontsize = 20)
    ax[0].set_ylabel(r'y [R$_{\oplus}$]', fontsize =20)
    ax[0].set_xticks([ midpoint_x+(scope/2.5), midpoint_x+(scope/2.5)/2, 0, midpoint_x-(scope/2.5)/2, midpoint_x-(scope/2.5)])
    ax[0].set_yticks([ midpoint_y+(scope/2.5), midpoint_y+(scope/2.5)/2, 0, midpoint_y-(scope/2.5)/2, midpoint_y-(scope/2.5)])
    if snap_time <0:
        time_str = f'{snap_time:.1f} h'
    else:
        time_str = f'+{snap_time:.1f} h'
    ax[0].text(3,7.5,time_str, fontsize = 24,color='white')
    ax[0].tick_params(axis='both', which='major', labelsize=20)
   
    # Plot 2: Create masks for Uranus particles
    mask_uranus_2 = np.zeros_like(matids, dtype=bool)
    mask_uranus_2[indicies_uranus] = True

    atmosphere_uranus_2 = (matids == 200) & mask_uranus_2
    mantle_uranus_2 = (matids == 900) & mask_uranus_2
    core_uranus_2 = (matids == 400) & mask_uranus_2

    # Create masks for Impactor 1 particles
    mask_impactor_2_1 = np.zeros_like(matids, dtype=bool)
    mask_impactor_2_1[indicies_impactor_1] = True
    
    mantle_impactor_2_1 = (matids == 900) & mask_impactor_2_1 
    core_impactor_2_1 = (matids == 400) & mask_impactor_2_1 

    if indicies_impactor_2.size!=0:
        # Create masks for Impactor 2 particles
        mask_impactor_2_2 = np.zeros_like(matids, dtype=bool)
        mask_impactor_2_2[indicies_impactor_2] = True
        mantle_impactor_2_2 = (matids == 900) & mask_impactor_2_2 
        core_impactor_2_2 = (matids == 400) & mask_impactor_2_2 
    else:
        # Empty bool array so that arr[mantle_impactor_2] and arr[core_impactor_2] return empty arrays
        mantle_impactor_2_2 = np.array([],dtype=bool)
        core_impactor_2_2 = np.array([],dtype=bool)

    # Combine all layers
    x_all = np.hstack([x[atmosphere_uranus_2], x[mantle_uranus_2], x[core_uranus_2], x[core_impactor_2_1], x[mantle_impactor_2_1], x[core_impactor_2_2], x[mantle_impactor_2_2]])
    y_all = np.hstack([y[atmosphere_uranus_2], y[mantle_uranus_2], y[core_uranus_2], y[core_impactor_2_1], y[mantle_impactor_2_1], y[core_impactor_2_2], y[mantle_impactor_2_2]])
    z_all = np.hstack([z[atmosphere_uranus_2], z[mantle_uranus_2], z[core_uranus_2], z[core_impactor_2_1], z[mantle_impactor_2_1], z[core_impactor_2_2], z[mantle_impactor_2_2]])

    # Assign colors to each layer
    # colors = np.hstack([
    #     np.full(x[atmosphere_uranus_2].shape[0], 'lightcyan'),    # Uranus atmosphere
    #     np.full(x[mantle_uranus_2].shape[0], '#6CBDFF'),       # Uranus mantle
    #     np.full(x[core_uranus_2].shape[0], 'slategray'),        # Uranus core
    #     np.full(x[core_impactor_2_1].shape[0], 'purple'),        # Impactor core
    #     np.full(x[mantle_impactor_2_1].shape[0], 'lightgreen'),    # Impactor mantle
    #     np.full(x[core_impactor_2_2].shape[0], 'olive'),        # Impactor 2 core
    #     np.full(x[mantle_impactor_2_2].shape[0], 'lemonchiffon')    # Impactor 2 mantle
    # ])

    # Assign colors to each layer
    colors = np.hstack([
        np.full(x[atmosphere_uranus_2].shape[0], 'lightblue'),    # Uranus atmosphere
        np.full(x[mantle_uranus_2].shape[0], '#3286C9'),       # Uranus mantle
        np.full(x[core_uranus_2].shape[0], '#225B89'),        # Uranus core
        np.full(x[core_impactor_2_1].shape[0], '#89225B'),        # Impactor core
        np.full(x[mantle_impactor_2_1].shape[0], '#C93286'),    # Impactor mantle
        np.full(x[core_impactor_2_2].shape[0], '#5B8922'),        # Impactor 2 core
        np.full(x[mantle_impactor_2_2].shape[0], '#86C932')    # Impactor 2 mantle
    ])

    # Sort by x coordinate
    sorted_indices_x = np.argsort(x_all)
    y_sorted_x = y_all[sorted_indices_x]
    z_sorted_x = z_all[sorted_indices_x]
    colors_sorted_x = colors[sorted_indices_x]

    # Move zero point to (0,0,0)
    y_sorted_x -= (boxsize[0]/2)
    z_sorted_x -= (boxsize[1]/2)

    ax[1].scatter(z_sorted_x,y_sorted_x,s=5,c=colors_sorted_x,marker='.',edgecolors='none',alpha=0.2)
    ax[1].text(-3,7.5,time_str, fontsize = 24,color='white')
    ax[1].tick_params(axis='both', which='major', labelsize=20)
    #ax[1].set_facecolor('white') # FOR REPORT FIGURE
    #ax[1].invert_xaxis() # Reverse z 

    # Ang mom arrows
    if indicies_impactor_2.size!=0:

        dy_init2 = ang_mom_arr[0,10]*scope/3
        dz_init2 = ang_mom_arr[0,11]*scope/3
        ax[1].arrow(midpoint_z, midpoint_y, dz_init2, dy_init2, head_width=0.2, head_length=0.2, fc='white', ec='white',zorder=6, linestyle=':',linewidth=2)
    
        # Rotate back by obliquity_after_first_impact to get obliquity before any impacts
        obliquity_after_first_impact_rad = np.radians(obliquity_after_first_impact)
        R_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(-obliquity_after_first_impact_rad), -np.sin(-obliquity_after_first_impact_rad)],
            [0, np.sin(-obliquity_after_first_impact_rad), np.cos(-obliquity_after_first_impact_rad)]
        ])
        init1 = np.array([0, dy_init2, dz_init2]) @ R_matrix.T
        ax[1].arrow(midpoint_z, midpoint_y, init1[2], init1[1], head_width=0.2, head_length=0.2, fc='white', ec='white',zorder=6,linewidth=2) # change to black for white plot

        # Ang mom now
        dy = ang_mom_arr[i,10]*scope/3
        dz = ang_mom_arr[i,11]*scope/3
        ax[1].arrow(midpoint_z, midpoint_y, dz, dy, head_width=0.2, head_length=0.2, fc='red', ec='red',zorder=7,linewidth=2)
        #ax.arrow(x, y, dx, dy, head_width, head_length, fc, ec)
    else:
        dy_init = ang_mom_arr[0,10]*scope/3
        dz_init = ang_mom_arr[0,11]*scope/3
        ax[1].arrow(midpoint_z, midpoint_y, dz_init, dy_init, head_width=0.2, head_length=0.2, fc='white', ec='white',zorder=6,linewidth=2)   # change to black for white plot
    
        dy = ang_mom_arr[i,10]*scope/3
        dz = ang_mom_arr[i,11]*scope/3
        ax[1].arrow(midpoint_z, midpoint_y, dz, dy, head_width=0.2, head_length=0.2, fc='red', ec='red',zorder=7,linewidth=2)

    ax[1].set_xlim([midpoint_z+(scope/2), midpoint_z-(scope/2)])
    ax[1].set_ylim([midpoint_y-(scope/2), midpoint_y+(scope/2)])
    ax[1].set_aspect('equal', 'box')
    ax[1].set_xlabel(r'z [R$_{\oplus}$]', fontsize =20)
    #ax[1].set_ylabel(r'y ($R_{\bigoplus}$)') 
    #ax[1].sharey(ax[0]) # Share the y axis with ax[0]
    ax[1].set_xticks([ midpoint_z+(scope/2.5), midpoint_z+(scope/2.5)/2, 0, midpoint_z-(scope/2.5)/2, midpoint_z-(scope/2.5)] )
    ax[1].set_yticks([ midpoint_y+(scope/2.5), midpoint_y+(scope/2.5)/2, 0, midpoint_y-(scope/2.5)/2, midpoint_y-(scope/2.5)] )


    # REMOVE LATER
    # ax[0].axes.xaxis.set_ticklabels([])
    #ax[1].axes.yaxis.set_ticklabels([])

    fig.tight_layout()
    name = f'{sim_folder}PNGs/snapshot_ids_{num}.png'
    try:
        fig.savefig(name,dpi=800)
    except Exception as err:
        sys.exit()
    plt.close()
    return()


# Helper function to process a single file
def process_snap_plotting(args):
    sim_folder, t, file, uranus_ids, impactor_1_ids, impactor_2_ids, phi, min_u, max_u, min_rho, max_rho, ang_mom_arr, obliquity_after_first_impact, plot_dict, verbose = args
    try:
        # Open HDF5
        data = sw.load(file)
    except Exception as err:
        print(err, '.\nCould not open ', file)
        return
    num = file.split('snapshot_')[1].split('.hdf5')[0]
    i = num.lstrip('0') or '0'
    i = int(i)

    # Metadata
    meta = data.metadata
    snap_time = meta.t
    snap_time = snap_time.to("hour").value
    snap_time -= t/3600
    num_particles = meta.n_gas
    boxsize = meta.boxsize
    boxsize = boxsize.to_ndarray()
    # print(type(snap_time), type(boxsize))
    # sys.exit()

    # Unit conversions
    data.gas.velocities.convert_to_mks()
    #data.gas.coordinates.convert_to_mks()
    data.gas.masses.convert_to_mks()
    data.gas.internal_energies.convert_to_mks()
    data.gas.densities.convert_to_mks()
    #data.gas.pressures.convert_to_mks()

    # Particle data
    parids = data.gas.particle_ids.to_ndarray()
    matids = data.gas.material_ids.to_ndarray()
    coords = data.gas.coordinates.to_ndarray()
    vels =  data.gas.velocities.to_ndarray()
    masses = data.gas.masses.to_ndarray()
    int_energies = data.gas.internal_energies.to_ndarray()
    densities = data.gas.densities.to_ndarray()
    #pressures = data.gas.pressures.to_ndarray()

    # Find indices of particles belonging to each body
    indicies_uranus = np.where(np.isin(parids, uranus_ids))[0]
    indicies_impactor_1 = np.where(np.isin(parids, impactor_1_ids))[0]
    if impactor_2_ids.size!=0:
        indicies_impactor_2 = np.where(np.isin(parids, impactor_2_ids))[0]
    else:
        indicies_impactor_2 = impactor_2_ids # empty int array
    
    #print(type(coords),type(time),type(matids),type(indicies_uranus),type(indicies_impactor),type(snap_time),type(ang_mom_arr),type(boxsize))

    if plot_dict['densities']:
        plot_densities()
    if plot_dict['ids']:
        plot_ids(sim_folder, i, coords, time, matids, indicies_uranus, indicies_impactor_1, indicies_impactor_2, snap_time, ang_mom_arr, boxsize,obliquity_after_first_impact)
    if plot_dict['energies']:
        plot_energies(sim_folder, i , coords, time, int_energies,indicies_uranus, indicies_impactor_1, indicies_impactor_2, min_u, max_u, snap_time, ang_mom_arr, boxsize, obliquity_after_first_impact)

    return () 

def pngs(sim_folder, t, uranus_ids, impactor_1_ids, impactor_2_ids, phi, min_u, max_u, min_rho, max_rho, ang_mom_arr, obliquity_after_first_impact,plot_dict, verbose):
    files = []
    print("Creating PNGs...")
    for filename in os.listdir(sim_folder+'output/'):
        if filename.endswith('hdf5'):
            files.append(sim_folder+'output/'+filename)
    files.sort()

    uranus_ids = np.array(uranus_ids)
    impactor_1_ids = np.array(impactor_1_ids)
    if impactor_2_ids.size!=0:
        impactor_2_ids = np.array(impactor_2_ids)

    # Check PNGs folder exists
    if not os.path.exists(sim_folder+'PNGs/'):
        os.makedirs(sim_folder+'PNGs/')

    # Prepare arguments for each worker
    pool_args = [(sim_folder,t, file, uranus_ids, impactor_1_ids, impactor_2_ids, phi, min_u, max_u, min_rho, max_rho, ang_mom_arr, obliquity_after_first_impact, plot_dict, verbose) for file in files]

    # create pool for multiprocessing
    with Pool(processes=8) as pool:
        results = list(tqdm(pool.imap(process_snap_plotting, pool_args), total=len(pool_args)))

    ## Or execute in series:
    # for arg in tqdm(pool_args):
    #     process_snap_plotting(arg)

def final_profiles(sim_folder, t, uranus_ids, impactor_1_ids, impactor_2_ids, phi, min_u, max_u, min_rho, max_rho, ang_mom_arr, obliquity_after_first_impact,scenario_str, threshold_rho, verbose):
    
    # FOCUS ON REMNANT OF 2ND IMPACT FIRST
    files = []
    print("Creating PNGs...")
    for filename in os.listdir(sim_folder+'output/'):
        if filename.endswith('hdf5'):
            files.append(sim_folder+'output/'+filename)
    files.sort()
    file = files[-1]     # final snap

    # Open file 
    try:
        # Open HDF5
        data = sw.load(file)
    except Exception as err:
        print(err, '.\nCould not open ', file)
        return
    num = file.split('snapshot_')[1].split('.hdf5')[0]
    i = num.lstrip('0') or '0'
    i = int(i)

    # Metadata
    meta = data.metadata
    snap_time = meta.t
    snap_time = snap_time.to("hour").value
    num_particles = meta.n_gas
    boxsize = meta.boxsize
    boxsize = boxsize.to_ndarray()
    # print(type(snap_time), type(boxsize))
    # sys.exit()

    # Unit conversions
    data.gas.velocities.convert_to_mks()
    data.gas.coordinates.convert_to_mks()
    data.gas.masses.convert_to_mks()
    data.gas.internal_energies.convert_to_mks()
    data.gas.densities.convert_to_mks()
    data.gas.potentials.convert_to_mks()  
    data.gas.pressures.convert_to_mks()
    data.gas.smoothing_lengths.convert_to_mks()

    # Particle data
    parids = data.gas.particle_ids.to_ndarray()
    matids = data.gas.material_ids.to_ndarray()
    coords = data.gas.coordinates.to_ndarray()
    vels =  data.gas.velocities.to_ndarray()
    masses = data.gas.masses.to_ndarray()
    int_energies = data.gas.internal_energies.to_ndarray()
    densities = data.gas.densities.to_ndarray()
    pots = data.gas.potentials.to_ndarray()
    pressures = data.gas.pressures.to_ndarray()
    h       = np.array(data.gas.smoothing_lengths)

    pos_centerM = np.sum(coords * masses[:,np.newaxis], axis=0) / np.sum(masses)
    vel_centerM = np.sum(vels * masses[:,np.newaxis], axis=0) / np.sum(masses)
    
    coords -= pos_centerM
    vels -= vel_centerM

    # Need to remove unbound particles
    bound_mask = bound_particles(coords, vels, pots, masses, parids)
    unbound_mask = ~bound_mask  # These are True/False arrays (masks) which can be used to filter out as below 

    # Keep these particles
    bound_ids = parids[bound_mask]
    bound_pos = coords[bound_mask]
    bound_vel = vels[bound_mask]
    bound_m = masses[bound_mask]
    bound_h = h[bound_mask]
    bound_rho = densities[bound_mask]
    bound_p = pressures[bound_mask]
    bound_u = int_energies[bound_mask]
    bound_matid = matids[bound_mask]
    bound_pots = pots[bound_mask]

    # Discarded particles - Do analysis on this later (is it even worth it?)
    unbound_ids = parids[unbound_mask]
    unbound_pos = coords[unbound_mask]
    unbound_vel = vels[unbound_mask]
    unbound_m = masses[unbound_mask]
    unbound_h = h[unbound_mask]
    unbound_rho = densities[unbound_mask]
    unbound_p = pressures[unbound_mask]
    unbound_u = int_energies[unbound_mask]
    unbound_matid = matids[unbound_mask]
    unbound_pots = pots[unbound_mask]
    # IF NOT EQUAL TO ZERO, SHOULD SAVE THESE TO A HDF5
    if len(unbound_m)>0:
        print(f'{len(unbound_m)} particles unbound! Removing...')
        with h5py.File(sim_folder+f"2_{round(phi,2)}_unbound.hdf5", "w") as f:
            custom_hdf5_save(f,unbound_ids,unbound_pos,unbound_vel,unbound_m,unbound_h,unbound_rho,unbound_p,unbound_u,unbound_matid,unbound_pots)
    else:
        print('No unbound particles!\n')

    xy = np.hypot(bound_pos[:,0],bound_pos[:,1])
    r  = np.hypot(xy,bound_pos[:,2])
    R, r_sorted, rho_sorted_r,rho_of_the_particle,rho_smoothed_r = radius_by_densities(sim_folder, bound_pos, bound_rho, threshold_rho, scenario_str)

    # Find indices of BOUND particles belonging to each body
    indicies_uranus = np.where(np.isin(bound_ids, uranus_ids))[0]
    indicies_impactor_1 = np.where(np.isin(bound_ids, impactor_1_ids))[0]
    if impactor_2_ids.size!=0:
        indicies_impactor_2 = np.where(np.isin(bound_ids, impactor_2_ids))[0]
    else:
        indicies_impactor_2 = impactor_2_ids # empty int array

    # Create masks for Uranus particles
    mask_uranus = np.zeros_like(bound_matid, dtype=bool)
    mask_uranus[indicies_uranus] = True

    atmosphere_uranus = (bound_matid == 200) & mask_uranus
    mantle_uranus = (bound_matid == 900) & mask_uranus
    core_uranus = (bound_matid == 400) & mask_uranus

    # Create masks for Impactor 1 particles
    mask_impactor_1 = np.zeros_like(bound_matid, dtype=bool)
    mask_impactor_1[indicies_impactor_1] = True
    
    mantle_impactor_1 = (bound_matid == 900) & mask_impactor_1 
    core_impactor_1 = (bound_matid == 400) & mask_impactor_1 

    if indicies_impactor_2.size!=0:
        # Create masks for Impactor 2 particles
        mask_impactor_2 = np.zeros_like(bound_matid, dtype=bool)
        mask_impactor_2[indicies_impactor_2] = True
        mantle_impactor_2 = (bound_matid == 900) & mask_impactor_2
        core_impactor_2 = (bound_matid == 400) & mask_impactor_2
    else:
        # Empty bool array so that arr[mantle_impactor_2] and arr[core_impactor_2] return empty arrays
        mantle_impactor_2 = np.array([],dtype=bool)
        core_impactor_2 = np.array([],dtype=bool)

    # Combine all layers
    r_all = np.hstack([r[atmosphere_uranus], r[mantle_uranus], r[core_uranus], r[core_impactor_1], r[mantle_impactor_1], r[core_impactor_2], r[mantle_impactor_2]])
    rho_all = np.hstack([bound_rho[atmosphere_uranus], bound_rho[mantle_uranus], bound_rho[core_uranus], bound_rho[core_impactor_1], bound_rho[mantle_impactor_1], bound_rho[core_impactor_2], bound_rho[mantle_impactor_2]])
    p_all = np.hstack([bound_p[atmosphere_uranus], bound_p[mantle_uranus], bound_p[core_uranus], bound_p[core_impactor_1], bound_p[mantle_impactor_1], bound_p[core_impactor_2], bound_p[mantle_impactor_2]])
    m_all =  np.hstack([bound_m[atmosphere_uranus], bound_m[mantle_uranus], bound_m[core_uranus], bound_m[core_impactor_1], bound_m[mantle_impactor_1], bound_m[core_impactor_2], bound_m[mantle_impactor_2]])
    h_all = np.hstack([bound_h[atmosphere_uranus], bound_h[mantle_uranus], bound_h[core_uranus], bound_h[core_impactor_1], bound_h[mantle_impactor_1], bound_h[core_impactor_2], bound_h[mantle_impactor_2]])
    m_uranus = np.hstack([bound_m[atmosphere_uranus], bound_m[mantle_uranus], bound_m[core_uranus]])
    r_uranus = np.hstack([r[atmosphere_uranus], r[mantle_uranus], r[core_uranus]])
    m_imp1 = np.hstack([bound_m[core_impactor_1], bound_m[mantle_impactor_1]])
    r_imp1 = np.hstack([r[core_impactor_1], r[mantle_impactor_1]])
    m_imp2 = np.hstack([bound_m[core_impactor_2], bound_m[mantle_impactor_2]])
    r_imp2 = np.hstack([r[core_impactor_2], r[mantle_impactor_2]])

    # Assign colors to each layer
    colors = np.concatenate([
        np.full(r[atmosphere_uranus].shape[0], 'lightcyan'),    # Uranus atmosphere
        np.full(r[mantle_uranus].shape[0], '#3286C9'),       # Uranus mantle
        np.full(r[core_uranus].shape[0], '#225B89'),        # Uranus core
        np.full(r[core_impactor_1].shape[0], '#89225B'),        # Impactor 1 core
        np.full(r[mantle_impactor_1].shape[0], '#C93286'),    # Impactor 1 mantle
        np.full(r[core_impactor_2].shape[0], '#5B8922'),        # Impactor 2 core
        np.full(r[mantle_impactor_2].shape[0], '#86C932')    # Impactor 2 mantle
    ])

    # Assign colors to each layer
    colors_uranus = np.concatenate([
        np.full(r[atmosphere_uranus].shape[0], 'lightcyan'),    # Uranus atmosphere
        np.full(r[mantle_uranus].shape[0], '#3286C9'),       # Uranus mantle
        np.full(r[core_uranus].shape[0], '#225B89')        # Uranus core
    ])
    colors_imp1 = np.concatenate([
        np.full(r[core_impactor_1].shape[0], '#89225B'),        # Impactor 1 core
        np.full(r[mantle_impactor_1].shape[0], '#C93286')    # Impactor 1 mantle
    ])
    # Assign colors to each layer
    colors_imp2 = np.concatenate([
        np.full(r[core_impactor_2].shape[0], '#5B8922'),        # Impactor 2 core
        np.full(r[mantle_impactor_2].shape[0], '#86C932')    # Impactor 2 mantle
    ])


    # Sort by R
    sorted_indices_r = np.argsort(r_all)
    r_sorted_r = r_all[sorted_indices_r]
    rho_sorted_r = rho_all[sorted_indices_r]
    p_sorted_r = p_all[sorted_indices_r]
    colors_sorted_r = colors[sorted_indices_r]
    m_sorted_r = m_all[sorted_indices_r]
    h_sorted_r = h[sorted_indices_r]

    # for uranus
    sorted_indices_r_uranus = np.argsort(r_uranus)
    r_sorted_r_uranus = r_uranus[sorted_indices_r_uranus]
    m_sorted_r_uranus = m_uranus[sorted_indices_r_uranus]
    sorted_colors_uranus = colors_uranus[sorted_indices_r_uranus]
    # for imp1
    sorted_indices_r_imp1 = np.argsort(r_imp1)
    r_sorted_r_imp1 = r_imp1[sorted_indices_r_imp1]
    m_sorted_r_imp1 = m_imp1[sorted_indices_r_imp1]
    sorted_colors_imp1 = colors_imp1[sorted_indices_r_imp1]
    # for imp1
    sorted_indices_r_imp2 = np.argsort(r_imp2)
    r_sorted_r_imp2 = r_imp2[sorted_indices_r_imp2]
    m_sorted_r_imp2 = m_imp2[sorted_indices_r_imp2]
    sorted_colors_imp2 = colors_imp2[sorted_indices_r_imp2]

    # Calculate estiamte for roche radius
    rho_m = np.min(bound_rho)   # use density floor for rho_m
    print(rho_m)
    rho_M = np.sum(bound_m[r < R]) /( (4/3)*np.pi*R**3 )
    roche_radius = R*(2*(rho_M/rho_m))**(1/3)

    # h plot
    fig, ax = plt.subplots()
    max_x_extent = 130
    # ax3
    h_sorted_r /= R_earth   # get in rearth 
    r_sorted_r_rearth = r_sorted_r / R_earth     # copy of r sorted r in r earth (r_sorted_r is needed later)

    print(r_sorted_r_rearth[r_sorted_r_rearth > 5])
    #print(np.unique(h_sorted_r[r_sorted_r_rearth > 5]))

    # Define bin edges for r_sorted_r in increments of 0.1 R_earth
    bin_edges = np.arange(0, np.max(r_sorted_r_rearth) + 4, 4)

    # Assign all particles to bins
    bin_indices_all = np.digitize(r_sorted_r_rearth, bin_edges)

    # Count total number of particles in each bin
    total_particles_per_bin = np.array([np.sum(bin_indices_all == i) for i in range(1, len(bin_edges))])


    h_sorted_r_filter = h_sorted_r[h_sorted_r >= 0.08]  # filter for hmax
    r_sorted_r_filter = r_sorted_r_rearth[h_sorted_r >= 0.08]   # mask for radius in rearth
    #print(h_sorted_r_filter[r_sorted_r_rearth > 5])

    # Compute total mass in each bin
    bin_indices_filtered = np.digitize(r_sorted_r_filter, bin_edges)  # Assign particles to bins
    #binned_smoothing = np.array([np.sum(h_sorted_r_filter[bin_indices == i])/np.sum(h_sorted_r[bin_indices == i]) for i in range(1, len(bin_edges))])
    filtered_particles_per_bin = np.array([np.sum(bin_indices_filtered == i) for i in range(1, len(bin_edges))])


    proportion_per_bin = np.divide(filtered_particles_per_bin, total_particles_per_bin, 
                               out=np.zeros_like(filtered_particles_per_bin, dtype=float), 
                               where=total_particles_per_bin > 0)
                               
    # Extend arrays to close the envelope
    bin_edges_extended = np.concatenate(([bin_edges[0]], bin_edges))  # Repeat last bin edge
    proportion_extended = np.concatenate(([0], proportion_per_bin, [0]))  # Ensure it drops to zero

    ax.step(bin_edges_extended, proportion_extended, where='pre', linewidth=1.5, color='black')    
    ax.set_xlim(0, max_x_extent)
    ax.set_ylim(0, 1)
    #ax[3].set_ylabel(r"Smoothing Length [R$_{\oplus}$]")
    ax.set_ylabel(r"Proportion of particles with h$_{\text{max}}$")
    ax.set_xlabel(r"Radius [R$_{\oplus}$]")
    ax.axvline(R/R_earth,ymin=0,ymax=1,color='b',linestyle='solid',linewidth=1,zorder=2000)   
    ax.axvline(roche_radius/R_earth,ymin=0,ymax=1,color='r',linestyle='solid',linewidth=1,zorder=2000)       
    
    fig.tight_layout()
    figname = f'{sim_folder}final_h_bins.png'    #f'/home/oo21461/Documents/tools/final_profiles.png'
    fig.savefig(figname,dpi=500)
    plt.close()
    print(figname + ' saved.\n')



    # big profile plots
    max_x_extent = 4.5 # R_earth
    fig, ax = plt.subplots(3, 1, figsize=(3.52,8),sharex=True)

    ax[0].plot(r_sorted/R_earth,rho_smoothed_r,color='black',zorder=100,linewidth=1.2,linestyle='--')
    ax[0].scatter(r_sorted_r/R_earth,rho_sorted_r,s=0.5,c=colors_sorted_r,marker='.',edgecolors='none',alpha=0.5)
    # need to do custom legend


    # ax[0].scatter(r[core_uranus]/R_earth,bound_rho[core_uranus],s=0.5,edgecolors='none',c='#225B89',alpha=0.4,label='Uranus core',zorder=4)
    # ax[0].scatter(r[mantle_uranus]/R_earth,bound_rho[mantle_uranus],s=0.5,edgecolors='none',c='#3286C9',alpha=0.4,label='Uranus mantle',zorder=7)
    # ax[0].scatter(r[atmosphere_uranus]/R_earth,bound_rho[atmosphere_uranus],s=0.5,edgecolors='none',c='cyan',alpha=0.4,label='Uranus atmosphere',zorder=10)

    # ax[0].scatter(r[core_impactor_1]/R_earth,bound_rho[core_impactor_1],s=0.5,edgecolors='none',c='#89225B',alpha=0.4,label='Imp1 core',zorder=5)
    # ax[0].scatter(r[mantle_impactor_1]/R_earth,bound_rho[mantle_impactor_1],s=0.5,edgecolors='none',c='#C93286',alpha=0.4,label='Imp1 mantle',zorder=8)

    # ax[0].scatter(r[core_impactor_2]/R_earth,bound_rho[core_impactor_2],s=0.5,edgecolors='none',c='#5B8922',alpha=0.4,label='Imp2 core',zorder=6)
    # ax[0].scatter(r[mantle_impactor_2]/R_earth,bound_rho[mantle_impactor_2],s=0.5,edgecolors='none',c='#86C932',alpha=0.4,label='Imp2 mantle',zorder=9)

    ax[0].set_xlim(0, max_x_extent)
    #ax.set_ylim(0, None)
    #ax[0].set_xlabel(r"Radius [$R_\oplus$]")
    ax[0].set_ylabel(r"Density [kg $\text{m}^{-3}$]")
    ax[0].set_yscale('log')
    _, max_y_extent = ax[0].get_ylim()
    ax[0].axvline(R/R_earth,ymin=0,ymax=1,color='r',linestyle='solid',linewidth=1,zorder=2000)       
    #ax[0].plot((R/R_earth, R/R_earth), (0, rho_of_the_particle),color='r',linestyle='dashed',linewidth=0.8,zorder=101)  # Put a line where radius is 
    ax[0].scatter(R/R_earth,rho_of_the_particle,marker='x',color='r',s=50,zorder=102)


    # # Create custom legend handles using Line2D with '-' as the marker
    # custom_legend = [
    #     Line2D([0], [0], color='#3286C9', linestyle='-', label="Uranus"),
    #     Line2D([0], [0], color='#C93286', linestyle='-', label="Imp1"),
    #     Line2D([0], [0], color='#86C932', linestyle='-', label="Imp2")
    # ]
    # lgnd = ax[0].legend(loc='upper right',handles=custom_legend)

    # #lgnd.legendHandles[0]._legmarker.set_markersize(MY_SIZE)
    # lgnd.legend_handles[0]._sizes = [30]
    # lgnd.legend_handles[1]._sizes = [30]
    # lgnd.legend_handles[2]._sizes = [30]
    # # lgnd.legend_handles[0].set_alpha(1)
    # lgnd.legend_handles[1].set_alpha(1)
    # lgnd.legend_handles[2].set_alpha(1)
    
    ax[1].scatter(r_sorted_r/R_earth,p_sorted_r,s=0.5,c=colors_sorted_r,marker='.',edgecolors='none',alpha=0.5)

    # ax[1].scatter(r[core_uranus]/R_earth,bound_p[core_uranus],s=0.5,edgecolors='none',c='#225B89',alpha=0.4,label='Uranus core',zorder=4)
    # ax[1].scatter(r[mantle_uranus]/R_earth,bound_p[mantle_uranus],s=0.5,edgecolors='none',c='#3286C9',alpha=0.4,label='Uranus mantle',zorder=7)
    # ax[1].scatter(r[atmosphere_uranus]/R_earth,bound_p[atmosphere_uranus],s=0.5,edgecolors='none',c='cyan',alpha=0.4,label='Uranus atmosphere',zorder=10)

    # ax[1].scatter(r[core_impactor_1]/R_earth,bound_p[core_impactor_1],s=0.5,edgecolors='none',c='#89225B',alpha=0.4,label='Imp1 core',zorder=5)
    # ax[1].scatter(r[mantle_impactor_1]/R_earth,bound_p[mantle_impactor_1],s=0.5,edgecolors='none',c='#C93286',alpha=0.4,label='Imp1 mantle',zorder=8)

    # ax[1].scatter(r[core_impactor_2]/R_earth,bound_p[core_impactor_2],s=0.5,edgecolors='none',c='#5B8922',alpha=0.4,label='Imp2 core',zorder=6)
    # ax[1].scatter(r[mantle_impactor_2]/R_earth,bound_p[mantle_impactor_2],s=0.5,edgecolors='none',c='#86C932',alpha=0.4,label='Imp2 mantle',zorder=9)

    ax[1].axvline(R/R_earth,ymin=0,ymax=1,color='r',linestyle='solid',linewidth=1,zorder=2000)       

    #ax[1].set_xlim(0, max_x_extent)
    ax[1].set_ylim((10**9)+5*10**8, (10**12)+5*10**11)
    ax[2].set_xlabel(r"Radius [R$_\oplus$]")
    ax[1].set_ylabel(r"Pressure [Pa]")
    ax[1].set_yscale('log')


    # enclosed mass
    #ax[2].scatter(r_sorted_r/R_earth,np.cumsum(m_sorted_r/M_earth),s=0.5,color='black',marker='.',edgecolors='none',label='Tot')
    ax[2].scatter(r_sorted_r_uranus/R_earth,np.cumsum(m_sorted_r_uranus/M_earth),s=2,c=sorted_colors_uranus,marker='.',edgecolors='none',label='Ur')
    ax[2].scatter(r_sorted_r_imp1/R_earth,np.cumsum(m_sorted_r_imp1/M_earth),s=2,c=sorted_colors_imp1,marker='.',edgecolors='none',label='Imp1')
    ax[2].scatter(r_sorted_r_imp2/R_earth,np.cumsum(m_sorted_r_imp2/M_earth),s=2,c=sorted_colors_imp2,marker='.',edgecolors='none',label='Imp2')
    
    # custom legend
    # Define custom legend elements
    legend_elements = [
        Line2D([0], [0], color='#3286C9', markerfacecolor='#3286C9', markersize=10,  label='Ur'),
        Line2D([0], [0], color='#C93286', markerfacecolor='#C93286', markersize=10, label='Imp1'),
        Line2D([0], [0], color='#86C932', markerfacecolor='#86C932', markersize=10, label='Imp2'),
    ]

    legend = ax[2].legend(handles=legend_elements,loc='lower right',framealpha=1)
    legend.set_zorder(3000)  # Ensure legend is above the vertical line

    ax[2].axvline(R/R_earth,ymin=0,ymax=1,color='r',linestyle='solid',linewidth=1,zorder=2000)       
    ax[2].set_ylabel(r"Enclosed Mass [M$_{\oplus}$]")
    ax[2].set_yscale('log')

    # plt.legend()
    plt.subplots_adjust(wspace=1)
    fig.tight_layout()
    figname = f'{sim_folder}final_profiles.png'    #f'/home/oo21461/Documents/tools/final_profiles.png'
    fig.savefig(figname,dpi=500)
    plt.close()
    print(figname + ' saved.\n')


    #here

    # # Radius vs density including radius definition
    # max_x_extent = 4.5 # R_earth
    # fig, ax = plt.subplots(figsize=(4,4))

    # #smoothed
    # ax.plot(r_sorted/R_earth,rho_smoothed_r,color='black',zorder=100,linewidth=1.2)

    # ax.scatter(r[core_uranus]/R_earth,bound_rho[core_uranus],s=0.5,edgecolors='none',c='#225B89',alpha=0.4,label='Uranus core',zorder=4)
    # ax.scatter(r[mantle_uranus]/R_earth,bound_rho[mantle_uranus],s=0.5,edgecolors='none',c='#3286C9',alpha=0.4,label='Uranus mantle',zorder=7)
    # ax.scatter(r[atmosphere_uranus]/R_earth,bound_rho[atmosphere_uranus],s=0.5,edgecolors='none',c='cyan',alpha=0.4,label='Uranus atmosphere',zorder=10)

    # ax.scatter(r[core_impactor_1]/R_earth,bound_rho[core_impactor_1],s=0.5,edgecolors='none',c='#89225B',alpha=0.4,label='Imp1 core',zorder=5)
    # ax.scatter(r[mantle_impactor_1]/R_earth,bound_rho[mantle_impactor_1],s=0.5,edgecolors='none',c='#C93286',alpha=0.4,label='Imp1 mantle',zorder=8)

    # ax.scatter(r[core_impactor_2]/R_earth,bound_rho[core_impactor_2],s=0.5,edgecolors='none',c='#5B8922',alpha=0.4,label='Imp2 core',zorder=6)
    # ax.scatter(r[mantle_impactor_2]/R_earth,bound_rho[mantle_impactor_2],s=0.5,edgecolors='none',c='#86C932',alpha=0.4,label='Imp2 mantle',zorder=9)
    # #ax.title.set_text(r'r vs $\rho$ for final snapshot of '+collision_str)
    # ax.set_xlim(0, max_x_extent)
    # #ax.set_ylim(0, None)
    # ax.set_xlabel(r"Radius [$R_\oplus$]")
    # ax.set_ylabel(r"Density [kg $\text{m}^{-3}$]")
    # ax.set_yscale('log')
    # _, max_y_extent = ax.get_ylim()
    # # ax.axvline(R/R_earth,ymin=0,ymax=np.log(threshold_rho)/np.log(max_y_extent),color='r',linestyle='dashed',linewidth=0.8)       
    # ax.plot((R/R_earth, R/R_earth), (0, rho_of_the_particle),color='r',linestyle='dashed',linewidth=0.8,zorder=101)  # Put a line where radius is 
    # ax.scatter(R/R_earth,rho_of_the_particle,marker='x',color='r',s=50,zorder=102)

    # fig.tight_layout()
    # figname = f'{sim_folder}density.png'
    # fig.savefig(figname,dpi=500)
    # plt.close()
    # print(figname + ' saved.\n')


    ## Bar plot
    fig, ax = plt.subplots(nrows=3,ncols=1,figsize=(4,6.2857),sharex=True) #4, 6.5
    ax[2].set_xlabel(r"Radius [R$_\oplus$]")
    ax[0].set_xlim(0, np.max(r/R_earth))

    ax[0].set_ylabel(r"Mass [M$_{\oplus}$]")
    ax[1].set_ylabel(r"Mass [M$_{\oplus}$]")
    ax[2].set_ylabel(r"Mass [M$_{\oplus}$]")
    ax[0].set_ylim(int(1e-5), int(1e1))
    ax[1].set_ylim(int(1e-5), int(1e1))
    ax[2].set_ylim(int(1e-5), int(1e1))
    
    bin_size_rearth = 4

    # ax0 for uranus
    uranus_min_depth = []
    uranus_external_mass = []
    uranus_internal_mass = []
    uranus_outside_roche = []
    labels = ['Core','Mantle','Atmosphere']
    zorders = [4,7,10] #,5,8,6,9]  #'#3286C9'
    linestyles = ['--','--','--'] #,'--','--','--','--','--']
    colours = ['black', 'dodgerblue','#d67f17'] #,'#89225B','#C93286','#5B8922','#86C932']
    for i, layer in enumerate([core_uranus,mantle_uranus,atmosphere_uranus]): #,core_impactor_1,mantle_impactor_1,core_impactor_2,mantle_impactor_2]):

        r_layer = r[layer]
        m_layer = bound_m[layer]
        uranus_min_depth.append(np.min(r_layer))

        # Mass past R condition
        past_R = r_layer > R
        below_R = ~past_R
        uranus_external_mass.append(np.sum(m_layer[past_R]))
        uranus_internal_mass.append(np.sum(m_layer[below_R]))

        # Mass past roche condition
        past_R = r_layer > roche_radius
        uranus_outside_roche.append(np.sum(m_layer[past_R]))

        # Sort data by radial distance
        sorted_indices = np.argsort(r_layer)
        sorted_r = r_layer[sorted_indices]
        sorted_m = m_layer[sorted_indices]

        # Define bin edges (e.g., every 0.25 M_earth)
        bin_edges = np.arange(0, np.max(sorted_r) + bin_size_rearth*R_earth, bin_size_rearth*R_earth)

        # Compute total mass in each bin
        bin_indices = np.digitize(sorted_r, bin_edges)  # Assign particles to bins
        binned_mass = np.array([np.sum(sorted_m[bin_indices == i]) for i in range(1, len(bin_edges))])

        # Extend arrays to close the envelope
        bin_edges_extended = np.append(bin_edges, bin_edges[-1])  # Repeat last bin edge
        binned_mass_extended = np.append(binned_mass, 0)  # Ensure it drops to zero

        ax[0].step(bin_edges_extended/R_earth, np.append([0], binned_mass_extended)/M_earth, where='pre', linewidth=1, color=colours[i],zorder=zorders[i],linestyle=linestyles[i],label=labels[i])

    # ax1 for imp1
    imp1_min_depth = []
    imp1_external_mass = []
    imp1_internal_mass = []
    imp1_outside_roche = []
    zorders = [5,8] #,6,9]  #'#3286C9'
    linestyles = ['--','--'] #,'--','--','--','--','--']
    colours = ['black','dodgerblue'] #,'#5B8922','#86C932']
    for i, layer in enumerate([core_impactor_1,mantle_impactor_1]): #,core_impactor_2,mantle_impactor_2]):

        r_layer = r[layer]
        m_layer = bound_m[layer]
        imp1_min_depth.append(np.min(r_layer))

        # Mass past R condition
        past_R = r_layer > R
        below_R = ~past_R
        imp1_external_mass.append(np.sum(m_layer[past_R]))
        imp1_internal_mass.append(np.sum(m_layer[below_R]))

        # Mass past roche condition
        past_R = r_layer > roche_radius
        imp1_outside_roche.append(np.sum(m_layer[past_R]))

        # Sort data by radial distance
        sorted_indices = np.argsort(r_layer)
        sorted_r = r_layer[sorted_indices]
        sorted_m = m_layer[sorted_indices]

        # Define bin edges (e.g., every 0.25 M_earth)
        bin_edges = np.arange(0, np.max(sorted_r) + bin_size_rearth*R_earth, bin_size_rearth*R_earth)

        # Compute total mass in each bin
        bin_indices = np.digitize(sorted_r, bin_edges)  # Assign particles to bins
        binned_mass = np.array([np.sum(sorted_m[bin_indices == i]) for i in range(1, len(bin_edges))])

        # Extend arrays to close the envelope
        bin_edges_extended = np.append(bin_edges, bin_edges[-1])  # Repeat last bin edge
        binned_mass_extended = np.append(binned_mass, 0)  # Ensure it drops to zero

        ax[1].step(bin_edges_extended/R_earth, np.append([0], binned_mass_extended)/M_earth, where='pre', linewidth=1, color=colours[i],zorder=zorders[i],linestyle=linestyles[i])

    # ax2 for imp2
    imp2_min_depth = []
    imp2_external_mass = []
    imp2_internal_mass = []
    imp2_outside_roche = []
    zorders = [6,9]  #'#3286C9'
    linestyles = ['--','--'] #,'--','--','--','--','--']
    colours = ['black','dodgerblue']
    for i, layer in enumerate([core_impactor_2,mantle_impactor_2]):

        r_layer = r[layer]
        m_layer = bound_m[layer]
        imp2_min_depth.append(np.min(r_layer))

        # Mass past R condition
        past_R = r_layer > R
        below_R = ~past_R
        imp2_external_mass.append(np.sum(m_layer[past_R]))
        imp2_internal_mass.append(np.sum(m_layer[below_R]))

        # Mass past roche condition
        past_R = r_layer > roche_radius
        imp2_outside_roche.append(np.sum(m_layer[past_R]))

        # Sort data by radial distance
        sorted_indices = np.argsort(r_layer)
        sorted_r = r_layer[sorted_indices]
        sorted_m = m_layer[sorted_indices]

        # Define bin edges (e.g., every 0.25 M_earth)
        bin_edges = np.arange(0, np.max(sorted_r) + bin_size_rearth*R_earth, bin_size_rearth*R_earth)

        # Compute total mass in each bin
        bin_indices = np.digitize(sorted_r, bin_edges)  # Assign particles to bins
        binned_mass = np.array([np.sum(sorted_m[bin_indices == i]) for i in range(1, len(bin_edges))])

        # Extend arrays to close the envelope
        bin_edges_extended = np.append(bin_edges, bin_edges[-1])  # Repeat last bin edge
        binned_mass_extended = np.append(binned_mass, 0)  # Ensure it drops to zero

        ax[2].step(bin_edges_extended/R_earth, np.append([0], binned_mass_extended)/M_earth, where='pre', linewidth=1, color=colours[i],zorder=zorders[i],linestyle=linestyles[i])

    # Create custom legend handles using Line2D with '-' as the marker
    custom_legend = [
        Line2D([0], [0], color='black', linestyle='--', label="Core"),
        Line2D([0], [0], color='dodgerblue', linestyle='--', label="Mantle"),
        Line2D([0], [0], color='#d67f17', linestyle='--', label="Atmosphere"),
        Line2D([0], [0], color='r', linestyle='-', label="Roche radius"),
        Line2D([0], [0], color='purple', linestyle='-', label="Moons")
    ]
    lgnd = ax[0].legend(loc='upper right',handles=custom_legend,fontsize=7)

    # Planet radius
    #ax[0].axvline(R/R_earth,0,1,color='r')
    #ax[1].axvline(R/R_earth,0,1,color='r')
    #ax[2].axvline(R/R_earth,0,1,color='r')
    # Roche
    ax[0].axvline(roche_radius/R_earth,0,1,color='r',linewidth=1)
    ax[1].axvline(roche_radius/R_earth,0,1,color='r',linewidth=1)
    ax[2].axvline(roche_radius/R_earth,0,1,color='r',linewidth=1)

    for i in range(3):
        # Plotting orbital radius of major satellites
        ax[i].axvline((129.9*10**6)/R_earth,0,1,color='purple',linewidth=1)
        ax[i].axvline((190.90*10**6)/R_earth,0,1,color='purple',linewidth=1)
        ax[i].axvline((266.0*10**6)/R_earth,0,1,color='purple',linewidth=1)
        ax[i].axvline((436.3*10**6)/R_earth,0,1,color='purple',linewidth=1)
        ax[i].axvline((583.5*10**6)/R_earth,0,1,color='purple',linewidth=1)

    ax[0].set_yscale("log")
    ax[0].set_ylim(1*10**(-5), None)
    ax[1].set_yscale("log")
    ax[1].set_ylim(1*10**(-5), None)
    ax[2].set_yscale("log")
    ax[2].set_ylim(1*10**(-5), None)


    #ax[0].legend()

    plt.subplots_adjust(wspace=1)
    #fig.tight_layout()
    figname = f'{sim_folder}mass_histogram.pdf'
    fig.savefig(figname,dpi=500,bbox_inches='tight')
    print(figname + ' saved.\n')
    return(R,
            np.array(uranus_internal_mass), np.array(imp1_internal_mass), np.array(imp2_internal_mass),
            np.array(uranus_external_mass),np.array(imp1_external_mass),np.array(imp2_external_mass),
            np.array(uranus_min_depth), np.array(imp1_min_depth), np.array(imp2_min_depth),
            np.array(uranus_outside_roche), np.array(imp1_outside_roche), np.array(imp2_outside_roche), roche_radius)
    
    
def main(verbose=True):

    plot_dict = {'densities': False ,'ids': True,'energies':True}
    assert True in plot_dict.values()   # You can't plot nothing

    # options: 
    # Use info.txt (initial distance) to determine which particles are impactor/target and save to a file.
    # Ang mom at each snap (mag, direction, tilt), SAVE to a npy or something. ALSO might as well save the max density / internal energies for normalising this plot later.
    # Plot x/y and y/z PNGs (on same fig)? Use saved ang mom to plot J (requirement?). Use saved tar/imp particle deterination file to color tar/imp differently. Save 
    # Animate? Automatically runs the ffmpeg and names it appropriately
    # Satellite mass / bound mass... eventually. Still not sure on how to do this. Need to define a radius
    # After getting radius, plot density vs radius, temp, etc

    do_separate_particles = False
    do_ang_mom = False
    do_pngs = False
    do_animate = False
    do_uranus_satellites = False

    # Gets the argument from the command line
    try:
        sim_folder = sys.argv[1]
        if sim_folder=='--help' or sim_folder=='-help' or sim_folder=='-h' or sim_folder=='-H':
            print('\n~~~ analysis.py help ~~~\nThis python script will do all simulation data analysis listed in the flags below.\n\nPlease enter the path to the simulation folder as the first argument to plot_snapshot.py\n\nTo do specfic analysis, please flag:\n -S (separate imp/tar particle analysis), \n -L (ang mom), \n -P (plot/save PNGs), \n -A (animate using ffmpeg), \n -I (resulting Uranus mass/radius and mass/composition of satellites).\n\nIf no flags are specified, all analysis will be performed (this may take a while!).')
            sys.exit()
    except Exception as err:
        print(err,"\nPlease enter the path to the simulation folder as the first argument to analysis.py")
        sys.exit()
    try:
        args = sys.argv[2:]
        valid = False   # used to make sure flags are correct in if statements below
    except Exception as err:
        print(f"No flags specified, running full analysis.")
        do_separate_particles = True
        do_ang_mom = True
        do_pngs = True
        do_animate = True
        do_uranus_satellites = True
    if len(args)==0:
        print(f"No flags specified, running full analysis.")
        do_separate_particles = True
        do_ang_mom = True
        do_pngs = True
        do_animate = True
        do_uranus_satellites = True
        valid = True
    if ('-S' in args) or ('-s' in args):
        do_separate_particles = True
        valid=True
    if ('-L' in args) or ('-l' in args):
        do_ang_mom = True
        valid=True
    if ('-P' in args) or ('-p' in args):
        do_pngs = True
        valid=True
    if ('-A' in args) or ('-a' in args):
        do_animate = True
        valid=True
    if ('-I' in args) or ('-i' in args):
        do_uranus_satellites = True
        valid=True

    # Check valid
    if not valid:
        print('Flags provided are invalid. Please use correct flags or run analysis.py --help for flag info.')
        sys.exit()
    if sim_folder[-1]!='/':
        sim_folder+='/'

    print('\n'+f'Simulation: {sim_folder}'+'\n'+f'Analysis specified: -S: {do_separate_particles}, -L: {do_ang_mom}, -P: {do_pngs}, -A: {do_animate}, -I: {do_uranus_satellites}'+'\n')

    ############ ^^^ ARGS STUFF ^^^ ##############
    
    # Get collision scenario from initial condition hdf5 basename
    scenario_str = get_hdf5_basename(sim_folder)
    scenario_str = scenario_str[:-5]
    angle_to_z = float(scenario_str.split('phi_')[1].split('_M_')[0])
    collision_num = int(scenario_str[0])


    # Open info.txt
    info_file = open(sim_folder + 'info_' + scenario_str + '.txt', "r")
    info = info_file.read()
    info_file.close()
    if verbose:
        print('~~ IC info ~~ \n'+info)
    M_i1_mearth = float(info.split('Mass of impactor 1: ')[1].split('M_earth')[0])
    M_i2_mearth = float(info.split('Mass of impactor 2: ')[1].split('M_earth')[0])
    d_rearth = float(info.split('Initial distance between target and impactor: ')[1].split('R_earth')[0])
    v_c = float(info.split('*v_esc = ')[1].split(' m/s')[0])
    t = float(info.split('Time to collision: t = ')[1].split(' s ')[0])

    # Silly formating stuff that I should've thought about when designing my file system but oh well
    mass_of_impactors_mearth = M_i1_mearth + M_i2_mearth
    if len(str(mass_of_impactors_mearth).split('.'))>1 and str(mass_of_impactors_mearth).split('.')[1]=='0':
        mass_of_impactors_mearth = int(mass_of_impactors_mearth)   

    if collision_num==2:
        try:
            obliquity_after_first_impact = float(info.split('Obliquity after first impact = ')[1].split(' deg')[0])
        except:
            obliquity_after_first_impact = float(info.split('Obliquity after first impact: ')[1].split(' deg')[0])
        phi_2 = float(info.split('Phi_2 = ')[1].split(' deg')[0])
        threshold_rho = float(info.split('Threshold rho = ')[1].split(' kg/m^3')[0])
        R_remnant = float(info.split('R_tar (calculated using threshold rho) = ')[1].split(' R_earth')[0])
        N_unbound = float(info.split('Number of unbound particles after impact 1 = ')[1].split(' ')[0])
    else:
        threshold_rho= -10  # will never be negative so use this as a flag for 1st / 2nd collision 

    # Running the analysis
    if collision_num==1 and (do_separate_particles or do_pngs or do_ang_mom):
        # Need to find out which particles belong to what body manually for the first impact 

        # Get radius of the proto-Uranus
        uranus_relax_info_file = open(f'/data/cluster4/oo21461/Planets/{mass_of_impactors_mearth}_uranus/relax_info.txt', "r")
        uranus_relax_info = uranus_relax_info_file.read()
        uranus_relax_info_file.close()
        #print(uranus_relax_info)
        R_uranus = float(uranus_relax_info.split('Estimated radius (averaging over outermost 100 particles) = ')[1].split('\n')[0]) / R_earth   # in units of R_earth
        
        uranus_ids,impactor_1_ids = separate_particles(sim_folder,d_rearth,R_uranus,verbose)
        impactor_2_ids=np.array([],dtype=int)   # empty array

    if collision_num==2 and (do_separate_particles or do_pngs or do_ang_mom):
        # Check the script that setup the 2nd impact has written the npy files of particle ids for each body
        if not os.path.exists(sim_folder+'particles_uranus.npy'):
            print(sim_folder+'particles_uranus.npy not found.')
            raise FileNotFoundError
        if not os.path.exists(sim_folder+'particles_impactor_1.npy'):
            print(sim_folder+'particles_impactor_1.npy not found.')
            raise FileNotFoundError 
        if not os.path.exists(sim_folder+'particles_impactor_2.npy'):
            print(sim_folder+'particles_impactor_2.npy not found.')
            raise FileNotFoundError
        uranus_ids = np.load(sim_folder+'particles_uranus.npy')
        impactor_1_ids = np.load(sim_folder+'particles_impactor_1.npy')
        impactor_2_ids = np.load(sim_folder+'particles_impactor_2.npy')


    if do_ang_mom:
        # This will also fail if separate_particles has not been used before
        try: # check if ang_mom and min_max files already exist
            ang_mom_arr = np.load(sim_folder + 'ang_mom_tilt.npy')
            print(ang_mom_arr[0])
            min_max_info = open(sim_folder + 'min_max_info.txt', "r")
            min_max_str = min_max_info.read()
            min_max_info.close()
            min_u = min_max_str.split('min_u=')[1].split('\n')[0]
            max_u = min_max_str.split('max_u=')[1].split('\n')[0]
            min_rho = min_max_str.split('min_rho=')[1].split('\n')[0]
            max_rho = min_max_str.split('max_rho=')[1].split('\n')[0]
            if verbose:
                print(f'Final angle to +Z = {ang_mom_arr[-1,6]} deg')
                print(f'Final period at equator = {ang_mom_arr[-1,7]} hours')
                print(f'Final obliquity = {ang_mom_arr[-1,8]} deg')
                print(f'Final L = {ang_mom_arr[-1,3:6]}')
        except Exception as err:
            if verbose:
                print('ang_mom_tilt.npy and / or min_max_info.txt not found. Looping through snapshots...')
            if collision_num == 1: # phi_1 is an int so can just use angle_to_z
                min_u, max_u, min_rho, max_rho, ang_mom_arr = ang_mom_tilt(sim_folder, t, uranus_ids, impactor_1_ids, impactor_2_ids, angle_to_z, verbose)
            elif collision_num==2: # unrounded phi_2 given by info txt 
                min_u, max_u, min_rho, max_rho, ang_mom_arr = ang_mom_tilt(sim_folder, t, uranus_ids, impactor_1_ids, impactor_2_ids, phi_2, verbose)


    if do_pngs:
        try: # check if ang_mom and min_max files already exist
            ang_mom_arr = np.load(sim_folder + 'ang_mom_tilt.npy')
            min_max_info = open(sim_folder + 'min_max_info.txt', "r")
            min_max_str = min_max_info.read()
            min_max_info.close()
            min_u = min_max_str.split('min_u=')[1].split('\n')[0]
            max_u = min_max_str.split('max_u=')[1].split('\n')[0]
            min_rho = min_max_str.split('min_rho=')[1].split('\n')[0]
            max_rho = min_max_str.split('max_rho=')[1].split('\n')[0]
        except Exception as err:
            if verbose:
                print('ang_mom_tilt.npy and / or min_max_info.txt not found. Looping through snapshots...')
            if collision_num == 1: # phi_1 is an int so can just use angle_to_z
                min_u, max_u, min_rho, max_rho, ang_mom_arr = ang_mom_tilt(sim_folder, t, uranus_ids, impactor_1_ids, impactor_2_ids, angle_to_z, verbose)
            elif collision_num==2: # unrounded phi_2 given by info txt 
                min_u, max_u, min_rho, max_rho, ang_mom_arr = ang_mom_tilt(sim_folder, t, uranus_ids, impactor_1_ids, impactor_2_ids, phi_2, verbose)
        
        if collision_num == 1: # phi_1 is an int so can just use angle_to_z, obliquity after first impact = 0
            pass
            pngs(sim_folder, t, uranus_ids, impactor_1_ids, impactor_2_ids, angle_to_z, min_u, max_u, min_rho, max_rho, ang_mom_arr, 0, plot_dict, verbose)
            #final_profiles(sim_folder, t, uranus_ids, impactor_1_ids, impactor_2_ids, angle_to_z, min_u, max_u, min_rho, max_rho, ang_mom_arr, 0, scenario_str, threshold_rho, verbose)
        elif collision_num == 2: # unrounded phi_2 given by info txt 
            R,uranus_internal_mass, imp1_internal_mass, imp2_internal_mass,uranus_external_mass,imp1_external_mass,imp2_external_mass,uranus_min_depth, imp1_min_depth, imp2_min_depth, uranus_outside_roche, imp1_outside_roche, imp2_outside_roche, roche_radius = final_profiles(sim_folder, t, uranus_ids, impactor_1_ids, impactor_2_ids, phi_2, min_u, max_u, min_rho, max_rho, ang_mom_arr, obliquity_after_first_impact, scenario_str, threshold_rho, verbose)
            pngs(sim_folder, t, uranus_ids, impactor_1_ids, impactor_2_ids, phi_2, min_u, max_u, min_rho, max_rho, ang_mom_arr, obliquity_after_first_impact, plot_dict, verbose)

    # THIS HAS BEEN COMMENTED OUT FOR THE REPORT ANALYSIS - UNCOMMENT IF YOU WANT TO ANIMATE
    # if do_animate:
    #     plot_trues = [key for key, value in plot_dict.items() if value]
    #     for plot_type in plot_trues:
    #         #os.system(f'cd {sim_folder}PNGs/')
    #         os.system(f'ffmpeg -r 20 -s 2560x1920 -i {sim_folder}PNGs/snapshot_{plot_type}_%04d.png -vcodec libx264 -pix_fmt yuv420p {sim_folder}PNGs/ani_{plot_type}.mp4')

    sys.exit()  # remove if you have to redo the analysis txts

    #Save analysis info text here
    to_write=f'''Remnant radius = {R/R_earth} R_earth
roche_radius = {roche_radius/R_earth} R_earth
uranus_outside_roche = {uranus_outside_roche/M_earth} M_earth
imp1_outside_roche = {imp1_outside_roche/M_earth} M_earth
imp2_outside_roche = {imp2_outside_roche/M_earth} M_earth
uranus_internal_mass = {uranus_internal_mass/M_earth} M_earth
imp1_internal_mass = {imp1_internal_mass/M_earth} M_earth
imp2_internal_mass = {imp2_internal_mass/M_earth} M_earth
uranus_external_mass = {uranus_external_mass/M_earth} M_earth
imp1_external_mass = {imp1_external_mass/M_earth} M_earth
imp2_external_mass = {imp2_external_mass/M_earth} M_earth
uranus_min_depth = {uranus_min_depth/M_earth} M_earth
imp1_min_depth = {imp1_min_depth/R_earth} R_earth
imp2_min_depth = {imp2_min_depth/R_earth} R_earth'''
    with open(sim_folder + 'analysis_info.txt', "w") as f:
        f.write(to_write)
    print(f'\nSaved {sim_folder}analysis_info.txt')



if __name__=='__main__':
    main()