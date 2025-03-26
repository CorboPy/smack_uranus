# This code opens up all txt info files related to each impactor scenario, prints key information and creates plots used in the report.
# I have not had time to make this code particularly readable  

import swiftsimio as sw
import sys
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.font_manager
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator
import multiprocessing
from multiprocessing import Pool, current_process
from scipy.signal import savgol_filter
import pickle
import unyt
import h5py
import time
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams["font.family"] = "Times New Roman"
import warnings

# Suppress DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from analysis import get_hdf5_basename

R_earth = 6.371e6   # m
M_earth = 5.9724e24  # kg
G = 6.67408e-11  # m^3 kg^-1 s^-2

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

if __name__=='__main__':
    root_folders = ['/data/echidna1/alex_corbett/SimulationsBC4/1_uranus/0.5-0.5', '/data/cluster4/oo21461/SimulationsBC4/1.5_uranus/0.5-1/','/data/echidna1/alex_corbett/SimulationsBC4/1.5_uranus/0.75-0.75' ] 
    
    # Big lists to catch all the stuff 
    big_list = []
    all_obliquities = []
    all_ang_moms = []
    flung_material = []
    flung_imp1_material = []
    flung_imp2_material = []
    flung_uranus_material = []
    all_vcs = []
    all_init_radii = []
    all_final_radii = []
    latex_table = ''
    for root_folder in root_folders:
        subfolders = [ f.path+'/' for f in os.scandir(root_folder) if f.is_dir() ]

        for sim_folder_1 in subfolders:
            # try:
            #     ang_mom_arr_1 = np.load(sim_folder_1 + 'ang_mom_tilt.npy')
            # except:
            #     print(f"ERROR: ang_mom_tilt.npy doesn't exist for {sim_folder_1}")
            #     continue

            subsubfolders_init = [ i.path+'/' for i in os.scandir(sim_folder_1) if (i.is_dir())]
            subsubfolders = []
            for folder in subsubfolders_init:
                num = folder.split('/')[-2].split('/'[-1])[0][0]=='2'
                if num:
                    subsubfolders.append(folder)

            not_included = []
            for j, sim_folder_2 in enumerate(subsubfolders):
                # print(f'Opened {sim_folder_2}')
                # exit_code = os.system(f'python analysis.py {sim_folder_2}')
                # print(exit_code, f' for {sim_folder_2}')
                # continue
                
                # Get collision scenario from initial condition hdf5 basename
                scenario_str = get_hdf5_basename(sim_folder_2)
                scenario_str = scenario_str[:-5]
                phi_1 = sim_folder_2.split('/')[7][2:5]

                try:
                    # Open ang mom array
                    ang_mom_arr_2 = np.load(sim_folder_2 + 'ang_mom_tilt.npy')
                except:
                    print(f"ERROR: ang_mom_tilt.npy doesn't exist for {sim_folder_2}")
                    not_included.append(sim_folder_2)
                    continue

                try:
                    # Open info.txt
                    info_file = open(sim_folder_2 + 'info_' + scenario_str + '.txt', "r")
                    info = info_file.read()
                    info_file.close()                
                except:
                    print(f"ERROR: info_{scenario_str}.txt doesn't exist for {sim_folder_2}")
                    not_included.append(sim_folder_2)
                    continue      

                # info.txt
                #print('~~ IC info ~~ \n'+info)
                M_i1_mearth = float(info.split('Mass of impactor 1: ')[1].split('M_earth')[0])
                M_i2_mearth = float(info.split('Mass of impactor 2: ')[1].split('M_earth')[0])
                d_rearth = float(info.split('Initial distance between target and impactor: ')[1].split('R_earth')[0])
                v_c = float(info.split('Collision velocity: ')[1].split('*v_esc')[0])
                t = float(info.split('Time to collision: t = ')[1].split(' s ')[0])
                try:
                    obliquity_after_first_impact = float(info.split('Obliquity after first impact = ')[1].split(' deg')[0])
                except:
                    obliquity_after_first_impact = float(info.split('Obliquity after first impact: ')[1].split(' deg')[0])
                phi_2 = float(info.split('Phi_2 = ')[1].split(' deg')[0])
                threshold_rho = float(info.split('Threshold rho = ')[1].split(' kg/m^3')[0])
                R_remnant = float(info.split('R_tar (calculated using threshold rho) = ')[1].split(' R_earth')[0])
                N_unbound = float(info.split('Number of unbound particles after impact 1 = ')[1].split(' ')[0])


                try:
                    # Open analysis_info.txt
                    info_file = open(sim_folder_2 + 'analysis_info.txt', "r")
                    info = info_file.read()
                    info_file.close()                
                except:
                    print(f"ERROR: analysis_info.txt doesn't exist for {sim_folder_2}")
                    not_included.append(sim_folder_2)
                    continue         
                
                # analysis_info.txt                         uranus_internal_mass = [ 1.0544189 10.763963   0.9205595] M_earth
                #print('~~ IC info ~~ \n'+info)
                R_rem = float(info.split('Remnant radius = ')[1].split('R_earth')[0])
                roche_radius =  float(info.split('roche_radius = ')[1].split('R_earth')[0])

                uranus_outside_roche =  np.fromstring(info.split('uranus_outside_roche = ')[1].split('M_earth')[0].strip("[]") , sep=" " )
                imp1_outside_roche = np.fromstring(info.split('imp1_outside_roche = ')[1].split('M_earth')[0].strip("[]") , sep=" " )
                imp2_outside_roche = np.fromstring(info.split('imp2_outside_roche = ')[1].split('M_earth')[0].strip("[]") , sep=" " )

                uranus_internal_mass = np.fromstring(info.split('uranus_internal_mass = ')[1].split('M_earth')[0].strip("[]") , sep=" " )
                imp1_internal_mass = np.fromstring(info.split('imp1_internal_mass = ')[1].split('M_earth')[0].strip("[]") , sep=" " )
                imp2_internal_mass = np.fromstring(info.split('imp2_internal_mass = ')[1].split('M_earth')[0].strip("[]") , sep=" " )
                uranus_external_mass = np.fromstring(info.split('uranus_external_mass = ')[1].split('M_earth')[0].strip("[]") , sep=" " )
                imp1_external_mass = np.fromstring(info.split('imp1_external_mass = ')[1].split('M_earth')[0].strip("[]") , sep=" " )
                imp2_external_mass = np.fromstring(info.split('imp2_external_mass = ')[1].split('M_earth')[0].strip("[]") , sep=" " )
                uranus_min_depth = np.fromstring(info.split('uranus_min_depth = ')[1].split('M_earth')[0].strip("[]") , sep=" " ) # error - was meant to write R_earth but oh weell
                imp1_min_depth = np.fromstring(info.split('imp1_min_depth = ')[1].split('M_earth')[0].strip("[]") , sep=" " ) # error - was meant to write R_earth but oh weell
                imp2_min_depth = np.fromstring(info.split('imp2_min_depth = ')[1].split('M_earth')[0].strip("[]") , sep=" " ) # error - was meant to write R_earth but oh weell

                #print(imp1_internal_mass, type(imp1_internal_mass))


                mass_of_impactors_mearth = M_i1_mearth + M_i2_mearth
                if len(str(mass_of_impactors_mearth).split('.'))>1 and str(mass_of_impactors_mearth).split('.')[1]=='0':
                    mass_of_impactors_mearth = int(mass_of_impactors_mearth)   
                loc_init = f'/data/cluster4/oo21461/Planets/{mass_of_impactors_mearth}_uranus/relax_info.txt'
                try:
                    # open init uranus
                    init_info_file = open(loc_init, "r")
                    init_info = init_info_file.read()
                    init_info_file.close()                  
                except:
                    print(f"ERROR: cannot open {loc_init}")
                    sys.exit()  
                R_init = float(init_info.split('Estimated radius (averaging over outermost 100 particles) = ')[1].split('\n')[0])/ R_earth

                if np.sum(imp2_internal_mass)/M_i2_mearth > np.sum(imp1_internal_mass)/M_i1_mearth:
                    print(np.sum(imp2_internal_mass)/M_i2_mearth,'More imp2 than imp1 ', np.sum(imp1_internal_mass)/M_i1_mearth)
                else:
                    print('More imp1 than imp2')

                ## Derotate such that +Z is in the direction of Uranus' orbital angular momentum
                # Rotation matrix. This is a counter-clockwise rotation so angle is +ve
                R_matrix = np.array([
                    [1, 0, 0],
                    [0, np.cos(np.deg2rad(float(phi_2))), -np.sin(np.deg2rad(float(phi_2)))],
                    [0, np.sin(np.deg2rad(float(phi_2))), np.cos(np.deg2rad(float(phi_2)))]
                ])
                #print('Before: ',l_norm)
                # Transform positions and velocities oppositely (derotate) 
                l = ang_mom_arr_2[-1,3:6] @ R_matrix.T                  # Rotate L

                print('Success for ',sim_folder_2)

                total= np.sum(uranus_outside_roche) + np.sum(imp1_outside_roche) + np.sum(imp2_outside_roche)

                flung_imp1_material.append(np.array(imp1_outside_roche))
                flung_imp2_material.append(np.array(imp2_outside_roche))
                flung_uranus_material.append(np.array(uranus_outside_roche))
                flung_material.append(total)
                all_vcs.append(v_c)
                big_list.append(sim_folder_2)
                all_obliquities.append(ang_mom_arr_2[-1,8]) 
                all_ang_moms.append(l)
                all_init_radii.append(R_init)
                all_final_radii.append(R_rem)
                phantom = r'\phantom{X}'
                latex_table+= f'{M_i1_mearth} & {phi_1} & {M_i2_mearth} & {round(phi_2,2)} & {v_c} & {phantom} & {round(obliquity_after_first_impact,2)} & {round(np.abs(ang_mom_arr_2[-1,8]),2)} & {round(np.linalg.norm(l)/10**36,2)} & {round(total*100,2)} & {round(R_rem,2)} {str(r'\\')}\n'
                #sys.exit()

    #print('DONE!')
    #print(all_obliquities)
    #print(latex_table)
    #sys.exit()
 
    big_list = np.array(big_list)
    all_ang_moms = np.array(all_ang_moms)
    all_obliquities = np.array(all_obliquities)
    all_vcs = np.array(all_vcs)
    flung_material = np.array( flung_material )
    flung_uranus_material = np.array(flung_uranus_material)
    flung_imp1_material = np.array(flung_imp1_material)
    flung_imp2_material = np.array(flung_imp2_material)
    all_init_radii = np.array(all_init_radii)
    all_final_radii = np.array(all_final_radii)

    rock_ice_hhe = np.zeros((len(big_list),3))
    for i in range(len(big_list)):
        rock = 0
        rock += flung_uranus_material[i,0]
        rock += flung_imp1_material[i,0]
        rock += flung_imp2_material[i,0]

        ice = 0
        ice += flung_uranus_material[i,1]
        ice += flung_imp1_material[i,1]
        ice += flung_imp2_material[i,1]

        hhe = flung_uranus_material[i,2]

        rock_ice_hhe[i,0] = rock
        rock_ice_hhe[i,1] = ice
        rock_ice_hhe[i,2] = hhe

    
    #print(rock_ice_hhe)

    

    #print(all_ang_moms)

    # mass masks
    mask_05_05 = ['0.5-0.5' in a for a in big_list]
    mask_05_1 = ['0.5-1' in a for a in big_list]
    mask_075_075 = ['0.75-0.75' in a for a in big_list]
    # vel masks
    mask_v1 = np.array([a == 1 for a in all_vcs])
    mask_v15 = np.array([a == 1.5 for a in all_vcs])

    #index_closest = np.where(big_list == '/data/echidna1/alex_corbett/SimulationsBC4/1.5_uranus/0.75-0.75/1_135/2_130.33/')[0][0]
    #print(index_closest)
    # boolean_mask = string_mask & float_mask

    # Closest orbiting mass to Uranus moons today
    mass_of_moons = ( (0.66 + 12.9 + 12.2 + 34.2 + 28.8) * 10 **20 ) / M_earth
    closest_flung_material = np.min(flung_material[flung_material > mass_of_moons])
    print(f'Closest flung material to Uranus today: {big_list[flung_material == closest_flung_material]} at a mass of {closest_flung_material} compared to {mass_of_moons} for Uranus moons today')
    closest_material_index = np.where(flung_material == closest_flung_material)[0][0]
    

    # Radius increase
    percentage_increase = (all_final_radii - all_init_radii) / all_init_radii
    print('Average increase in radius (%): ', np.mean(percentage_increase) * 100)
    percentage_increase_1 = (all_final_radii[mask_v1] - all_init_radii[mask_v1]) / all_init_radii[mask_v1]
    print('Average increase in radius (%): ', np.mean(percentage_increase_1) * 100)
    percentage_increase_15 = (all_final_radii[mask_v15] - all_init_radii[mask_v15]) / all_init_radii[mask_v15]
    print('Average increase in radius (%): ', np.mean(percentage_increase_15) * 100)

    max_extent = np.max(np.abs(all_ang_moms))

    # L of uranus today
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

    # FIND CLOSEST RESULT!
    distances = np.linalg.norm(all_ang_moms - l_uranus_today, axis=1)
    closest_index = np.argmin(distances)
    #print('CLOSEST: ',big_list[closest_index],f'at a distance of \n{np.min(distances)} kg m^2 s^-1')
    index_closest = np.where(big_list == big_list[closest_index])[0][0]
    #print(index_closest)

    # 0.5-0.5
    distances_05_05 = distances[mask_05_05]
    min_distance_05_05 = np.min(distances_05_05)
    index_closest_05_05 = np.where(distances == min_distance_05_05)[0][0]
    print('CLOSEST 05-05: ',big_list[index_closest_05_05],f'at a distance of \n{min_distance_05_05} kg m^2 s^-1')

    # flung_material_05_05 = flung_material[mask_05_05]
    # max_flung_material_05_05 = np.max(flung_material_05_05)
    # index_material_05_05 = np.where(flung_material == max_flung_material_05_05)[0][0]
    # print(f'05-05: Sim with max flung material is {big_list[index_material_05_05]} at a mass of {max_flung_material_05_05} M_earth')

    condition = flung_material > mass_of_moons
    closest_flung_material = np.min(flung_material[mask_05_05 & condition])
    print(f'05-05: Closest flung material to Uranus today: {big_list[flung_material == closest_flung_material]} at a mass of {closest_flung_material} compared to {mass_of_moons} for Uranus moons today')
    index_material_05_05 = np.where(flung_material == closest_flung_material)[0][0]
    print(rock_ice_hhe[:,1]/rock_ice_hhe[:,2]) # ice to atmosphere ratio (ICE / ATMOSPHERE)

    big = '/data/cluster4/oo21461/SimulationsBC4/1.5_uranus/0.5-1/1_155_1.5/2_127.58/'
    big_index = np.where(big_list == big)[0][0]
    print(rock_ice_hhe[big_index,1]/rock_ice_hhe[big_index,2])

    print(np.mean(rock_ice_hhe[mask_05_1 & mask_v15,1]/rock_ice_hhe[mask_05_1 & mask_v15,2]))
    print(np.mean(rock_ice_hhe[mask_075_075 & mask_v15,1]/rock_ice_hhe[mask_075_075 & mask_v15,2]))
    print(np.mean(rock_ice_hhe[mask_05_05 & mask_v15,1]/rock_ice_hhe[mask_05_05 & mask_v15,2]))

    #sys.exit()

    # 0.5-0.5 v_c = v_m
    distances_05_05_v1 = distances[mask_05_05 & mask_v1]
    min_distance_05_05_v1 = np.min(distances_05_05_v1)
    index_closest_05_05_v1 = np.where(distances == min_distance_05_05_v1)[0][0]
    phi_needed = (360/(2*np.pi))* np.arccos( np.dot([0,0,1],all_ang_moms[index_closest_05_05_v1,:]) / ( np.linalg.norm([0,0,1]) * np.linalg.norm(all_ang_moms[index_closest_05_05_v1,:]) ) )
    #print('CLOSEST 05-05 v_c=v_m: ',big_list[index_closest_05_05_v1],f'at a distance of \n{min_distance_05_05_v1} kg m^2 s^-1. Phi_needed = {phi_needed}')

    # 0.75-0.75
    distances_075_075 = distances[mask_075_075]
    min_distance_075_075 = np.min(distances_075_075)
    index_closest_075_075 = np.where(distances == min_distance_075_075)[0][0]
    print('CLOSEST 075-075: ',big_list[index_closest_075_075],f'at a distance of \n{min_distance_075_075} kg m^2 s^-1')

    # flung_material_075_075 = flung_material[mask_075_075]
    # max_flung_material_075_075 = np.max(flung_material_075_075)
    # index_material_075_075 = np.where(flung_material == max_flung_material_075_075)[0][0]
    # print(f'075-075: Sim with max flung material is {big_list[index_material_075_075]} at a mass of {max_flung_material_075_075} M_earth')

    closest_flung_material = np.min(flung_material[mask_075_075 & condition])
    print(f'075-075: Closest flung material to Uranus today: {big_list[flung_material == closest_flung_material]} at a mass of {closest_flung_material} compared to {mass_of_moons} for Uranus moons today')
    index_material_075_075 = np.where(flung_material == closest_flung_material)[0][0]

    # 0.5-1
    distances_05_1 = distances[mask_05_1]
    min_distance_05_1 = np.min(distances_05_1)
    index_closest_05_1 = np.where(distances == min_distance_05_1)[0][0]
    print('CLOSEST 05-1: ',big_list[index_closest_05_1],f'at a distance of \n{min_distance_05_1} kg m^2 s^-1')

    # flung_material_05_1 = flung_material[mask_05_1]
    # max_flung_material_05_1 = np.max(flung_material_05_1)
    # index_material_05_1 = np.where(flung_material == max_flung_material_05_1)[0][0]
    # print(f'05-1: Sim with max flung material is {big_list[index_material_05_1]} at a mass of {max_flung_material_05_1} M_earth')

    closest_flung_material = np.min(flung_material[condition & mask_05_1])
    print(f'05-1: Closest flung material to Uranus today: {big_list[flung_material == closest_flung_material]} at a mass of {closest_flung_material} compared to {mass_of_moons} for Uranus moons today')
    index_material_05_1 = np.where(flung_material == closest_flung_material)[0][0]

    ## Get initial ang mom of each uranus 
    # FOR 1 URANUS
    phi_1 = 155     # arbitrary
    pos_init, vel_init, h_init, m_init, rho_init, p_init, u_init, matid_init, parids_init, R_init, pots_init = load_to_woma(f'/data/cluster4/oo21461/Planets/1_uranus/chuck_in_swift/spunup_{phi_1}/output/swift_spunup_{phi_1}_0001.hdf5') # Open initial uranus
    total_atmosphere_1 = np.sum(m_init[matid_init == 200]) / M_earth
    l_init, l_init_norm, L_init_angle_to_z = ang_mom(pos_init,vel_init,m_init,matid_init,0)
    print(f'This should be close to {phi_1}: ',L_init_angle_to_z)
    R_matrix = np.array([
    [1, 0, 0],
    [0, np.cos(np.deg2rad(float(phi_1))), -np.sin(np.deg2rad(float(phi_1)))],
    [0, np.sin(np.deg2rad(float(phi_1))), np.cos(np.deg2rad(float(phi_1)))]
])    
    l_init_1 = l_init @ R_matrix.T    # This should now be pointing in the +Z direction
    #l_1 = l - l_init   # change in L

    # FOR 1.5 URANUS
    phi_1 = 155     # arbitrary
    pos_init, vel_init, h_init, m_init, rho_init, p_init, u_init, matid_init, parids_init, R_init, pots_init = load_to_woma(f'/data/cluster4/oo21461/Planets/1.5_uranus/chuck_in_swift/spunup_{phi_1}/output/swift_spunup_{phi_1}_0001.hdf5') # Open initial uranus
    total_atmosphere_15 = np.sum(m_init[matid_init == 200]) / M_earth
    l_init, l_init_norm, L_init_angle_to_z = ang_mom(pos_init,vel_init,m_init,matid_init,0)
    print(f'This should be close to {phi_1}: ',L_init_angle_to_z)
    R_matrix = np.array([
    [1, 0, 0],
    [0, np.cos(np.deg2rad(float(phi_1))), -np.sin(np.deg2rad(float(phi_1)))],
    [0, np.sin(np.deg2rad(float(phi_1))), np.cos(np.deg2rad(float(phi_1)))]
])    
    l_init_15 = l_init @ R_matrix.T    # This should now be pointing in the +Z direction
    #l_1 = l - l_init   # change in L
    #sys.exit()

    # Atmosphere loss
    print(f'Percentage of flung atmosphere for 05_05 v1: {100* np.mean( rock_ice_hhe[mask_05_05 & mask_v1,2] / total_atmosphere_1 )}')
    print(f'Percentage of flung atmosphere for 05_05 v15: {100*np.mean( rock_ice_hhe[mask_05_05 & mask_v15,2] / total_atmosphere_1 )}')

    print(f'Percentage of flung atmosphere for 05_1 v1: {100*np.mean( rock_ice_hhe[mask_05_1 & mask_v1,2] / total_atmosphere_15 )}')
    print(f'Percentage of flung atmosphere for 05_1 v15: {100*np.mean( rock_ice_hhe[mask_05_1 & mask_v15,2] / total_atmosphere_15 )}')

    print(f'Percentage of flung atmosphere for 075_075 v1: {100*np.mean( rock_ice_hhe[mask_075_075 & mask_v1,2] / total_atmosphere_15 )}')
    print(f'Percentage of flung atmosphere for 075_075 v15: {100*np.mean( rock_ice_hhe[mask_075_075 & mask_v15,2] / total_atmosphere_15 )}')


    #sys.exit()

    # Plot step histograms
    fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(4.5,3))

    # mask_05_05 
    # mask_05_1 
    # mask_075_075 

    max_x_extent = np.max(flung_material)
    mass_v1_05_05_ = flung_material[mask_v1 & mask_05_05] 
    mass_v15_05_05_ = flung_material[mask_v15 & mask_05_05] 
    mass_v1_05_05 = mass_v1_05_05_[mass_v1_05_05_ != 0]
    mass_v15_05_05 = mass_v15_05_05_[mass_v15_05_05_ != 0]

    mass_v1_05_1_ = flung_material[mask_v1 & mask_05_1] 
    mass_v15_05_1_ = flung_material[mask_v15 & mask_05_1] 
    mass_v1_05_1 = mass_v1_05_1_[mass_v1_05_1_ != 0]
    mass_v15_05_1 = mass_v15_05_1_[mass_v15_05_1_ != 0]

    mass_v1_075_075_ = flung_material[mask_v1 & mask_075_075] 
    mass_v15_075_075_ = flung_material[mask_v15 & mask_075_075] 
    mass_v1_075_075 = mass_v1_075_075_[mass_v1_075_075_ != 0]
    mass_v15_075_075 = mass_v15_075_075_[mass_v15_075_075_ != 0]

    # for i in range(len(flung_material[flung_material < 0.01])):
    #     print(big_list[flung_material < 0.01][i], ' with mass ',flung_material[flung_material < 0.01][i])
    # #print(big_list[flung_material < 0.01],flung_material[flung_material < 0.01])

    # Define bins
    # bins_1 = np.linspace(min(mass_v1_075_075.min(), mass_v1_05_1.min(), mass_v1_05_05.min()), max(mass_v1_075_075.max(), mass_v1_05_1.max(), mass_v1_05_05.max()), 50)
    # bins_15 = np.linspace(min(mass_v15_05_05.min(), mass_v15_05_1.min(), mass_v15_075_075.min()), max(mass_v15_05_05.max(), mass_v15_05_1.max(), mass_v15_075_075.max()), 50)
    bins_1 = np.linspace(0, 0.040, 40)
    bins_15 = np.linspace(0, 0.040, 40)

    # print(f'{max(mass_v1_075_075.max(), mass_v1_05_1.max(), mass_v1_05_05.max()):3e}')
    # print(f'{max(mass_v15_05_05.max(), mass_v15_05_1.max(), mass_v15_075_075.max())}')
    
    # ax[0].hist(mass_v1_05_1, bins=bins_1, histtype='step', label=r"$\beta$", linewidth=2,color='red')
    # ax[0].hist(mass_v1_075_075, bins=bins_1, histtype='step', label=r"$\gamma$", linewidth=2,color='blue')
    # ax[0].hist(mass_v1_05_05, bins=bins_1, histtype='step', label=r"$\alpha$", linewidth=2,color='black')
    
    ax.hist(mass_v15_05_1, bins=bins_15, histtype='step', label=r"$\alpha$", linewidth=1,color='black')
    ax.hist(mass_v15_075_075, bins=bins_15, histtype='step', label=r"$\beta$", linewidth=1,color='blue')
    ax.hist(mass_v15_05_05, bins=bins_15, histtype='step', label=r"$\gamma$", linewidth=1,color='red')

    print('alpha ',len(mass_v15_05_1))
    print('beta ',len(mass_v15_075_075))
    print('gamma ',len(mass_v15_05_05))

    # Labels and title
    ax.set_xlabel(r"Mass Beyond Roche Radius [M$_{\oplus}$]")
    ax.set_ylabel("Number of simulations")
    ax.set_xlim(0,0.04)

    # need custom legend
    alpha_handle = mlines.Line2D([], [], color='black', linestyle='-', linewidth=1.5, label=r'$\alpha$')
    beta_handle = mlines.Line2D([], [], color='blue', linestyle='-', linewidth=1.5, label=r'$\beta$')
    gamma_handle = mlines.Line2D([], [], color='red', linestyle='-', linewidth=1.5, label=r'$\gamma$')
    ax.legend(handles=[alpha_handle, beta_handle, gamma_handle], loc='upper right')
    #plt.grid(True)

    # Show the plot
    fig.tight_layout()
    figname = f'/home/oo21461/Documents/tools/flung_mass_hist.png'    #f'/home/oo21461/Documents/tools/final_profiles.png'
    fig.savefig(figname,dpi=800)
    plt.close()
    print(figname + ' saved.\n')



    # # material flung out histograms
    # fig, ax = plt.subplots()

    # # Sort by flung material
    # sorted_indices = np.argsort(flung_material_v1)
    # sorted_flung_material_v1 = flung_material_v1[sorted_indices]
    # sorted_indices = np.argsort(flung_material_v15)
    # sorted_flung_material_v15 = flung_material_v15[sorted_indices]

    # # Define bin edges (e.g., every 0.25 M_earth)
    # bin_edges = np.arange(0, max_x_extent + 0.005, 0.005)

    # # Compute total mass in each bin
    # bin_indices_v1 = np.digitize(sorted_flung_material_v1, bin_edges)  # Assign particles to bins
    # binned_v1 = np.array([np.sum(sorted_m[bin_indices_v1 == i]) for i in range(1, len(bin_edges))])

    # # Extend arrays to close the envelope
    # bin_edges_extended = np.append(bin_edges, bin_edges[-1])  # Repeat last bin edge
    # binned_mass_extended = np.append(binned_mass, 0)  # Ensure it drops to zero

    # ax[0].step(bin_edges_extended/R_earth, np.append([0], binned_mass_extended)/M_earth, where='pre', linewidth=1, color=colours[i],zorder=zorders[i],linestyle=linestyles[i],label=labels[i])


    # ax.step(bin_edges_extended, proportion_extended, where='pre', linewidth=1.5, color='black')    
    # ax.set_xlim(0, max_x_extent)
    # ax.set_ylim(0, 1)
    # #ax[3].set_ylabel(r"Smoothing Length [R$_{\oplus}$]")
    # ax.set_ylabel(r"Proportion of particles with h$_{\text{max}}$")
    # ax.set_xlabel(r"Radius [R$_{\oplus}$]")
    # ax.axvline(R/R_earth,ymin=0,ymax=1,color='b',linestyle='solid',linewidth=1,zorder=2000)   
    # ax.axvline(roche_radius/R_earth,ymin=0,ymax=1,color='r',linestyle='solid',linewidth=1,zorder=2000)       
    
    # fig.tight_layout()
    # figname = f'{sim_folder}final_h_bins.png'    #f'/home/oo21461/Documents/tools/final_profiles.png'
    # fig.savefig(figname,dpi=500)
    # plt.close()
    # print(figname + ' saved.\n')









    # divide by e36
    all_ang_moms/= 1e36
    l_init_15 /= 1e36
    l_init_1 /= 1e36
    l_uranus_today /= 1e36

    # Three differentsub plots for ang mom
    fig, ax = plt.subplots(nrows=3,ncols=1,figsize=(3.52,8),sharex=True) #4, 6.5
    ax[0].set_xlim(1.75, -0.75)
    ax[0].set_ylim(-2, 0.5)
    ax[1].set_ylim(-2, 0.5)
    ax[2].set_ylim(-2, 0.5)
    ax[0].set_aspect('equal', 'box')
    ax[1].set_aspect('equal', 'box')
    ax[2].set_aspect('equal', 'box')
    ax[2].set_xticks([1.5,1,0.5,0,-0.5])
    ax[0].text(-0.4,0.1,r'$\alpha$',fontsize=20)
    ax[1].text(-0.4,0.1,r'$\beta$',fontsize=20)
    ax[2].text(-0.4,0.1,r'$\gamma$',fontsize=20)

    ax[0].set_ylabel(r"S$_y$ [10$^{36}$ kg m$^2$ s$^{-1}$]")
    ax[1].set_ylabel(r"S$_y$ [10$^{36}$ kg m$^2$ s$^{-1}$]")
    ax[2].set_ylabel(r"S$_y$ [10$^{36}$ kg m$^2$ s$^{-1}$]")
    ax[2].set_xlabel(r"S$_z$ [10$^{36}$ kg m$^2$ s$^{-1}$]")

    # 0.5-1
    ax[0].scatter(all_ang_moms[mask_05_1 & mask_v1,2],all_ang_moms[mask_05_1 & mask_v1,1],marker='x',color='black')
    ax[0].scatter(all_ang_moms[mask_05_1 & mask_v15,2],all_ang_moms[mask_05_1 & mask_v15,1],marker='.',color='black')
    ax[0].arrow(0,0,l_init_15[2],l_init_15[1],color='#C00000',head_width=0.05,length_includes_head=True)
    ax[0].arrow(0,0,l_uranus_today[2],l_uranus_today[1],color='dodgerblue',head_width=0.05,length_includes_head=True,zorder=0)
    ax[0].scatter(all_ang_moms[index_closest_05_1,2],all_ang_moms[index_closest_05_1,1],marker='x',color='orange',zorder=10)
    ax[0].scatter(all_ang_moms[index_material_05_1,2],all_ang_moms[index_material_05_1,1],marker='.',color='#C00000',zorder=10)


    # 0.75-0.75
    ax[1].scatter(all_ang_moms[mask_075_075 & mask_v1,2],all_ang_moms[mask_075_075 & mask_v1,1],marker='x',color='black')
    ax[1].scatter(all_ang_moms[mask_075_075 & mask_v15,2],all_ang_moms[mask_075_075 & mask_v15,1],marker='.',color='black')
    ax[1].arrow(0,0,l_init_15[2],l_init_15[1],color='#C00000',head_width=0.05,length_includes_head=True)
    ax[1].arrow(0,0,l_uranus_today[2],l_uranus_today[1],color='dodgerblue',head_width=0.05,length_includes_head=True,zorder=0)
    ax[1].scatter(all_ang_moms[index_closest_075_075,2],all_ang_moms[index_closest_075_075,1],marker='x',color='orange',zorder=10)
    ax[1].scatter(all_ang_moms[index_material_075_075,2],all_ang_moms[index_material_075_075,1],marker='.',color='#C00000',zorder=10)


    # 0.5-0.5
    ax[2].scatter(all_ang_moms[mask_05_05 & mask_v1,2],all_ang_moms[mask_05_05 & mask_v1,1],marker='x',color='black',label=r'1 v$_{\text{esc}}$')
    ax[2].scatter(all_ang_moms[mask_05_05 & mask_v15,2],all_ang_moms[mask_05_05 & mask_v15,1],marker='.',color='black',label=r'1.5 v$_{\text{esc}}$')
    ax[2].arrow(0,0,l_init_1[2],l_init_1[1],color='#C00000',head_width=0.05,length_includes_head=True,label='init')
    ax[2].arrow(0,0,l_uranus_today[2],l_uranus_today[1],color='dodgerblue',head_width=0.05,length_includes_head=True,label='tar',zorder=0)
    ax[2].scatter(all_ang_moms[index_closest_05_05,2],all_ang_moms[index_closest_05_05,1],marker='.',color='orange',zorder=10)
    ax[2].scatter(all_ang_moms[closest_material_index,2],all_ang_moms[closest_material_index,1],marker='.',color='#C00000',zorder=10)


    #custom legend
# Create custom arrow patches
    arrow_init = mlines.Line2D([], [], color='#C00000', marker='>', markersize=8, linestyle='-', linewidth=2, label=r'$\vec{S}_u^{init}$',markevery=[1])
    arrow_tar = mlines.Line2D([], [], color='dodgerblue', marker='>', markersize=8, linestyle='-', linewidth=2, label=r'$\vec{S}_u^{tar}$',markevery=[1])
    scatter_v1 = mlines.Line2D([], [], color='black', marker='x', linestyle='None', label=r'1 v$_{{m}}$')
    scatter_v15 = mlines.Line2D([], [], color='black', marker='.', linestyle='None', label=r'1.5 v$_{{m}}$')
    #scatter_closest = mlines.Line2D([], [], color='orange', marker='.', linestyle='None', label=r'Closest to $\vec{S}_u^{tar}$')
    ax[0].legend(handles=[scatter_v1, scatter_v15, arrow_init, arrow_tar], loc='lower left')



    #print(all_obliquities,all_ang_moms)
    # need to decide what to group by - how to plot?
    # 3 different plots for 0.5-0.5, 0.5-1, 0.75-0.75
    # colour = phi
    # marker type = velocity

    fig.tight_layout()
    path = '/home/oo21461/Documents/tools/'
    figname = path + 'final_ang_moms.pdf'
    plt.savefig(figname,dpi=800)
    print(figname + ' saved.\n')