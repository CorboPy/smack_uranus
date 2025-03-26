# Sanity check angular momentum

import h5py
import sys
import math
import woma
import numpy as np
import os
import swiftsimio as sw
import unyt
from tqdm import tqdm
import matplotlib.pyplot as plt
from analysis import separate_particles

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
    parids =  data.gas.particle_ids.to_ndarray()


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
    
    return pos, vel, h, m, rho, p, u, matid, R, snap_time, parids

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
    parids = f['PartType0']['ParticleIDs'][()]


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

    return positions, velocities, masses, ids, rhos, internal_energies, R, parids

def compute_angular_momentum(masses, positions, velocities):
    """Compute the total angular momentum of the system of particles."""
    angular_momentum = np.sum(
        masses[:,np.newaxis] * np.cross(positions, velocities), axis=0
    )
    return angular_momentum

def compute_moi(masses,positions):
    mag_rs = np.linalg.norm(positions, axis=1)
    #print(mag_rs)
    I = np.sum(masses * mag_rs * mag_rs)
    return I

def plot_vels(pos,vel,R,path,angle,ids,verbose=True):
    plt.style.use('default')
    fig = plt.figure(figsize=(8,4))
    gs = fig.add_gridspec(1,2)

    atmosphere = np.where(ids == 200)
    mantle = np.where(ids == 900)
    core = np.where(ids == 400)
    theta = np.radians(angle)  # Convert angle to radians
    pos = pos/R_earth

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

    if verbose:
        # Print out periods
        print(np.shape(r),np.shape(v_norm))
        periods = (2*np.pi*R) * np.divide(r*R_earth,v_norm)
        print('Average period: ',np.mean(periods)/3600, 'h')
        print('Std dev: ',np.std(periods)/3600 ,'h')
        print('Max period: ',np.max(periods)/3600)
        print('Min period: ',np.min(periods)/3600)

    interval = 1e0
    ax1 = fig.add_subplot(gs[0,0])
    ax1.scatter(r[core][::int(interval)],v_norm[core][::int(interval)],marker='.',s=1,label='ANEOS_forsterite',zorder=2)
    ax1.scatter(r[mantle][::int(interval)],v_norm[mantle][::int(interval)],marker='.',s=1,label='AQUA',zorder=1)
    ax1.scatter(r[atmosphere][::int(interval)],v_norm[atmosphere][::int(interval)],marker='.',s=1,label='HM80_HHe',zorder=0)
    plt.legend() 
    ax1.set_xlabel(r"Radius, $r$ $[R_{Earth}]$")
    ax1.set_ylabel(r"Velocity, $v$ [m/s]")
    ax1.set_xlim(0, None)
    ax1.set_ylim(0, None)
    ax1.set_title(f'v vs r from origin (inclined at {angle:.2f} deg)')

    pos_inclined = pos_inclined[::int(1e3)]
    vel_inclined = vel_inclined[::int(1e3)]

    #v_xy = vel[:,0:2]
    norm = np.power(np.add(np.power(vel_inclined[:,0:1],2), np.power(vel_inclined[:,1:2],2)),0.5)
    ax2 = fig.add_subplot(gs[0,1])
    ax2.quiver(pos_inclined[:,0:1],pos_inclined[:,1:2], vel_inclined[:,0:1]/norm, vel_inclined[:,1:2]/norm)
    ax2.set_xlabel(r"$x$ $[R_{Earth}]$")
    ax2.set_ylabel(r"$y$ $[R_{Earth}]$")
    ax2.set_xlim( - (R/R_earth+1) , (R/R_earth+1) )
    ax2.set_ylim( - (R/R_earth+1) , (R/R_earth+1) )
    ax2.set_title(f'v vectors inclined at {angle:.2f} degrees')

    plt.tight_layout()
    figname = path+'test_vels.png'
    fig.savefig(figname,dpi=400)
    plt.close()

def plot_densities(pos,rho,angle,ids,path):
    atmosphere = np.where(ids == 200)
    mantle = np.where(ids == 900)
    core = np.where(ids == 400)
    theta = np.radians(angle)  # Convert angle to radians
    pos = pos/R_earth
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
    plt.style.use('default')
    ax = plt.gca()
    ax.scatter(r[core],rho[core],marker='.',s=1,label='ANEOS_forsterite',zorder=2)
    ax.scatter(r[mantle],rho[mantle],marker='.',s=1,label='AQUA',zorder=1)
    ax.scatter(r[atmosphere],rho[atmosphere],marker='.',s=1,label='HM80_HHe',zorder=0)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    #ax.set_yscale('log')
    ax.set_xlabel(r"Radius, $r$ $[R_{Earth}]$")
    ax.set_ylabel(r"Density, $\rho$ [arbitrary units]")

    plt.legend()
    plt.tight_layout()
    figname = path+'test_rhos.png'
    plt.savefig(figname,dpi=400)
    plt.close()

def period_at_equator(pos,vel,R,angle,ids):
    theta = np.radians(angle)  # Convert angle to radians
    atmosphere = np.where(ids == 200)
    mantle = np.where(ids == 900)
    core = np.where(ids == 400)


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

    print(R)
    print(np.mean(r_100))
    print(np.mean(v_100))
    print(f'Period at equator = {( ((2*np.pi*R)/np.mean(v_100)) )/ 3600}')


#loc=f'/home/oo21461/Documents/tools/tests/spunup_0.hdf5'
#loc_end = '/home/oo21461/Documents/tools/tests/chuck_in_swift/output/swift_spunup_110_0001.hdf5'
loc = "/home/oo21461/Documents/tools/tests/73_1_demo_impact_n60_0000.hdf5"
loc_end = '/home/oo21461/Documents/tools/tests/73_1_demo_impact_n60_442222_0000.hdf5'
loc_init = '/home/oo21461/Documents/tools/tests/73_1_demo_impact_n60.hdf5'

# pos, vel, m, ids, rho, u, R = load_hdf5(loc)
# pos_end, vel_end, m_end, ids_end, rho_end, u_end, R_end = load_hdf5(loc_end) # load in final snapshot to compare angular velocities as a func of r (plot_vels)
pos_init, vel_init, m_init, ids_init, rho_init, u_init, R_init, parids_init = load_hdf5(loc_init ) 
pos, vel, h, m, rho, p, u, ids, R, snap_time, parids = load_to_woma(loc)
pos_end, vel_end, h_end, m_end, rho_end, p_end, u_end, ids_end, R_end, snap_time_end, parids_end = load_to_woma(loc_end)

print('Uranus selected (mass of impactors): ', round(14.5 - np.sum(m)/M_earth))
print('Uranus mass (M_earth): ',np.sum(m)/M_earth)
print('Example pos / vel: ',pos[10:11:],vel[10:11:])
#print(np.mean(u)/1e7,np.mean(u_end)/1e7,(np.mean(u_end)-np.mean(u))/1e7)

# Remove later - uranus ids for lauren
uranus_ids,impactor_ids = separate_particles(loc,18,3.98,True)
indicies_uranus = np.where(np.isin(parids, uranus_ids))[0]
indicies_impactor = np.where(np.isin(parids, impactor_ids))[0]

# COM correction for Lauren
pos_centerM = np.sum(pos[indicies_uranus] * m[indicies_uranus,np.newaxis], axis=0) / np.sum(m[indicies_uranus])
vel_centerM = np.sum(vel[indicies_uranus] * m[indicies_uranus,np.newaxis], axis=0) / np.sum(m[indicies_uranus])

pos_uranus = pos[indicies_uranus]
vel_uranus = vel[indicies_uranus]
m_uranus = m[indicies_uranus]
print(f'NUMBER OF URANUS: {len(m_uranus)}')
print(f'NUMBER OF IMPACTOR: {len(m[indicies_impactor])}')

pos_uranus -= pos_centerM
vel_uranus -= vel_centerM

print('AVERAGE POS:',np.mean(pos_uranus))

# Remove later - COM correction for Lauren
indicies_uranus_end = np.where(np.isin(parids_end, uranus_ids))[0]
indicies_impactor_end = np.where(np.isin(parids_end, impactor_ids))[0]

pos_centerM = np.sum(pos_end[indicies_uranus_end] * m_end[indicies_uranus_end,np.newaxis], axis=0) / np.sum(m_end[indicies_uranus_end])
vel_centerM = np.sum(vel_end[indicies_uranus_end] * m_end[indicies_uranus_end,np.newaxis], axis=0) / np.sum(m_end[indicies_uranus_end])

pos_uranus_end = pos_end[indicies_uranus_end]
vel_uranus_end = vel_end[indicies_uranus_end]
m_uranus_end = m_end[indicies_uranus_end]
print(f'NUMBER OF URANUS: {len(m_uranus_end)}')
print(f'NUMBER OF IMPACTOR: {len(m_end[indicies_impactor_end])}')

pos_uranus_end -= pos_centerM
vel_uranus_end -= vel_centerM

# Remove later - COM correction for Lauren
indicies_uranus_init = np.where(np.isin(parids_init, uranus_ids))[0]
indicies_impactor_init = np.where(np.isin(parids_init, impactor_ids))[0]

pos_centerM = np.sum(pos_init[indicies_uranus_init] * m_init[indicies_uranus_init,np.newaxis], axis=0) / np.sum(m_init[indicies_uranus_init])
vel_centerM = np.sum(vel_init[indicies_uranus_init] * m_init[indicies_uranus_init,np.newaxis], axis=0) / np.sum(m_init[indicies_uranus_init])

pos_uranus_init = pos_init[indicies_uranus_init]
vel_uranus_init = vel_init[indicies_uranus_init]
m_uranus_init = m_init[indicies_uranus_init]
print(f'NUMBER OF URANUS: {len(m_uranus_init)}')
print(f'NUMBER OF IMPACTOR: {len(m_init[indicies_impactor_init])}')

pos_uranus_init -= pos_centerM
vel_uranus_init -= vel_centerM

print('AVERAGE POS:',np.mean(pos_uranus_init))

L_init = compute_angular_momentum(m_uranus_init,pos_uranus_init,vel_uranus_init)
L = compute_angular_momentum(m_uranus,pos_uranus,vel_uranus)
L_end = compute_angular_momentum(m_uranus_end,pos_uranus_end,vel_uranus_end)
print('L_init = ',L_init,' kg m^2 s^-1')
print('L = ',L,' kg m^2 s^-1')
print('L_end = ',L_end,' kg m^2 s^-1')

angle = (360/(2*np.pi))* np.arccos( np.dot([0,0,1],L) / ( np.linalg.norm([0,0,1]) * np.linalg.norm(L) ) )
print(f'Spin axis inclined at {angle} deg')
angle_end = (360/(2*np.pi))* np.arccos( np.dot([0,0,1],L_end) / ( np.linalg.norm([0,0,1]) * np.linalg.norm(L_end) ) )
print(f'END: Spin axis inclined at {angle_end} deg')
angle_init = (360/(2*np.pi))* np.arccos( np.dot([0,0,1],L_init) / ( np.linalg.norm([0,0,1]) * np.linalg.norm(L_init) ) )
print(f'INIT: Spin axis inclined at {angle_init} deg')



plot_vels(pos_uranus,vel_uranus,3.98*R_earth,loc.split('tests')[0],angle,ids[indicies_uranus],verbose=True)
plot_vels(pos_uranus_end,vel_uranus_end,3.98*R_earth,loc.split('tests')[0]+'end_',angle_end,ids_end[indicies_uranus_end],verbose=True)
plot_vels(pos_uranus_init,vel_uranus_init,3.98*R_earth,loc.split('tests')[0]+'init_',angle_init,ids_init[indicies_uranus_init],verbose=True)

sys.exit()
# plot_densities(pos,rho,angle,ids,loc.split('test')[0])
# plot_densities(pos_end,rho_end,angle_end,ids_end,loc.split('test')[0]+'end_')
sys.exit()
I = compute_moi(m,pos)
print('I = ',I, 'kg m^2')
print('I factor (I/MR^2) = ',I/(np.sum(m)*R*R))
omega = L/I
print('omega = ',omega,'s^-1')
print(np.linalg.norm(omega))
period = (2*np.pi)/np.linalg.norm(omega)
print('Period = ',period/3600,' h')

I_end = compute_moi(m_end,pos_end)
print('I_end = ',I_end, 'kg m^2')
print('I_end factor (I/MR^2) = ',I_end/(np.sum(m_end)*R*R))
omega_end = L_end/I_end
print('omega_end = ',omega_end,'s^-1')
print(np.linalg.norm(omega_end))
period_end = (2*np.pi)/np.linalg.norm(omega_end)
print('Period end = ',period_end/3600,' h')

# Period analysis by averaging outer most 100 particles
period_at_equator(pos,vel,R,angle,ids)
period_at_equator(pos_end,vel_end,R_end,angle_end,ids_end)

# Period analysis
folder = '/home/oo21461/Documents/tools/tests/chuck_in_swift/output'

files = []
print("Retrieving file info...")
for filename in os.listdir(folder):
    if filename.endswith('hdf5'):
        files.append(folder+'/'+filename)
print(files[0])
print(len(files),' snapshots found in ',folder)
files = sorted(files)

theta = np.radians(angle)  # Convert angle to radians
R_matrix = np.array([
    [1, 0, 0],
    [0, np.cos(theta), -np.sin(theta)],
    [0, np.sin(theta), np.cos(theta)]
])

polars = np.zeros((len(files),len(m)))
times = np.zeros(len(files))
for i, file in enumerate(tqdm(files)):
    # This loads all metadata but explicitly does _not_ read any particle data yet
    try:
        data = sw.load(file)
    except Exception as err:
        print(err, '.\nCould not open ',file)
        sys.exit()

    # Unit conversions here
    data.gas.coordinates.convert_to_mks()

    # Get data
    meta = data.metadata # Gets metadata of the file (not used in plotting anything here but might be useful in future for organising simulations)
    snap_time = meta.t / 3600 # in hours
    pos_snap = data.gas.coordinates.to_ndarray()
    masses = data.gas.masses.to_ndarray()

    pos_centerM = np.sum(pos_snap * masses[:,np.newaxis], axis=0) / np.sum(masses)    
    pos_snap -= pos_centerM

    # Transform frame
    pos_inclined = pos_snap @ R_matrix.T  # Rotate positions

    # Get polar coordinates of each paricle
    x = pos_inclined[:, 0]
    y = pos_inclined[:, 1]
    polar_angles = np.arctan2(y, x)
    # print(np.average(x)/R_earth,np.average(y)/R_earth)
    # print(np.unique(polar_angles))
    # sys.exit()
    times[i] = snap_time
    polars[i,:] = polar_angles

print(np.shape(times),np.shape(polars))
selected_polars = polars[:,::int(1e4)]
print('Selected polars: ',np.shape(selected_polars))

fig, ax = plt.subplots()
for i in range(len(selected_polars[0])):
    particle_polars = selected_polars[:,i]
    #print(np.shape(particle_polars))

    ax.plot(times,particle_polars)

ax.title.set_text('Polars v time')
ax.set_xlabel(r"Time $h$")
ax.set_ylabel(r"Polars")
ax.set_xlim(0, None)
ax.set_ylim(-np.pi, np.pi)
plt.tight_layout()
figname = loc.split('tests')[0]+'test_polars.png'
fig.savefig(figname,dpi=400)