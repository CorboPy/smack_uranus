# Script for creating proto-Uranus

import woma
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np

R_earth = 6.371e6   # m
M_earth = 5.972e24  # kg

def plot_spherical_profiles(planet,path):    
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
    plt.close()

def plot_spinning_profiles(sp):    
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
    fig.savefig('/data/cluster4/lauren_flemons/SWIFT-1.0.0/test/tests/uranus/spinning_uranus_plot_test.png')

def plot_ICs(planet,particleplanet,pos,ids,boxsize,path):
    plt.style.use('default')
    vx, vy, vz = particleplanet.A2_vel.T
    pos /= R_earth
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
N = {particleplanet.N_particles}
boxsize = {boxsize/R_earth}
mat_ids = {np.unique(particleplanet.A1_mat_id)}
datetime = {now}
"""
#s = {np.min(particleplanet.A1_s)},  {np.max(particleplanet.A1_s)}

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
    plt.close()

#path_to_planets = f'/data/cluster4/oo21461/Planets/{1.5}_uranus'


planet = woma.Planet(
    A1_mat_layer    = ["ANEOS_forsterite", "AQUA", "HM80_HHe"],
    A1_T_rho_type   = ["adiabatic", "power=0.9", "adiabatic"],
    P_s             = 1e5,
    T_s             = 70,
    M               = (14.54 * M_earth)-(1.5*M_earth),#- for alex 
    A1_R_layer      = [1.01 * R_earth, 2.72 * R_earth, 3.98 * R_earth],
)

#planet.name = f"{1.5}_uranus"

# Generate the profiles
#planet.gen_prof_L2_find_R1_given_M_R(verbosity=0)
# Generate the profiles
planet.gen_prof_L3_find_R1_given_M_R_R2()
# Create the sets of particles
#particles_low_res = woma.ParticlePlanet(planet, 1e4, verbosity=0)
# Plot the results
#plot_spherical_profiles(planet)

#plot_spherical_profiles(planet,path_to_planets+'/relax_sim/')

particles_high_res = woma.ParticlePlanet(planet, 1e6, verbosity=0)
particles_high_res.A1_mat_id[particles_high_res.A1_mat_id==304]=900 #setting equal to custom ID

print(particles_high_res.A2_pos[122] / R_earth, "R_earth")
print(particles_high_res.A1_m[122] / M_earth, "M_earth")
print(particles_high_res.A1_rho[122], "kg m^-3")
print(particles_high_res.A1_T[122], "K")

#plot_ICs(planet,particles_high_res,particles_high_res.A2_pos,particles_high_res.A1_mat_id,10*R_earth,path_to_planets+'/relax_sim/')


#print("%.3e" % particles_low_res.N_particles)
print("%.3e" % particles_high_res.N_particles)
particles_high_res.save("/home/oo21461/Documents/tools/1.5_uranus_woma.hdf5", boxsize=10*R_earth, file_to_SI=woma.Conversions(M_earth, R_earth, 1), do_entropies=True)
#particles_low_res = woma.ParticlePlanet(planet, 1e4, N_ngb=48, verbosity=0)

#print(particles_low_res.A1_h[122] / R_earth, "R_earth")

#spin_planet = woma.SpinPlanet(
    #planet      = planet,
    #period      = 16.87,  # h
    #verbosity   = 0,
#)




#plot_spinning_profiles(spin_planet)

#particles = woma.ParticlePlanet(spin_planet, 1e6, N_ngb=48)

#print(particles.A2_pos[122] / R_earth, "R_earth")

#particles.save("/data/cluster4/lauren_flemons/SWIFT-1.0.0/test/tests/uranus/hdf5/particles_spinning.hdf5", boxsize=10*R_earth, file_to_SI=woma.Conversions(M_earth, R_earth, 1))