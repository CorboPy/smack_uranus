# Plotting cubic splie kernal for report

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

def cubic_spline_kernel(r, h=1.0):
    q = np.abs(r) / h
    sigma = 1 / (h * (2/3))  # Normalization factor for 1D
    W = np.zeros_like(q)
    
    mask1 = (q >= 0) & (q < 1)
    W[mask1] = sigma * ((2/3) - q[mask1]**2 + (1/2) * q[mask1]**3)
    
    mask2 = (q >= 1) & (q < 2)
    W[mask2] = sigma * ((1/6) * (2 - q[mask2])**3)
    
    return W

# Define the range of r values
h = 1.0
r_values_1 = np.linspace(-2*h, 2*h, 100)
W_values_1 = cubic_spline_kernel(r_values_1, h)

h = 2.0
r_values_2 = np.linspace(-2*h, 2*h, 100)
W_values_2 = cubic_spline_kernel(r_values_2, h)

h = 0.5
r_values_05 = np.linspace(-2*h, 2*h, 100)
W_values_05 = cubic_spline_kernel(r_values_05, h)

# Plot the kernel
fig = plt.figure(figsize=(4, 2.25))
plt.plot(r_values_05, W_values_05, label=r'$h=0.5$', color='r',zorder=3)
plt.plot(r_values_1, W_values_1, label=r'$h=1$', color='g',zorder=2)
plt.plot(r_values_2, W_values_2, label=r'$h=2$', color='b',zorder=1)
plt.xlabel(r'$r$')
plt.ylabel(r'$W(\mathbf{r}, h)$')
#plt.title('Cubic Spline Kernel Function for SPH')
plt.axhline(0, color='k', linewidth=0.5, linestyle='--')
plt.axvline(0, color='k', linewidth=0.5, linestyle='--')
plt.legend()
plt.grid()
fig.tight_layout()
plt.savefig('C:/Users/alexc/Downloads/plot_kernel.pdf',dpi=500)
