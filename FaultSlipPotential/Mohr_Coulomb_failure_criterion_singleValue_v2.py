from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')  # Clear all variables before the script begins to run.

import numpy as np
import matplotlib.pyplot as plt
import math
import os
import matplotlib.ticker as ticker

# Set working directory (update as needed)
os.chdir(
    'C:/Users/Xiaoming Zhang/Desktop/postdoc_Xiaoming Zhang/'
    'NoamWork/Forge_Noam_project/UtahFORGE_allData/fittingFaults/Mohrs_circle'
)

# Function to read dip angles and dip directions from a file
def read_dip_data(filename):
    data = np.loadtxt(filename, delimiter='\t', skiprows=1)  # Assuming tab-separated format
    theta_values = data[:, 0]  # First column: Dip angles
    phi_values = data[:, 1]    # Second column: Dip directions
    return theta_values, phi_values

# Given stress gradients in psi/ft
pPressure_gradient = 0.4  # Pore pressure gradient

Sv_gradient = 1.1
SHmax_gradient = 0.9
Shmin_gradient = 0.6

# Sv_gradient = 1.09
# SHmax_gradient = 1.6
# Shmin_gradient = 0.8

# Reference depth in feet
ref_depth = 8120  # ft

# Compute total stresses
Sv = Sv_gradient * ref_depth
SHmax = SHmax_gradient * ref_depth
Shmin = Shmin_gradient * ref_depth
pPressure = pPressure_gradient * ref_depth

# Compute effective stresses
sigma_v = Sv - pPressure
sigma_SH = SHmax - pPressure
sigma_Sh = Shmin - pPressure

# Assign principal stresses in descending order
sigma_values = [sigma_v, sigma_SH, sigma_Sh]
sigma_values.sort(reverse=True)  # Sort from largest to smallest

# Assign sigma_1, sigma_2, sigma_3
sigma_1_psi, sigma_2_psi, sigma_3_psi = sigma_values

# Conversion factor from psi to MPa
psi_to_MPa = 0.00689476

# Convert principal stresses to MPa
sigma_1 = sigma_1_psi * psi_to_MPa
sigma_2 = sigma_2_psi * psi_to_MPa
sigma_3 = sigma_3_psi * psi_to_MPa

# Read dip angles and directions from a file
filename = "dip_data.txt"  # Update with the actual filename
theta_values, phi_values = read_dip_data(filename)

# Initialize lists to store computed stresses
sigma_n_values = []
tau_values = []

# Compute normal and shear stress for each dip angle
for theta, phi in zip(theta_values, phi_values):
    theta_rad = np.radians(theta)
    phi_rad = np.radians(phi)

    # Compute unit normal vector components
    n1 = np.sin(theta_rad) * np.cos(phi_rad)
    n2 = np.sin(theta_rad) * np.sin(phi_rad)
    n3 = np.cos(theta_rad)

    # Compute normal stress (σ_n)
    sigma_n = sigma_1 * n1**2 + sigma_2 * n2**2 + sigma_3 * n3**2
    sigma_n_values.append(sigma_n)

    # Compute shear stress (τ)
    tau = np.sqrt(
        (sigma_1 - sigma_2)**2 * n1**2 * n2**2 +
        (sigma_2 - sigma_3)**2 * n2**2 * n3**2 +
        (sigma_3 - sigma_1)**2 * n3**2 * n1**2
    )
    tau_values.append(tau)

# Convert to numpy arrays
sigma_n_values = np.array(sigma_n_values)
tau_values = np.array(tau_values)

# Mohr's Circle parameters
centers = [(sigma_1 + sigma_3) / 2, (sigma_2 + sigma_3) / 2, (sigma_1 + sigma_2) / 2]
radii = [(sigma_1 - sigma_3) / 2, (sigma_2 - sigma_3) / 2, (sigma_1 - sigma_2) / 2]

# Generate Mohr's circle (only upper half where y >= 0)
theta_circle = np.linspace(0, np.pi, 200)

mohr_sigma = [center + radius * np.cos(theta_circle) for center, radius in zip(centers, radii)]
mohr_tau = [radius * np.sin(theta_circle) for radius in radii]

# Define multiple friction coefficients
frictionCoeffi_values = [0.65]
cohesive_strength = 5

phi_Coulomb_radian = np.arctan(np.array(frictionCoeffi_values))
phi_Coulomb_degree = np.degrees(phi_Coulomb_radian)

colors = ['black']  # Colors for different friction lines

# Store pore pressure data as a list of lists
pore_pressure = []

# Plot configuration
plt.rcParams.update({
    'ytick.labelsize': 16,
    'xtick.labelsize': 16,
    'figure.dpi': 144
})
plt.figure(figsize=(8, 6))
plt.axes(aspect='equal')

# Plot Mohr’s Circles
for sigma, tau in zip(mohr_sigma, mohr_tau):
    plt.plot(sigma, tau, color='black', linewidth=2)

# Plot Mohr-Coulomb failure envelopes for multiple friction coefficients
for i, frictionCoeffi in enumerate(frictionCoeffi_values):
    phi_Coulomb_radian = math.atan(frictionCoeffi)
    
    # Mohr-Coulomb Failure Envelope
    # sigma_n_mohr = np.linspace(0, 1.2*sigma_1, 100)
    sigma_n_mohr = np.linspace(0, sigma_1 + 5, 100)
    tau_mohr = cohesive_strength + sigma_n_mohr * np.tan(phi_Coulomb_radian)
    
    # Set line style: solid if μ = 0.7, dashed otherwise
    # line_style = '-' if frictionCoeffi == 0.7 else '--'
    line_style = '-'
    
    plt.plot(sigma_n_mohr, tau_mohr, color=colors[i], 
             linewidth=2, linestyle=line_style, 
             label=f'MC: $μ$ = {frictionCoeffi}, '
             f'c = {cohesive_strength} MPa')
    
    # Compute pore pressure: σ_n - σ_n_mohr_line
    sigma_n_mohr_line = (tau_values - cohesive_strength) / np.tan(phi_Coulomb_radian)
    distances_x_Coulomb = sigma_n_values - sigma_n_mohr_line
    pore_pressure.append(distances_x_Coulomb)  # Append to list

# Convert list to a NumPy array for easier indexing
pore_pressure = np.array(pore_pressure)

# Plot computed (σ_n, τ) points, colored by distance to failure (for μ = 0.7)
sc = plt.scatter(
    sigma_n_values, tau_values, c=pore_pressure[0], cmap='coolwarm_r', 
    edgecolors='black', linewidth=0.8, s=60, vmin=0, vmax=30
)

plt.xlim(0, 50)
plt.ylim(0, 40)

# Set axis ticks
# x_ticks = [0, 20, 40, 60]
# x_labels = ['0', '20', '40', '60'] 
# y_ticks = [0, 20, 40]
# y_labels = ['0', '20', '40'] 

# plt.xticks(x_ticks, x_labels) 
# plt.yticks(y_ticks, y_labels) 

plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(10))
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(10))
plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

# Colorbar
cbar = plt.colorbar(sc)

num_ticks = 4 # Specify the number of ticks you want
# Generate evenly spaced values 
# tick_values = np.floor(np.linspace(np.ceil(distances_x.min()), distances_x.max(), num_ticks))

tick_values = np.floor(np.linspace(0, 30, num_ticks))

cbar.set_ticks(tick_values)
cbar.set_ticklabels(tick_values)
# cbar.set_label('Fault stability: distance to failure envelope', fontsize=16)
cbar.set_label('Pore pressure to slip', fontsize=16)

# Labels and legend
plt.xlabel(r'$\sigma_n$ (MPa)', fontsize=16)
plt.ylabel(r'$\tau$ (MPa)', fontsize=16)
# plt.title("Mohr-Coulomb failure criterion", fontsize=18)
plt.legend(frameon=False, fontsize=14)

# Remove grid and show plot
plt.grid(False)
plt.show()

# Write pore pressure data to a file with left-aligned columns
output_filename = "pore_pressure_Mohr_Coulomb.txt"
with open(output_filename, 'w') as f:
    # Create a header row with left-aligned column names
    header = ["Fault ID"] + [f'μ = {friction}' for friction in frictionCoeffi_values]
    f.write("".join(f"{col:<12}" for col in header) + "\n")  # Left align (width=12)
    
    # Write each row of data
    for i in range(len(sigma_n_values)):
        row = [f"Fault {i+1}"] + [f"{pore_pressure[j][i]:.2f}" for j in range(len(frictionCoeffi_values))]
        f.write("".join(f"{col:<12}" for col in row) + "\n")  # Left align (width=12)

print(f"Pore pressure data saved to {output_filename}")

# # Compute perpendicular distance of (σ_n, τ) points to Mohr-Coulomb failure line
# A = -np.tan(phi_Coulomb_radian)  # Coefficient for σ_n
# B = 1             # Coefficient for τ
# C = -c            # Constant term

# distances = np.abs(A * sigma_n_values + B * tau_values + C) / np.sqrt(A**2 + B**2)
