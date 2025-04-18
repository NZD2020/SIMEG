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

def calculate_fault_stress(S_1, S_2, S_3, SHMaxAzi, fault_type, strike, dip, Ppore, biot=1.0):
    # Stress tensor in principal coordinates
    S_P = np.diag([S_1, S_2, S_3])
    
    print('fault_type = ', fault_type)

    # Determine rotation angles (degrees)
    if fault_type == 'normal':  # Normal fault
        rot_alpha = SHMaxAzi + 90
        rot_beta = 90
        rot_gamma = 0
    elif fault_type == 'strike-slip':  # Strike-slip fault
        rot_alpha = SHMaxAzi
        rot_beta = 0
        rot_gamma = 90
    elif fault_type == 'reverse':  # Reverse fault
        rot_alpha = SHMaxAzi
        rot_beta = 0
        rot_gamma = 0
    else:
        raise ValueError("Invalid fault type")

    # Convert to radians
    rot_alpha_rad = np.radians(rot_alpha)
    rot_beta_rad = np.radians(rot_beta)
    rot_gamma_rad = np.radians(rot_gamma)

    # Rotation matrix components
    R_PG = np.array([
        [np.cos(rot_alpha_rad) * np.cos(rot_beta_rad), np.sin(rot_alpha_rad) * np.cos(rot_beta_rad), -np.sin(rot_beta_rad)],
        [np.cos(rot_alpha_rad)*np.sin(rot_beta_rad)*np.sin(rot_gamma_rad) - np.sin(rot_alpha_rad)*np.cos(rot_gamma_rad),
         np.sin(rot_alpha_rad)*np.sin(rot_beta_rad)*np.sin(rot_gamma_rad) + np.cos(rot_alpha_rad)*np.cos(rot_gamma_rad),
         np.cos(rot_beta_rad)*np.sin(rot_gamma_rad)],
        [np.cos(rot_alpha_rad)*np.sin(rot_beta_rad)*np.cos(rot_gamma_rad) + np.sin(rot_alpha_rad)*np.sin(rot_gamma_rad),
         np.sin(rot_alpha_rad)*np.sin(rot_beta_rad)*np.cos(rot_gamma_rad) - np.cos(rot_alpha_rad)*np.sin(rot_gamma_rad),
         np.cos(rot_beta_rad)*np.cos(rot_gamma_rad)]
    ])
    
    # print(R_PG)

    # Stress tensor in geographic coordinates
    S_G = R_PG.T @ S_P @ R_PG

    # Fault plane orientation vectors
    strike_rad = np.radians(strike)
    dip_rad = np.radians(dip)

    n_n = np.array([
        -np.sin(strike_rad) * np.sin(dip_rad),
        np.cos(strike_rad) * np.sin(dip_rad),
        -np.cos(dip_rad)
    ])
    n_s = np.array([
        np.cos(strike_rad),
        np.sin(strike_rad),
        0
    ])
    n_d = np.array([
        -np.sin(strike_rad) * np.cos(dip_rad),
        np.cos(strike_rad) * np.cos(dip_rad),
        np.sin(dip_rad)
    ])

    # Traction vector on the fault plane
    t_fault = S_G @ n_n

    # Normal and shear stresses
    S_n = np.dot(t_fault, n_n)
    sigma_n = S_n - biot * Ppore
    tau_d = np.dot(t_fault, n_d)
    tau_s = np.dot(t_fault, n_s)
    tau = np.sqrt(tau_d ** 2 + tau_s ** 2)

    rake_rad = np.arctan2(tau_d, tau_s)
    rake_deg = np.degrees(rake_rad)

    return {
        "t_fault": t_fault,
        "S_n": S_n,
        "sigma_n": sigma_n,
        "tau": tau,
        "tau_d": tau_d,
        "tau_s": tau_s,
        "rake": rake_deg
    }

# Given stress gradients in psi/ft
Pp_gradient = 0.4  # Pore pressure gradient

Sv_gradient = 1.1
SHmax_gradient = 0.9
Shmin_gradient = 0.6

fault_type = 'normal'

# Sv_gradient = 1.09
# SHmax_gradient = 1.6
# Shmin_gradient = 0.8

# Reference depth in feet
ref_depth = 8120  # ft

# Compute total stresses
Sv = Sv_gradient * ref_depth
SHmax = SHmax_gradient * ref_depth
Shmin = Shmin_gradient * ref_depth

Pp = Pp_gradient * ref_depth

# Compute effective stresses
sigma_v = Sv - Pp
sigma_SH = SHmax - Pp
sigma_Sh = Shmin - Pp

# Assign sigma_1, sigma_2, sigma_3
sigma_1_psi, sigma_2_psi, sigma_3_psi = sigma_v, sigma_SH, sigma_Sh

# Conversion factor from psi to MPa
psi_to_MPa = 0.00689476

# Convert principal stresses to MPa
sigma_1 = sigma_1_psi * psi_to_MPa
sigma_2 = sigma_2_psi * psi_to_MPa
sigma_3 = sigma_3_psi * psi_to_MPa

# Read dip angles and directions from a file
filename = "dip_data.txt"
dip_values, dip_directions = read_dip_data(filename)

strike_values = dip_directions - 90

Pp_MPa = Pp*psi_to_MPa
S1_MPa = Sv*psi_to_MPa
S2_MPa = SHmax*psi_to_MPa
S3_MPa = Shmin*psi_to_MPa

SHMaxAzi = 25
biot = 1
# Compute normal and shear stress for each dip angle
sigma_n_values = []
tau_values = []
for strike, dip in zip(strike_values, dip_values):
    results = calculate_fault_stress(S1_MPa, S2_MPa, S3_MPa, SHMaxAzi, fault_type, strike, dip, Pp_MPa, biot)
    sigma_n_values.append(results["sigma_n"])
    tau_values.append(results["tau"])

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
