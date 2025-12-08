"""Step responses, to be used to understand noise. 

J R Forbes, 2025/09/10
"""

# %%
# Libraries
import numpy as np
# import control
# from scipy import signal
from matplotlib import pyplot as plt
# from scipy import fft
# from scipy import integrate
import pathlib

# Set up plots directory
plots_dir = pathlib.Path("/Users/aidan1/Documents/McGill/MECH412/MECH 412 Pump Project/plots")
plots_dir.mkdir(exist_ok=True)

# %%
# Plotting parameters
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=14)
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')

# %%
# Common parameters

# Conversion
rps2Hz = lambda w: w / 2 / np.pi
Hz2rps = lambda w: w * 2 * np.pi

# Golden ratio
gr = (1 + np.sqrt(5)) / 2

# Figure height
height = 4.25


# %%
# Read in all input-output (IO) data
path = pathlib.Path('DATA_noise/')
all_files = sorted(path.glob("*.csv"))
# all_files.sort()
data = [
    np.loadtxt(
        filename,
        dtype=float,
        delimiter=',',
        skiprows=1,
        usecols=(0, 1, 2),
        # max_rows=1100,
    ) for filename in all_files
]
data = np.array(data)

# %%
# 

N_data = data.shape[0]
max_input_output_std = np.zeros((N_data, 7))

u_arr=[]
y_arr=[]

SD_arr=[]

for i in range(N_data):  # N_data
    # Data
    data_read = data[i, :, :]

    t_full = data_read[:, 0]
    target_time = 0  # s
    t_start_index = np.argmin(np.abs(t_full - target_time))

    # Extract time
    t = data_read[t_start_index:-1, 0]
    T = t[1] - t[0]

    # Extract input and output
    u_raw = data_read[t_start_index:-1, 1]  # V, volts
    y_raw = data_read[t_start_index:-1, 2]  # LMP, force
    print("shape of y_raw: ",np.shape(y_raw))
    # Average y after 5 seconds:
    y_avg = np.mean(y_raw[t > 5])
    u_avg = np.mean(u_raw[t > 5])
    u_arr.append(u_avg)
    y_arr.append(y_avg)
    print("Average y after 5 seconds dataset", i, ":", y_avg)
    print("Average u after 5 seconds dataset", i, ":", u_avg)

    # Plotting: time domain
    fig, ax = plt.subplots(2, 1)
    fig.set_size_inches(height * gr, height, forward=True)
    ax[0].plot(t, u_raw)
    ax[1].plot(t, y_raw)
    ax[0].set_xlabel(r'$t$ (s)')
    ax[1].set_xlabel(r'$t$ (s)')
    ax[0].set_ylabel(r'$\tilde{u}(t)$ (V)')
    ax[1].set_ylabel(r'$\tilde{y}(t)$ (LPM)')
    fig.tight_layout()
    fig.savefig(plots_dir / f'time_domain_{i}.pdf')

    # Compute and plot PSD of y
    from scipy.signal import welch

    # Only use signal past 5 seconds to calculate the PSD
    mask = t > 5
    y_for_psd = y_raw[mask]
    t_for_psd = t[mask]
    
    # Check if we have enough data points after 5 seconds
    if len(y_for_psd) > 10:  # Need at least 10 points for meaningful PSD
        if len(t_for_psd) > 1:
            fs = 1.0 / (t_for_psd[1] - t_for_psd[0])  # Sampling frequency from time step
        else:
            fs = 1.0  # fallback if insufficient points
        print("fs: ", fs)
        
        # Ensure we have enough points for welch (needs at least nperseg points)
        nperseg = min(1024, len(y_for_psd))
        if nperseg >= 4:  # welch needs at least 4 points
            f, Pxx = welch(y_for_psd, fs=fs, nperseg=nperseg)
            # Get standard deviation of noise
            if len(Pxx) > 0 and np.all(np.isfinite(Pxx)):
                SD = np.sqrt(np.mean(Pxx) * (fs / 2))
                SD_arr.append(SD)
                print("standard dev: ", SD)
                
                # Plot PSD
                fig_psd = plt.figure()
                plt.semilogy(f, Pxx)
                plt.xlabel('Frequency [Hz]')
                plt.ylabel('PSD [$y^2$/Hz]')
                plt.title(f'Power Spectral Density of y (Dataset {i}) (t > 5s)')
                plt.grid(True, which='both', ls='--')
                plt.tight_layout()
                fig_psd.savefig(plots_dir / f'psd_{i}.pdf')
                plt.close(fig_psd)
            else:
                print(f"Warning: Invalid PSD for dataset {i}, skipping")
        else:
            print(f"Warning: Insufficient points for PSD calculation in dataset {i}")
    else:
        print(f"Warning: Not enough data points after 5s for dataset {i} (only {len(y_for_psd)} points)")
#avg SD - only compute if we have valid values
if len(SD_arr) > 0:
    # Filter out any NaN or inf values
    SD_arr_clean = [sd for sd in SD_arr if np.isfinite(sd)]
    if len(SD_arr_clean) > 0:
        SD_avg = np.mean(SD_arr_clean)
        print("average standard deviation: ", SD_avg)
    else:
        print("Warning: No valid standard deviation values found")
        SD_avg = np.nan
else:
    print("Warning: No standard deviation values computed")
    SD_avg = np.nan


#form ax=b problem with u_avg and y_avg
# Perform linear regression (least squares) to get constant k: y = k*u
A = np.array(u_arr).reshape(-1, 1)   # predictor (u), must be 2D for lstsq
b = np.array(y_arr)                  # response (y)
k, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
# print("Linear regression constant (k) relating y = k*u: ", k[0])

# Plot the regression results
plt.figure()
plt.plot(u_arr, y_arr, 'o', label='Data (u, y)')
u_line = np.linspace(0, 5, 100)
y_fit = k[0] * u_line
plt.plot(u_line, y_fit, '-', label=f'Fit: y = {k[0]:.3f}*u')
plt.xlabel('u')
plt.ylabel('y')
plt.title('Linear Regression: y = k*u')
plt.legend()
plt.savefig(plots_dir / 'linear_regression.pdf')
plt.show()