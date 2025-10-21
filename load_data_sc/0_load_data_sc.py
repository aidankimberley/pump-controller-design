"""MECH 412 sample code.

J R Forbes, 2025/10/13

This code loads the data.
"""

# %%
# Libraries
import numpy as np
import control
from scipy import signal
from matplotlib import pyplot as plt
import pathlib


# %%
# Plotting parameters

plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')
# %%
# Read in all input-output (IO) data
path = pathlib.Path('load_data_sc/PRBS_DATA/')
all_files = sorted(path.glob("*.csv"))
# all_files.sort()
data = [
    np.loadtxt(
        filename,
        dtype=float,
        delimiter=',',
        skiprows=1,
        usecols=(0, 1, 2),
    ) for filename in all_files
]
data = np.array(data)

# %%
# Load a dataset
k_train = 0
data_read = data[k_train, :, :]

# Extract time
t = data_read[:, 0]
N = t.size
T = t[1] - t[0]

# Extract input and output
u_raw = data_read[:, 1]  # V, volts
y_raw = data_read[:, 2]  # LPM, liters per minute

# Plot data, just to see what it looks like
fig, ax = plt.subplots(2, 1)
ax[0].set_ylabel(r'$u(t)$ (V)')
ax[1].set_ylabel(r'$y(t)$ (LPM)')
# Plot data
ax[0].plot(t, u_raw, label='input', color='C0')
ax[1].plot(t, y_raw, label='output', color='C1')
for a in np.ravel(ax):
    a.set_xlabel(r'$t$ (s)')
    a.legend(loc='best')
fig.tight_layout()
# This ell variable will allow you to save a plot with the number ell in the plot name
ell = k_train
# fig.savefig('test_plot_%s.pdf' % ell)


# %%
# Show plots
plt.show()
