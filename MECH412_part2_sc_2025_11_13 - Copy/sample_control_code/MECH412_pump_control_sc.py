"""Pump control sample code, MECH 412.

James Forbes
2025/10/15
"""
# %%
# Packages
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
from scipy import signal
from scipy import fft
import control

# Custom packages
import siso_rob_perf as srp
import pathlib

# %%
# Plotting parameters
# # Set up plots directory
# plots_dir = pathlib.Path("/Users/aidan1/Documents/McGill/MECH412/MECH 412 Pump Project/plots")
# plots_dir.mkdir(exist_ok=True)

# Get the directory where this script is located
# script_dir = pathlib.Path(__file__).parent
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

# Laplace variable
s = control.tf('s')

# Frequencies for Bode plot, in rad/s
w_shared_low, w_shared_high, N_w = np.log10(Hz2rps(10**(-2))), np.log10(Hz2rps(10**(2))), 500
w_shared = np.logspace(w_shared_low, w_shared_high, N_w)

# Frequencies for Nyquist plot, in rad/s
w_shared_low_2, w_shared_high_2, N_w_2 = np.log10(Hz2rps(10**(-4))), np.log10(Hz2rps(10**(4))), 5000
w_shared_2 = np.logspace(w_shared_low_2, w_shared_high_2, N_w_2)

# %%
# Extract uncertainty weight and nominal model.

# Uncertainty weight
# Dummy uncertainty weight! You must change!
W2 = control.TransferFunction(
    [1.019, 49.4, 890.9, 784.3, 6525],
    [1, 34.54, 3599, 1765, 2.659e+04]
)
# W2 = control.TransferFunction(
#     [1.006, 42.29, 530.4, 4749, 1.171e4],
#     [1, 40.4, 804.8, 7099, 4.725e4]
# )
W2_inv = 1 / W2
print("W_2(s) = ", W2)

# Nominal model, which is NOT normalized, and has units of LPM / V.
m, n = 0, 1
P_tilde = control.TransferFunction([26.26], [1, 13.65])
DC_gain = P_tilde.dcgain() 
print("DC gain = ", DC_gain)
max_V = 5
slope = 2.437 #From step response data
max_LPM = max_V * slope  # Dummy value. You must change. 

P = P_tilde  # The ``tilde" means ``with units". This sample code has not done any normalization. 

print("P(s) = ", P)
N_off_nom = 10
gain_pert = np.linspace(-1, 1, N_off_nom)
all_pass_pz = np.linspace(0.1, 10, N_off_nom)
# P_off_nom = [P * (1 + W2 * gain_range[i]) for i in range(-N_off_nom, N_off_nom + 1, 1)]
Delta = []
for i in range(N_off_nom):
    for j in range(N_off_nom):
        Delta.append(gain_pert[i] * control.TransferFunction([1, -all_pass_pz[j]], [1, all_pass_pz[j]]))

# P_off_nom = [P * (1 + W2 * gain_pert[i]) for i in range(N_off_nom)]
P_off_nom = [P * (1 + W2 * Delta[i]) for i in range(len(Delta))]

# %%
# Performance

#normalization constants:
SD_noise = 0.107 #LPM

r_nor = max_LPM #LPM
u_nor_r = max_V #V
e_nor_r = 0.05 * max_LPM #LPM

n_nor = SD_noise #LPM
u_nor_n = 0.02*max_V #V 
e_nor_n = n_nor #LPM

d_nor = 0.256 # V
u_nor_d = d_nor
e_nor_d = 0.01 * max_LPM #LPM

#Get this from fft of reference
w_r_h_Hz = 0.015  # Hz
w_n_Hz = 3 # Hz

# Noise and reference bounds
# gamma_n, w_n_l = e_nor_n/e_nor_r, Hz2rps(w_r_h_Hz * 100)
# gamma_r, w_r_h = 1, Hz2rps(w_r_h_Hz)
# gamma_u, w_u_l = u_nor_r/u_nor_n, w_r_h
# gamma_d, w_d_h = 1, w_r_h / 10 #TODO: MAYBE CHANGE THIS

#gamma_n = e_nor_n/n_nor

gamma_n, w_n_l = e_nor_n/n_nor, Hz2rps(w_n_Hz)
gamma_r, w_r_h = e_nor_r/r_nor, Hz2rps(w_r_h_Hz)
gamma_u, w_u_l = e_nor_r*u_nor_n/n_nor/u_nor_r, Hz2rps(w_n_Hz)#w_r_h_Hz*250)
gamma_d, w_d_h = e_nor_d/d_nor, w_r_h #TODO: MAYBE CHANGE THIS



# Set up design specifications plot
w_r = np.logspace(w_shared_low, np.log10(w_r_h), 100)
w_d = np.logspace(w_shared_low, np.log10(w_d_h), 100)
w_n = np.logspace(np.log10(w_n_l), w_shared_high, 100)
w_u = np.logspace(np.log10(w_u_l), w_shared_high, 100)

# In dB
gamma_r_dB = 20 * np.log10(gamma_r) * np.ones(w_r.shape[0],)
gamma_d_dB = 20 * np.log10(gamma_d) * np.ones(w_d.shape[0],)
gamma_n_dB = 20 * np.log10(gamma_n) * np.ones(w_n.shape[0],)
gamma_u_dB = 20 * np.log10(gamma_u) * np.ones(w_u.shape[0],)

# Weight W_1(s) according to Zhou et al.
k = 2
epsilon = 10**(-30 / 20)
M1 = 10**(5 / 20)
w1 = Hz2rps(w_r_h_Hz + 0.2)
W1 = ((s / M1**(1 / k) + w1) / (s + w1 * (epsilon)**(1 / k)))**k
W1_inv = 1 / W1


# %%
# Plot both weights, W1 and W2 (and their inverses).
fig, ax = srp.bode_mag_W1_W2(W1, W2, w_d_h, w_n_l, w_shared, Hz = True)
fig.set_size_inches(height * gr, height, forward=True)
ax.legend(loc='upper right')
# fig.savefig(plots_dir / 'bode_W1_W2.pdf')

fig, ax = srp.bode_mag_W1_inv_W2_inv(W1, W2, gamma_r, w_r_h, w_d_h, gamma_n, w_n_l, w_shared_low, w_shared_high, w_shared, Hz = True)
fig.set_size_inches(height * gr, height, forward=True)
ax.legend(loc='lower right')
# fig.savefig(plots_dir / 'bode_W1_inv_W2_inv.pdf')


# %%
# Nyquist of open-loop plant without control
wmin, wmax, N_w_robust_nyq = np.log10(Hz2rps(10**(-4))), np.log10(Hz2rps(10**(4))), 1000
count, fig, ax = srp.robust_nyq(P, P_off_nom, W2, wmin, wmax, N_w_robust_nyq)
fig.tight_layout()
# fig.savefig(plots_dir / 'robust_nyquist.pdf')

# %%
# Control design.
# Dummy controller, you must change!
#w_c = np.sqrt(w_n_l * w_d_h)*1

tau_0 = 0.13
tau_1 = 0.01
tau_2 = 0.01
w_c = 5.15
#L_des = w_c/s * (1/s * (tau_1*s+1))

# Simple PI + rolloff controller
L_des = w_c/ (s*(tau_0*s+1))
print("L_des = ", L_des)


C = L_des / P
print("C = ", C, "\n")

fig_L, ax = srp.bode_mag_L(P, C, gamma_r, w_r_h, gamma_n, w_n_l, w_shared_low, w_shared_high, w_shared, Hz = False)
fig_L.set_size_inches(height * gr, height, forward=True)
ax.legend(loc='lower left')

fig_RP, ax = srp.bode_mag_rob_perf(P, C, W1, W2, w_shared, Hz = False)
fig_RP.set_size_inches(height * gr, height, forward=True)
ax.legend(loc='lower left')

fig_RP_RD, ax = srp.bode_mag_rob_perf_RD(P, C, W1, W2, w_shared, Hz = False)
fig_RP_RD.set_size_inches(height * gr, height, forward=True)
ax.legend(loc='upper right')

# %%
fig_S_T, ax = srp.bode_mag_S_T(P, C, gamma_r, w_r_h, w_d_h, gamma_n, w_n_l, w_shared_low, w_shared_high, w_shared, Hz = True)
ax.legend(loc='lower center')
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

fig_S_T_W1_inv_W2_inv, ax = srp.bode_mag_S_T_W1_inv_W2_inv(P, C, W1, W2, gamma_r, w_r_h, gamma_d, w_d_h, gamma_n, w_n_l, gamma_u, w_u_l, w_shared_low, w_shared_high, w_shared, Hz = True)
fig_S_T_W1_inv_W2_inv.set_size_inches(height * gr, height, forward=True)
ax.legend(loc='lower center')
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

fig_L_P, ax = srp.bode_mag_L_P(P, C, gamma_d, w_d_h, gamma_u, w_u_l, w_shared, Hz = True)
fig_L_P.set_size_inches(height * gr, height, forward=True)
ax.legend(loc='lower left')

fig_L_P_C, ax = srp.bode_mag_L_P_C(P, C, gamma_r, w_r_h, gamma_n, w_n_l, w_shared_low, w_shared_high, w_shared, Hz = True)
fig_L_P_C.set_size_inches(height * gr, height, forward=True)
ax.legend(loc='lower left')

fig_margins, ax, gm, pm, vm, wpc, wgc, wvm = srp.bode_margins(P, C, w_shared, Hz = True)
fig_margins.set_size_inches(height * gr, height, forward=True)
print(f'\nGain margin is', 20 * np.log10(gm),
      '(dB) at phase crossover frequency', wpc, '(rad/s)')
print(f'Phase margin is', pm, '(deg) at gain crossover frequency',
      wgc, '(rad/s)')
print(f'Vector margin is', vm, 'at frequency', wvm, '(rad/s)\n')

fig_Gof4, ax = srp.bode_mag_Gof4(P, C, gamma_r, w_r_h, gamma_d, w_d_h, gamma_n, w_n_l, gamma_u, w_u_l, w_shared_low, w_shared_high, w_shared, Hz = True)
fig_Gof4.set_size_inches(height * gr, height, forward=True)

# Nyquist
fig_Nyquist, ax_Nyquist = plt.subplots()
count, contour = control.nyquist_plot(control.minreal(P * C),
                                      omega=w_shared_2,
                                      plot=True,
                                      return_contour=True)
# ax_Nyquist.axis('equal')
fig_Nyquist.tight_layout()

# Print vector, gain, and phase margins
print(f"Vector margin: {vm}")
print(f"Gain margin: {gm} (linear), {20 * np.log10(gm):.2f} dB")
print(f"Phase margin: {pm} deg")



# fig_L.savefig(plots_dir / 'bode_L.pdf')
# fig_RP.savefig(plots_dir / 'robust_performance.pdf')
# fig_RP_RD.savefig(plots_dir / 'robust_performance_RD.pdf')
# fig_S_T_W1_inv_W2_inv.savefig(plots_dir / 'S_T_weights.pdf')
# fig_S_T.savefig(plots_dir / 'S_T.pdf')
# fig_L_P.savefig(plots_dir / 'L_P.pdf')
# fig_L_P_C.savefig(plots_dir / 'L_P_C.pdf')
# fig_margins.savefig(plots_dir / 'margins.pdf')
# fig_Gof4.savefig(plots_dir / 'Gof4.pdf')
# fig_Nyquist.savefig(plots_dir / 'nyquist.pdf')



# %%
# Reference
# Reference

data = np.loadtxt(
    "RL_temp_motor_mod.csv",
    dtype=float,
    delimiter=',',
    skiprows=1,
    usecols=(0, 1),
    # max_rows=1100,
)

# Extract time and temperature data
N_temp_data = data.shape[0]
dt = 0.02

# Must specify time, because the data has some numerical rounding issues. 
t_raw = np.linspace(0, dt * N_temp_data, N_temp_data)

# All the temperatures
temp_raw_raw = data[:, 1]

# Extract a subset of time 
t_start = 900  # s
t_end = 1900  # s
# t_end = 1200  # s
t_start_index = np.where(np.abs(t_raw - t_start) <= 0.02)[0][-1]
t_end_index = np.where(np.abs(t_raw - t_end) <= 0.02)[0][-1]

# Extract time over the desired interval
t = t_raw[t_start_index:t_end_index]

# Extract temperature data over the desired interval. 
temp_raw = temp_raw_raw[t_start_index:t_end_index]

# Plotting
# Plot raw data time domain
fig, ax = plt.subplots()
fig.set_size_inches(height * gr, height, forward=True)
ax.plot(t, temp_raw)
ax.set_xlabel(r'$t$ (s)')
ax.set_ylabel(r'Temperature (°C)')
fig.tight_layout()
# fig.savefig(plots_dir / 'temperature_raw.pdf')

# Temperature bounds for normalization
temp_raw_max = np.max(np.abs(temp_raw))
temp_raw_min = np.min(np.abs(temp_raw))
print("The max (in absolute value) output is", temp_raw_max, '(°C)')
print("The min (in absolute value) output is", temp_raw_min, '(°C)')

# Round the max up to 90 C
temp_raw_max = 90  # C

# Extract temperature data 
temp_raw = temp_raw_raw[t_start_index:t_end_index]

# Normalize and center temperature data
temp_norm = (temp_raw - temp_raw_min) / (temp_raw_max - temp_raw_min)

# Plot Normalized temperature profile
fig, ax = plt.subplots()
fig.set_size_inches(height * gr, height, forward=True)
ax.plot(t, temp_norm)
ax.set_xlabel(r'$t$ (s)')
ax.set_ylabel(r'Normalized Temperature (unitless)')
fig.tight_layout()
# fig.savefig(plots_dir / 'temperature_normalized.pdf')

# %% 
# Extract frequency content of temperature profile
N_fft = fft.next_fast_len(temp_norm.size, real=True)

T_fft = fft.rfft(temp_norm, n=N_fft) / N_fft  # same units as u
T_mag = np.abs(T_fft)  # compute the magnitude of each u_fft
T_mag[1:] = 2 * T_mag[1:]  # multiply all mag's by 2, but the zero frequency

f = fft.rfftfreq(N_fft, d=dt)  # the frequencies in Hz
omega = f * 2 * np.pi   # the frequencies in rad/s

#Plots
# FFT magnitude 
fig, ax = plt.subplots()
fig.set_size_inches(height * gr, height, forward=True)
ax.plot(f, T_mag)
ax.set_xlabel(r'$f$ (Hz)')
ax.set_ylabel(r'$|T(f)|$ (unitless)')
fig.tight_layout()

# FFT zoomed in on dominant frequencies
fig, ax = plt.subplots()
fig.set_size_inches(height * gr, height, forward=True)
ax.plot(f, T_mag)
ax.vlines(w_r_h_Hz, 0, 1, color='r', linestyle='--')
ax.set_xlim(0, 0.05) # 1 / 10s
ax.set_xlabel(r'$f$ (Hz)')
ax.set_ylabel(r'$|T(\omega)|$ (unitless)')
fig.tight_layout()

# Semilog
fig, ax = plt.subplots()
fig.set_size_inches(height * gr, height, forward=True)
ax.semilogx(omega, T_mag)
ax.set_xlabel(r'$\omega$ (rad/s)')
ax.set_ylabel(r'$|T(\omega)|$ (unitless)')
fig.tight_layout()

# Loglog
fig, ax = plt.subplots()
fig.set_size_inches(height * gr, height, forward=True)
ax.loglog(omega, T_mag)
ax.vlines(w_r_h_Hz, 0, 1, color='r', linestyle='--')
ax.set_xlabel(r'$\omega$ (rad/s)')
ax.set_ylabel(r'$|T(\omega)|$ (unitless)')
fig.tight_layout()

# %% 
# Define Reference in terms of flow rate
# Convert temp into a reference LPM
r_raw_tilde = temp_norm * max_LPM

# Now filter using w_r_h
a = w_r_h
#a = 0.5
_, r_tilde = control.forced_response(1 / (1 / a * s + 1), t, r_raw_tilde, 0)

r_raw = r_raw_tilde  # r_raw would be unfiltered, without units, if you did some normalization
r = r_tilde  # r would be filtered, without units, if you did some normalization

fig, ax = plt.subplots()
ax.set_xlabel(r'$t$ (s)')
ax.set_ylabel(r'$r(t)$ (LPM)')
# Plot data
ax.plot(t, r, '--', label='$r(t)$', color='C3')
ax.legend(loc='upper right')
fig.tight_layout()
# fig.savefig(plots_dir / 'reference.pdf')
#plt.show()

# Noise
np.random.seed(123321)
noise_raw = np.random.normal(0, 1, t.shape[0])
sigma_n = SD_noise  # LPM, dummy value. You must change. 
noise =  sigma_n * noise_raw * 1  # Change the 1 to a zero to ``turn off" noise in order to debug. 


# %%
# Set up simulation.
"""
You should not need to change this part of the code. Just run the 
simulation once you've found the reference, noise, controller, 
bounds max_V and max_LPM, etc. 
"""

# Note, the simulation does take awhile to run. 
# When debugging, set t_end = 1200, to shorten the simulation. 

u_range = np.array([0, max_V])
z_range = np.array([0, max_LPM])

def simulate(P, C, t, r_raw, r, noise, u_range, z_range):
    """Nonlinear simulation.
    
    The plant is not linear, it's affine. This is why this 
    type of simulation is needed. 
    """

    # Time needs to be redefined as a new variable for solve_ivp.
    time = t

    # Plant state-space form.
    P_ss = control.tf2ss(np.array(P.num).ravel(), np.array(P.den).ravel())
    n_x_P = np.shape(P_ss.A)[0]
    
    # Control state-space form.
    C_ss = control.tf2ss(np.array(C.num).ravel(), np.array(C.den).ravel())
    n_x_C = np.shape(C_ss.A)[0]

    # ICs for plant and control. 
    x_P_IC = np.zeros((n_x_P, 1))
    x_C_IC = np.zeros((n_x_C, 1))
    
    # Set up closed-loop ICs.
    x_cl_IC = np.block([[x_P_IC], [x_C_IC]]).ravel()

    # Define closed-loop system. This will be passed to solve_ivp.
    def closed_loop(t, x):
        """Closed-loop system"""

        # Reference at current time.
        r_now = np.interp(t, time, r).reshape((1, 1))

        # Noise at current time.
        n_now = np.interp(t, time, noise).reshape((1, 1))

        # Split state.
        x_P = (x[:n_x_P]).reshape((-1, 1))
        x_C = (x[n_x_P:]).reshape((-1, 1))

        # Interpolation of u_bar and y_bar.
        # Note, r_raw is used, because we are ``linearizing" about
        # the current reference point.
        z_bar = np.interp(t, time, r_raw).reshape((1, 1))
        u_bar = np.interp(z_bar, z_range, u_range).reshape((1, 1))
        
        # Plant output, with noise.
        delta_z = P_ss.C @ x_P
        y = z_bar + delta_z + n_now
        
        # Compute error.
        error = r_now - y
        
        # Compute control signal. 
        delta_u = C_ss.C @ x_C + C_ss.D @ error
        u = u_bar + delta_u

        # Advance system state.
        dot_x_sys = P_ss.A @ x_P + P_ss.B @ u - P_ss.B @ u_bar
        
        # Advance controller state.
        dot_x_ctrl = C_ss.A @ x_C + C_ss.B @ error

        # Concatenate state derivatives.
        x_dot = np.block([[dot_x_sys], [dot_x_ctrl]]).ravel()

        return x_dot


    # Find time-domain response by integrating the ODE
    sol = integrate.solve_ivp(
        closed_loop,
        (t_start, t_end),
        x_cl_IC,
        t_eval=t,
        rtol=1e-8,
        atol=1e-6,
        method='RK45',
    )

    # Extract states.
    sol_x = sol.y
    x_P = sol_x[:n_x_P, :]
    x_C = sol_x[n_x_P:, :]

    # Compute plant output, control signal, and ideal error.
    y = np.zeros(t.shape[0],)
    u = np.zeros(t.shape[0],)
    e = np.zeros(t.shape[0],)

    for i in range(time.size):

        # Reference at current time.
        r_now = np.interp(t[i], time, r).reshape((1, 1))
        
        # Noise at current time.
        n_now = np.interp(t[i], time, noise).reshape((1, 1))

        # Interpolation of u_bar and y_bar
        # Note, r_raw is used, because we are ``linearizing" about the current reference point.
        z_bar = np.interp(t[i], time, r_raw).reshape((1, 1))
        u_bar = np.interp(z_bar, z_range, u_range).reshape((1, 1))

        # Plant output, with noise.
        delta_z = P_ss.C @ x_P[:, [i]]
        y[i] = (z_bar + delta_z + n_now).ravel()[0]

        # Compute error.
        error = r_now - (z_bar + delta_z + n_now)
        e[i] = error.ravel()[0]

        # Compute control.
        delta_u = C_ss.C @ x_C[:, [i]] + C_ss.D @ error
        u[i] = (u_bar + delta_u).ravel()[0]

    return y, u, e


# Run simulation
y, u, e = simulate(P, C, t, r_raw, r, noise, u_range, z_range)


# %%
# Plots

y_tilde = y
u_tilde = u
e_tilde = e

# Max acceptable error and control values. 
e_nor_ref = e_nor_r 
u_nor_ref = 5  # V


# Plot
fig, ax = plt.subplots(2, 1, figsize=(height * gr, height))
ax[0].set_ylabel(r'$\tilde{y}(t)$ (LPM)')
ax[1].set_ylabel(r'$\tilde{u}(t)$ (V)')
for a in np.ravel(ax):
    a.set_xlabel(r'$t$ (s)')

ax[0].plot(t, y_tilde, '-', label=r'$\tilde{y}(t)$', color='C0')
ax[0].plot(t, r_tilde, '--', label=r'$\tilde{r}(t)$', color='C3')
ax[1].plot(t, u_tilde, '-', label=r'$\tilde{u}(t)$', color='C1')
ax[1].plot(t, u_nor_ref * np.ones(t.shape[0],), '--', label=r'$u_{nor, r}$', color='C6')
ax[0].legend(loc='lower right')
ax[1].legend(loc='lower right')
fig.tight_layout()
# fig.savefig(plots_dir / 'y_u_time_dom_response_tilde.pdf')

# Plot
fig, ax = plt.subplots(figsize=(height * gr, height))
ax.set_ylabel(r'$\tilde{y}(t)$ (LPM)')
ax.set_xlabel(r'$t$ (s)')
ax.plot(t, y_tilde, '-', label=r'$\tilde{y}(t)$', color='C0')
ax.plot(t, r_tilde, '--', label=r'$\tilde{r}(t)$', color='C3')
ax.legend(loc='best')
fig.tight_layout()
# fig.savefig(plots_dir / 'y_time_dom_response_tilde.pdf')

# Plot
fig, ax = plt.subplots(figsize=(height * gr, height))
ax.set_ylabel(r'$\tilde{e}(t)$ (LPM)')
ax.set_xlabel(r'$t$ (s)')
ax.plot(t, e_tilde, '-', label=r'$\tilde{e}(t)$', color='C0')
ax.plot(t, e_nor_ref * np.ones(t.shape[0],), '--', label=r'$e_{nor, r}$', color='C6')
ax.plot(t, -e_nor_ref * np.ones(t.shape[0],), '--', color='C6')
ax.legend(loc='upper right')
fig.tight_layout()
# fig.savefig(plots_dir / 'e_time_dom_response_tilde.pdf')

# Plot
fig, ax = plt.subplots(2, 1, figsize=(height * gr, height))
ax[0].set_ylabel(r'$\tilde{e}(t)$ (LPM)')
ax[1].set_ylabel(r'$\tilde{u}(t)$ (V)')
for a in np.ravel(ax):
    a.set_xlabel(r'$t$ (s)')

ax[0].plot(t, e_tilde, '-', label=r'$\tilde{e}(t)$', color='C0')
ax[0].plot(t, e_nor_ref * np.ones(t.shape[0],), '--', label=r'$e_{nor, r}$', color='C6')
ax[0].plot(t, -e_nor_ref * np.ones(t.shape[0],), '--', color='C6')
ax[1].plot(t, u_tilde, '-', label=r'$\tilde{u}(t)$', color='C1')
ax[1].plot(t, u_nor_ref * np.ones(t.shape[0],), '--', label=r'$u_{nor, r}$', color='C6')
ax[0].legend(loc='lower right')
ax[1].legend(loc='lower right')
fig.tight_layout()
# fig.savefig(plots_dir / 'u_e_time_dom_response_tilde.pdf')



# %%
# Energy estimate

power = np.zeros(t.size,)
for i in range(t.size):
    power[i] = u_tilde[i]**2

# Plot
fig, ax = plt.subplots(figsize=(height * gr, height))
ax.set_ylabel(r'$P(t)$ (V$^2$)')
ax.set_xlabel(r'$t$ (s)')
ax.plot(t, power, '-', label=r'$P(t)$', color='C0')
ax.legend(loc='best')
fig.tight_layout()
# fig.savefig(plots_dir / 'power_fb_vs_time.pdf')

# Integrate using Simpson's rule to get total ``energy" in units of V^2 s
energy = integrate.simpson(power, x=t)
print("The total energy consumed when using feedback control is", energy, '(V^2 s).')

# Compute total energy when pump is at max voltage all the time, or ``all out". 
energy_ao = integrate.simpson(max_V**2 * np.ones(t.size,), x=t)
print("The total energy consumed when using the max voltage all the time is", energy_ao, '(V^2 s).')
print("The percent energy saved relative to the max voltage all the time is", (energy_ao - energy) / energy_ao * 100, '(%).')


# %%
# Some stats

# print("The mean error is", np.mean(e_tilde), '(LPM). \n')
print("The mean error relative to the max flow rate is", np.mean(e_tilde) / max_LPM * 100, '(%). \n')

# print("The error standard deviation is", np.std(e_tilde), '(LPM). \n')
print("The error standard deviation relative to the max flow rate is", np.std(e_tilde) / max_LPM * 100, '(%). \n')

# print("The mean control effort is", np.mean(u_tilde), '(V). \n')
print("The mean control effort relative to the max control input is", np.mean(u_tilde) / max_V * 100, '(%). \n')

# print("The control effort standard deviation is", np.std(u_tilde), '(V). \n')
print("The control effort standard deviation relative to the max control input is", np.std(u_tilde) / max_V * 100, '(%). \n')

# print("The max error is", np.max(e_tilde), '(LPM). \n')
print("The max error relative to the max flow rate is", np.max(e_tilde) / max_LPM * 100, '(%). \n')

# print("The max control effort is", np.max(u_tilde), '(V). \n')
print("The max control effort relative to the max control input is", np.max(u_tilde) / max_V * 100, '(%). \n')


# %%
# Plot
#plt.show()

# %%
