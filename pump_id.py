# %%
# Libraries
import numpy as np
import control
from matplotlib import pyplot as plt
import pathlib
# Custom libraries 
import d2c
'''


TODO: FIX UNITS ON PLOT
TODO: LOOK INTO SCALING
TODO: IM GETTING RLY BAD ERRORS

import data
define train and test data
normalize -> form A and b
solve for x
compute stats
compute TF
train error
test error
plot etc

'''

#%%
#Import Data
#Plotting parameters
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')

path = pathlib.Path('load_data_sc/PRBS_DATA/')
all_files = sorted(path.glob("*.csv"))
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
'''
data_set1 = np.array(data[0])
data_set2 = np.array(data[1])
data_set3 = np.array(data[2])
data_set4 = np.array(data[3])
print("shape of data_set1",np.shape(data_set1))
# Find the max of the 1st (voltage) and 2nd (LPM) column across datasets 1-4
all_data = np.concatenate([data_set1, data_set2, data_set3, data_set4], axis=0)
u_max = np.max(all_data[:, 1])
y_max = np.max(all_data[:, 2])


print("Max Voltage (input u) across all datasets:", u_max)
print("Max LPM (output y) across all datasets:", y_max)
'''

#%%
#User Inputs
#Training data
training_data_sets = [1,2]

#n>=m ALWAYS
n=1 #order of denominator
m=0 #order of numerator


#%%
#Functions

def normalize_data(data):
    # Create a COPY to avoid modifying original data
    data_copy = data.copy()
    data_set1 = np.array(data[0])
    data_set2 = np.array(data[1])
    data_set3 = np.array(data[2])
    data_set4 = np.array(data[3])
    # Find the max of the 1st (voltage) and 2nd (LPM) column across datasets 1-4
    all_data = np.concatenate([data_set1, data_set2, data_set3, data_set4], axis=0)
    u_max = np.max(all_data[:, 1])
    y_max = np.max(all_data[:, 2])
    data_copy[:, :, 1] /= u_max
    data_copy[:, :, 2] /= y_max
    return data_copy, u_max, y_max

def form_A_b(y_train, u_train, n, m):
    N = y_train.shape[0]
    A = np.zeros((N-n,n+m+1))
    b = np.zeros((N-n,1))
    for k in range(N-n): #k is the index of the data points
        for j in range(n): 
            A[k,j] = -y_train[k+n-j-1]
        for j in range(m+1):
            A[k,n+j] = u_train[k+m-j]
    for k in range(N-n):
        b[k] = y_train[k+n]
    return A, b

def get_CT_tf(A, b, n, u_max, y_max, dt):
    N = A.shape[0]
    cond_A = np.linalg.cond(A)
    print("cond_A = ",cond_A)
    if cond_A > 1e10:
        print("The A matrix is ill-conditioned")
        exit()
    else:
        print("The A matrix is well-conditioned")
    
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    
    # Check if solution has significant imaginary parts
    if np.any(np.abs(np.imag(x)) > 1e-10):
        print("WARNING: Solution has imaginary components! Taking real part.")
        print(f"Max imaginary part: {np.max(np.abs(np.imag(x))):.2e}")
        x = np.real(x)

    # Compute the MSE, MSO, NMSE on normalized data
    MSE = 1/N*(np.linalg.norm(b-A@x)**2)
    MSO = 1/N*(np.linalg.norm(b)**2)
    NMSE = MSE/MSO

    print('The MSE for training is', MSE)
    print('The MSO for training is', MSO)
    print('The NMSE for training is', NMSE, '\n')
    '''
    # Compute the uncertainty and relative uncertainty. 
    # Compute the uncertainty and relative uncertainty. 
    sigma = 1/(N-(n+m+1))* np.linalg.norm(b-A@x)*np.linalg.inv(A.T@A)
    sigma_diag = np.diag(sigma)
    #Divides element wise
    rel_unc = sigma_diag/np.abs(x)*100
    print('The uncertainty is         ', sigma_diag)
    print('The relative uncertainty is', rel_unc, '%\n')
    '''
    Pd_ID_den = np.hstack([1, x[0:n].reshape(-1,)])  # denominator coefficients of DT TF
    Pd_ID_num = x[n:].reshape(-1,)  # numerator coefficients of DT TF
    Pd_ID = y_max / u_max * control.tf(Pd_ID_num, Pd_ID_den, dt)
    Pc_ID = d2c.d2c(Pd_ID)
    return Pc_ID, Pd_ID, u_max, y_max

def get_tf(data,n,m,training_data_sets):
    dt = data[0, 1, 0] - data[0, 0, 0]
    print("dt = ",dt)
    norm_data, u_max, y_max = normalize_data(data)
    A_stacked = None
    b_stacked = None
    for i in training_data_sets:
        data_set = np.array(norm_data[i])
        y_train = data_set[:, 2]
        u_train = data_set[:, 1]
        A, b = form_A_b(y_train, u_train, n, m)
        if A_stacked is None:
            A_stacked = A
            b_stacked = b
        else:
            A_stacked = np.vstack([A_stacked, A])
            b_stacked = np.vstack([b_stacked, b])
    Pc_ID, Pd_ID, u_max, y_max = get_CT_tf(A_stacked, b_stacked, n, u_max, y_max, dt)
    return Pc_ID, Pd_ID

def test_error(Pd_ID, data, plot=False, test_indices=[0,1,2,3]):
    dt = data[0, 1, 0] - data[0, 0, 0]
    for i in test_indices:
        test_data = np.array(data[i])
        t = test_data[:, 0]
        y_test = test_data[:, 2]
        u_test = test_data[:, 1]
        # Use discrete-time TF with the original unnormalized data
        td_ID_test, yd_ID_test = control.forced_response(Pd_ID, t, u_test)
        if plot:
            fig, ax = plt.subplots(2, 1)
            ax[0].set_ylabel(r'$u(t)$ (V)')
            ax[1].set_ylabel(r'$y(t)$ (LPM)')
            ax[0].plot(t, u_test, '--', label='input', color='C0')
            ax[1].plot(t, y_test, label='output', color='C1')
            ax[1].plot(td_ID_test, yd_ID_test, '-.', label="IDed output", color='C2')
            ax[0].set_title(f'Dataset {i}')
            for a in np.ravel(ax):
                a.set_xlabel(r'$t$ (s)')
                a.legend(loc='upper right')
            fig.tight_layout()
            plt.show()
        error = yd_ID_test - y_test
        y_max_dataset = np.max(np.abs(y_test))
        print(f"Dataset {i}: abs avg error = {np.mean(np.abs(error)):.4f}, rel avg error = {np.mean(np.abs(error/y_max_dataset))*100:.2f}%")
'''
print("Training on data set: ", training_data_sets)
print("="*50)
Pc_ID, Pd_ID = get_tf(data,n,m,training_data_sets)
print("\nContinuous-time TF:")
print(Pc_ID)
print("\nDiscrete-time TF:")
print(Pd_ID)
print("\n" + "="*50)
print("Testing on all datasets:")
print("="*50)
test_error(Pd_ID, data, plot=True)
'''
#Test on datasets seperately
num_arr = np.array([])
den_arr = np.array([])
Pc_ID3, Pd_ID3 = get_tf(data,n,m,[3])
num_arr = np.append(num_arr, Pc_ID3.num[0])
den_arr = np.append(den_arr, Pc_ID3.den[0][0][1])
Pc_ID2, Pd_ID2 = get_tf(data,n,m,[2])
num_arr = np.append(num_arr, Pc_ID2.num[0])
den_arr = np.append(den_arr, Pc_ID2.den[0][0][1])
Pc_ID1, Pd_ID1 = get_tf(data,n,m,[1])
num_arr = np.append(num_arr, Pc_ID1.num[0])
den_arr = np.append(den_arr, Pc_ID1.den[0][0][1])
Pc_ID0, Pd_ID0 = get_tf(data,n,m,[0])
num_arr = np.append(num_arr, Pc_ID0.num[0])
den_arr = np.append(den_arr, Pc_ID0.den[0][0][1])

#%%
#plot bode plots of all datasets on same plot using Pc_ID3,2,1,0
N = 1000
w = np.linspace(0.01, 1000, N)

mag_Pc_ID3_bode = 20*np.log10(np.abs(Pc_ID3(1j*w)))
mag_Pc_ID2_bode = 20*np.log10(np.abs(Pc_ID2(1j*w)))
mag_Pc_ID1_bode = 20*np.log10(np.abs(Pc_ID1(1j*w)))
mag_Pc_ID0_bode = 20*np.log10(np.abs(Pc_ID0(1j*w)))
phase_Pc_ID3_bode = np.angle(Pc_ID3(1j*w))
phase_Pc_ID2_bode = np.angle(Pc_ID2(1j*w))
phase_Pc_ID1_bode = np.angle(Pc_ID1(1j*w))
phase_Pc_ID0_bode = np.angle(Pc_ID0(1j*w))
fig, ax = plt.subplots(2, 1)
ax[0].semilogx(w, mag_Pc_ID3_bode, label='Pc_ID3')
ax[0].semilogx(w, mag_Pc_ID2_bode, label='Pc_ID2')
ax[0].semilogx(w, mag_Pc_ID1_bode, label='Pc_ID1')
ax[0].semilogx(w, mag_Pc_ID0_bode, label='Pc_ID0')
ax[0].set_ylabel(r'$|P_c(j\omega)|$ (dB)')
ax[1].semilogx(w, phase_Pc_ID3_bode, label='Pc_ID3')
ax[1].semilogx(w, phase_Pc_ID2_bode, label='Pc_ID2')
ax[1].semilogx(w, phase_Pc_ID1_bode, label='Pc_ID1')
ax[1].semilogx(w, phase_Pc_ID0_bode, label='Pc_ID0')
ax[1].set_ylabel(r'$\angle P_c(j\omega)$ (rad)')
ax[0].set_xlabel(r'$\omega$ (rad/s)')
ax[1].set_xlabel(r'$\omega$ (rad/s)')
ax[0].set_title("Magnitude Bode Plot of Pc_ID3,2,1,0")
ax[1].set_title("Phase Bode Plot of Pc_ID3,2,1,0")
ax[0].legend(loc='upper right')
ax[1].legend(loc='upper right')
fig.tight_layout(pad=2.0)  # Add some vertical space to prevent overlapping
plt.show()

# %%


#LS NOT WORKING RN, GOING TO TAKE AVERAGE FOR NOMINAL MODEL FOR NOW

#Formulate New LS to find nominal model
#First assume 1st order model
#SOMETHING WRONG WITH THIS, P_NOM IS NOT INBETWEEN THE OTHERS
N = 1000
w = np.logspace(-2, 3, N) #Use logspace to get even spacing in log scale
A_nominal = np.zeros((N*4,2), dtype=complex)
b_nominal = np.zeros((N*4,1), dtype=complex)
for i in range(N):
    A_nominal[4*i,0] = 1
    A_nominal[4*i+1,0] = 1
    A_nominal[4*i+2,0] = 1
    A_nominal[4*i+3,0] = 1
    A_nominal[4*i,1] = -Pc_ID0(1j*w[i])*w[i]*1j
    A_nominal[4*i+1,1] = -Pc_ID1(1j*w[i])*w[i]*1j
    A_nominal[4*i+2,1] = -Pc_ID2(1j*w[i])*w[i]*1j
    A_nominal[4*i+3,1] = -Pc_ID3(1j*w[i])*w[i]*1j
    b_nominal[4*i,0] = Pc_ID0(1j*w[i])
    b_nominal[4*i+1,0] = Pc_ID1(1j*w[i])
    b_nominal[4*i+2,0] = Pc_ID2(1j*w[i])
    b_nominal[4*i+3,0] = Pc_ID3(1j*w[i])

A_real = np.real(A_nominal)
A_imag = np.imag(A_nominal)
b_real = np.real(b_nominal)
b_imag = np.imag(b_nominal)
A1 = np.vstack([A_real, A_imag])
b1 = np.vstack([b_real, b_imag])
#check condition number of A1
cond_A1 = np.linalg.cond(A1)
print("cond_A1 = ", cond_A1)
#solve for x
x = np.linalg.lstsq(A1, b1, rcond=None)[0]
print("x = ", x)
print("x = ", x)
k_nom = float(x[0])
tau_nom = float(x[1])

Pc_nominal = control.tf([k_nom/tau_nom], [1, 1/tau_nom])
print("Pc_nominal = ", Pc_nominal)
# %%

#%%
#Bode with nominal model
N = 1000
w = np.linspace(0.01, 1000, N)

mag_Pc_ID3_bode = 20*np.log10(np.abs(Pc_ID3(1j*w)))
mag_Pc_ID2_bode = 20*np.log10(np.abs(Pc_ID2(1j*w)))
mag_Pc_ID1_bode = 20*np.log10(np.abs(Pc_ID1(1j*w)))
mag_Pc_ID0_bode = 20*np.log10(np.abs(Pc_ID0(1j*w)))
mag_Pc_nominal_bode = 20*np.log10(np.abs(Pc_nominal(1j*w)))
phase_Pc_ID3_bode = np.angle(Pc_ID3(1j*w))
phase_Pc_ID2_bode = np.angle(Pc_ID2(1j*w))
phase_Pc_ID1_bode = np.angle(Pc_ID1(1j*w))
phase_Pc_ID0_bode = np.angle(Pc_ID0(1j*w))
phase_Pc_nominal_bode = np.angle(Pc_nominal(1j*w))
fig, ax = plt.subplots(2, 1)
ax[0].semilogx(w, mag_Pc_ID3_bode, label='Pc_ID3')
ax[0].semilogx(w, mag_Pc_ID2_bode, label='Pc_ID2')
ax[0].semilogx(w, mag_Pc_ID1_bode, label='Pc_ID1')
ax[0].semilogx(w, mag_Pc_ID0_bode, label='Pc_ID0')
ax[0].semilogx(w, mag_Pc_nominal_bode, label='Pc_nominal')
ax[0].set_ylabel(r'$|P_c(j\omega)|$ (dB)')
ax[1].semilogx(w, phase_Pc_ID3_bode, label='Pc_ID3')
ax[1].semilogx(w, phase_Pc_ID2_bode, label='Pc_ID2')
ax[1].semilogx(w, phase_Pc_ID1_bode, label='Pc_ID1')
ax[1].semilogx(w, phase_Pc_ID0_bode, label='Pc_ID0')
ax[1].semilogx(w, phase_Pc_nominal_bode, label='Pc_nominal')
ax[1].set_ylabel(r'$\angle P_c(j\omega)$ (rad)')
ax[0].set_xlabel(r'$\omega$ (rad/s)')
ax[1].set_xlabel(r'$\omega$ (rad/s)')
ax[0].set_title("Magnitude Bode Plot of Pc_ID3,2,1,0 and Pc_nominal")
ax[1].set_title("Phase Bode Plot of Pc_ID3,2,1,0 and Pc_nominal")
ax[0].legend(loc='upper right')
ax[1].legend(loc='upper right')
fig.tight_layout(pad=2.0)  # Add some vertical space to prevent overlapping
plt.show()
# %%

#%%
#plot the residuals
residuals0 = Pc_ID0(1j*w)/Pc_nominal(1j*w)-1
residuals1 = Pc_ID1(1j*w)/Pc_nominal(1j*w)-1
residuals2 = Pc_ID2(1j*w)/Pc_nominal(1j*w)-1
residuals3 = Pc_ID3(1j*w)/Pc_nominal(1j*w)-1
#Magnitude of residuals in dB
mag_residuals0_dB = 20*np.log10(np.abs(residuals0))
mag_residuals1_dB = 20*np.log10(np.abs(residuals1))
mag_residuals2_dB = 20*np.log10(np.abs(residuals2))
mag_residuals3_dB = 20*np.log10(np.abs(residuals3))

plt.semilogx(w, mag_residuals0_dB, label=f'Residuals of Pc_ID0')
plt.semilogx(w, mag_residuals1_dB , label=f'Residuals of Pc_ID1')
plt.semilogx(w, mag_residuals2_dB, label=f'Residuals of Pc_ID2')
plt.semilogx(w, mag_residuals3_dB, label=f'Residuals of Pc_ID3')

plt.legend(loc='upper right')
plt.xlabel(r'$\omega$ (rad/s)')
plt.ylabel(r'$\Delta P_c(j\omega)$ (dB)')
plt.title("Magnitude of Residuals of Pc_ID3,2,1,0")
plt.show()

#%%
#Use DECAR unc_bound.py to compute uncertainty bounds
import unc_bound
P = Pc_nominal
P_off_nom  = [Pc_ID0, Pc_ID1, Pc_ID2, Pc_ID3]
N = len(P_off_nom)
R = unc_bound.residuals(P, P_off_nom)
mag_max_dB, mag_max_abs = unc_bound.residual_max_mag(R, w)

# Bode plot
N_w = 500
w_shared = np.logspace(-1, 3, N_w)

# Compute magnitude part of R(s) in both dB and in absolute units
mag_max_dB, mag_max_abs = unc_bound.residual_max_mag(R, w_shared)

# Plot Bode magnitude plot in dB and in absolute units
fig, ax = plt.subplots(2, 1)
ax[0].set_xlabel(r'$\omega$ (rad/s)')
ax[0].set_ylabel(r'Magnitude (dB)')
ax[1].set_xlabel(r'$\omega$ (rad/s)')
ax[1].set_ylabel(r'Magnitude (absolute)')
for i in range(N):
    mag_abs, _, _ = control.frequency_response(R[i], w_shared)
    mag_dB = 20 * np.log10(mag_abs)
    # Magnitude plot (dB)
    ax[0].semilogx(w_shared, mag_dB, '--', color='C0', linewidth=1)
    # Magnitude plot (absolute).
    ax[1].semilogx(w_shared, mag_abs, '--', color='C0', linewidth=1)

# Magnitude plot (dB).
ax[0].semilogx(w_shared, mag_max_dB, '-', color='C4', label='upper bound')
# Magnitude plot (absolute).
ax[1].semilogx(w_shared, mag_max_abs, '-', color='C4', label='upper bound')
ax[0].legend(loc='best')
ax[1].legend(loc='best')
# fig.savefig(path.joinpath('1st_order_unstable_R.pdf'))


# %%
# Find W2

# Order of W2
nW2 = 3

# Calculate optimal upper bound transfer function.
W2 = (unc_bound.upperbound(omega=w_shared, upper_bound=mag_max_abs, degree=nW2)).minreal()
print("The optimal weighting function W_2(s) is ", W2)


# Plot the Bode magnitude plot of the optimal W_2(s) transfer function

# Compute magnitude part of W_2(s) in absolute units
mag_W2_abs, _, _ = control.frequency_response(W2, w_shared)
# Compute magnitude part of W_2(s) in dB
mag_W2_dB = 20 * np.log10(mag_W2_abs)

fig, ax = plt.subplots(2, 1)
ax[0].set_xlabel(r'$\omega$ (rad/s)')
ax[0].set_ylabel(r'Magnitude (dB)')
ax[1].set_xlabel(r'$\omega$ (rad/s)')
ax[1].set_ylabel(r'Magnitude (absolute)')
for i in range(N):
    mag_abs, _, _ = control.frequency_response(R[i], w_shared)
    mag_dB = 20 * np.log10(mag_abs)
    # Magnitude plot (dB)
    ax[0].semilogx(w_shared, mag_dB, '--', color='C0', linewidth=1)
    # Magnitude plot (absolute).
    ax[1].semilogx(w_shared, mag_abs, '--', color='C0', linewidth=1)

# Magnitude plot (dB).
ax[0].semilogx(w_shared, mag_max_dB, '-', color='C4', label='upper bound')
# Magnitude plot (absolute).
ax[1].semilogx(w_shared, mag_max_abs, '-', color='C4', label='upper bound')
# Magnitude plot (dB).
ax[0].semilogx(w_shared, mag_W2_dB, '-', color='seagreen', label='optimal bound')
# Magnitude plot (absolute).
ax[1].semilogx(w_shared, mag_W2_abs, '-', color='seagreen', label='optimal bound')
ax[0].legend(loc='best')
ax[1].legend(loc='best')
# fig.tight_layout()
# fig.savefig(path.joinpath('1st_order_unstable_W2.pdf'))
plt.show()