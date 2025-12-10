# %%
# Libraries
import numpy as np
import control
from matplotlib import pyplot as plt
import pathlib

# Set up plots directory
plots_dir = pathlib.Path("/Users/aidan1/Documents/McGill/MECH412/MECH 412 Pump Project/plots")
plots_dir.mkdir(exist_ok=True)
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
def standardize_data(data):
    """
    Standardize data by centering each dataset around its OWN mean (equilibrium).
    
    For perturbation modeling, each dataset should be centered around its own 
    equilibrium point (per-dataset mean), not a global mean. This ensures the 
    model captures perturbation dynamics that generalize across operating points.
    """
    # Create a COPY to avoid modifying original data
    data_copy = data.copy()
    
    # Center each dataset around its OWN mean (equilibrium point)
    for i in range(len(data)):
        u_mean_i = np.mean(data[i][:, 1])
        y_mean_i = np.mean(data[i][:, 2])
        data_copy[i, :, 1] = data[i, :, 1] - u_mean_i
        data_copy[i, :, 2] = data[i, :, 2] - y_mean_i
    
    # Find scaling factors (max of centered data across all datasets)
    u_max = max(np.max(np.abs(data_copy[i][:, 1])) for i in range(len(data)))
    y_max = max(np.max(np.abs(data_copy[i][:, 2])) for i in range(len(data)))
    
    # Normalize by max
    data_copy[:, :, 1] /= u_max
    data_copy[:, :, 2] /= y_max
    
    return data_copy, u_max, y_max

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

def get_tf(data,n,m,training_data_sets,standardize=True):
    """
    Identify transfer function from PRBS data.
    
    When standardize=True (default), each dataset is centered around its own mean
    before identification. This is appropriate for perturbation modeling.
    """
    dt = data[0, 1, 0] - data[0, 0, 0]
    print("dt = ",dt)
    if standardize:
        norm_data, u_max, y_max = standardize_data(data)
    else:
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
    """
    Test the identified model on data.
    
    Each test dataset is centered around its OWN mean (equilibrium point),
    matching the per-dataset centering used during training.
    
    Args:
        Pd_ID: Discrete-time transfer function
        data: All datasets
        plot: Whether to plot results
        test_indices: Which dataset indices to test on
    """
    dt = data[0, 1, 0] - data[0, 0, 0]
    for i in test_indices:
        test_data = np.array(data[i])
        t = test_data[:, 0]
        y_test = test_data[:, 2].copy()
        u_test = test_data[:, 1].copy()
        
        # Center around THIS dataset's mean (its equilibrium point)
        u_test_mean = np.mean(u_test)
        y_test_mean = np.mean(y_test)
        u_test_centered = u_test - u_test_mean
        y_test_centered = y_test - y_test_mean
        
        # Use discrete-time TF with mean-centered data
        td_ID_test, yd_ID_test = control.forced_response(Pd_ID, t, u_test_centered)
        
        if plot:
            fig, ax = plt.subplots(2, 1)
            ax[0].set_ylabel(r'$\Delta u(t)$ (V)')
            ax[1].set_ylabel(r'$\Delta y(t)$ (LPM)')
            ax[0].plot(t, u_test_centered, '--', label='input (centered)', color='C0')
            ax[1].plot(t, y_test_centered, label='output (centered)', color='C1')
            ax[1].plot(td_ID_test, yd_ID_test, '-.', label="model output", color='C2')
            ax[0].set_title(f'Dataset {i} (centered around its mean)')
            for a in np.ravel(ax):
                a.set_xlabel(r'$t$ (s)')
                a.legend(loc='upper right')
            fig.tight_layout()
            fig.savefig(plots_dir / f'dataset_{i}_test_error.pdf')
            plt.show()
        
        error = yd_ID_test - y_test_centered
        y_max_dataset = np.max(np.abs(y_test_centered))
        #get %VAF
        var_e = 1/len(error)*np.sum(error**2)
        var_y = 1/len(error)*np.sum(y_test_centered**2)
        VAF = 1 - var_e/var_y
        print(f"Dataset {i}: %VAF = {VAF*100:.2f}%")
        print(f"Dataset {i}: abs avg error = {np.mean(np.abs(error)):.4f}, rel avg error = {np.mean(np.abs(error/y_max_dataset))*100:.2f}%")

def create_hf_offnominal(Pc_base, omega_n, zeta=0.3):
    """
    Create a stable higher-order off-nominal model by adding a 2nd-order
    resonance to a stable 1st-order base model.
    
    This is useful for uncertainty modeling at high frequencies where you want
    to capture unmodeled dynamics without introducing instability.
    
    The resulting model has:
    - 3rd order denominator (1st order base + 2nd order resonance)
    - Constant numerator (strictly proper)
    - Guaranteed stable poles
    
    Args:
        Pc_base: A stable 1st-order CT transfer function (e.g., K/(s+a))
        omega_n: Natural frequency of the resonance (rad/s)
        zeta: Damping ratio (0 < zeta < 1 for underdamped resonance)
              Lower zeta = sharper resonance peak = more HF uncertainty
    
    Returns:
        Pc_hf: A stable 3rd-order CT transfer function
    
    Example:
        >>> Pc_base = control.tf([20], [1, 10])  # 20/(s+10)
        >>> Pc_hf = create_hf_offnominal(Pc_base, omega_n=50, zeta=0.2)
        >>> # Pc_hf is now 3rd order with resonance at 50 rad/s
    """
    # 2nd-order resonance: omega_n^2 / (s^2 + 2*zeta*omega_n*s + omega_n^2)
    # This has unity DC gain, so it preserves low-frequency behavior
    resonance = control.tf([omega_n**2], [1, 2*zeta*omega_n, omega_n**2])
    return (Pc_base * resonance).minreal()

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

Pc_ID3, Pd_ID3 = get_tf(data,n,m,[3])
print("Pc_ID3 = ", Pc_ID3)

Pc_ID2, Pd_ID2 = get_tf(data,n,m,[2])
print("Pc_ID2 = ", Pc_ID2)

Pc_ID1, Pd_ID1 = get_tf(data,n,m,[1])
print("Pc_ID1 = ", Pc_ID1)

Pc_ID0, Pd_ID0 = get_tf(data,n,m,[0])
print("Pc_ID0 = ", Pc_ID0)


#Test Generalization (each dataset centered around its own mean)
# test_error(Pd_ID3, data, plot=True)
# test_error(Pd_ID2, data, plot=True)
# test_error(Pd_ID1, data, plot=True)
# test_error(Pd_ID0, data, plot=True)

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
fig.savefig(plots_dir / 'bode_all_datasets.pdf')
plt.show()

# %%


#LS Working only for 1st order currently

#Formulate New LS to find nominal model
#First assume 1st order model
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

dt = data[0, 1, 0] - data[0, 0, 0]
Pc_nominal = control.tf([k_nom/tau_nom], [1, 1/tau_nom])
print("Pc_nominal = ", Pc_nominal)
Pd_nominal = control.c2d(Pc_nominal, dt, method='zoh')
print("Pd_nominal = ", Pd_nominal)

#Test Generalization (each dataset centered around its own mean)
test_error(Pd_nominal, data, plot=True)

# %%

#%%
#Bode with nominal model
# Add a sharp cutoff frequency at 10 rad/s to the nominal model
# Pc_ID7 is now the nominal plant multiplied by a lowpass filter with cutoff at 10 rad/s
#Pc_ID7 = Pc_nominal * control.TransferFunction([10], [1, 10])
# Create family of HF off-nominal models
Pc_hf1 = create_hf_offnominal(Pc_nominal, omega_n=60, zeta=0.3)

test_error(Pc_hf1, data, plot=True)

Pc_hf2 = create_hf_offnominal(Pc_ID1, omega_n=20, zeta=0.9)
Pc_hf3 = create_hf_offnominal(Pc_ID2, omega_n=20, zeta=0.9)
Pc_hf4 = create_hf_offnominal(Pc_ID3, omega_n=20, zeta=0.9)

# test_error(Pc_hf1, data, plot=True)
# test_error(Pc_hf2, data, plot=True)
# test_error(Pc_hf3, data, plot=True)
# test_error(Pc_hf4, data, plot=True)

# Include in your uncertainty set
P_off_nom = [Pc_ID0, Pc_ID1, Pc_ID2, Pc_ID3, Pc_hf1]#, Pc_hf2]#, Pc_hf3, Pc_hf4]# Pc_hf2, Pc_hf3]
Pc_ID7 = Pc_hf1

print("Pc_ID7 = ", Pc_ID7)
#Pc_ID4, Pd_ID4 = get_tf(data,n+1,m+1,[0])
#Pc_ID5, Pd_ID5 = get_tf(data,n+2,m+2,[0])
#Pc_ID6, Pd_ID6 = get_tf(data,n+3,m+3,[0])

N = 1000
w = np.linspace(0.01, 1000, N)

mag_Pc_ID3_bode = 20*np.log10(np.abs(Pc_ID3(1j*w)))
mag_Pc_ID2_bode = 20*np.log10(np.abs(Pc_ID2(1j*w)))
mag_Pc_ID1_bode = 20*np.log10(np.abs(Pc_ID1(1j*w)))
mag_Pc_ID0_bode = 20*np.log10(np.abs(Pc_ID0(1j*w)))
mag_Pc_nominal_bode = 20*np.log10(np.abs(Pc_nominal(1j*w)))
mag_Pc_ID7_bode = 20*np.log10(np.abs(Pc_ID7(1j*w)))
# mag_Pc_ID4_bode = 20*np.log10(np.abs(Pc_ID4(1j*w)))
# mag_Pc_ID5_bode = 20*np.log10(np.abs(Pc_ID5(1j*w)))
# mag_Pc_ID6_bode = 20*np.log10(np.abs(Pc_ID6(1j*w)))
phase_Pc_ID3_bode = np.angle(Pc_ID3(1j*w))
phase_Pc_ID2_bode = np.angle(Pc_ID2(1j*w))
phase_Pc_ID1_bode = np.angle(Pc_ID1(1j*w))
phase_Pc_ID0_bode = np.angle(Pc_ID0(1j*w))
phase_Pc_nominal_bode = np.angle(Pc_nominal(1j*w))
phase_Pc_ID7_bode = np.angle(Pc_ID7(1j*w))
# phase_Pc_ID4_bode = np.angle(Pc_ID4(1j*w))
# phase_Pc_ID5_bode = np.angle(Pc_ID5(1j*w))
# phase_Pc_ID6_bode = np.angle(Pc_ID6(1j*w))
fig, ax = plt.subplots(2, 1)
ax[0].semilogx(w, mag_Pc_ID3_bode, label='Pc_ID3')
ax[0].semilogx(w, mag_Pc_ID2_bode, label='Pc_ID2')
ax[0].semilogx(w, mag_Pc_ID1_bode, label='Pc_ID1')
ax[0].semilogx(w, mag_Pc_ID0_bode, label='Pc_ID0')
ax[0].semilogx(w, mag_Pc_nominal_bode, label='Pc_nominal')
ax[0].semilogx(w, mag_Pc_ID7_bode, label='Pc_ID7')
# ax[0].semilogx(w, mag_Pc_ID4_bode, label='Pc_ID4')
# ax[0].semilogx(w, mag_Pc_ID5_bode, label='Pc_ID5')
# ax[0].semilogx(w, mag_Pc_ID6_bode, label='Pc_ID6')
ax[0].set_ylabel(r'$|P_c(j\omega)|$ (dB)')
ax[1].semilogx(w, phase_Pc_ID3_bode, label='Pc_ID3')
ax[1].semilogx(w, phase_Pc_ID2_bode, label='Pc_ID2')
ax[1].semilogx(w, phase_Pc_ID1_bode, label='Pc_ID1')
ax[1].semilogx(w, phase_Pc_ID0_bode, label='Pc_ID0')
ax[1].semilogx(w, phase_Pc_nominal_bode, label='Pc_nominal')
ax[1].semilogx(w, phase_Pc_ID7_bode, label='Pc_ID7')
# ax[1].semilogx(w, phase_Pc_ID4_bode, label='Pc_ID4')
# ax[1].semilogx(w, phase_Pc_ID5_bode, label='Pc_ID5')
# ax[1].semilogx(w, phase_Pc_ID6_bode, label='Pc_ID6')
ax[1].set_ylabel(r'$\angle P_c(j\omega)$ (rad)')
ax[0].set_xlabel(r'$\omega$ (rad/s)')
ax[1].set_xlabel(r'$\omega$ (rad/s)')
ax[0].set_title("Magnitude Bode Plot of Pc_ID3,2,1,0 and Pc_nominal")
ax[1].set_title("Phase Bode Plot of Pc_ID3,2,1,0 and Pc_nominal")
ax[0].legend(loc='upper right')
ax[1].legend(loc='upper right')
fig.tight_layout(pad=2.0)  # Add some vertical space to prevent overlapping
fig.savefig(plots_dir / 'bode_nominal_and_datasets.pdf')
plt.show()



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
plt.savefig(plots_dir / 'residuals_magnitude.pdf')
plt.show()

#%%
#Use DECAR unc_bound.py to compute uncertainty bounds
import unc_bound
P = Pc_nominal


#generate higher order off nominal models


#P_off_nom  = [Pc_ID0, Pc_ID1, Pc_ID2, Pc_ID3, Pc_ID7]

N = len(P_off_nom)
R = unc_bound.residuals(P, P_off_nom)
mag_max_dB, mag_max_abs = unc_bound.residual_max_mag(R, w)

# Bode plot
N_w = 500
w_shared = np.logspace(-1, 3, N_w)
R2 = unc_bound.residuals(Pc_ID2, [Pc_ID0, Pc_ID1, Pc_ID3])
mag_max_dB2, mag_max_abs2 = unc_bound.residual_max_mag(R2, w_shared)
print("average max magnitude in dB for Pc_ID2 = ", np.mean(mag_max_dB2))

# Compute magnitude part of R(s) in both dB and in absolute units
mag_max_dB, mag_max_abs = unc_bound.residual_max_mag(R, w_shared)
print("average max magnitude in dB = ", np.mean(mag_max_dB))


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
fig.savefig(plots_dir / 'residuals_upper_bound.pdf')


# %%
# Find W2

# Order of W2
nW2 = 4

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
fig.tight_layout()
fig.savefig(plots_dir / 'uncertainty_W2.pdf')
plt.show()
# %%
