# %%
# Libraries
import numpy as np
import control
from matplotlib import pyplot as plt
import pathlib
import warnings
# Custom libraries 
import d2c

# Suppress complex warning from d2c conversion
warnings.filterwarnings('ignore', 'Casting complex values to real discards the imaginary part')

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

#%%
#User Inputs
#Training data
training_data_sets = [0,1,2,3]  # Use all for robust training
test_data_set = 4
#n>=m ALWAYS
n=2 #order of denominator (keep simple - higher orders overfit!)
m=1 #order of numerator

# NEW: Option to remove initial transient
skip_initial_samples = 150  # Skip first 15 seconds (150 samples @ 0.1s) - INCREASED for better results
remove_dc_offset = True  # Work with deviations from mean

#%%
#Functions

def preprocess_data(u, y, skip_initial=0, remove_dc=False):
    """
    Preprocess data to handle initial transients and DC offsets
    
    Parameters:
    -----------
    u : array - input signal
    y : array - output signal  
    skip_initial : int - number of initial samples to skip
    remove_dc : bool - whether to remove DC component (work with deviations)
    
    Returns:
    --------
    u_proc, y_proc : processed signals
    u_offset, y_offset : DC offsets removed (0 if remove_dc=False)
    """
    # Skip initial transient
    u_proc = u[skip_initial:].copy()
    y_proc = y[skip_initial:].copy()
    
    # Remove DC offset if requested
    if remove_dc:
        u_offset = np.mean(u_proc)
        y_offset = np.mean(y_proc)
        u_proc = u_proc - u_offset
        y_proc = y_proc - y_offset
    else:
        u_offset = 0.0
        y_offset = 0.0
    
    return u_proc, y_proc, u_offset, y_offset

def normalize_data(data, training_data_sets, skip_initial=0, remove_dc=False):
    """
    Normalize data using only training datasets
    Also applies preprocessing (skip initial transient, remove DC)
    """
    # First preprocess and find max values from training data only
    u_max_list = []
    y_max_list = []
    u_offsets = []
    y_offsets = []
    
    for i in training_data_sets:
        data_set = np.array(data[i])
        u = data_set[:, 1]
        y = data_set[:, 2]
        
        u_proc, y_proc, u_offset, y_offset = preprocess_data(u, y, skip_initial, remove_dc)
        
        u_max_list.append(np.max(np.abs(u_proc)))
        y_max_list.append(np.max(np.abs(y_proc)))
        u_offsets.append(u_offset)
        y_offsets.append(y_offset)
    
    u_max = np.max(u_max_list)
    y_max = np.max(y_max_list)
    
    # Use average offsets from training data
    u_offset_avg = np.mean(u_offsets)
    y_offset_avg = np.mean(y_offsets)
    
    # Create normalized copy of all data
    data_copy = []
    for i in range(len(data)):
        data_set = np.array(data[i])
        t = data_set[:, 0]
        u = data_set[:, 1]
        y = data_set[:, 2]
        
        # Apply same preprocessing to all datasets
        u_proc, y_proc, _, _ = preprocess_data(u, y, skip_initial, remove_dc)
        t_proc = t[skip_initial:]
        
        # Normalize
        u_norm = u_proc / u_max
        y_norm = y_proc / y_max
        
        data_copy.append(np.column_stack([t_proc, u_norm, y_norm]))
    
    return np.array(data_copy), u_max, y_max, u_offset_avg, y_offset_avg

def form_A_b(y_train, u_train, n, m):
    N = y_train.shape[0]
    A = np.zeros((N-n,n+m+1))
    b = np.zeros((N-n,1))
    for k in range(N-n):
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

    # Compute the uncertainty and relative uncertainty
    sigma = 1/(N-(n+m+1))* np.linalg.norm(b-A@x)*np.linalg.inv(A.T@A)
    sigma_diag = np.sqrt(np.abs(np.diag(sigma)))  # Take sqrt for standard deviation
    rel_unc = sigma_diag/np.abs(x.flatten())*100
    print('The uncertainty is         ', sigma_diag)
    print('The relative uncertainty is', rel_unc, '%\n')

    Pd_ID_den = np.hstack([1, x[0:n].reshape(-1,)])
    Pd_ID_num = x[n:].reshape(-1,)
    
    # Create normalized TF first
    Pd_ID_norm = control.tf(Pd_ID_num, Pd_ID_den, dt)
    
    # Then scale to get actual TF
    Pd_ID = y_max / u_max * Pd_ID_norm
    
    # Check poles
    poles_d = control.poles(Pd_ID)
    print("Discrete-time poles:", poles_d)
    print("Pole magnitudes:", np.abs(poles_d))
    stable = np.all(np.abs(poles_d) < 1)
    print(f"System is {'stable' if stable else 'UNSTABLE'}\n")
    
    Pc_ID = d2c.d2c(Pd_ID)
    return Pc_ID, Pd_ID, u_max, y_max

def get_tf(data, n, m, training_data_sets, skip_initial=0, remove_dc=False):
    dt = data[0, 1, 0] - data[0, 0, 0]
    print("dt = ",dt)
    print(f"Skipping first {skip_initial} samples ({skip_initial*dt:.1f}s)")
    print(f"Remove DC offset: {remove_dc}\n")
    
    norm_data, u_max, y_max, u_offset, y_offset = normalize_data(
        data, training_data_sets, skip_initial, remove_dc
    )
    
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
    
    return Pc_ID, Pd_ID, norm_data, u_offset, y_offset

def test_error(Pd_ID, data, norm_data, u_offset, y_offset, skip_initial=0, remove_dc=False, plot=False):
    """
    Test the identified model on datasets
    
    NOTE: When remove_dc=True, the model works on deviations from mean.
    We need to add back the offset for comparison with original data.
    """
    dt = data[0, 1, 0] - data[0, 0, 0]
    
    for i in range(data.shape[0]):
        # Get original data
        test_data_orig = np.array(data[i])
        t_orig = test_data_orig[:, 0]
        u_orig = test_data_orig[:, 1]
        y_orig = test_data_orig[:, 2]
        
        # Get preprocessed/normalized data
        test_data = np.array(norm_data[i])
        t = test_data[:, 0]
        u_test_norm = test_data[:, 1]
        y_test_norm = test_data[:, 2]
        
        # Simulate on preprocessed data (deviations if remove_dc=True)
        # Note: Pd_ID expects UNNORMALIZED deviations
        u_test_proc = u_test_norm * np.max(np.abs(u_test_norm))  # Scale back from normalized
        
        # For the model: If we removed DC, model expects deviations
        # Get the actual preprocessed signal (not normalized)
        u_proc, y_proc, _, _ = preprocess_data(u_orig, y_orig, skip_initial, remove_dc)
        
        # Use discrete-time TF
        td_ID_test, yd_ID_test = control.forced_response(Pd_ID, t, u_proc)
        
        # If we removed DC offset, add it back for comparison
        if remove_dc:
            yd_ID_test = yd_ID_test + y_offset
            y_compare = y_proc + y_offset
        else:
            y_compare = y_proc
        
        if plot:
            fig, ax = plt.subplots(2, 1, figsize=(10, 6))
            ax[0].set_ylabel(r'$u(t)$ (V)')
            ax[1].set_ylabel(r'$y(t)$ (LPM)')
            ax[0].plot(t, u_proc if not remove_dc else u_proc + u_offset, '--', label='input', color='C0')
            ax[1].plot(t, y_compare, label='measured', color='C1', linewidth=1.5)
            ax[1].plot(td_ID_test, yd_ID_test, '-.', label="IDed output", color='C2', linewidth=1.5)
            ax[0].set_title(f'Dataset {i}')
            for a in np.ravel(ax):
                a.set_xlabel(r'$t$ (s)')
                a.legend(loc='upper right')
            fig.tight_layout()
            plt.savefig(f'result_dataset_{i}.png', dpi=150)
            plt.show()
        
        error = yd_ID_test - y_compare
        y_max_dataset = np.max(np.abs(y_compare))
        rms_error = np.sqrt(np.mean(error**2))
        rel_rms = rms_error / y_max_dataset * 100
        
        print(f"Dataset {i}:")
        print(f"  abs avg error = {np.mean(np.abs(error)):.4f} LPM")
        print(f"  RMS error = {rms_error:.4f} LPM")
        print(f"  rel RMS error = {rel_rms:.2f}%")

print("="*60)
print("SYSTEM IDENTIFICATION - FIXED VERSION")
print("="*60)
print("Training on data sets: ", training_data_sets)
print("="*60)

Pc_ID, Pd_ID, norm_data, u_offset, y_offset = get_tf(
    data, n, m, training_data_sets, 
    skip_initial=skip_initial_samples,
    remove_dc=remove_dc_offset
)

print("\nContinuous-time TF:")
print(Pc_ID)
print("\nDiscrete-time TF:")
print(Pd_ID)
print("\n" + "="*60)
print("Testing on all datasets:")
print("="*60)

test_error(Pd_ID, data, norm_data, u_offset, y_offset,
          skip_initial=skip_initial_samples,
          remove_dc=remove_dc_offset,
          plot=True)

print("\n" + "="*60)
print("NOTES:")
print("="*60)
print("1. Skipped initial transient to avoid zero initial conditions")
print("2. Removed DC offsets to handle different operating points")
print("3. Model now works on DEVIATIONS from mean operating point")
print("4. This assumes system is approximately linear around each operating point")

