# Multi-model approach: Separate models for different operating regions
import numpy as np
import control
from matplotlib import pyplot as plt
import pathlib
import warnings
import d2c

warnings.filterwarnings('ignore', 'Casting complex values to real discards the imaginary part')

plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')

# Load data
path = pathlib.Path('load_data_sc/PRBS_DATA/')
all_files = sorted(path.glob("*.csv"))
data = [
    np.loadtxt(filename, dtype=float, delimiter=',', skiprows=1, usecols=(0, 1, 2))
    for filename in all_files
]
data = np.array(data)

# User parameters
n, m = 2, 1  # Keep it simple
skip_initial_samples = 150  # Skip MORE samples (15 seconds)
remove_dc_offset = True

# Define operating regions based on analysis
# Low flow: Datasets 0,1 (mean voltage 2.5-3.7V, mean flow 6.5 LPM)
# High flow: Datasets 2,3 (mean voltage 4.3-4.5V, mean flow 10-12 LPM)
operating_regions = {
    'low_flow': [0, 1],
    'high_flow': [2, 3]
}

def preprocess_data(u, y, skip_initial=0, remove_dc=False):
    u_proc = u[skip_initial:].copy()
    y_proc = y[skip_initial:].copy()
    
    if remove_dc:
        u_offset = np.mean(u_proc)
        y_offset = np.mean(y_proc)
        u_proc = u_proc - u_offset
        y_proc = y_proc - y_offset
    else:
        u_offset = 0.0
        y_offset = 0.0
    
    return u_proc, y_proc, u_offset, y_offset

def normalize_data_region(data, training_data_sets, skip_initial=0, remove_dc=False):
    u_max_list = []
    y_max_list = []
    
    for i in training_data_sets:
        u = data[i][:, 1]
        y = data[i][:, 2]
        u_proc, y_proc, _, _ = preprocess_data(u, y, skip_initial, remove_dc)
        
        u_max_list.append(np.max(np.abs(u_proc)))
        y_max_list.append(np.max(np.abs(y_proc)))
    
    u_max = np.max(u_max_list)
    y_max = np.max(y_max_list)
    
    return u_max, y_max

def form_A_b(y_train, u_train, n, m):
    N = y_train.shape[0]
    A = np.zeros((N-n, n+m+1))
    b = np.zeros((N-n, 1))
    for k in range(N-n):
        for j in range(n):
            A[k,j] = -y_train[k+n-j-1]
        for j in range(m+1):
            A[k,n+j] = u_train[k+m-j]
    for k in range(N-n):
        b[k] = y_train[k+n]
    return A, b

def train_model_for_region(data, training_sets, n, m, skip_initial, remove_dc):
    dt = data[0, 1, 0] - data[0, 0, 0]
    
    # Get normalization factors
    u_max, y_max = normalize_data_region(data, training_sets, skip_initial, remove_dc)
    
    # Stack data from all datasets in this region
    A_stacked = None
    b_stacked = None
    
    for i in training_sets:
        u = data[i][:, 1]
        y = data[i][:, 2]
        
        u_proc, y_proc, _, _ = preprocess_data(u, y, skip_initial, remove_dc)
        
        # Normalize
        u_norm = u_proc / u_max
        y_norm = y_proc / y_max
        
        A, b = form_A_b(y_norm, u_norm, n, m)
        
        if A_stacked is None:
            A_stacked = A
            b_stacked = b
        else:
            A_stacked = np.vstack([A_stacked, A])
            b_stacked = np.vstack([b_stacked, b])
    
    # Solve
    x = np.linalg.lstsq(A_stacked, b_stacked, rcond=None)[0]
    
    # Compute errors
    N = A_stacked.shape[0]
    MSE = 1/N*(np.linalg.norm(b_stacked-A_stacked@x)**2)
    MSO = 1/N*(np.linalg.norm(b_stacked)**2)
    NMSE = MSE/MSO
    
    # Create TF
    Pd_den = np.hstack([1, x[0:n].reshape(-1,)])
    Pd_num = x[n:].reshape(-1,)
    Pd = y_max / u_max * control.tf(Pd_num, Pd_den, dt)
    
    try:
        Pc = d2c.d2c(Pd)
    except:
        Pc = None
    
    poles_d = control.poles(Pd)
    
    return Pd, Pc, u_max, y_max, NMSE, poles_d

def test_model(Pd, data, dataset_idx, skip_initial, remove_dc, u_max, y_max):
    t_orig = data[dataset_idx][:, 0]
    u_orig = data[dataset_idx][:, 1]
    y_orig = data[dataset_idx][:, 2]
    
    # Preprocess
    t = t_orig[skip_initial:]
    u_proc, y_proc, u_offset, y_offset = preprocess_data(u_orig, y_orig, skip_initial, remove_dc)
    
    # Simulate
    _, y_pred = control.forced_response(Pd, t, u_proc)
    
    # Add DC back if needed
    if remove_dc:
        y_pred = y_pred + y_offset
        y_compare = y_proc + y_offset
    else:
        y_compare = y_proc
    
    error = y_pred - y_compare
    rms_error = np.sqrt(np.mean(error**2))
    y_max_dataset = np.max(np.abs(y_compare))
    rel_rms = rms_error / y_max_dataset * 100
    
    return rms_error, rel_rms, t, u_proc, y_compare, y_pred, u_offset, y_offset

# Train models for each region
print("="*70)
print("MULTI-MODEL SYSTEM IDENTIFICATION")
print("="*70)
print(f"Skipping first {skip_initial_samples} samples ({skip_initial_samples*0.1:.1f}s)")
print(f"Remove DC offset: {remove_dc_offset}\n")

models = {}
for region_name, training_sets in operating_regions.items():
    print(f"\n{region_name.upper()}: Training on datasets {training_sets}")
    print("-"*70)
    
    Pd, Pc, u_max, y_max, NMSE, poles = train_model_for_region(
        data, training_sets, n, m, skip_initial_samples, remove_dc_offset
    )
    
    models[region_name] = {
        'Pd': Pd,
        'Pc': Pc,
        'u_max': u_max,
        'y_max': y_max,
        'training_sets': training_sets
    }
    
    print(f"NMSE: {NMSE:.4f} ({NMSE*100:.2f}%)")
    print(f"Poles: {poles}")
    print(f"Stable: {np.all(np.abs(poles) < 1)}")
    if Pc is not None:
        print(f"Continuous TF:\n{Pc}")

# Test all models on all datasets
print("\n\n" + "="*70)
print("TESTING RESULTS")
print("="*70)

results = {}
for region_name, model_info in models.items():
    print(f"\n{region_name.upper()} MODEL:")
    print("-"*70)
    
    Pd = model_info['Pd']
    training_sets = model_info['training_sets']
    
    for i in range(len(data)):
        rms_error, rel_rms, t, u_proc, y_compare, y_pred, u_off, y_off = test_model(
            Pd, data, i, skip_initial_samples, remove_dc_offset,
            model_info['u_max'], model_info['y_max']
        )
        
        is_training = "TRAIN" if i in training_sets else "TEST "
        print(f"Dataset {i} [{is_training}]: RMS = {rms_error:.4f} LPM ({rel_rms:.2f}%)")
        
        # Store for later plotting
        if region_name not in results:
            results[region_name] = {}
        results[region_name][i] = {
            'rms': rms_error,
            'rel_rms': rel_rms,
            't': t,
            'u': u_proc + (u_off if remove_dc_offset else 0),
            'y_meas': y_compare,
            'y_pred': y_pred
        }

# Determine best model for each dataset
print("\n\n" + "="*70)
print("BEST MODEL SELECTION FOR EACH DATASET")
print("="*70)

for i in range(len(data)):
    print(f"\nDataset {i}:")
    u_mean = np.mean(data[i][skip_initial_samples:, 1])
    y_mean = np.mean(data[i][skip_initial_samples:, 2])
    print(f"  Operating point: {u_mean:.2f}V -> {y_mean:.2f} LPM")
    
    best_region = None
    best_error = float('inf')
    
    for region_name in results.keys():
        rel_rms = results[region_name][i]['rel_rms']
        if rel_rms < best_error:
            best_error = rel_rms
            best_region = region_name
    
    print(f"  Best model: {best_region.upper()} ({best_error:.2f}% error)")
    
    # Plot with best model
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    result = results[best_region][i]
    
    ax[0].plot(result['t'], result['u'], '--', label='input', color='C0')
    ax[0].set_ylabel('u(t) (V)')
    ax[0].legend()
    ax[0].grid(True, linestyle='--', alpha=0.5)
    
    ax[1].plot(result['t'], result['y_meas'], label='measured', color='C1', linewidth=1.5)
    ax[1].plot(result['t'], result['y_pred'], '-.', label=f'{best_region} model', color='C2', linewidth=1.5)
    ax[1].set_ylabel('y(t) (LPM)')
    ax[1].set_xlabel('t (s)')
    ax[1].legend()
    ax[1].grid(True, linestyle='--', alpha=0.5)
    ax[1].set_title(f'Dataset {i} - Best: {best_region.upper()} ({best_error:.2f}%)')
    
    fig.tight_layout()
    plt.savefig(f'multi_model_dataset_{i}.png', dpi=150)
    plt.close()

print("\n\n" + "="*70)
print("SUMMARY & RECOMMENDATIONS")
print("="*70)
print("\nThis multi-model approach:")
print("- Trains separate models for low and high flow regions")
print("- Accounts for nonlinear pump behavior")
print("- Should give better accuracy than single model")
print("\nFor best results:")
print("- Use skip_initial_samples = 150 (more aggressive)")
print("- Keep n=2, m=1 (higher orders overfit)")
print("- Select model based on operating voltage:")
print("  * Use LOW_FLOW model for u < 3.5V")
print("  * Use HIGH_FLOW model for u >= 3.5V")
print("\nPlots saved: multi_model_dataset_0.png through multi_model_dataset_3.png")

