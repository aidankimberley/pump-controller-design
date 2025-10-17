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
test_data_set = 4
#n>=m ALWAYS
n=2 #order of denominator
m=1 #order of numerator


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

    # Compute the uncertainty and relative uncertainty. 
    # Compute the uncertainty and relative uncertainty. 
    sigma = 1/(N-(n+m+1))* np.linalg.norm(b-A@x)*np.linalg.inv(A.T@A)
    sigma_diag = np.diag(sigma)
    #Divides element wise
    rel_unc = sigma_diag/np.abs(x)*100
    print('The uncertainty is         ', sigma_diag)
    print('The relative uncertainty is', rel_unc, '%\n')

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

def test_error(Pd_ID, data, plot=False):
    dt = data[0, 1, 0] - data[0, 0, 0]
    for i in range(data.shape[0]):
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
            for a in np.ravel(ax):
                a.set_xlabel(r'$t$ (s)')
                a.legend(loc='upper right')
            fig.tight_layout()
            plt.show()
        error = yd_ID_test - y_test
        y_max_dataset = np.max(np.abs(y_test))
        print(f"Dataset {i}: abs avg error = {np.mean(np.abs(error)):.4f}, rel avg error = {np.mean(np.abs(error/y_max_dataset))*100:.2f}%")

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
exit()






#Ignore below this point
# Normalize the three training datasets
u_train1_norm, y_train1_norm, u_max1, y_max1 = normalize_data(data_train1[0])
u_train2_norm, y_train2_norm, u_max2, y_max2 = normalize_data(data_train2[0])
u_train3_norm, y_train3_norm, u_max3, y_max3 = normalize_data(data_train3[0])

# Form A and b matrices for each training dataset
A1, b1 = form_A_b(y_train1_norm, u_train1_norm, n, m)
A2, b2 = form_A_b(y_train2_norm, u_train2_norm, n, m)
A3, b3 = form_A_b(y_train3_norm, u_train3_norm, n, m)

# Concatenate A and b matrices
A = np.vstack([A1, A2, A3])
b = np.vstack([b1, b2, b3])

print("\nConcatenated matrix shapes:")
print(f"A: {A.shape}, b: {b.shape}")

cond_A = np.linalg.cond(A)
print("cond_A:", cond_A)

Pc_ID = get_CT_tf(A, b, n, T, u_max, y_max)

x = np.linalg.lstsq(A, b)[0]
print("x:", x)

# Compute the uncertainty and relative uncertainty. 
sigma = 1/(N-(n+m+1))* np.linalg.norm(b-A@x)*np.linalg.inv(A.T@A)
sigma_diag = np.diag(sigma)

#Divides element wise
rel_unc = sigma_diag/np.abs(x)*100


print('The uncertainty is         ', sigma_diag)
print('The relative uncertainty is', rel_unc, '%\n')

# Compute the MSE, MSO, NMSE.
MSE = 1/N*(np.linalg.norm(b-A@x)**2)
MSO = 1/N*(np.linalg.norm(b)**2)
NMSE = MSE/MSO

print('The MSE fortraining is', MSE)
print('The MSO for training is', MSO)
print('The NMSE for training is', NMSE, '\n')

N_x = x.shape[0]
Pd_ID_den = np.hstack([1, x[0:n].reshape(-1,)])  # denominator coefficients of DT TF
print(Pd_ID_den)
Pd_ID_num = x[n:].reshape(-1,)  # numerator coefficients of DT TF
print(Pd_ID_num)

# Compute DT TF (and remember to ``undo" the normalization).
u_bar = u_max  # You change.
y_bar = y_max  # You change.
Pd_ID = y_bar / u_bar * control.tf(Pd_ID_num, Pd_ID_den, T)
#Pd_ID = control.tf(Pd_ID_num, Pd_ID_den, T) #discrete transfer function
print('The discrete-time TF is,', Pd_ID)

# Compute the CT TF
Pc_ID = d2c.d2c(Pd_ID) #continuous transfer function
print('The continuous-time TF is,', Pc_ID)
#%%
#Test error on training data

td_ID_train, yd_ID_train = control.forced_response(Pd_ID, t, u_train)

error = yd_ID_train - y_train
print("abs average error:", np.mean(np.abs(error)))
print("rel average error:", np.mean(np.abs(error/y_max))*100,"%")
# Plot training data
fig, ax = plt.subplots(2, 1)   
ax[0].set_ylabel(r'$u(t)$ (Pa)')
ax[1].set_ylabel(r'$y(t)$ (N)')
# Plot data
ax[0].plot(t, u_train, '--', label='input', color='C0')
ax[1].plot(t, y_train, label='output', color='C1')
ax[1].plot(td_ID_train, yd_ID_train, '-.', label="IDed output", color='C2')
for a in np.ravel(ax):
    a.set_xlabel(r'$t$ (s)')
    a.legend(loc='upper right')
fig.tight_layout()


#plot error
fig, ax = plt.subplots(2, 1)
ax[0].set_ylabel(r'$e(t)$ (N)')
ax[1].set_ylabel(r'$e(t)/y(t)$ (unitless)')
ax[0].plot(t, error)
ax[1].plot(t, error/y_train)
fig.tight_layout()

#%%
#Test error on test data
td_ID_test, yd_ID_test = control.forced_response(Pd_ID, t, u_test)
error = yd_ID_test - y_test
print("abs average error:", np.mean(np.abs(error)))
print("rel average error:", np.mean(np.abs(error/y_max))*100,"%")
# Plot test data
fig, ax = plt.subplots(2, 1)
ax[0].set_ylabel(r'$u(t)$ (Pa)')
ax[1].set_ylabel(r'$y(t)$ (N)')
# Plot data
ax[0].plot(t, u_test, '--', label='input', color='C0')
ax[1].plot(t, y_test, label='output', color='C1')
ax[1].plot(td_ID_test, yd_ID_test, '-.', label="IDed output", color='C2')
for a in np.ravel(ax):
    a.set_xlabel(r'$t$ (s)')
    a.legend(loc='upper right')
fig.tight_layout()

#plot error
fig, ax = plt.subplots(2, 1)
ax[0].set_ylabel(r'$e(t)$ (N)')
ax[1].set_ylabel(r'$e(t)/y(t)$ (unitless)')
ax[0].plot(t, error)
ax[1].plot(t, error/y_test)
fig.tight_layout()
