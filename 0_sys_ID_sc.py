"""
SISO system ID sample code for students
J R Forbes, 2025/10/02
"""

# %%
# Libraries
import numpy as np
import control
from matplotlib import pyplot as plt

# Custom libraries 
import d2c

# %%
# A demo on how to use d2c. You will need to use this to go from your DT IDed model to a CT IDed model.
# Pd is a first oder, DT transfer function.
# Pd = control.tf(0.09516, np.array([1, -0.9048]), 0.01)
# # Using the custom command d2c.d2c convert Pd to Pc where Pc is a CT transfer function.
# Pc = d2c.d2c(Pd)
# print(Pd, Pc)
# %% 
# Plotting parameters
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=14)
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')

# %% 
# Read in input-output (IO) data
data_read = np.loadtxt('IO_data.csv',
                        dtype=float,
                        delimiter=',',
                        skiprows=1,
                        usecols=(0, 1, 2))

# print(data_read)  # print the data, just to get a feel for the data.

# Extract time
t = data_read[:, 0]
N = t.size
print("Number of data points:", N)
T = t[1] - t[0]


n=2
m=1
# Extract input and output
u = data_read[:, 1]
y = data_read[:, 2]


# %% 
# System ID
#out of curiousity, before normalization...\
A = np.zeros((N-2,4))
for i in range(N-2): #N-1 is last index
    A[i,0] = -y[i]
    A[i,1] = -y[i-1]
    A[i,2] = u[i]
    A[i,3] = u[i-1]
cond_A = np.linalg.cond(A)
print("cond_A before normalization:", cond_A)


#Normalize u and y
u_max = np.max(np.abs(u))
y_max = np.max(np.abs(y))
print("u_max:", u_max)
print("y_max:", y_max)
u_norm = u/u_max
y_norm = y/y_max

# Form the A and b matrix. (You might want create a function to form A and b given u and y.)
A = np.zeros((N-2,4))
for i in range(N-2): #N-1 is last index, 
    A[i,0] = -y_norm[i+1]
    A[i,1] = -y_norm[i]
    A[i,2] = u_norm[i+1]
    A[i,3] = u_norm[i]

b = y_norm[2:] #y_2...y_N

# Is the A matrix "good"? How can you check?
cond_A = np.linalg.cond(A)
print("cond_A after normalization:", cond_A)
if cond_A > 1e10:
    print("The A matrix is ill-conditioned")
    exit()
else:
    print("The A matrix is well-conditioned")

# Solve for x.
x = np.linalg.lstsq(A, b)[0]
print('The parameter estimates are\n', x,'\n')

n=2
m=1
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

# %% 
# Compute TF 
# Extract denominator and numerator coefficients.
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

# %% 
# Response of DT IDed system to (training) input data
# Use unnormalized u and y
td_ID_train, yd_ID_train = control.forced_response(Pd_ID, t, u)

error = yd_ID_train - y
print("abs average error:", np.mean(np.abs(error)))
print("rel average error:", np.mean(np.abs(error/y_max))*100,"%")
# Plot training data
fig, ax = plt.subplots(2, 1)   
ax[0].set_ylabel(r'$u(t)$ (Pa)')
ax[1].set_ylabel(r'$y(t)$ (N)')
# Plot data
ax[0].plot(t, u, '--', label='input', color='C0')
ax[1].plot(t, y, label='output', color='C1')
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
ax[1].plot(t, error/y)
fig.tight_layout()
# %%
# Test
# Read in input-output (IO) data
data_read = np.loadtxt('IO_data2.csv',
                        dtype=float,
                        delimiter=',',
                        skiprows=1,
                        usecols=(0, 1, 2))

# print(data_read)  # print the data, just to get a feel for the data.

# Extract time
t_test = data_read[:, 0]
N_test = t_test.size
T_test = t_test[1] - t_test[0]



# Extract input and output, add noise if wanted
u_test = data_read[:, 1]
y_test = data_read[:, 2]

N_test = t_test.size
print("Number of test data points:", N_test)
u_test_norm = u_test/u_max
y_test_norm = y_test/y_max
# %%
# Compute various error metrics

# Form the A and b matrix using test data.
# Form the A and b matrix. (You might want create a function to form A and b given u and y.)
A = np.zeros((N_test-2,4))
#A is testing data y and u vals
for i in range(N_test-2): #N-1 is last index
    A[i,0] = -y_test_norm[i+1]
    A[i,1] = -y_test_norm[i]
    A[i,2] = u_test_norm[i+1]
    A[i,3] = u_test_norm[i]

b = y_test_norm[2:] #y_2...y_N

#x is the parameter estimates from training data

# Compute the MSE, MSO, NMSE using test data
MSE_test = 1/N_test*(np.linalg.norm(b-A@x)**2)
MSO_test = 1/N_test*(np.linalg.norm(b)**2)
NMSE_test = MSE_test/MSO_test

# Error associated with Ax = b
print('The MSE in test is', MSE_test)
print('The MSO in test is', MSO_test)
print('The NMSE in test is', NMSE_test, '\n')

# Forced response of IDed system using test data
td_ID_test, yd_ID_test = control.forced_response(Pd_ID, t_test, u_test)

# Compute error
e = yd_ID_test  - y_test

# Compute %VAF
var_e = 1/N_test*np.sum(e**2)
var_y = 1/N_test*np.sum(y_test**2)
VAF_test = 1/N_test*(1-var_e/var_y)
print('The %VAF is', VAF_test)

# Compute and plot errors
e_abs = np.abs(e)
e_rel = np.zeros(N_test)
y_max = np.max(np.abs(y_test))
for i in range(N_test):    
    e_rel[i] = e_abs[i] / y_max * 100  # or / np.std(y)

# Plot test data
fig, ax = plt.subplots(2, 1)   
ax[0].set_ylabel(r'$u(t)$ (Pa)')
ax[1].set_ylabel(r'$y(t)$ (N)')
# Plot data
ax[0].plot(t_test, u_test, '--', label='input', color='C0')
ax[1].plot(t_test, y_test, label='output', color='C1')
#title
ax[1].set_title('Test Data')
ax[1].plot(td_ID_test, yd_ID_test, '-.', label="IDed output", color='C2')
for a in np.ravel(ax):
    a.set_xlabel(r'$t$ (s)')
    a.legend(loc='upper right')
fig.tight_layout()

# Plot error
fig, ax = plt.subplots(2, 1)
# Format axes
for a in np.ravel(ax):
    a.set_xlabel(r'$t$ (s)')
ax[0].set_ylabel(r'$e_{abs}(t)$ (N)')
ax[1].set_ylabel(r'$e_{rel}(t) \times 100\%$ (unitless)')
# Plot data
ax[0].plot(t_test, e)
ax[1].plot(t_test, e_rel)
# for a in np.ravel(ax):
#     a.legend(loc='lower right')
fig.tight_layout()


# %%
# Show plots
plt.show()

