import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import time
import qutip as qp
from functions import kron_commutator,sk_kron,row_to_col,stat_aver,kron_anticom

# Parameters
fac = 1000
invf= 1/fac
w1 = 2*np.pi*10*fac
dw = 2*np.pi*5*fac
w0 = 2*np.pi*100*100*fac
w = w0 + dw
sig= w +w0
tau_c = 1e-6*invf


I_x = np.array([[0, 1], [1, 0]]) / 2
I_y = np.array([[0, -1j], [1j, 0]]) / 2
I_z = np.array([[1, 0], [0, -1]]) / 2

idm = np.eye(I_x.ndim)
# Ladder operators
I_plus = np.array([[0, 1], [0, 0]])
I_minus = np.array([[0, 0], [1, 0]])


def gamma(theta,a):
    if a ==1:
        return (tau_c + 1j*(theta)*tau_c*tau_c )/(1 + ((tau_c*theta)**2)) 
    if a==2:
        return  (tau_c - 1j*(theta)*(tau_c**2) )/(1 + ((tau_c*theta)**2)) 

def expo(a, t):
    return np.exp(1j * a * t)



c = ((w1)**2)/4
I_kron_pl = kron_commutator(I_plus,idm)
I_kron_mi = kron_commutator(I_minus,idm)
I_kr_anti_plmi =kron_anticom(I_plus @ I_minus,idm)
I_kr_anti_mipl =kron_anticom(I_minus @ I_plus,idm)
pl_mi_I = I_kron_pl @ I_kron_mi
mi_pl_I = I_kron_mi @ I_kron_pl
pl_pl_I = I_kron_pl @ I_kron_pl
mi_mi_I = I_kron_mi @ I_kron_mi 
p_plus = 0.6 
p_minus = 0.4
jw = 100
gamma_sig= gamma(sig,1)
gamma_sigs = gamma(sig,2)
gamma_dw = gamma(dw,1)
gamma_dws = gamma(dw,2)

def f1(t,rho):
    # First order drive term 
    F =-expo(-dw,t) *I_plus - I_minus*expo(dw,t)
    L1 = 1j*(w1/2)*kron_commutator(F,idm)

    # Second order drive term 
    L2 = -c*(gamma_sig*(pl_mi_I) + gamma_sigs* (mi_pl_I) )
    L3 = -(4*c)* (gamma_dw*(pl_pl_I )* expo(-2*dw,t) + gamma_dws*(mi_mi_I)*expo(2*dw,t))
    L4 = -(4*c)* (gamma_dws*pl_mi_I +gamma_dw* mi_pl_I)

    # Second order Coupling Terms
    L5 = -(p_plus*jw*((I_kr_anti_plmi/2) - sk_kron(I_minus,I_plus)))
    L6 = -(p_minus*jw*((I_kr_anti_mipl/2) - sk_kron(I_plus,I_minus)))

    L = L1+ L2 + L3 + L4 


    return np.dot(L,rho)


def rrk4(f, x, y, h):
    k1 = h*f(x, y)
    k2 = h*f(x + (h/2), y + (k1/2))
    k3 = h*f(x + (h/2), y + (k2/2))
    k4 = h*f(x + h, y + k3)
    return y + ((k1 + 2*k2 + 2*k3 + k4) / 6)


def evolution(method, fn, y_ini, xs, h):
    ys = np.zeros((len(xs), 4, 1),dtype=complex)
    y = y_ini
    iteration =0
    for i in range(len(xs)):
        
        x = xs[i]
        ys[i] = y
        y = method(fn, x, y, h)
        iteration+= 1
    print(iteration)
    return ys

# Initial Conditions
rho1 = np.array([1,0,0,1])/2 + 0.4*np.array([0,1,1,0])
rho_0 = row_to_col(rho1)
h = 1e-4 * invf  # Time step

t_start = 0.0 * invf
t_stop = 5.0 * invf
ts = np.arange(t_start, t_stop + h, h)


# pp1_array = np.array(pp1)
st = time.time()
# Reshaping the coloumn vector of 4 dimensional into a matrix of 2 dimensional (row-wise)
pp1 = np.array(evolution(rrk4, f1, rho_0, ts, h))

et = time.time()
print('Elasped Time:',et-st)

st = time.time()


rho_t = pp1.reshape(-1, 2, 2)
Fx= [(I_plus * expo(-dw, t) + I_minus * expo(dw, t)) / 2 for t in ts]
Fy =[(I_plus * expo(-dw, t) - I_minus * expo(dw, t)) / (2*1j) for t in ts]
# Fz = I_z # So I have used it directly in M_z


# stat_aver[A,B] = np.real(np.trace(np.dot(A,B)))

Mx = [stat_aver(Fx[p],rho_t[p]) for p in range(len(ts))]
My = [stat_aver(Fy[p],rho_t[p]) for p in range(len(ts))]
Mz = [stat_aver(I_z,rho_t[p]) for p in range(len(ts))]

print(Mx[:5])


# Create a 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the vectors
ax.plot3D(Mx, My, Mz, color='blue')
plt.savefig("bloch_qme.png")


# Plot all vectors in one plot
plt.figure(figsize=(8, 6))
ts = ts*fac
plt.plot(ts, Mx, label='X component', color='blue')
plt.plot(ts, My, label='Y component', color='black')
plt.plot(ts, Mz, label='Z component', color='red')
plt.xlabel('$time(milisecond)$')
plt.savefig("bloch_qme_time.png")
et = time.time()
print('Elasped Time:',et-st)
plt.show()





