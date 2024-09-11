import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from functions import kron_commutator,sk_kron,row_to_col,stat_aver,kron_anticom


# Define the time-dependent matrix A(t)
# Parameters
fac = 1000
invf= 1/fac

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

def sqrt_density_matrix(rho):
    # Perform eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(rho)
    
     # Set small eigenvalues to zero
    threshold = 1e-10
    eigenvalues = np.where(eigenvalues < threshold, 0, eigenvalues)
    
    # Compute the square roots of the eigenvalues
    sqrt_eigenvalues = np.sqrt(eigenvalues)
    
    # Reconstruct the square root of the density matrix
    sqrt_rho = eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.T.conj()
    
    return sqrt_rho
def fidelity(p,fp):
    sq_p = sqrt_density_matrix(p)
    return (np.trace(sqrt_density_matrix(sq_p @ fp @ sq_p)))**2


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
jw = 10
gamma_sig= gamma(sig,1)
gamma_sigs = gamma(sig,2)
gamma_dw = gamma(dw,1)
gamma_dws = gamma(dw,2)

def liouvilian(w1,t):
    # First order drive term 
    c = ((w1)**2)/4
    F =-expo(-dw,t) *I_plus - I_minus*expo(dw,t)
    L1 = 1j*(w1/2)*kron_commutator(F,idm)

    # Second order drive term 
    L2 = -c*(gamma_sig*(pl_mi_I) + gamma_sigs* (mi_pl_I) )
    L3 = -(4*c)* (gamma_dw*(pl_pl_I )* expo(-2*dw,t) + gamma_dws*(mi_mi_I)*expo(2*dw,t))
    L4 = -(4*c)* (gamma_dws*pl_mi_I +gamma_dw* mi_pl_I)

    # Second order Coupling Terms
    L5 = -(p_plus*jw*((I_kr_anti_plmi/2) - sk_kron(I_minus,I_plus)))
    L6 = -(p_minus*jw*((I_kr_anti_mipl/2) - sk_kron(I_plus,I_minus)))

    L = L1+ L2 + L3 + L4 + L5 + L6


    return L


def fidelity_cal(w1_l,f,ts,rho_f,rho_0):
    h = ts[1]- ts[0]
    fidelity_m = np.zeros((len(ts),len(w1_l)))
    for j,w1 in enumerate(w1_l):
        i = 0
        p= rho_0
        for i,t in enumerate(ts):
            
            
            p= expm(f(w1,t) * h) @ p
            # print(p)
            z = np.squeeze(p.reshape(-1,2, 2))
            # print(z)
            fidelity_m[i,j] = fidelity(z,rho_f)
           
        print(fidelity_m[len(ts)-1,j],w1/(2*np.pi*fac))
    return fidelity_m
    
rho1 = np.array([1,0,0,0])
rho_0 = row_to_col(rho1)
rho_f = np.array([[0,0],[0,1]])
w1_l = np.arange( 2*np.pi*1*fac,  2*np.pi*100*fac,  2*np.pi*1*fac)
h = 1e-4 * invf  # Time step
t_start = 0.0 * invf
t_stop = np.pi/w1_l[2]
ts = np.arange(t_start, t_stop + h, h)
fidelity_matrix = fidelity_cal(w1_l,liouvilian,ts,rho_f,rho_0)    



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x, y = np.meshgrid(w1_l, ts)
ax.plot_surface(x, y, fidelity_matrix, cmap='viridis', edgecolor='none')

ax.set_xlabel('w1')
ax.set_ylabel('ts')
ax.set_zlabel('Fidelity')

plt.show()
