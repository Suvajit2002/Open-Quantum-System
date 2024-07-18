import numpy as np
import matplotlib.pyplot as plt
import qutip as qp
from functions import stat_aver,I_x,I_y,I_z



I_x = np.array([[0, 1], [1, 0]]) / 2
I_y = np.array([[0, -1j], [1j, 0]]) / 2
I_z = np.array([[1, 0], [0, -1]]) / 2

ground_state = np.outer([1,0],[1,0])
excited_state = np.outer([0,1],[0,1])


initial_state = np.array([[1/2,0.4],[0.4,1/2]])

# stat_aver(A,B) = np.trace(np.dot(A,B))

# Calculate Mx_0, My_0, and Mz_0 in one line
Mx_0, My_0, Mz_0 = [stat_aver(mat, initial_state) for mat in [I_x, I_y, I_z]]



# Parameters
fac = 1000
invf= 1/fac
w1 = 2*np.pi*10*fac
dw = 2*np.pi*5*fac
w0 = 2*np.pi*100*100*fac
w = w0 + dw
sig= w +w0
tau_c = 1e-6*invf
nx = (w1**2)* ((tau_c/2)/(1 +((tau_c*sig)**2) ))
ny = (w1**2)* tau_c * ((1/(2*(1+ ((sig*tau_c)**2)))) + (1/(1+((dw*tau_c)**2))))
nz =  (w1**2)* tau_c * ((1/(1+ ((sig*tau_c)**2))) + (1/(1+((dw*tau_c)**2))))
dwx = dw - ((((tau_c*w1)**2) * (sig/2))/(1 +((tau_c*sig)**2)))
dwy = dwx + ((((tau_c*w1)**2) * (dw))/(1 +((tau_c*dw)**2)))

def f1(t, M):
    Mx, My, Mz = M
    return np.array([dwx*My- nx*Mx, -dwy*Mx - w1*Mz - ny*My, w1*My-nz*Mz], dtype=object)

def rk4w(f, x, y, h):
    k1 = h*f(x, y)
    k2 = h*f(x + (h/2), y + (k1/2))
    k3 = h*f(x + (h/2), y + (k2/2))
    k4 = h*f(x + h, y + k3)
    return y + ((k1 + 2*k2 + 2*k3 + k4) / 6)

def evolution(method, fn, y_ini, xs, h):
    ys = np.zeros(shape=(len(xs), len(y_ini)),dtype=complex)
    y = y_ini
    for i in range(len(xs)):
        x = xs[i]
        ys[i] = y
        y = method(fn, x, y, h)
    return ys

# Initial Conditions
y_initial = np.array([Mx_0, My_0, Mz_0])
h = 1e-4*invf
t_start = 0.0* invf
t_stop = 10*invf
ts = np.arange(t_start, t_stop+h, h)

bloch_vectors = evolution(rk4w, f1, y_initial, ts, h)
Mx_t = bloch_vectors[:, 0]
My_t = bloch_vectors[:, 1]
Mz_t = bloch_vectors[:, 2]



# Create a 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the vectors
ax.plot3D(Mx_t, My_t, Mz_t, color='blue')

# Set labels and title
ax.set_xlabel('$M_x$')
ax.set_ylabel('$M_y$')
ax.set_zlabel('$M_z$')
ax.set_title('3D plot of Bloch Vectors')


# Add a legend
ax.legend()

# Save the plot as an image file
plt.savefig("bloch_cde.png")

# Plot all vectors in one plot
plt.figure(figsize=(8, 6))
ts = ts*fac
plt.plot(ts, Mx_t, label='X component', color='blue')
plt.plot(ts, My_t, label='Y component', color='black')
plt.plot(ts, Mz_t, label='Z component', color='red')
plt.xlabel('$time(milisecond)$')
plt.savefig("bloch_cde_time.png")
ax.set_ylabel('$Bloch Vectors$')
# Show plot
plt.show()
