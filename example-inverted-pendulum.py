import numpy as np
import autograd.numpy as np
from autograd import  jacobian
import matplotlib.pyplot as plt

import lqr

# constants
# gravity
G = 9.81  
# length of the pendulum
L = 1  
# mass of the pendulum
M = 1   

STATE_DIMENSIONS = 2
CONTROL_DIMENSIONS = 1


SAMPLING_TIME = 0.1
N_STEPS = 1000

def dynamics(x, u, dt=SAMPLING_TIME, control_noise=None, dynamics_noise=None):
    if control_noise is not None:
        u += control_noise*np.random.randn(*u.shape)
    theta, theta_dot = x
    theta_dot_dot = G/L * np.sin(theta) + (1/(M*L**2)) * u
    x_new = np.array([theta + theta_dot*dt + 0.5*theta_dot_dot*dt**2, theta_dot + theta_dot_dot*dt])
    if dynamics_noise is not None:
        x_new += dynamics_noise*np.random.randn(*x_new.shape)
    return x_new.reshape(-1)

# sysytem dynamics are non linear, so we need to linearize the system around x*,u* using Taylor series expansion
# x[t+1] = f(x[t], u[t]) ~= f[x*, u*] + f'_x[x*, u*](x[t] - x*) + f'_u[x*, u*](u[t] - u*)
#    where f'_x is the jacobian of f with respect to x and f'_u is the jacobian of f with respect to u
# Once linerarized, we have our A and B matrices of the form
# A' = f'_x[x*, u*]
# B' = f'_u[x*, u*]
# z[t] = x[t] - x*
# v[t] = u[t] - u*
# g(z[t], v[t]) = z[t].T * Q * z[t] + v[t].T * R * v[t]
# And the lineaized dynamics can be written as
# z[t+1] = A'z[t] + B'v[t]

# A
def A(x, dt=SAMPLING_TIME):
    theta_t = x[0]
    cost = np.cos(theta_t)
    return np.array([
        [1 + 0.5 * (G/L) * cost * dt**2, dt],
        [(G/L) * cost * dt, 1]
    ])


# B
def B(dt=SAMPLING_TIME):
    return np.array([
        [0.5 * dt**2*(1/(M*L**2))],
        [dt*(1/(M*L**2))]
    ])


def Q():
    return np.identity(STATE_DIMENSIONS)

def R():
    return np.identity(CONTROL_DIMENSIONS)

def move(x, u, A_, B_):
    return A_.dot(x.T) + B_.dot(u)


if __name__ == "__main__":

    ##### 
    x = np.zeros((N_STEPS,STATE_DIMENSIONS))
    u = np.zeros((N_STEPS,CONTROL_DIMENSIONS))

    # Linearized system
    # z = x - x_ref
    # v = u - u_ref
    z = np.zeros((N_STEPS,STATE_DIMENSIONS))
    v = np.zeros((N_STEPS,CONTROL_DIMENSIONS))

    # Linear system around these x,u positions
    x_ref = np.array([0.0,-0.0])
    u_ref = np.array([0.0])

    dyn_x = jacobian(dynamics, 0)
    dyn_u = jacobian(dynamics, 1)

    _Q = Q()
    _R = R()
    _A = dyn_x(x_ref, u_ref)
    _B = dyn_u(x_ref, u_ref)

    x_0 =[0.5, -0.5]
    control_noise = 0.5
    dynamics_noise = np.array([[0.0] , [0.05]])


    k_lqr,_ = lqr.LQR(_A, _B, _Q, _R, N_STEPS)
    x[0] = x_0
    for i in range(N_STEPS-1):
        z[i] = x[i] - x_ref
        v[i] = k_lqr[:,:,i].dot(z[i])
        u[i] = v[i] + u_ref
        x[i+1] = dynamics(x[i], u[i], control_noise=control_noise, dynamics_noise=dynamics_noise)

    print("Final state: ", x[-1:])
    plt.subplot(2, 1, 1)
    plt.plot(x[:,0], x[:,1],'r')
    plt.subplot(2, 1, 2)
    plt.plot(u)
    plt.show()

    #np.save("./data/inverted_pendulum/u_noisy.npy", u)
    #np.save("./data/inverted_pendulum/x_noisy.npy", x)


    #############################################
    # Penalize High frequency control
    x = np.zeros((N_STEPS,STATE_DIMENSIONS))
    u = np.zeros((N_STEPS,CONTROL_DIMENSIONS))

    # Linearized system
    # z = x - x_ref
    # v = u - u_ref
    z = np.zeros((N_STEPS,STATE_DIMENSIONS))
    v = np.zeros((N_STEPS,CONTROL_DIMENSIONS))

    # Augmented state space
    z_aug = np.zeros((N_STEPS,STATE_DIMENSIONS+CONTROL_DIMENSIONS))
    v_aug = np.zeros((N_STEPS,CONTROL_DIMENSIONS))

    # Linear system around these x,u positions
    x_ref = np.array([0.0,-0.0])
    u_ref = np.array([0.0])

    control_noise = 0.5
    dynamics_noise = np.array([[0.0] , [0.05]])

    dyn_x = jacobian(dynamics, 0)
    dyn_u = jacobian(dynamics, 1)

    _Q1 = Q()
    _R1 = R()
    _A1 = dyn_x(x_ref, u_ref)
    _B1 = dyn_u(x_ref, u_ref)

    _R = 1*_R1
    _Q = np.zeros((STATE_DIMENSIONS+CONTROL_DIMENSIONS,STATE_DIMENSIONS+CONTROL_DIMENSIONS))
    _Q[:STATE_DIMENSIONS,:STATE_DIMENSIONS] = _Q1
    _Q[STATE_DIMENSIONS:, STATE_DIMENSIONS:] = _R
    _B = np.zeros((STATE_DIMENSIONS + CONTROL_DIMENSIONS,CONTROL_DIMENSIONS))
    _B[:STATE_DIMENSIONS] = _B1
    _B[STATE_DIMENSIONS:] = np.eye(CONTROL_DIMENSIONS)
    _A = np.zeros((STATE_DIMENSIONS+CONTROL_DIMENSIONS, STATE_DIMENSIONS+CONTROL_DIMENSIONS))
    _A[:STATE_DIMENSIONS,:STATE_DIMENSIONS] = _A1
    _A[:STATE_DIMENSIONS,STATE_DIMENSIONS:] = _B1
    _A[STATE_DIMENSIONS:,:STATE_DIMENSIONS] = np.zeros((CONTROL_DIMENSIONS,STATE_DIMENSIONS))
    _A[STATE_DIMENSIONS:,STATE_DIMENSIONS:] = np.eye(CONTROL_DIMENSIONS)


    x_0 =[-3.5, 0]

    k_lqr = lqr.LQR(_A, _B, _Q, _R, N_STEPS)
    x[0] = x_0
    u[0] = [0]
    
    # v_aug[t] = [ u[t] - u_ref - u[t-1] + u_ref] 
    for i in range(1,N_STEPS-1):
        z[i] = x[i] - x_ref
        z_aug[i] = np.concatenate((z[i], u[i] - u_ref))
        v_aug[i] = k_lqr[:,:,i].dot(z_aug[i])
        u[i] = v_aug[i] + u[i-1]
        x[i+1] = dynamics(x[i], u[i], control_noise=control_noise, dynamics_noise=dynamics_noise)

    print("Final state: ", x[-1:])
    plt.subplot(2, 1, 1)
    plt.plot(x[:,0], x[:,1],'r')
    plt.subplot(2, 1, 2)
    plt.plot(u[1:] - u[:-1])
    plt.show()

     


