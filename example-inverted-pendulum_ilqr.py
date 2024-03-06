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
N_STEPS = 100
N_ITER = 10

def dynamics(x, u, dt=SAMPLING_TIME, control_noise=None, dynamics_noise=None):
    if control_noise is not None:
        u += control_noise*np.random.randn(*u.shape)
    theta, theta_dot = x
    theta_dot_dot = G/L * np.sin(theta) + (1/(M*L**2)) * u
    x_new = np.array([theta + theta_dot*dt + 0.5*theta_dot_dot*dt**2, theta_dot + theta_dot_dot*dt])
    if dynamics_noise is not None:
        x_new += dynamics_noise*np.random.randn(*x_new.shape)
    return x_new.reshape(-1)


def Q():
    return np.identity(STATE_DIMENSIONS)

def R():
    return np.identity(CONTROL_DIMENSIONS)*10

def move(x, u, A_, B_):
    return A_.dot(x.T) + B_.dot(u)


if __name__ == "__main__":

    ##### 
    x = np.zeros((N_STEPS,STATE_DIMENSIONS))
    u = np.zeros((N_STEPS,CONTROL_DIMENSIONS))

    # Linearized system
    # z = x - x_ref
    # v = u - u_ref
    z = np.zeros((N_STEPS,STATE_DIMENSIONS+1))
    v = np.zeros((N_STEPS,CONTROL_DIMENSIONS))

    dyn_x = jacobian(dynamics, 0)
    dyn_u = jacobian(dynamics, 1)

    _Q = np.eye(STATE_DIMENSIONS+1)
    _Q[-1,-1] = 0
    _R = R()
    _A_t = np.zeros((STATE_DIMENSIONS + CONTROL_DIMENSIONS,STATE_DIMENSIONS + CONTROL_DIMENSIONS,N_STEPS))
    _B_t = np.zeros((STATE_DIMENSIONS + CONTROL_DIMENSIONS,CONTROL_DIMENSIONS,N_STEPS))

    control_noise = 0.0
    dynamics_noise = np.array([[0.0] , [0.0]])

    # Initial actions to record the trajectory
    t_target_x = np.load('x_clean.npy')[:N_STEPS]
    t_target_u = np.load('u_clean.npy')[:N_STEPS]

    t_current_x = np.zeros(t_target_x.shape)
    t_current_u = np.zeros(t_target_u.shape)
    #t_current_x = np.load('./data/inverted_pendulum/x_clean.npy')[:N_STEPS]
    #t_current_u = np.load('./data/inverted_pendulum/u_clean.npy')[:N_STEPS]

    #t_target_x = t_current_x
    #t_target_u = t_current_u

    t_current_u = np.random.randn(N_STEPS,CONTROL_DIMENSIONS)
    for i in range(N_STEPS-1):
        #t_target_x[i+1] = dynamics(t_target_x[i], t_target_u[i])
        t_current_x[i+1] = dynamics(t_current_x[i], t_current_u[i])

    x_ref = t_current_x
    u_ref = t_current_u


    plt.subplot(2, 2, 1)
    plt.plot(x_ref[:,0], x_ref[:,1],'b')
    plt.subplot(2, 2, 2)
    plt.plot(t_target_x[:,0], t_target_x[:,1],'r')
    plt.subplot(2, 2, 3)
    plt.plot(u_ref,'b')
    plt.subplot(2, 2, 4)
    plt.plot(t_target_u,'r')
    plt.show()
    
    for i in range(N_ITER):
        # build A and B matrices
        _A1_t = np.array([dyn_x(x_ref[i], u_ref[i]) for i in range(N_STEPS)]).reshape(STATE_DIMENSIONS,STATE_DIMENSIONS,-1)
        _A2_t = np.vstack((t_current_x[1:] - t_target_x[1:], t_current_x[-1] - t_target_x[-1])).reshape(STATE_DIMENSIONS,1,-1)
        _A2_t += np.einsum('ijk,jk->ik',_A1_t,(t_target_x - t_current_x).T).reshape(STATE_DIMENSIONS,1,-1)
        _A2_t += np.einsum('ijk,jk->ik',_A1_t, (t_target_u - t_current_u).T).reshape(STATE_DIMENSIONS,1,-1)
        _A_t[:STATE_DIMENSIONS, :STATE_DIMENSIONS] = _A1_t
        _A_t[:STATE_DIMENSIONS, STATE_DIMENSIONS:] = _A2_t
        _A_t[STATE_DIMENSIONS:, :STATE_DIMENSIONS] = np.zeros((CONTROL_DIMENSIONS, CONTROL_DIMENSIONS))
        _A_t[STATE_DIMENSIONS:, STATE_DIMENSIONS:] = np.identity(CONTROL_DIMENSIONS)
    
    
        _B1_t = np.array([dyn_u(x_ref[i], u_ref[i]) for i in range(N_STEPS)]).reshape(STATE_DIMENSIONS,CONTROL_DIMENSIONS,-1)
        _B_t[:STATE_DIMENSIONS] = _B1_t
    
        x[0] = t_current_x[0]
        alpha = 0.1
        _Q = np.eye(STATE_DIMENSIONS+1)
        for i in range(N_STEPS-1):
            k_lqr, _Q = lqr.LQR(_A_t[:,:,i], _B_t[:,:,i], _Q, _R, 1)
            z[i] = np.append(x[i] - t_target_x[i],1.0)
            v[i] = k_lqr[:,:,0].dot(z[i]).reshape(-1,1)
            u[i] = v[i] + t_target_u[i]
            x[i+1] = dynamics(x[i], u[i], control_noise=control_noise, dynamics_noise=dynamics_noise)
            cost = z[i].T.dot(_Q).dot(z[i]) + v[i].T.dot(_R).dot(v[i])
            _Q = _Q / (np.sqrt(np.sum(_Q**2, axis=1, keepdims=True)) + 1e-6)
            #regularization_term = alpha * (np.linalg.norm(x[i] - x_ref[i])**2 + np.linalg.norm(u[i] - u_ref[i])**2)
            #_Q = (1-alpha)*_Q + alpha * np.sqrt(np.sum(x[i] - x_ref[i])**2,) + np.sqrt(np.sum(u[i] - u_ref[i])**2)
            

        x_ref = x
        u_ref = u

        """
        plt.subplot(2, 2, 1)
        plt.plot(x_ref[:,0], x_ref[:,1],'b')
        plt.subplot(2, 2, 2)
        plt.plot(t_target_x[:,0], t_target_x[:,1],'r')
        plt.subplot(2, 2, 3)
        plt.plot(u_ref,'b')
        plt.subplot(2, 2, 4)
        plt.plot(t_target_u,'r')
        plt.show()
        """
        
    
    
plt.subplot(2, 2, 1)
plt.plot(x_ref[:,0], x_ref[:,1],'b')
plt.subplot(2, 2, 2)
plt.plot(t_target_x[:,0], t_target_x[:,1],'r')
plt.subplot(2, 2, 3)
plt.plot(u_ref,'b')
plt.subplot(2, 2, 4)
plt.plot(t_target_u,'r')
plt.show()
    

  