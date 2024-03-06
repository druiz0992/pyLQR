import numpy as np
import autograd.numpy as np
from autograd import  jacobian
import matplotlib.pyplot as plt

import lqr

# constants
WHEEL_RADIUS_LEFT  = 0.1
WHEEL_RADIUS_RIGHT = 0.1
WHEEL_DISTANCE     = 0.5

STATE_DIMENSIONS = 3
CONTROL_DIMENSIONS = 2


SAMPLING_TIME = 0.01
N_STEPS = 100
N_ITER = 10

def dynamics(x, u, dt=SAMPLING_TIME, control_noise=None, dynamics_noise=None):
    if control_noise is not None:
        u += control_noise*np.random.randn(*u.shape)
    x_c, y_c, theta = x
    x_c_new = x_c + np.pi * (WHEEL_RADIUS_RIGHT * u[0] + WHEEL_RADIUS_LEFT * u[1]) * np.cos(theta) * dt
    y_c_new = y_c + np.pi * (WHEEL_RADIUS_RIGHT * u[0] + WHEEL_RADIUS_LEFT * u[1]) * np.sin(theta) * dt
    theta_new = theta + 2 * np.pi * (WHEEL_RADIUS_RIGHT * u[0] - WHEEL_RADIUS_LEFT * u[1])/WHEEL_DISTANCE * dt
    x_new = np.array([x_c_new, y_c_new, theta_new])
    if dynamics_noise is not None:
        x_new += dynamics_noise.reshape(-1)*np.random.randn(*x_new.shape)
    return x_new.reshape(-1)

def cost(x, u, x_ref, u_ref, Q, R):
    return (x - x_ref).T.dot(Q).dot(x - x_ref) + (u - u_ref).T.dot(R).dot(u - u_ref)


def move(x, u, A_, B_):
    return A_.dot(x.T) + B_.dot(u)

def build_trajectory(n_steps, step_size = 0.1):
    x = np.zeros((n_steps,STATE_DIMENSIONS))
    u = np.zeros((n_steps,CONTROL_DIMENSIONS))
    U = np.array([
        [0,0,0],
        [10,10, 30],
        [10,30, 20],
        [20,20, 20],
        [20,10, 20],
        [10,10, 10]
    ]) * np.array([1,1, n_steps/100])

    bins = np.cumsum(U[:,2])
    for i in range(n_steps-1):
        for bin_idx,bin in enumerate(bins[:-1]):
            if np.floor(i * 100/n_steps) >= bin and np.floor(i * 100/n_steps) < bins[bin_idx+1]:
                u[i] = U[bin_idx+1][:CONTROL_DIMENSIONS]
                break
        x[i+1] = dynamics(x[i], u[i], dt=step_size)
    return u,x

if __name__ == "__main__":
    show_changes = False

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

    _Q = np.diag([0,0.1,0.1,1])
    _R = np.diag([0.1])
    _QT = np.diag([0,100,100,100])

    _A_t = np.zeros((STATE_DIMENSIONS + 1,STATE_DIMENSIONS + 1,N_STEPS))
    _B_t = np.zeros((STATE_DIMENSIONS + 1,CONTROL_DIMENSIONS,N_STEPS))

    control_noise = np.array([0.2, 0.3])
    dynamics_noise = np.array([[0.03] , [0.2], [0.1]])

    t_current_u = np.random.randn(N_STEPS,CONTROL_DIMENSIONS)
    t_current_x = np.zeros((N_STEPS,STATE_DIMENSIONS))

    t_target_u, t_target_x = build_trajectory(N_STEPS, step_size = SAMPLING_TIME)

    for i in range(N_STEPS-1):
        t_current_x[i+1] = dynamics(t_current_x[i], t_current_u[i])

    x_ref = t_current_x
    u_ref = t_current_u


    plt.subplot(2, 2, 1)
    plt.plot(x_ref[:,0], x_ref[:,1],'b', linewidth=1.0)
    plt.subplot(2, 2, 2)
    plt.plot(t_target_x[:,0], t_target_x[:,1],'r')
    plt.subplot(2, 2, 3)
    plt.plot(u_ref[:,0], u_ref[:,1],'b')
    plt.subplot(2, 2, 4)
    plt.plot(t_target_u[:,0], t_target_u[:,1],'r')
    plt.show()

    for i in range(N_ITER):
        # build A and B matrices
        _A1x_t = np.array([dyn_x(x_ref[i], u_ref[i]) for i in range(N_STEPS)]).reshape(STATE_DIMENSIONS,STATE_DIMENSIONS,-1)
        _A1u_t = np.array([dyn_u(x_ref[i], u_ref[i]) for i in range(N_STEPS)]).reshape(CONTROL_DIMENSIONS,STATE_DIMENSIONS,-1)
        _A2_t = np.vstack((t_current_x[1:] - t_target_x[1:], t_current_x[-1] - t_target_x[-1])).reshape(STATE_DIMENSIONS,1,-1)
        _A2_t += np.einsum('ijk,jk->ik',_A1x_t,(t_target_x - t_current_x).T).reshape(STATE_DIMENSIONS,1,-1)
        _A2_t += np.einsum('ijk,ik->jk',_A1u_t, (t_target_u - t_current_u).T).reshape(STATE_DIMENSIONS,1,-1)
        _A_t[:STATE_DIMENSIONS, :STATE_DIMENSIONS] = _A1x_t
        _A_t[:STATE_DIMENSIONS, STATE_DIMENSIONS:] = _A2_t
        _A_t[STATE_DIMENSIONS:, :STATE_DIMENSIONS] = np.zeros((1, STATE_DIMENSIONS, N_STEPS))
        _A_t[STATE_DIMENSIONS:, STATE_DIMENSIONS:] = 1
    
    
        _B1_t = np.array([dyn_u(x_ref[i], u_ref[i]) for i in range(N_STEPS)]).reshape(STATE_DIMENSIONS,CONTROL_DIMENSIONS,-1)
        _B_t[:STATE_DIMENSIONS] = _B1_t
    
        x[0] = t_current_x[0]
        alpha = 0.1
        _Q = np.eye(STATE_DIMENSIONS+1)
        for i in range(N_STEPS-1):
            k_lqr, _Q = lqr.LQR(_A_t[:,:,i], _B_t[:,:,i], _Q, _R, 1)
            z[i] = np.append(x[i] - t_target_x[i],1.0)
            v[i] = k_lqr[:,:,0].dot(z[i])
            u[i] = v[i] + t_target_u[i]
            x[i+1] = dynamics(x[i], u[i], control_noise=control_noise, dynamics_noise=dynamics_noise)
            cost = z[i].T.dot(_Q).dot(z[i]) + v[i].T.dot(_R).dot(v[i])
            _Q = _Q / (np.sqrt(np.sum(_Q**2, axis=1, keepdims=True)) + 1e-6)
            #regularization_term = alpha * (np.linalg.norm(x[i] - x_ref[i])**2 + np.linalg.norm(u[i] - u_ref[i])**2)
            #_Q = (1-alpha)*_Q + alpha * np.sqrt(np.sum(x[i] - x_ref[i])**2,) + np.sqrt(np.sum(u[i] - u_ref[i])**2)
            

        x_ref = x
        u_ref = u

        if show_changes:
            plt.subplot(2, 2, 1)
            plt.plot(x_ref[:,0], x_ref[:,1],'b')
            plt.subplot(2, 2, 2)
            plt.plot(t_target_x[:,0], t_target_x[:,1],'r')
            plt.subplot(2, 2, 3)
            plt.plot(u_ref[:,0], u_ref[:,1],'b')
            plt.subplot(2, 2, 4)
            plt.plot(t_target_u[:,0], t_target_u[:,1],'r')
            plt.show()
        
    
    
plt.subplot(2, 2, 1)
plt.plot(x_ref[:,0], x_ref[:,1],'b')
plt.subplot(2, 2, 2)
plt.plot(t_target_x[:,0], t_target_x[:,1],'r')
plt.subplot(2, 2, 3)
plt.plot(u_ref[:,0], u_ref[:,1],'b')
plt.subplot(2, 2, 4)
plt.plot(t_target_u[:,0], t_target_u[:,1],'r')
plt.show()
    

  