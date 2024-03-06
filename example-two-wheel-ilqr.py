import numpy as np
import autograd.numpy as np
from autograd import jacobian
import matplotlib.pyplot as plt


from ilqr import iLQR

# constants
WHEEL_RADIUS_LEFT  = 0.1
WHEEL_RADIUS_RIGHT = 0.1
WHEEL_DISTANCE     = 0.5

STATE_DIMENSIONS = 3
CONTROL_DIMENSIONS = 2


SAMPLING_TIME = 0.01
N_STEPS = 100
N_ITER = 30

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
    L = np.dot((x - x_ref).T, np.dot(Q,(x - x_ref))) + np.dot((u - u_ref).T, np.dot(R,(u - u_ref))) 
    return L

def cost_final(x, x_ref, Q):
    return np.dot((x - x_ref).T, np.dot(Q,(x - x_ref)))

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
    _Q = np.diag([0.1,0.1,10])
    _R = np.diag([1,1])
    _QT = np.diag([100,100,100])

    control_noise = np.array([0.2, 0.3])
    dynamics_noise = np.array([[0.03] , [0.2], [0.1]])

    t_current_u = np.random.randn(N_STEPS,CONTROL_DIMENSIONS)
    t_current_x = np.zeros((N_STEPS,STATE_DIMENSIONS))

    t_target_u, t_target_x = build_trajectory(N_STEPS, step_size = SAMPLING_TIME)

    for i in range(N_STEPS-1):
        t_current_x[i+1] = dynamics(t_current_x[i], t_current_u[i])
    
    dyn_x = jacobian(dynamics, 0)
    dyn_u = jacobian(dynamics, 1)
    fa = np.array([dyn_x(t_current_x[i], t_current_u[i]) for i in range(N_STEPS)]).reshape(CONTROL_DIMENSIONS,STATE_DIMENSIONS,-1)
    fu = np.array([dyn_u(t_current_x[i], t_current_u[i]) for i in range(N_STEPS)]).reshape(CONTROL_DIMENSIONS,STATE_DIMENSIONS,-1)
    ilqr = iLQR(dynamics, cost, cost_final, t_target_x, t_target_u, _Q, _R, _QT, control_noise, dynamics_noise, N_ITER)
    t_current_x, t_current_u, cost_trace = ilqr.optimize(t_current_x, t_current_u)

    plt.subplot(2, 2, 1)
    plt.plot(t_current_x[:,0], t_current_x[:,1],'b', linewidth=1.0)
    plt.subplot(2, 2, 2)
    plt.plot(t_target_x[:,0], t_target_x[:,1],'r')
    plt.subplot(2, 2, 3)
    plt.plot(t_current_u[:,0], t_current_u[:,1],'b')
    plt.subplot(2, 2, 4)
    plt.plot(t_target_u[:,0], t_target_u[:,1],'r')
    plt.show()
    

  