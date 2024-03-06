import numpy as np
import matplotlib.pyplot as plt

import lqr

# constants
WHEEL_RADIUS_LEFT  = 0.1
WHEEL_RADIUS_RIGHT = 0.1
WHEEL_DISTANCE     = 0.5

STATE_DIMENSIONS = 3
CONTROL_DIMENSIONS = 2


SAMPLING_TIME = 0.01
N_STEPS = 1000

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
def A(x, u, dt=SAMPLING_TIME):
    theta_t = x[2]
    u1, u2 = u
    cost = np.cos(theta_t)
    sint = np.sin(theta_t)
    return np.array([
        [1, 0, -np.pi * (WHEEL_RADIUS_RIGHT * u1 + WHEEL_RADIUS_LEFT * u2) * sint * dt],
        [0, 1,  np.pi * (WHEEL_RADIUS_RIGHT * u1 + WHEEL_RADIUS_LEFT * u2) * cost * dt],
        [0, 0, 1]
    ])


# B
def B(x, dt=SAMPLING_TIME):
    theta_t = x[2]
    cost = np.cos(theta_t)
    sint = np.sin(theta_t)
    return np.array([
        [np.pi * WHEEL_RADIUS_RIGHT * cost * dt, np.pi * WHEEL_RADIUS_LEFT * cost * dt],
        [np.pi * WHEEL_RADIUS_RIGHT * sint * dt, np.pi * WHEEL_RADIUS_LEFT * sint * dt],
        [0, 0]
    ])


def Q():
    return np.identity(STATE_DIMENSIONS)

def R():
    return np.identity(CONTROL_DIMENSIONS)

def move(x, u, A_, B_):
    return A_.dot(x.T) + B_.dot(u)


def define_trajectory(n_points, step_size = SAMPLING_TIME):
    x = np.linspace(0, n_points*step_size, n_points)
    y = 5 * np.sin(x) + 10 
    theta = np.mod(np.arctan2(np.diff(y), np.diff(x)), 2*np.pi)

    return (x[:n_points-1], y[:n_points-1], theta[:n_points-1])

def plot_trajectory(trajectory):
    x, y, _ = trajectory
    plt.plot(x, y)
    plt.show()



def plot_movement(x):
    plt.plot(x[:,0], x[:,1])
    plt.show()



if __name__ == "__main__":

       # initial state
    x = np.zeros((N_STEPS,STATE_DIMENSIONS))

    trajectory = define_trajectory(N_STEPS+1)
    t = np.column_stack(trajectory)

    _Q = Q()
    _R = R()

    x_star = t[0]
    u_star = [0, 0]
    # track the trajectory for a linear system
    for i in range(N_STEPS-1):
        _A = A(x_star, u_star)
        _B = B(x_star)
        z = x[i] - x_star
        k_lqr = lqr.LQR(_A, _B, _Q, _R)
        u = k_lqr.dot(z)
        x[i+1] = move(x[i], u, _A, _B)
        x_star = t[i]
        u_star = u

    subplot = plt.subplot(2,1, 1)
    plt.plot(x[:,0], x[:,1],'r')
    subplot = plt.subplot(2,1, 2)
    plt.plot(t[:,0], t[:,1],'b')
    plt.show()

    print("error", np.sum((x - t)**2)/(N_STEPS))


    # LQR drives the system to the origin
    x = np.zeros((N_STEPS,STATE_DIMENSIONS))
    x[0] = [1, 1,0]

    k_lqr = lqr.LQR(_A, _B, _Q, _R)
    for i in range(N_STEPS-1):
        u = k_lqr.dot(x[i])
        x[i+1] = move(x[i], u, _A, _B)
        if np.sum(x[i+1]**2) < 1e-4:
            break

    print("State at the end of the trajectory", x[i+1], "Steps", i+1)

