import numpy as np
import matplotlib.pyplot as plt

import lqr

STATE_DIMENSIONS = 2
CONTROL_DIMENSIONS = 2


SAMPLING_TIME = 0.01
N_STEPS = 1000

# A
def A():
    return np.array([
        [1, 0],
        [0, 1],
    ])

# B
def B():
    return np.array([
        [1, 0],
        [0, 1],
    ])


def Q():
    return np.identity(STATE_DIMENSIONS)

def R():
    return np.identity(CONTROL_DIMENSIONS)

def move(x, u, A_, B_):
    return A_.dot(x.T) + B_.dot(u)


def define_trajectory(n_points, step_size = SAMPLING_TIME):
    x1 = np.linspace(0, n_points*step_size//2, n_points//2)
    x = np.concatenate((x1, x1[-1::-1]))
    sign = np.array([(-1)**np.random.choice([1, 0],p=[0.6, 0.4]) for i in range(n_points)])
    y = np.cumsum(np.random.rand(n_points)*sign)

    return (x, y)

def plot_trajectory(trajectory):
    x, y = trajectory
    plt.plot(x, y)
    plt.show()



def plot_movement(x):
    plt.plot(x[:,0], x[:,1])
    plt.show()



if __name__ == "__main__":

    # initial state
    x = np.zeros((N_STEPS,STATE_DIMENSIONS))
    x[0] = [0, 0]

    trajectory = define_trajectory(N_STEPS)
    t = np.column_stack(trajectory)

    _A = A()
    _Q = Q()
    _R = R()
    _B = B()

    # track the trajectory for a linear system
    k_lqr = lqr.LQR(_A, _B, _Q, _R)[:,:,0]
    for i in range(N_STEPS-1):
        u = k_lqr.dot(x[i]-t[i])
        x[i+1] = move(x[i], u, _A, _B)

    subplot = plt.subplot(2,1, 1)
    plt.plot(x[:,0], x[:,1],'r')
    subplot = plt.subplot(2,1, 2)
    plt.plot(t[:,0], t[:,1],'b')
    plt.show()

    print("error", np.sum((x - t)**2)/(N_STEPS))


    # LQR drives the system to the origin
    x = np.zeros((N_STEPS,STATE_DIMENSIONS))
    x[0] = [1, 1]

    k_lqr = lqr.LQR(_A, _B, _Q, _R)[:,:,0]
    for i in range(N_STEPS-1):
        u = k_lqr.dot(x[i])
        x[i+1] = move(x[i], u, _A, _B)
        if np.sum(x[i+1]**2) < 1e-4:
            break

    print("State at the end of the trajectory", x[i+1], "Steps", i+1)




    