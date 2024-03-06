import numpy as np
import autograd.numpy as np
from autograd import  jacobian
import matplotlib.pyplot as plt

class iLQR:
    def __init__(self, dynamics, cost, cost_final, x_target, u_target, Q, R, QT, control_noise, dynamics_noise, N):
        self.dynamics = dynamics
        self.cost = cost
        self.cost_final = cost_final
        self.x_target = x_target
        self.u_target = u_target
        self.Q = Q
        self.R = R
        self.QT = QT
        self.control_noise = control_noise
        self.dynamics_noise = dynamics_noise
        self.N = N

    def forward_pass(self, x, u, K, k):
        T = u.shape[0]
        u_new = np.empty_like(u)
        x_new = np.empty_like(x)
        x_new[0] = x[0]
        cost_new = 0.0

        for n in range(T):
            u_new[n] = u[n] + k[n].T + K[n].dot(x_new[n] - x[n])
            x_new[n+1] = self.dynamics(x_new[n], u_new[n], dynamics_noise=self.dynamics_noise, control_noise=self.control_noise)
            cost_new += self.cost(x_new[n], u_new[n], self.x_target[n], self.u_target[n], self.Q, self.R)
        cost_new += self.cost_final(x_new[-1], self.x_target[-1], self.QT)

        return x_new, u_new, cost_new

    def backward_pass(self, x, u):
        k = np.zeros((u.shape[0], self.R.shape[0],1))
        K = np.zeros((u.shape[0], self.R.shape[0], self.Q.shape[0]))
        delta_V = 0

        V_x = jacobian(self.cost_final, 0)
        V_xx = jacobian(V_x, 0)

        _V_x = (V_x(x[-1], self.x_target[-1], self.QT)).reshape(-1,1)
        _V_xx = V_xx(x[-1], self.x_target[-1], self.QT)
        noise = np.array([0.1, 0.2, 0.3])

        for n in range(u.shape[0]-1, -1, -1):
            f_x = jacobian(self.dynamics, 0)
            f_u = jacobian(self.dynamics, 1)

            l_x = jacobian(self.cost, 0)
            l_u = jacobian(self.cost, 1)
            l_xx = jacobian(l_x, 0)
            l_ux = jacobian(l_u, 0)
            l_uu = jacobian(l_u, 1)


            _f_x = f_x(x[n], u[n])
            _f_u = f_u(x[n], u[n])
            _l_x = l_x(x[n], u[n], self.x_target[n], self.u_target[n], self.Q, self.R).reshape(-1,1)
            _l_u = l_u(x[n], u[n], self.x_target[n], self.u_target[n], self.Q, self.R).reshape(-1,1)
            _l_xx = l_xx(x[n], u[n], self.x_target[n], self.u_target[n], self.Q, self.R)
            _l_ux = l_ux(x[n], u[n], self.x_target[n], self.u_target[n], self.Q, self.R)
            _l_uu = l_uu(x[n], u[n], self.x_target[n], self.u_target[n], self.Q, self.R)


            Q_x = _l_x + _f_x.T.dot(_V_x)
            Q_u = _l_u + _f_u.T.dot(_V_x)
            Q_xx = _l_xx + _f_x.T.dot(_V_xx).dot(_f_x)
            Q_ux = _l_ux + _f_u.T.dot(_V_xx).dot(_f_x)
            Q_uu = _l_uu + _f_u.T.dot(_V_xx).dot(_f_u)
            Q_uu_inv = np.linalg.inv(Q_uu)

            k[n] = -Q_uu_inv.dot(Q_u)
            K[n] = -Q_uu_inv.dot(Q_ux)


            _V_x = Q_x + K[n].T.dot(Q_u) + Q_ux.T.dot(k[n]) + K[n].T.dot(Q_uu).dot(k[n])
            _V_xx = Q_xx + 2 * K[n].T.dot(Q_ux) + K[n].T.dot(Q_uu).dot(K[n])
            delta_V += Q_u.T.dot(k[n]) + 0.5*k[n].T.dot(Q_uu).dot(k[n])

        return K, k, delta_V

    def optimize(self, x_init, u_init):
        u = u_init
        x, J_old = self.rollout(x_init[0], u)
        T = x_init.shape[0]

        for i in range(self.N):
            K, k, expected_cost_reduction = self.backward_pass(x, u)
            x_new, u_new, J_new = self.forward_pass(x, u, K, k)
            x = x_new
            u = u_new
        
        return x, u, J_new
        

    def rollout(self, x0, u):
        T = u.shape[0]
        x = np.zeros((T+1, x0.shape[0]))
        x[0] = x0
        cost = 0
        for i in range(T):
            x[i+1] = self.dynamics(x[i], u[i], dynamics_noise=self.dynamics_noise, control_noise=self.control_noise)
            cost += self.cost(x[i], u[i], self.x_target[i], self.u_target[i], self.Q, self.R)
        cost += self.cost_final(x[-1], self.x_target[-1], self.QT)
        return x, cost