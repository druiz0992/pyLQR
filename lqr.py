import numpy as np


def LQR(A, B, Q, R, N=100):
    V = Q.copy()
    K = np.zeros((R.shape[0], Q.shape[0], N))
    for i in range(N-1, -1, -1):
        K[:,:,i] = -np.linalg.inv(R + B.T.dot(V).dot(B)).dot(B.T).dot(V).dot(A)
        V = Q + K[:,:,i].T.dot(R).dot(K[:,:,i]) + (A + B.dot(K[:,:,i])).T.dot(V).dot(A + B.dot(K[:,:,i]))
    return K, V 
