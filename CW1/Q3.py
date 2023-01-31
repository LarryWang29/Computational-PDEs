import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt

def central_difference_O4(epsilon, N):
    h = 1/(N-1)
    r = 1/(12*h)
    p = epsilon/(12*h**2) # Redifine some variables
    x = np.linspace(0, 1, N)
    x_bar = 3/4
    w_0 = 1/4
    U = x - x_bar + w_0*np.tanh(w_0*(x-x_bar)/(2*epsilon))
    # U = x - 3/4
    U[0] = -1
    U[N-1] = 1/2
    for j in range(1000):
        G = np.array([(p*(11*U[0] - 20*U[1] + 6*U[2] + 4*U[3] - U[4]) + \
            U[1]*(r*(-3*U[0] - 10*U[1] + 18*U[2] - 6*U[3] + U[4]) - 1))] + \
            (p*(-U[:N-4] + 16*U[1:N-3] - 30*U[2:N-2] + 16*U[3:N-1] -U[4:]) + \
            U[2:N-2]*(r*(U[:N-4] - 8*U[1:N-3] + 8*U[3:N-1] - U[4:]) - 1)).tolist() + \
            [(p*(11*U[-1] - 20*U[-2] + 6*U[-3] + 4*U[-4] - U[-5]) - \
            U[-2]*(r*(-3*U[-1] - 10*U[-2] + 18*U[-3] - 6*U[-4] + U[-5]) - 1))])
        sub_diag1 = (16*p - (8*r)*U[2:N-2]).tolist() + \
                    [6*p - (18*r)*U[N-2]]
        sub_diag2 = (-p + (r)*U[3:N-2]).tolist() + \
                    [4*p - (-6*r)*U[N-2]]
        sub_diag3 = [0] * (N-6) + [-p - U[N-2]*r]
        main_diag = [-20*p + r*(-3*U[0] - 20*U[1] + 18*U[2]-6*U[3]+U[4]) - 1] + \
                    (-30*p + r*(U[:N-4] - 8*U[1:N-3] + 8*U[3:N-1] - U[4:]) - 1).tolist() + \
                    [-20*p - r*(-3*U[-1] - 20*U[-2] + 18*U[-3]-6*U[-4]+U[-5]) - 1]
        sup_diag1 = [6*p + (18*r)*U[1]] + \
                    (16*p + (8*r)*U[2:N-2]).tolist()
        sup_diag2 = [4*p + (-6*r)*U[1]] + \
                    (-p - (r)*U[2:N-3]).tolist()
        sup_diag3 = [-p + U[1]*r] + [0] * (N-6)
        Diagonals = [sub_diag3, sub_diag2, sub_diag1, main_diag, sup_diag1, sup_diag2, sup_diag3]
        J = scipy.sparse.diags(Diagonals, [-3, -2, -1, 0, 1, 2, 3], format="csr")
        # print(scipy.sparse.linalg.norm(scipy.sparse.linalg.inv(J)))
        # J1 = J.toarray()
        # print(np.linalg.norm(np.linalg.inv(J1)))
        # print(np.linalg.cond(J))
        if np.linalg.norm(scipy.sparse.linalg.spsolve(J, -G)) < 1e-8:
            print(j)
            break
        U[1:N-1] += scipy.sparse.linalg.spsolve(J, -G)
    return U

# central_difference_O4(0.01, 1001)
x = np.linspace(0,1,1001)
plt.plot(np.linspace(0,1,101), central_difference_O4(0.01, 101))
plt.plot(x, x-1)
plt.plot(x, x-1/2)
plt.show()