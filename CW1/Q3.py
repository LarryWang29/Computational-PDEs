import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt

def central_difference_O4(epsilon, N):
    h = 1/(N-1)
    r = 1/h
    p = epsilon/h**2 # Redifine some variables
    x = np.linspace(0, 1, N)
    x_bar = 3/4
    w_0 = 1/4
    U = x - x_bar + w_0*np.tanh(w_0*(x-x_bar)/(2*epsilon))
    # U = x - 3/4
    U[0] = -1
    U[N-1] = 1/2
    for j in range(1000):
        G = np.array([(p*(11*U[0]/12 - 5*U[1]/3 + U[2]/2 + U[3]/3 - U[4]/12) + \
            U[1]*(r*(-U[0]/4 - 5*U[1]/6 + 3*U[2]/2 - U[3]/2 + U[4]/12) - 1))] + \
            (p*(-U[:N-4]/12 + 4*U[1:N-3]/3 - 5*U[2:N-2]/2 + 4*U[3:N-1]/3 -U[4:]/12) + \
            U[2:N-2]*(r*(U[:N-4]/12 - 2*U[1:N-3]/3 + 2*U[3:N-1]/3 - U[4:]/12) - 1)).tolist() + \
            [(p*(11*U[-1]/12 - 5*U[-2]/3 + U[-3]/2 + U[-4]/3 - U[-5]/12) + \
            U[1]*(r*(-U[N-1]/4 - 5*U[N-2]/6 + 3*U[N-3]/2 - U[N-4]/2 + U[N-5]/12) - 1))])
        sub_diag1 = (4*p/3 - (2*r/3)*U[2:N-2]).tolist() + \
            [p/2 - (3*r/2)*U[N-1]]
        sub_diag2 = (-p/12 + (r/12)*U[3:N-2]).tolist() + \
                    [p/3 + (-r/2)*U[1]]
        sub_diag3 = [-p/12 + U[1]*r/12] + [0] * (N-6)
        main_diag = [-5*p/3 + r*(-U[0]/4 - (5/3)*U[1] + (3/2)*U[2]-U[3]/2+U[4]/12) - 1] + \
                    (-5*p/2 + r*(U[:N-4]/12 - 2*U[1:N-3]/3 + 2*U[3:N-1]/3 - U[4:]/12) - 1).tolist() + \
                    [-5*p/3 - r*(-U[-1]/4 - (5/3)*U[-2] + (3/2)*U[-3]-U[-4]/2+U[-5]/12) - 1]
        sup_diag1 = [p/2 + (3*r/2)*U[1]] + \
                    (4*p/3 + (2*r/3)*U[2:N-2]).tolist()
        sup_diag2 = [p/3 - (-r/2)*U[1]] + \
                    (-p/12 - (r/12)*U[2:N-3]).tolist()
        sup_diag3 =  [0] * (N-6) + [-p/12 - U[N-1]*r/12]
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
plt.plot(np.linspace(0,1,1001), central_difference_O4(0.001, 1001))
plt.plot(x, x-1)
plt.plot(x, x-1/2)
plt.show()