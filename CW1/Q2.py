import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg

epsilon_array = np.array([10 ** -i for i in range(1,5)])
def upper_bound(x):
    return x-1/2

def lower_bound(x):
    return x-1

def Q2_discretisation_scheme(epsilon, N):
    h = 1/(N-1)
    x = np.linspace(0, 1, N)
    # x_bar = 3/4
    # w_0 = 1/4
    # U = x - x_bar + w_0*np.tanh(w_0*(x-x_bar)/(2*epsilon))
    U = x - 3/4
    U[0] = -1
    U[N-1] = 1/2
    for j in range(100):
        G = np.array([epsilon * (U[i-1]-2*U[i]+U[i+1]) / h**2 + \
                    U[i]*((U[i+1]-U[i-1])/(2*h) -1) for i in range(1,N-1)])
        main_diag = [-2*epsilon/h**2 + (1/2*h)*(U[i+1]-U[i-1]) for i in range(1,N-1)]
        sup_diag = [epsilon/h**2 + (1/(2*h))*U[i] for i in range(1,N-2)]
        sub_diag = [epsilon/h**2 - (1/(2*h))*U[i] for i in range(2,N-1)]
        J = scipy.sparse.diags([sub_diag, main_diag, sup_diag], [-1,0,1], format="csr")
        if np.linalg.norm(scipy.sparse.linalg.spsolve(J, -G)) < 1e-8:
            print(j)
            break
        print(np.linalg.norm(scipy.sparse.linalg.spsolve(J, -G)))
        U[1:N-1] += scipy.sparse.linalg.spsolve(J, -G)
    return U

x = np.linspace(0,1,1001)
plt.plot(np.linspace(0,1,10001), Q2_discretisation_scheme(0.0001, 10001))
plt.plot(x, x-1)
plt.plot(x, x-1/2)
plt.show()

# N = 10001
# h = 1/(N-1)
# x = np.linspace(0, 1, N)
# x_bar = 3/4
# w_0 = 1/4
# U = x - x_bar + w_0*np.tanh(w_0*(x-x_bar)/(2*0.01))
# plt.plot(x, U)
# plt.plot(x, x-1)
# plt.plot(x, x-1/2)
# plt.show()