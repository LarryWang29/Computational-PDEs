import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse

def explicit_scheme(k, N, t):
    """
    This function takes implements the explicit discretisation scheme and returns the numerical
    solution at time t

    :param k: magnitude of the time step
    :param N: number of grid points in the X-direction
    :param t: time level to calculate the numerical solution at

    :return U: the numerical solution at time t
    """
    h = 1/(N-1) # Initiate the horizontal step length
    r = k / h**2 # Create the variable r
    M = int(t/k) # Obtain the number of iterations
    U = np.linspace(0, 1, N) # Initiate the initial array
    U = 2 - 1.5*U + np.sin(np.pi * U) # Set up the initial array at time 0
    diagonals = [np.sqrt(2) * r, 1-2*np.sqrt(2) * r, np.sqrt(2) * r] # Set up the diagonals of the matrix
    A = scipy.sparse.diags(diagonals, offsets = (-1, 0, 1), shape = [N, N], format="csr") # Create a triadiagonal matrix
    A[0,0], A[0,1], A[N-1, N-1], A[N-1, N-2] = 1, 0, 1, 0 # Changing specfic entries of the matrix
    for i in range(M):
        U = A @ U
    return U

def steady_state_sol(N):
    x = np.linspace(0, 1, N)
    return (-3/2) * x + 2

# Q1c Plots
Times = np.array([0.001, 0.01, 0.1, 1])
N = 11
x = np.linspace(0, 1, N)
steady = steady_state_sol(N)
for i in Times:
    plt.plot(x, explicit_scheme(0.001, N, i) - steady)
plt.show()


# Q1d Plots
# N_array = np.array([51, 101, 251, 501])
# k_indices = [8, 6, 4, 2]
# k_array = np.array([10 ** -i for i in k_indices])
# for i in N_array:
#     errors = []
#     idx = int((i-1) / 2)
#     anal_sol = steady_state_sol(i)[idx]
#     for j in k_array:
#         numerical_sol = explicit_scheme(j, i, 1)[idx]
#         errors.append(numerical_sol-anal_sol)
#     plt.plot(k_array, errors)
# plt.show()

# Q1e code

# def solve_tridiagonal(A, b):
#     """Solve the tridiagonal system Ax = b for x
    
#     :param A: the matrix A
#     :param b: the vector b

#     :return x: the solution to the system
#     """


def implicit_scheme(k, N, t):
    """
    This function takes implements the implicit discretisation scheme and returns the numerical
    solution at time t

    :param k: magnitude of the time step
    :param N: number of grid points in the X-direction
    :param t: time level to calculate the numerical solution at

    :return U: the numerical solution at time t
    """
    h = 1/(N-1) # Initiate the horizontal step length
    r = k / h**2 # Create the variable r
    M = int(t/k) # Obtain the number of iterations
    U = np.linspace(0, 1, N) # Initiate the initial array
    U = 2 - 1.5*U + np.sin(np.pi * U) # Set up the initial array at time 0
    diagonals = [-np.sqrt(2) * r, 1+2*np.sqrt(2) * r, -np.sqrt(2) * r] # Set up the diagonals of the matrix
    A = scipy.sparse.diags(diagonals, offsets = (-1, 0, 1), shape = [N, N], format="csr") # Create a triadiagonal matrix
    A[0,0], A[0,1], A[N-1, N-1], A[N-1, N-2] = 1, 0, 1, 0 # Changing specfic entries of the matrix
    for i in range(M):
        U = scipy.sparse.linalg.spsolve(A, U)
    return U

def CN_scheme(k, N, t):
    """
    This function takes implements the Crank-Nicholson discretisation scheme and returns the numerical
    solution at time t

    :param k: magnitude of the time step
    :param N: number of grid points in the X-direction
    :param t: time level to calculate the numerical solution at

    :return U: the numerical solution at time t
    """
    h = 1/(N-1) # Initiate the horizontal step length
    r = k / h**2 # Create the variable r
    M = int(t/k) # Obtain the number of iterations
    U = np.linspace(0, 1, N) # Initiate the initial array
    U = 2 - 1.5*U + np.sin(np.pi * U) # Set up the initial array at time 0
    A_diagonals = [-r/np.sqrt(2), 1+np.sqrt(2)*r, -r/np.sqrt(2)] # Set up the diagonals of the matrix
    B_diagonals = [r/np.sqrt(2), 1-np.sqrt(2)*r, r/np.sqrt(2)]
    A = scipy.sparse.diags(A_diagonals, offsets = (-1, 0, 1), shape = [N, N], format="csr") # Create a triadiagonal matrix
    B = scipy.sparse.diags(B_diagonals, offsets = (-1, 0, 1), shape = [N, N], format="csr")
    A[0,0], A[0,1], A[N-1, N-1], A[N-1, N-2] = 1, 0, 1, 0 # Changing specfic entries of the matrix
    B[0,0], B[0,1], B[N-1, N-1], B[N-1, N-2] = 1, 0, 1, 0
    for i in range(M):
        U = scipy.sparse.linalg.spsolve(A, B @ U)
    return U