
#%%
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Function to define the constraints for points in C
def constraint_c(n_and_d, C):
    n, d = n_and_d[:-1], n_and_d[-1]
    return np.dot(C, n) + d - 0.01  # Adding a small margin

# Function to define the constraints for points in Q
def constraint_q(n_and_d, Q):
    n, d = n_and_d[:-1], n_and_d[-1]
    return -(np.dot(Q, n) + d) - 0.01 # Adding a small margin

# Dummy objective function
def dummy_objective(n_and_d):
    return 0.0

# # Constraint to ensure n is a unit vector
# def unit_vector_constraint(n_and_d):
#     n = n_and_d[:-1]
#     return np.linalg.norm(n) - 1

def find_separation_plane(C, Q):
    # Initial guess for n and d
    n_dim = C.shape[1]
    initial_guess = np.zeros(n_dim + 1)
    initial_guess[:-1] = 1 / np.sqrt(n_dim)  # Normalized initial guess for n

    # Set up constraints
    constraints = [
        {'type': 'ineq', 'fun': lambda x: constraint_c(x, C)},
        {'type': 'ineq', 'fun': lambda x: constraint_q(x, Q)},
        # {'type': 'eq', 'fun': unit_vector_constraint}
    ]

    # Perform the optimization
    result = minimize(dummy_objective, initial_guess, constraints=constraints, method='SLSQP', options={'ftol': 1e-4, 'maxiter': 1000})

    if not result.success:
        return None

    # Extract the result for n and d (normalise by size of n)
    n_and_d = result.x / np.linalg.norm(result.x[:-1])

    # now we need to move the plane to 'stick' to the obstacle for less conservatism
    # closest vertex to the plane on the obstacle C:
    # n = n_and_d[:-1]
    # d = n_and_d[-1]
    # d_min = np.min(np.abs(np.dot(C, n)+d))
    # n_and_d[-1] -= d_min # this line would give the min jerk optimiser more space and freedom, but seems to slow it down.

    

    return n_and_d

def plot_separation_line(C, Q, n, d):
    # Generate x values for plotting the line
    x_vals = np.linspace(min(C[:, 0].min(), Q[:, 0].min()) - 1, max(C[:, 0].max(), Q[:, 0].max()) + 1, 100)

    if n is not None and d is not None:
        # Calculate y values based on the line equation n[0] * x + n[1] * y + d = 0
        y_vals = -(n[0] * x_vals + d) / (n[1]+1e-6)
        # Plot the separating line
        plt.plot(x_vals, y_vals, 'k--', label='Separating Plane')

    # Plot the points in C and Q
    plt.scatter(C[:, 0], C[:, 1], color='b', label='Set C')
    plt.scatter(Q[:, 0], Q[:, 1], color='r', label='Set Q')


    # Set labels and legend
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Separating Line Visualization')
    plt.show()


if __name__ == '__main__':

    #modified minvo to work with t from 0 to 1 like most other splines
    minvo3 = np.array([
        [-3.4416, 6.9896, -4.46236, 0.91439], 
        [6.6792, -11.846, 5.2524, 0.0001],
        [-6.6792, 8.1916, -1.598, 0.0856],
        [3.4416, -3.3352, 0.80796, -0.0007903]
    ]).T

    bspline3 = np.array([
        [-1, 3, -3, 1],
        [3, -6, 3, 0],
        [-3, 0, 3, 0],
        [1, 4, 1, 0]]) / 6
        # Example points in sets C and Q
    Q = np.array([[1, 2], [2, 2], [3, 2]]) - np.array([0,2])
    C = np.array([[1, 0], [2, 0.], [3, 0]]) - np.array([0,2])

    Qbs=np.array([[ 0.        , -5.        ],
       [ 0.        , -4.66666667],
       [-0.16666667, -4.33333333],
       [-0.72892415, -4.64930875]])
    Qmv = np.linalg.inv(minvo3)@bspline3@Qbs
    print(f"{Qmv=}")
    

    # C = np.array([[-1.5,-3.], [0, -4.4], [0,-3.4], [-1.5, -3.4]])
    # Find the separating plane
    n_and_d = find_separation_plane(C, Q)

    if n_and_d is not None:
        n, d = n_and_d[:-1], n_and_d[-1]
        print("Found a separating plane:")
        print("n:", n)
        print("d:", d)
        plot_separation_line(C, Q, n, d)
    else:
        print("Failed to find a separating plane")

# %%