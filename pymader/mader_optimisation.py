#%%
import numpy as np
from scipy.optimize import minimize
from matrices import Matrix3, Matrix2

#modified minvo to work with t from 0 to 1 like most other splines
minvo3 = np.array([
    [-3.4416, 6.9896, -4.46236, 0.91439], 
    [6.6792, -11.846, 5.2524, 0.0001],
    [-6.6792, 8.1916, -1.598, 0.0856],
    [3.4416, -3.3352, 0.80796, -0.0007903]
]).T

minvo3inv = np.linalg.inv(minvo3)

minvo2 = np.array([
    [ 1.5      , -3.       ,  1.5      ],
    [-2.3660254,  3.       , -0.6339746],
    [ 0.9330127,  0.       ,  0.0669873]])

minvo2inv = np.linalg.inv(minvo2)

def optimize_spline(Q, all_planes, goal, r, vmax, amax, knots):
    # `Q` is the array of control points of shape (n+1, ndim)
    # `planes` is an array of shape (num_intervals, ndim+1) containing the normal vectors and distances for each interval
    # `r` is the maximum allowable distance from the starting point
    p = 3
    num_intervals = len(all_planes[0])
    ndim = Q.shape[1]
    fixed_points = Q[:3]
    starting_point = Q[0]

    omega = 2  # Scale factor for the distance term in the objective

    def objective(x):
        # Reshape x to match the shape of Q, excluding the fixed points and last three points
        x_reshaped = x.reshape(-1, ndim)
        Q_opt = np.vstack([fixed_points, x_reshaped, np.tile(x_reshaped[-1], (2, 1))])
        jerks = [np.array([[6,0,0,0]])@Matrix3(i,knots)@Q_opt[i:i+p+1] for i in range(num_intervals)]

        # Calculate the overall jerk
        overall_jerk = sum(np.linalg.norm(jerk)**2 for jerk in jerks)
        # Calculate the distance from the third-to-last point to the goal
        distance_to_goal = np.linalg.norm(Q_opt[-3] - goal)

        # print(f"{overall_jerk=}")

        # Add the distance to the objective, scaled by omega
        return overall_jerk + omega * distance_to_goal

    # Constraints
    constraints = []

    # Vectorized Plane constraints
    def plane_constraints(x):
        return_value = np.empty((0,num_intervals))
        # print(f"empty {return_value=}")
        for planes in all_planes:
            # print(planes.shape)
            x_reshaped = x.reshape(-1, ndim)
            Q_opt = np.vstack([fixed_points, x_reshaped, np.tile(x_reshaped[-1], (2, 1))])
            
            # Extract the control points for each interval. We use minvo control points here
            Qmv_intervals = np.array([minvo3inv@Matrix3(j,knots)@Q_opt[j:j+4] for j in range(num_intervals)])
            # print(Qmv_intervals.shape)
            
            # Apply the plane constraints
            n = planes[:, :-1]
            d = planes[:, -1]
            plane_dots = np.einsum('ijk,ik->ij', Qmv_intervals, n) + d[:, None]
            # print(f"{np.min(-plane_dots, axis=1)=}")
            # print(f"{np.min(-plane_dots, axis=1).shape=}")
            # print(f"{return_value.shape=}")
            return_value = np.vstack((return_value, np.min(-plane_dots, axis=1)))
        # print(f"final {return_value=}")
        return return_value.flatten()
        
    
    constraints.append({'type': 'ineq', 'fun': plane_constraints})

    # Distance constraint
    def distance_constraint(x):
        x_reshaped = x.reshape(-1, ndim)
        # Q_opt = np.vstack([fixed_points, x_reshaped, np.tile(x_reshaped[-1], (2, 1))])
        # distances = np.linalg.norm(Q_opt - starting_point, axis=1)
        # only check the points we are optimising, the rest is fixed
        distances = np.linalg.norm(x_reshaped - starting_point, axis=1)

        # print(f"{r - np.max(distances)=}")
        return r - np.max(distances)

    constraints.append({'type': 'ineq', 'fun': distance_constraint})

    # Velocity and acceleration constraints
    def velocity_acceleration_constraints(x):
        x_reshaped = x.reshape(-1, ndim)
        # print(f"{x_reshaped.shape=}")
        Q_opt = np.vstack([fixed_points, x_reshaped, np.tile(x_reshaped[-1], (2, 1))])


        # Velocity calculation
        Vbs = p * (Q_opt[1:] - Q_opt[:-1]) / (knots[p+1:-1] - knots[1:-(p+1)])[:, None]
        # shape of Vmv_intervals = (num_intervals, p , ndim) eg (15,3,2)
        Vmv_intervals = np.array([minvo2inv@Matrix2(j,knots[1:-1])@Vbs[j:j+3] for j in range(num_intervals)])
        # Vmv = Vmv_intervals.flatten() #shape = (num_intervals*p*ndim)

        # Acceleration calculation (bspline and minvo control points are the same)
        Abs = (p-1) * (Vbs[1:] - Vbs[:-1]) / (knots[p+1:-2] - knots[2:-(p+1)])[:, None]

        # print(f"{np.concatenate([v_constraints, a_constraints]).shape=}")
        # Constraints for velocities
        v_constraints = np.vstack([vmax - Vmv_intervals, Vmv_intervals + vmax]).flatten() # shape = 2 * (num_intervals*p*ndim)

        # Constraints for accelerations
        a_constraints = np.vstack([amax - Abs, Abs + amax]).flatten() # shape = 2 * (n-1) * ndim

        # print(f"{np.concatenate([v_constraints, a_constraints])=}")
        # import sys
        # sys.exit()
        return np.concatenate([v_constraints, a_constraints])

    constraints.append({'type': 'ineq', 'fun': velocity_acceleration_constraints})

    # Initial guess for the optimization #from 3rd to third to last point, rest is fixed
    initial_guess = Q[3:-2].flatten()

    # Define the options with max iterations
    options = {'maxiter': 15}
    # options = {}

    # Perform the optimization
    result = minimize(objective, initial_guess, constraints=constraints, method='SLSQP', options=options)

    # Extract the optimized control points
    optimized_control_points = np.vstack([fixed_points, result.x.reshape(-1, ndim), np.tile(result.x.reshape(-1, ndim)[-1], (2, 1))])

    return optimized_control_points

# # Example usage
# # Define your variables here: Q, planes, goal, r, vmax, amax, knots, T_jerk, Matrix

# optimized_control_points = optimize_spline(Q, planes, goal, r, vmax, amax, knots)
# # print("Optimized Control Points:\n", optimized_control_points)


# print("Optimized Control Points:\n", optimized_control_points)

# # print(optimized_control_points-Q)

# Qopt = optimized_control_points

# #%%
# import matplotlib.pyplot as plt

# print(Q.shape, Qopt.shape)
# Qbs = np.array([[0,0], [1.2,1], [2,0], [3,1]])
# T = [[t**3, t**2,t,1] for t in np.linspace(0,Dt,10000)]
# spline = np.vstack([T@Matrix(j,knots)@Qopt[j:j+4] for j in range(num_intervals)])
# # spline = np.vstack([T@Matrix(j,knots)@Qbs[j:j+4] for j in range(1)])
# plt.plot(spline[:,0],spline[:,1])
# plt.scatter(Q[:,0], Q[:,1])
# plt.scatter(Qopt[:,0], Qopt[:,1])
# # plt.scatter(Qbs[:,0], Qbs[:,1])

# plt.show()