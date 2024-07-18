#%%
import numpy as np

def get_velocity_control_points(positional_points, knots, p):
    """
    Computes the velocity control points for a B-spline.

    :param positional_points: List or array of positional control points q.
    :param delta_t: time to traverse one segment of the uniform spline
    :param p: Degree of the B-spline.
    :return: Array of velocity control points v.
    """
    positional_points = np.array(positional_points)    
    # n = len(pos_points) - 1
    # m = n+p+1
    # Calculate differences between consecutive positional control points
    delta_q = positional_points[1:] - positional_points[:-1]
    # len(delta_q) = n = m-(p+1)
    delta_t = knots[p+1:-1] - knots[1:-(p+1)]
    # len(delta_t) = m+1 - p - 2 = m-(p+1)
    # Compute the velocity control points
    velocity_points = p * (delta_q / delta_t[:, None])
    # print(f'{velocity_points.shape=}')
    return velocity_points

def get_acceleration_control_points(velocity_points, knots, p):
    """
    Computes the acceleration control points for a B-spline.

    :param velocity_points: List or array of velocity control points v.
    :param delta_t: time to traverse one segment of the uniform spline
    :param p: Degree of the B-spline.
    :return: Array of acceleration control points a.
    """
    # velocity_points = np.array(velocity_points)    
    # Calculate differences between consecutive positional control points
    delta_v = velocity_points[1:] - velocity_points[:-1]
    #len(delta_v) = n-1 = m-p-2
    delta_t = knots[p+1:-2] - knots[2:-(p+1)]
    # len(delta_t) = m-p-2
    # Compute the acceleration control points
    acceleration_points = (p-1) * delta_v / delta_t[:, None]
    return acceleration_points

def voxel_based_sampling_vectorized(vmin, vmax, num_samples, eps_v):
    # Calculate the number of voxels along each dimension
    num_voxels = np.ceil((vmax-vmin) / eps_v).astype(int)
    # print(f"{num_voxels=} {eps_v=}")
    
    # Calculate the total number of voxels
    total_voxels = np.prod(num_voxels)

    # Generate an initial batch of candidate samples
    initial_samples = np.random.uniform(vmin, vmax, (total_voxels, len(vmax)))

    # Compute the voxel indices for each candidate sample. first ensure all are above [0,0] by subtracting vmin
    voxel_indices = ((initial_samples - vmin) // eps_v).astype(int)
    # print(f"{voxel_indices.shape=}")

    # Remove duplicates by finding unique voxel indices
    _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)

    # Select the unique samples
    unique_samples = initial_samples[unique_indices]
    # unique_samples = initial_samples[:]

    # print(f"{len(unique_indices)=}")
    # print(f"{unique_samples.shape=}")
    # # If we have more unique samples than needed, select a random subset
    if len(unique_samples) > num_samples:
        selected_indices = np.random.choice(len(unique_samples), num_samples, replace=False)
        final_samples = unique_samples[selected_indices]
    else:
        final_samples = unique_samples

    return final_samples
    # return unique_samples



def uniformly_sample_velocity_with_acceleration_check(vmax, amax, knots,l, num_samples, vprev, p):
    """
    Uniformly samples velocities satisfying vmax and amax constraints in each direction (x, y, z).
    REMEMBER epsilon is in the velocity space!!
    :param vmax: Array of maximum velocities for each direction [vmax_x, vmax_y, vmax_z].
    :param amax: Array of maximum accelerations for each direction [amax_x, amax_y, amax_z].
    :param num_samples: Number of velocity samples to generate.
    :param epsilon: minimum L_infinity distance between points
    :param previous_velocity: Previous velocity control point [vx, vy, vz].
    :param delta_t: Time to traverse one segment of the uniform spline.
    :param p: Degree of the B-spline.
    :return: Array of sampled velocity control points that satisfy the acceleration constraint.
    """
    # Ensure vmax and amax are numpy arrays
    # vmax = np.array(vmax)
    # amax = np.array(amax)
    # previous_velocity = np.array(previous_velocity)

    # eps_v = epsilon*p/(knots[l+p+1] - knots[l+1])

    delta_v_max = (knots[l+p+1] - knots[l+2])/(p-1) * amax# change in velocity that is limited by amax #NOTE why 2*
    # Compute the range based on dv constraint
    vnext_min_dv = vprev - delta_v_max
    vnext_max_dv = vprev + delta_v_max

    # Apply the vmax constraint
    vnext_min = np.maximum(-vmax, vnext_min_dv)
    vnext_max = np.minimum(vmax, vnext_max_dv)
    smallest_range = np.min(vnext_max-vnext_min)
    eps_v = smallest_range/np.sqrt(num_samples) # this value of eps_v will make a grid with at least num_samples voxels

    # vmax = np.minimum(vmax,vm) # pick the lower of the two element wise.
    # print(f"{vmax=} {vnext_min=} {vnext_max=}")
    # print(f"{previous_velocity=}")

    # Calculate the total number of samples to generate to ensure we have enough valid ones
    # oversample_factor = 1  # Adjust this factor as needed to ensure enough valid samples
    # total_samples = num_samples # * oversample_factor

    # Sample velocities for each dimension using NumPy's vectorized operations
    # print(vnext_min, vnext_max, num_samples, eps_v)
    velocities = voxel_based_sampling_vectorized(vnext_min, vnext_max, num_samples, eps_v)

    return velocities


def remove_close_points(Q, new_points, epsilon):
    # Compute the L_infinity distance matrix between all points in P and all points in Q
    # This results in a matrix of shape (m, n) where m is the number of points in P and n is the number of points in Q
    distance_matrix = np.max(np.abs(new_points[:, np.newaxis, :] - Q), axis=2)
    
    # Find the minimum distance for each point in P to any point in Q
    min_distances = np.min(distance_matrix, axis=1)
    
    # Select the points in P where the minimum distance is greater than or equal to eps1
    valid_points = new_points[min_distances >= epsilon]

    
    return valid_points


def sample_velocity_control_points(num_samples, knots, l, v_l_minus_1, vmax, amax, p=3):
    """
    Sample a new velocity control point satisfying the given constraints.
    
    Parameters:
        v_l_minus_1 (numpy.ndarray): Previous velocity control point of shape (1, n_dim).
        vmax (numpy.ndarray): Maximum velocity for each dimension of shape (n_dim,).
        Dt (float): Time interval.
        a_max (float): Maximum allowable acceleration.
        n_dim (int): Number of dimensions.
        p (int): Degree of the spline (default is 3).
    
    Returns:
        numpy.ndarray: New velocity control point of shape (1, n_dim).
    """
    ndim = vmax.shape[0]
    dt = knots[l+p+1] - knots[l+2]
    # print(f"{dt=}")
    # Compute the maximum velocity allowed by the acceleration constraint for each dimension
    max_v_allowed_by_acc = v_l_minus_1 + (amax * dt) / (p - 1)
    min_v_allowed_by_acc = v_l_minus_1 - (amax * dt) / (p - 1)
    # print(f"{v_l_minus_1=}")
    # print(f"{max_v_allowed_by_acc=}, {min_v_allowed_by_acc=}")
    # The actual maximum velocity is the minimum of vmax and max_v_allowed_by_acc
    actual_vmax = np.minimum(vmax, max_v_allowed_by_acc)
    actual_vmin = np.maximum(-vmax, min_v_allowed_by_acc)  # Assuming velocity cannot be negative

    # print(f"{actual_vmax=}, {actual_vmin=}")
    
    # Sample uniformly within the valid range
    v_l = np.random.uniform(actual_vmin, actual_vmax, (num_samples, ndim))
    
    return v_l

def collect_positions(point, depth:int):
    positions = []
    current = point
    while depth > 0 and current is not None:
        positions.append(current.position)
        current = current.prev
        depth -= 1
    return np.vstack(positions)[::-1]

def collect_planes(point, depth: int):
    planes = []
    current = point
    while depth > 0 and current is not None:
        planes.append(np.array(current.planes))
        current = current.prev
        depth -= 1

    # Stack the planes along the second axis to have shape (num_obstacles, num_intervals, ndim+1)
    stacked_planes = np.stack(planes, axis=1)

    # Reverse the order to match the original desired outcome
    return stacked_planes[:,::-1,:]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    vmax = np.array([10,10])
    amax = np.array([10,10])
    vprev = np.array([10,10])
    Dt = 1.

    # eps1 = 0.000833
    p=3
    knots = np.hstack(([0]*(p+1), np.linspace(Dt,p*Dt,p)))
    num_samples = 25
    eps1 = 2*Dt**2/p * np.max(amax)/np.sqrt(num_samples)
    # print(f"{2*Dt**2/p * np.max(amax)/20=}")
    # print(f"{vmax*Dt+0.5*amax*Dt**2=}")
    # a = np.minimum([2,3],[1,5])
    # print(a)
    # vm = 2*Dt/(p-1) *amax + vprev
    # print(vm)
    v = uniformly_sample_velocity_with_acceleration_check(vmax,amax,knots, 3-1, num_samples,eps1,vprev,p)
    print(v.shape)
    plt.scatter(v[:,0], v[:,1])
    plt.show()
# %%
