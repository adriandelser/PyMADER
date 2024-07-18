#%%
import numpy as np
from scipy.spatial import ConvexHull
from separate import find_separation_plane
import heapq
from scipy.interpolate import BSpline
# from collections import namedtuple
from octopus_utils import remove_close_points, sample_velocity_control_points, collect_positions, collect_planes,get_velocity_control_points
from mader_optimisation import optimize_spline
# from optimisation.mader_optimise_nlopt import optimize_spline
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
import time
from matrices import Matrix3
import sys




#modified minvo to work with t from 0 to 1 like most other splines
minvo3 = np.array([
    [-3.4416, 6.9896, -4.46236, 0.91439], 
    [6.6792, -11.846, 5.2524, 0.0001],
    [-6.6792, 8.1916, -1.598, 0.0856],
    [3.4416, -3.3352, 0.80796, -0.0007903]
]).T

minvo3inv = np.linalg.inv(minvo3)


goal = np.array([0.,8.])
# initial and final conditions
p0, v0, a0 = np.array([0,-5]), np.array([0,2]), np.array([0,0])
pf, vf, af = goal, np.array([0,0]), np.array([0,0])

r = np.linalg.norm(p0-goal).astype(np.float64)# max radius that we are planning to (see MADER paper Table 1)

vmax = np.array([5., 5.]) # 10m/s in x y and z (more if combined)
amax = np.array([10., 10.]) #approx 0.5g in x y and z (more if combined)
tin = 0.0
tf = tin + np.linalg.norm(goal-p0)/np.max(vmax) *1.8
print(f"{tin=}, {tf=}")

p = 3 # cubic spline
# define how many intervals we want 
num_intervals = 15
Dt = (tf-tin)/num_intervals #time allocated per interval t_p+j+1-t_p+j
n = num_intervals+p-1
print(f"{n=}")
m = n+p+1
knots = np.hstack(([tin]*p, np.linspace(tin, tf,2+m-2*p-1),[tf]*p)).astype(np.float64)

print(f"{Dt=}")

##################################################################################################
# OBSTACLES

v_obstacle0 = np.array([3.,0.2])*1. # m/s
#now obstacle intervals
building_verts0 = np.array([[-1,-1],[1,-1],[1,1],[0,2],[-2,1.5]])*1.
building_start0 = np.array([-3,-1])  
verts = building_verts0 + building_start0

C0 = []
num_obstacle_intervals = 10000
for i in range(num_obstacle_intervals):
    hull_verts = np.vstack((verts, verts+v_obstacle0*Dt))
    hull = ConvexHull(hull_verts)
    C0.append(hull_verts[hull.vertices])
    verts += v_obstacle0*Dt

v_obstacle1 = np.array([-3.,0.])*.7 # m/s
#now obstacle intervals
building_verts1 = np.array([[-1,-1],[1,-1],[1,1],[0,2],[-2,1.5]])*0.6
building_start1 = np.array([7,3])  
verts = building_verts1 + building_start1

C1= []
for i in range(num_obstacle_intervals):
    hull_verts = np.vstack((verts, verts+v_obstacle1*Dt))
    hull = ConvexHull(hull_verts)
    C1.append(hull_verts[hull.vertices])
    verts += v_obstacle1*Dt

obstacles = [C0, C1]
##################################################################################################


PVAinit = np.vstack((p0,v0,a0))
PVAfinal = np.vstack((pf,vf,af))


eps0 = 3  # bias factor between cost to come and cost to go
# eps1 is min distance ql can be to another q already added to Q
eps1 = 0.2 
eps2 = 0.5 #min distance ql can be to the goal


class Point:
    def __init__(self, full_cost,cost_to_come,position,l,prev,planes:list) -> None:
        self.full_cost = full_cost
        self.cost_to_come = cost_to_come
        self.position = position
        self.l = l
        self.prev = prev
        self.planes = planes

    def __lt__(self, other):
        return self.full_cost < other.full_cost
    
    def __repr__(self) -> str:
        return f"{self.l=}, {self.full_cost=}, {self.position=}, {self.planes=}"

# Point = namedtuple('Point', ['full_cost', 'cost_to_come', 'position', 'l', 'prev','planes'])

initial_cost_to_come = 0  # g for the initial point
initial_full_cost = initial_cost_to_come + eps0 * np.linalg.norm(goal-p0)

q0 = Point(full_cost = initial_full_cost, cost_to_come=0,position=p0,l=0,prev=None, planes=[])
q1 = Point(initial_full_cost, 0, q0.position +v0/3,1, q0, [])
q2 = Point(initial_full_cost, 0, a0/6 +2*q1.position - q0.position, 2, q1, [])


discarded = np.empty((0,2))


num_samples = 5


# Create a priority queue
Q = []
# Example: Add the initial point
heapq.heappush(Q, q2)


print(f'{r=}')
qwerty=1
paths:list[np.ndarray] = []
list_of_planes:list[np.ndarray] = []
closest_distance_to_goal = r+1 #shortest path (set to anything slightly greater than r)
we_out = False


np.random.seed(10)

num_paths = 0
discarded = np.empty((0,2))
accepted:list[Point] = []
best_path_idx = 0
start_time = time.perf_counter()
measured = 0.

    
while Q and qwerty<10000:
    qwerty+=1

    ql:Point = heapq.heappop(Q)
    l = ql.l
    # get last velocity control point (needed for acceleration constraint)
    vprev = p*(ql.position - ql.prev.position)/(knots[l+p] - knots[l])

    M = sample_velocity_control_points(num_samples, knots,l,vprev,vmax,amax,p)

    
    Qbs = collect_positions(ql,depth=p+1)
    
    ############# CONDITIONS 1-6 #######################
    conditions_passed = True
    # condition 1
    if 2<l<=n-2:
        Mbs = Matrix3(l-3, knots)
        Qmv = minvo3inv@Mbs@Qbs
        for i, C in enumerate(obstacles):
            n_and_d = find_separation_plane(C[l-3],Qmv)
            if n_and_d is None:
                conditions_passed = False
                break
            else:
                ql.planes.append(n_and_d)


    #conditions 2 and 3
    if l == n-2:
        #check second to last interval
        q_n_minus1 = Point(ql.full_cost,ql.cost_to_come,ql.position,l+1,ql, [])
        Qbs = collect_positions(q_n_minus1,depth=p+1) # last three points and repeat the last one once
        Mbs = Matrix3(n-4, knots) 
        Qmv = minvo3inv@Mbs@Qbs

        for i, C in enumerate(obstacles):
            n_and_d = find_separation_plane(C[n-4],Qmv)
            if n_and_d is None:
                conditions_passed = False
                break
            else:
                q_n_minus1.planes.append(n_and_d)

        # n_and_d = find_separation_plane(C0[n-4],Qmv)
        # # Create a new named tuple with the modified attribute
        # q_n_minus1 = q_n_minus1._replace(plane=n_and_d)
        # if n_and_d is None: 
        #     conditions_passed = False

        if conditions_passed:
            # check last interval
            q_n = Point(ql.full_cost,ql.cost_to_come,ql.position,l+2,q_n_minus1, [])
            Qbs = collect_positions(q_n,depth=p+1)
            Mbs = Matrix3(n-3, knots)
            Qmv = minvo3inv@Mbs@Qbs

            for i, C in enumerate(obstacles):
                n_and_d = find_separation_plane(C[n-3],Qmv)
                if n_and_d is None:
                    conditions_passed = False
                    break
                else:
                    q_n.planes.append(n_and_d)

            # n_and_d = find_separation_plane(C0[n-3],Qmv)
            # q_n = q_n._replace(plane=n_and_d)
            # if n_and_d is None: 
            #     conditions_passed = False

    # condition 4
    if np.linalg.norm(ql.position - p0)>r and conditions_passed: # condition 4
        conditions_passed = False 

    # condition 5 is handled by removing close points directly beforehand

    # condition 6. Unclear if I can have an empty M, need to check the maths
    if len(M) == 0 and conditions_passed: #condition 6
        print("HOW DID WE GET HERE")
        conditions_passed = False

    if not conditions_passed:
        discarded = np.vstack((discarded, ql.position))
        continue
   
    if l==n-2:
        accepted.extend([ql, q_n_minus1, q_n])
        if np.linalg.norm(ql.position-goal)<eps2:
            print("DONEEE")
            break
        else:
            print("NOT REACHED GOAL, CONTINUING")
            continue
    elif l<n-2:
        # if not isinstance(ql, Point):
        #     print(ql, "\n")
        accepted.append(ql)

    # conditions have passed so we have qn and qn minus 1

    ############# GET NEW qls from vls #######################

    # Generate new points and calculate their costs
    # note ql has now been added to the control points


    ql_next = ql.position + (knots[l+p+1]-knots[l+1]) / p * M

    m_t = time.perf_counter()
    if len(Q)!=0:
        Q_array = np.vstack([point.position for point in Q]) #this line is a bit slow
        measured+=time.perf_counter()-m_t    
        ql_next = remove_close_points(Q_array, ql_next, eps1) # this line is also slow
   
    new_g_costs = ql.cost_to_come + np.linalg.norm(ql_next - ql.position, axis=1)
    h_costs = np.linalg.norm(ql_next - goal, axis=1)
    f_costs = new_g_costs + eps0 * h_costs
    # Push all new points and their costs into the priority queue
    
    for new_g_cost, f_cost, ql_n in zip(new_g_costs, f_costs, ql_next):
        heapq.heappush(Q, Point(f_cost,new_g_cost,ql_n,l+1,ql,[]))
    

    # if l == 3 and len(control_points)==l+1:
    #     break
#     if l == 3 and len(paths)==1:
#         we_out = True
#         break
# if we_out:
#     break



total_t = time.perf_counter()-start_time

# ########################################################################################################################################
print(f"{total_t=}, {measured=}")


print('###########END########\n\n')


q_n  = min(accepted, key=lambda q: q.full_cost-q.cost_to_come)
accepted_qn = [q for q in accepted if q.l==n]
print(f"{len(accepted_qn)=}")
if accepted_qn:
    q_n  = min(accepted_qn, key=lambda q: np.linalg.norm(q.position-goal))
else:
    print("Failed to find a path with all control points. Exitting")
    sys.exit()




control_points_pre_opt = collect_positions(q_n, depth=q_n.l+1)
planes = collect_planes(q_n, depth=q_n.l-p+1)



# also collect all other paths with n+1 control points:
other_control_points = []
for q in accepted_qn:
    other_control_points.append(collect_positions(q, depth=q_n.l+1))




# print(f"{planes=}")
print(f"{planes.shape=}")
print(f"{len(control_points_pre_opt)=}", f"{n=}, {q_n.l=}")
print(f"{num_intervals=}")

control_points = optimize_spline(control_points_pre_opt, planes, goal, r, vmax, amax, knots)
velocity_points = get_velocity_control_points(control_points, knots, p)

# make the splines
pos_spline = BSpline(knots ,control_points, p)
vel_spline = BSpline(knots[1:-1], velocity_points, p-1)

total_t = time.perf_counter()-start_time
print(f"{total_t=}")


print(f"{l=}, {len(Q)=}")





# Create a figure with 2 rows and 3 columns of subplots
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 8), gridspec_kw={'width_ratios': [1, 1, 1], 'wspace': 0.2})

# Adjust the space between the subplots for better visualization
# plt.subplots_adjust(hspace=0.3, wspace=0.3)


acc = np.vstack([q.position for q in accepted if q.l==n])

ax[0].scatter(acc[:,0], acc[:,1], color='g')
ax[0].scatter(discarded[:,0], discarded[:,1], color='r')

for cps in other_control_points:
    ax[1].plot(cps[:,0], cps[:,1], alpha = 0.4, lw = 1, marker='', color='gray')

ax[1].plot(control_points_pre_opt[:,0], control_points_pre_opt[:,1], alpha = 0.4, lw = 2, marker='s')


t = np.linspace(0, knots[-1], 1000)
positions = pos_spline(t)

velocity = np.linalg.norm(vel_spline(t), axis=1)
# Create line segments for coloring
pts = positions.reshape(-1, 1, 2)

segments = np.concatenate([pts[:-1], pts[1:]], axis=1)

# Normalize velocity for color mapping
norm = Normalize(vmin=velocity.min(), vmax=velocity.max())
lc = LineCollection(segments, cmap='jet', norm=norm)
lc.set_array(velocity)
lc.set_linewidth(4)

ax[2].add_collection(lc)

cbar = fig.colorbar(lc, ax=ax[2], orientation='horizontal', fraction=0.05, pad=0.1)
cbar.set_label('Velocity')






# Add a scatter plot point to each subplot
for i, axis in enumerate(ax[:2].flat):
    # axis.set_title(f'Subplot {i+1}')
    axis.scatter(*goal, color='red', label='Goal')
    axis.scatter(*p0, color='blue', label='Start')

    axis.set_aspect('equal', adjustable='box')
    axis.axis([-10, 10, -10, 10])
    axis.minorticks_on()
    # Add grid lines
    axis.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    axis.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')


    #building plot
    for C in obstacles:
        for idx, c in enumerate(C[:n]):
            polygon = plt.Polygon(c, facecolor='none' if idx!=l-3 else 'c', alpha = 1 if idx!=l-3 else 0.3, edgecolor='black')
            axis.add_patch(polygon)

    axis.legend()

# plt.scatter(*goal)
# plt.scatter(*p0)

# print(f"{ql=}")
# plt.scatter(*ql, color = 'c')
if Q:
    # print("Q is not empty")
    q = np.vstack([k.position for k in Q])
    ax[0].scatter(q[:,0],q[:,1])
    # Extract costs and points
    costs = np.array([item.full_cost for item in Q])
    points = np.array([item.position for item in Q])
    # print(points)

    # Normalize the costs to get a relative scale for color-coding
    norm = plt.Normalize(costs.min(), costs.max())
    colors = plt.cm.viridis(norm(costs))
    # Create the plot with an explicit Axes
    sc = ax[0].scatter(points[:, 0], points[:, 1], c=colors, s=100, cmap='viridis', edgecolor='k')
    cbar = fig.colorbar(sc, ax=ax[0], orientation='horizontal', fraction=0.05, pad=0.1)
    cbar.set_label('Relative Cost')


# M = M*Dt/p
# ax.scatter(M[:, 0], M[:, 1])
# print(f"{M.shape=}")
# print(f"{len(Q)=}")
# print(f"{p/Dt=}")
# point_samples = control_points[-1]+ Dt/p*M
# ax.scatter(point_samples[:, 0], point_samples[:, 1])
# ax.scatter(*control_points[-1])
# print(f"{poly_Q=}")
# ax.add_patch(plt.Polygon(POLYQ.exterior.coords, facecolor = 'none', edgecolor='red'))
# hull_bs = ConvexHull(QBS,qhull_options='QJ')
# poly_Qbs = QBS[hull_bs.vertices]
# ax.add_patch(plt.Polygon(poly_Qbs,facecolor = 'none', edgecolor='blue'))
# plt.scatter(Qmv[:,0], Qmv[:,1])
# plt.scatter(Qbs[:,0], Qbs[:,1])









# plt.show()










########################################################################################################################################
# dynamic plot

# Define figure and axis
# ax[2].scatter(control_points[:,0], control_points[:,1], alpha = 1, marker='s', color='k')


ax[2].plot([p0[0], goal[0]],[p0[1], goal[1]], alpha = 0.7, lw = 2, marker=None,color='k', linestyle='--',label='Planned Path')

# Number of frames in the animation
num_frames = len(C0) #any number will do, this doesn't matter

# Plot the spline curve
T = np.linspace(0, num_intervals*Dt, 1000)

spline_points = np.array([pos_spline(ti) for ti in T])

# Plot the initial point on the spline
drone_pos, = ax[2].plot([], [], 'ro', label='Drone Position')

# Plot the initial obstacle polygon
obstacle0 = plt.Polygon(building_verts0+building_start0, closed=True, color='g')
obstacle1 = plt.Polygon(building_verts1+building_start1, closed=True, color='g')

polygon0 = plt.Polygon(C0[0], closed=True, color='k', alpha=0.3)
polygon1 = plt.Polygon(C1[0], closed=True, color='k', alpha=0.3)

for obstacle in [obstacle0,obstacle1]:
    ax[2].add_patch(obstacle)
for polygon in [polygon0,polygon1]:
    ax[2].add_patch(polygon)


# ax[2].legend()
ax[2].set_xlim(-10, 10)
ax[2].set_ylim(-10, 10)
ax[2].minorticks_on()
# Add grid lines
ax[2].grid(which='major', linestyle='-', linewidth='0.5', color='black')
ax[2].grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
ax[2].set_aspect('equal')
# ax[2].scatter(*goal)
# ax[2].scatter(*p0)

# plot planes
# Define the range for x values
x_vals = np.linspace(-10, 10, 400)

plane_plots = [[],[]]
# Plot each plane as a line
for idx, plane_list in enumerate(planes):
    for nx, ny, d in plane_list:
        # Calculate y values from the plane equation nx*x + ny*y = d
        y_vals = -(nx * x_vals + d) / (ny+1e-6)

        # y_vals = (d - nx * x_vals) / (ny+1e-6)
        pl, = ax[2].plot(x_vals, y_vals, label=f'{nx}x + {ny}y = {d}', linestyle='-', color='r', visible=False)
        plane_plots[idx].append(pl)


time_dilation = 1
start_time = time.time()/time_dilation
# Animation function


def animate(frame):
    global start_time
    elapsed_time = time.time() - start_time
    # elapsed_time /= time_dilation
    if elapsed_time>knots[-(p+1)]:
        start_time = time.time()
        elapsed_time = time.time() - start_time

    # t = frame * Dt
    spline_time = elapsed_time#/(n_intervals*Dt)
    # print(spline_time)
    point = pos_spline(spline_time) 
    
    # Update the spline point
    drone_pos.set_data([point[0]], [point[1]])
    
    building_idx = int(elapsed_time//Dt)

    # Update the obstacle polygon
    polygon0.set_xy(C0[building_idx])
    polygon1.set_xy(C1[building_idx])

    obstacle0.set_xy(building_verts0+building_start0+v_obstacle0*elapsed_time)
    obstacle1.set_xy(building_verts1+building_start1+v_obstacle1*elapsed_time)

    try:
        for i in range(2):
            plane_plots[i][building_idx-1].set_visible(False)
            plane_plots[i][building_idx].set_visible(True)
    except Exception as e:
        print(e)
        pass
    
    ax[2].set_title(f"Time: {elapsed_time:.2f}")

# Create animation
ani = FuncAnimation(fig, animate, frames=num_frames, interval=20)


plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.25)

plt.show()

#%%

