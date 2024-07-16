#%%
import numpy as np
from scipy.spatial import ConvexHull
from separate import find_separation_plane
import heapq
from scipy.interpolate import BSpline
from collections import namedtuple
from octopus_utils import remove_close_points, sample_velocity_control_points, collect_positions, collect_planes
from mader_optimisation import optimize_spline
# from optimisation.mader_optimise_nlopt import optimize_spline
from matplotlib.animation import FuncAnimation
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


v_obstacle = np.array([3.,0.2])*1. # m/s
#now obstacle intervals
building_verts = np.array([[-1,-1],[1,-1],[1,1],[0,2],[-2,1.5]])*1.
building_start = np.array([-3,-1])  
verts = building_verts + building_start
# verts = building_verts + np.array([10,0])  

C = []
num_obstacle_intervals = 10000
for i in range(num_obstacle_intervals):
    hull_verts = np.vstack((verts, verts+v_obstacle*Dt))
    hull = ConvexHull(hull_verts)
    C.append(hull_verts[hull.vertices])
    verts += v_obstacle*Dt

PVAinit = np.vstack((p0,v0,a0))
PVAfinal = np.vstack((pf,vf,af))


eps0 = 3  # bias factor between cost to come and cost to go
# eps1 is min distance ql can be to another q already added to Q
eps1 = 0.2 # 2*Dt**2/p * np.max(amax)/np.sqrt(num_samples) #see maths in notebook
eps2 = 0.2 #min distance ql can be to the goal

Point = namedtuple('Point', ['full_cost', 'cost_to_come', 'position', 'l', 'prev','plane'])

initial_cost_to_come = 0  # g for the initial point
initial_full_cost = initial_cost_to_come + eps0 * np.linalg.norm(goal-p0)

q0 = Point(full_cost = initial_full_cost, cost_to_come=0,position=p0,l=0,prev=None, plane=None)
q1 = Point(initial_full_cost, 0, q0.position +v0/3,1, q0, None)
q2 = Point(initial_full_cost, 0, a0/6 +2*q1.position - q0.position, 2, q1, None)


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


# np.random.seed(4)

num_paths = 0
discarded = np.empty((0,2))
accepted:list[Point] = []
best_path_idx = 0
start_time = time.perf_counter()
measured = 0.

    
while Q and qwerty<10000:
    qwerty+=1
    # get last velocity control point (needed for acceleration constraint)

    ql:Point = heapq.heappop(Q)
    l = ql.l
    # print(f"{n=}, {l=}, {ql.position=}")
    vprev = p*(ql.position - ql.prev.position)/(knots[l+p] - knots[l])

    M = sample_velocity_control_points(num_samples, knots,l,vprev,vmax,amax,p)

    
    Qbs = collect_positions(ql,depth=p+1) # last 4 control points

    ############# CONDITIONS 1-6 #######################
    conditions_passed = True
    # condition 1 (if l=2 it's q2, the first point we added, so automatically skip all condition checking and accept)
    if 2<l<=n-2:
        Mbs = Matrix3(l-3, knots)
        Qmv = minvo3inv@Mbs@Qbs
        n_and_d = find_separation_plane(C[l-3],Qmv)
        ql = ql._replace(plane=n_and_d)
        if n_and_d is None:
            conditions_passed = False

    #conditions 2 and 3
    if l == n-2:
        #check second to last interval
        q_n_minus1 = Point(ql.full_cost,ql.cost_to_come,ql.position,l+1,ql, None)
        Qbs = collect_positions(q_n_minus1,depth=p+1) # last three points and repeat the last one once
        Mbs = Matrix3(n-4, knots) 
        Qmv = minvo3inv@Mbs@Qbs
        n_and_d = find_separation_plane(C[n-4],Qmv)
        # Create a new named tuple with the modified attribute
        q_n_minus1 = q_n_minus1._replace(plane=n_and_d)
        if n_and_d is None: 
            conditions_passed = False


        if conditions_passed:
            # check last interval
            q_n = Point(ql.full_cost,ql.cost_to_come,ql.position,l+2,q_n_minus1, None)
            Qbs = collect_positions(q_n,depth=p+1)
            Mbs = Matrix3(n-3, knots)
            Qmv = minvo3inv@Mbs@Qbs
            n_and_d = find_separation_plane(C[n-3],Qmv)

            q_n = q_n._replace(plane=n_and_d)
            if n_and_d is None: 
                conditions_passed = False
            else:
                condition = True
            # else:
    # measured+=time.perf_counter()-m_t    

    # condition 4
    if np.linalg.norm(ql.position - p0)>r and conditions_passed: # condition 4
        conditions_passed = False 

    # condition 5 is handled by removing close points directly beforehand

    # condition 6. Unclear if I can have an empty M, need to check the maths
    if len(M) == 0 and conditions_passed: #condition 6
        print("HOW DID WE GET HERE")
        conditions_passed = False

    
    # since the first point added to Q is q2, we don't need to add it to q
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
    # if len(discarded)!=0:
    #     ql_next = remove_close_points(discarded, ql_next, eps1)
    # if len(accepted)!=0:
    #     acc = np.vstack([q.position for q in accepted])
    #     ql_next = remove_close_points(acc, ql_next, eps1)


    new_g_costs = ql.cost_to_come + np.linalg.norm(ql_next - ql.position, axis=1)
    h_costs = np.linalg.norm(ql_next - goal, axis=1)
    f_costs = new_g_costs + eps0 * h_costs
    # Push all new points and their costs into the priority queue
    for new_g_cost, f_cost, ql_n in zip(new_g_costs, f_costs, ql_next):
        heapq.heappush(Q, Point(f_cost,new_g_cost,ql_n,l+1,ql,None))


total_t = time.perf_counter()-start_time

print(f"{total_t=}, {measured=}")


print('###########END########\n\n')


q_n  = min(accepted, key=lambda q: q.full_cost-q.cost_to_come)
accepted_full = [q for q in accepted if q.l==n]
print(f"{len(accepted_full)=}")
if accepted_full:
    q_n  = min(accepted_full, key=lambda q: np.linalg.norm(q.position-goal))
else:
    print("Failed to find a path with all control points. Exitting")
    sys.exit()

control_points = collect_positions(q_n, depth=q_n.l+1)
planes = collect_planes(q_n, depth=q_n.l-p+1)
print(f"{len(control_points)=}", f"{n=}, {q_n.l=}")
print(f"{num_intervals=}")

control_points = optimize_spline(control_points, planes, goal, r, vmax, amax, knots)
total_t = time.perf_counter()-start_time
print(f"{total_t=}")


# np.save('Q.npy', control_points)
# np.save('planes.npy', planes)
n = len(control_points)-1
# m+1 = n+p+2 but if fewer control points we need to truncate the knots
# we need to truncate to n+p+2-(n+1-len(control_points)) = p+1+len(control_points)
knots = knots[:n+p+2]
# print(planes)
# n = len(control_points)-1
num_intervals = n-p+1
# print(f"{num_intervals=}, {best_path_idx=}")

# print(f"{len(control_points)=}")
# print(f"{knots=}")



# print(f"{control_points=}, {len(control_points)=}")
# print(f"{knots=}")
print(f"{l=}, {len(Q)=}")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 6))

# control_points = np.vstack(control_points)
# n = len(control_points)-1

# np.save('cp.npy',control_points)
# print(control_points, len(control_points))
# print(np.linspace(0,1,len(control_points)-1))
# knots = np.hstack(([0]*p, np.linspace(0,1,2+n-p), [1]*p))
# knots = np.hstack(([0]*p, np.linspace(0,1,len(control_points)),[1]))  #uncomment for truncated spline

# print(f"{knots=}")
# knots = knots if l>=n-2 else knots[:l+p+1]

acc = np.vstack([q.position for q in accepted if q.l==n])

plt.scatter(acc[:,0], acc[:,1], color='g')
plt.scatter(discarded[:,0], discarded[:,1], color='r')
plt.plot(control_points[:,0], control_points[:,1], alpha = 0.4, lw = 2, marker='s')
# # Draw arrow from A to B
# plt.annotate('', xy=goal, xytext=p0,
#             arrowprops=dict(arrowstyle='->', color='blue'))

# plot all paths
# for cps in paths:
#     if len(cps)>=4:
#         n = len(cps)-1
#         kts = np.hstack(([0]*p, np.linspace(0,1,2+n-p), [1]*p))
#         splne = BSpline(kts ,cps, 3)
#         t = np.linspace(0,1, 1000)
#         f = np.vstack([splne(x) for x in t])
#         # plt.plot(f[:,0],f[:,1],alpha = 0.5, lw = 1, marker='None')#, color ='gray')
#         plt.plot(cps[:,0], cps[:,1], alpha = 0.5, lw = 1, marker='None', color ='gray')


plt.scatter(*goal)
plt.scatter(*p0)

# print(f"{ql=}")
# plt.scatter(*ql, color = 'c')
if Q:
    # print("Q is not empty")
    q = np.vstack([k for _,_,k,_,_,_ in Q])
    plt.scatter(q[:,0],q[:,1])
    # Extract costs and points
    costs = np.array([item[0] for item in Q])
    points = np.array([item[2] for item in Q])
    # print(points)

    # Normalize the costs to get a relative scale for color-coding
    norm = plt.Normalize(costs.min(), costs.max())
    colors = plt.cm.viridis(norm(costs))
    # Create the plot with an explicit Axes
    sc = ax.scatter(points[:, 0], points[:, 1], c=colors, s=100, cmap='viridis', edgecolor='k')
    plt.colorbar(sc, ax=ax, label='Cost')

# M = M*Dt/p
ax.scatter(M[:, 0], M[:, 1])
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




#building plot
for idx, c in enumerate(C[:n]):
    polygon = plt.Polygon(c, facecolor='none' if idx!=l-3 else 'c', alpha = 1 if idx!=l-3 else 0.3, edgecolor='black')
    ax.add_patch(polygon)

ax.set_aspect('equal', adjustable='box')
ax.axis([-10, 10, -10, 10])
ax.minorticks_on()
# Add grid lines
ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')


plt.show()










########################################################################################################################################
# dynamic plot

if len(knots)>=8:
    spline = BSpline(knots ,control_points, p)
    # print(knots[:l+p+1])
    t = np.linspace(0,num_intervals*Dt, 1000)
    # print(spline(0.1).shape)
    f = np.vstack([spline(x) for x in t])
    # plt.plot(f[:,0],f[:,1])

# Define figure and axis
fig, ax = plt.subplots()
plt.plot(control_points[:,0], control_points[:,1], alpha = 0.4, lw = 2, marker='s')
# planned_start = p0+np.array([0,-3])
# planned_end = goal+np.array([0,3])

plt.plot([p0[0], goal[0]],[p0[1], goal[1]], alpha = 0.7, lw = 2, marker=None,color='k', linestyle='--',label='Planned Path')
# plt.annotate('', xy=goal+np.array([0,2]), xytext=p0+np.array([0,-3]),
#             arrowprops=dict(arrowstyle='->', color='k',lw=3, linestyle='--'))
# old_cp = np.load('cp.npy')
# plt.plot(old_cp[:,0], old_cp[:,1], alpha = 0.4, lw = 2, marker='s')

# Number of frames in the animation
num_frames = len(C) #any number will do, this doesn't matter

# Plot the spline curve
# T = np.linspace(0, (len(C) - 1) * Dt, 100)
T = np.linspace(0, num_intervals*Dt, 1000)

spline_points = np.array([spline(ti) for ti in T])
spline_line, = ax.plot(spline_points[:, 0], spline_points[:, 1], 'b-', label='Alternative Path')

# Plot the initial point on the spline
spline_point, = ax.plot([], [], 'ro', label='Drone Position')

# Plot the initial obstacle polygon
obstacle = plt.Polygon(building_verts+building_start, closed=True, color='g')
polygon = plt.Polygon(C[0], closed=True, color='k', alpha=0.3)

ax.add_patch(obstacle)
ax.add_patch(polygon)


ax.legend()
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.minorticks_on()
# Add grid lines
ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')

plt.scatter(*goal)
plt.scatter(*p0)

# plot planes
# Define the range for x values
x_vals = np.linspace(-10, 10, 400)

plane_plots = []
# Plot each plane as a line
for nx, ny, d in planes:
    # Calculate y values from the plane equation nx*x + ny*y = d
    y_vals = -(nx * x_vals + d) / (ny+1e-6)

    # y_vals = (d - nx * x_vals) / (ny+1e-6)
    pl, = ax.plot(x_vals, y_vals, label=f'{nx}x + {ny}y = {d}', linestyle='-', color='r', visible=False)
    plane_plots.append(pl)

start_time = time.time()
# Animation function

# print(f"{knots[-1]*n*Dt=}")
# print(len(C), len(plane_plots))

def animate(frame):
    global start_time
    elapsed_time = time.time() - start_time
    # elapsed_time /= 1
    # print(f"{knots[-(p+1)]*n_intervals*Dt=}")
    if elapsed_time>knots[-(p+1)]:
        start_time = time.time()
        elapsed_time = time.time() - start_time

    # t = frame * Dt
    spline_time = elapsed_time#/(n_intervals*Dt)
    # print(spline_time)
    point = spline(spline_time) 
    
    # Update the spline point
    spline_point.set_data([point[0]], [point[1]])
    
    building_idx = int(elapsed_time//Dt)
    # print(elapsed_time, building_idx, f"{knots[-1]*n_intervals*Dt=}")
    # print(building_idx)
    # Update the obstacle polygon
    polygon.set_xy(C[building_idx])
    obstacle.set_xy(building_verts+building_start+v_obstacle*elapsed_time)
    # print(f"{obstacle.get_verts()=}")
    # print(f"{building_verts+building_start+v_obstacle*elapsed_time}")
    # print(f"{building_idx-1=}, {len(plane_plots)=}")
    # plane_plots[building_idx-1].set_visible(False)
    # plane_plots[building_idx].set_visible(True)
    try:
        plane_plots[building_idx-1].set_visible(False)
        plane_plots[building_idx].set_visible(True)
    except Exception:
        pass
    
    ax.set_title(f"Time: {elapsed_time:.2f}")

# Create animation
ani = FuncAnimation(fig, animate, frames=num_frames, interval=20)

plt.show()

#%%

