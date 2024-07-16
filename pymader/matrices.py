#%%
import numpy as np
import matplotlib.pyplot as plt

t=np.array([0,0,0,0,1,2,3,4,5,6,7,8,9,9,9,9])

def safe_divide(numerator, denominator):
    """Safely divide two numbers, returning 0 if both are zero."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.true_divide(numerator, denominator)
        result = np.where((numerator == 0) & (denominator == 0), 0, result)
        result = np.where(np.isnan(result), 0, result)
    return result

def Matrix3(interval: int, t: np.ndarray):
    # interval j starts at 0, i starts at 3
    i = interval+3
    #i should be at least 2
    M = np.zeros((4, 4))
    
    M[0, 0] = safe_divide((t[i+1] - t[i])**2, (t[i+1] - t[i-1]) * (t[i+1] - t[i-2]))
    M[0, 2] = safe_divide((t[i] - t[i-1])**2, (t[i+2] - t[i-1]) * (t[i+1] - t[i-1]))
    M[1, 2] = 3 * safe_divide((t[i+1] - t[i]) * (t[i] - t[i-1]), (t[i+2] - t[i-1]) * (t[i+1] - t[i-1]))
    M[2, 2] = 3 * safe_divide((t[i+1] - t[i])**2, (t[i+2] - t[i-1]) * (t[i+1] - t[i-1]))
    M[3, 3] = safe_divide((t[i+1] - t[i])**2, (t[i+3] - t[i]) * (t[i+2] - t[i]))
    
    M[1, 0] = -3 * M[0, 0]
    M[2, 0] = 3 * M[0, 0]
    M[3, 0] = - M[0, 0]
    M[0, 1] = 1 - M[0, 0] - M[0, 2]
    M[1, 1] = 3 * M[0, 0] - M[1, 2]
    M[2, 1] = -3 * M[0, 0] - M[2, 2]
    M[3, 2] = -safe_divide(M[2, 2], 3) - M[3, 3] - safe_divide((t[i+1] - t[i])**2, (t[i+2] - t[i]) * (t[i+2] - t[i-1]))
    M[3, 1] = M[0, 0] - M[3, 2] - M[3, 3]
    
    return M[::-1]


def Matrix2(interval: int, t: np.ndarray):
    # interval j starts at 0, i starts at 3
    i = interval+2 # check this
    #i should be at least 2
    M = np.zeros((3, 3))
    M[0, 0] = safe_divide((t[i+1] - t[i]), (t[i+1] - t[i-1]))
    M[0, 1] = safe_divide((t[i] - t[i-1]), (t[i+1] - t[i-1]))
    M[0, 2] = 0.
    M[1, 0] = -2 * M[0, 0]
    M[1, 1] = 2 * M[0, 0]
    M[1, 2] = 0.
    M[2, 0] = M[0, 0]
    M[2, 1] = -(t[i+1]-t[i])*(1/(t[i+1]-t[i-1])+1/(t[i+2]-t[i]))
    M[2, 2] = (t[i+1]-t[i])/(t[i+2]-t[i-1])
    return M[::-1]


# %%
# m = n+p+1 (number of knots = m+1)
# n + 1 number of control points
# J = m - 2p = n-p+1 number of intervals 
# num internal knots = m - 2p - 1 = n-p

if __name__ == "__main__":
    Q = np.array([[0,0],[0,1],[1,1],[1,0],[2,1],[3,0],[3,0],[3,0]])
    n = len(Q)-1
    p = 3
    m = n+p+1
    i = 0
    j = 0
    knots=np.hstack(([0]*(p-j), np.linspace(0,1,2+m-2*p-1+i+j), [1]*(p-i))) # we add 2 to internal knots for the 0 and the 1 on the edges
    print(f"{knots=}")
    # print([i for i in range(0,m-2*p)])
    matrices = [Matrix3(j,knots) for j in range(m-2*p)]

    # print(matrices)
    # M1 = Matrix(0,knots)
    # print(M1)
    # np.save('Mbs0.npy', M)

    T = np.vstack(([[x**3,x**2,x,1] for x in np.linspace(0,1,100)]))
    # print([Q[j:j+3] for j, M in enumerate(matrices)])
    s = np.vstack([T@M@Q[j:j+4] for j, M in enumerate(matrices)])
    interval = 4
    c = T@Matrix3(interval, knots)@Q[interval:interval+p+1] 
    plt.plot(s[:,0], s[:,1])
    plt.plot(c[:,0], c[:,1])

    plt.scatter(Q[:,0],Q[:,1])
# %%
