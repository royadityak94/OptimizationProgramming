# Solving second-order cone program using cvxpy
# Installing the required packages
import numpy as np
import cvxpy as cp

# Setting up the Data
m, n = 5, 20
n_i = p = 10
x0 = np.random.randn(n)
f = np.random.randn(n)
A, b, c, d = [], [], [], []

for i in range(m):
    A.append(np.random.rand(n_i, n))
    b.append(np.random.randn(n_i))
    c.append(np.random.randn(n))
    d.append(cp.norm(A[i]@x0 + b, 2) - c[i].T@x0)
F = np.random.randn(p, n)
g = F@x0

# Constructing the problem
x = cp.Variable(n)
soc_constraints = [
    cp.SOC(c[i].T@x+d[i], A[i]@x+b[i])
    for i in range(m)
]

prob = cp.Problem(cp.Minimize(f.T@x), soc_constraints+[F@x == g])
prob.solve()

print ("Found Optimal Solution = ", prob.value)
for i in range(m):
    print ("SOC constraint %d dual variable solution" % i)
    print (soc_constraints[i].dual_value)
