# Python program to solve non-trivial linear program
# Importing the required packages
import numpy as np
import cvxpy as cp

# Ensuring re-reproducibility
np.random.seed(42)

# Setting up data
m, n = 30, 20
s0 = np.maximum(np.random.randn(m), 0)
lambda0 = np.maximum(-np.random.randn(m), 0)
x0 = np.random.randn(n)

A = np.random.rand(m, n)
b = A @ x0 + s0
c = -A.T @ lambda0

# Setting up the problem and solve
x = cp.Variable(n)
prob = cp.Problem(cp.Minimize(c.T@x), [A@x <= b])
prob.solve(solver=cp.CVXOPT)

print ("Found optimal value = ", x.value)
print ("Norm of the residual = ", cp.norm(A @ x - b, p=2).value)
