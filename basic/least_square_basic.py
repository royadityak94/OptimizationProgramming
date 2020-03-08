# Simple least square problem with box-constraint using CVXPY over CVXOPT solver
# Importing the required packages
import numpy as np
import cvxpy as cp

# Ensuring reproducibility
np.random.seed(42)

# Data-points
m, n = 30, 20
A = np.random.randn(m, n)
B = np.random.randn(m)

# Problem construction
x = cp.Variable(n)
objective = cp.Minimize(cp.sum_squares(A@x - B))
constraints = [x >= 0, x <= 1]
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.CVXOPT)

print ("Found optimal value = ", prob.value)
print (constraints[0].dual_value)
print ("The norm of the residual is ", cp.norm(A@x - B, p=2).value)
