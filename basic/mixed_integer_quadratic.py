# Solving mixed integer least-squares problem
import numpy as np
import cvxpy as cp

# Ensuring reproducibility
np.random.seed(42)

# Setting up data
m, n = 30, 20
A = np.random.randn(m, n)
b = np.random.randn(m)

# Constructing the problem
x = cp.Variable(n, integer=True)
objective = cp.Minimize(cp.sum_squares(A@x - b))
prob = cp.Problem(objective)
result = prob.solve()

print ("Found Optimal Value = ", x.value)
print ("The norm of the residual is = ", cp.norm(A@x-b, p=2).value)
