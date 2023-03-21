import numpy as np
import cvxpy as cp

n = int(input("Enter the number of assets: "))
L = np.tril(np.random.uniform(low=0.00001, high=0.1, size=(n, n)), k=-1)
cov = np.array(L + L.T + np.eye(n)/2)
mu = [np.random.uniform(0, 1) for _ in range(n)]
w = cp.Variable(n)
r = np.array(mu).T @ w
alpha = 0.1
cov += alpha * np.eye(n)

# Define function for Newton-Raphson method and projections onto convex sets

def proj(x):
    return x / np.sum(x)

# Newton-Raphson method and projections onto convex sets
w = np.ones(n) / n
while True:
    grad = mu
    hess = cov
    w_new = proj(w - np.linalg.solve(cov, mu))
    if np.linalg.norm(w_new - w) < 1e-6:
        break
    w = w_new

print("Optimal portfolio weights:")
for i in range(n):
    print(f"Asset {i+1}: {w[i]:.5f}")
