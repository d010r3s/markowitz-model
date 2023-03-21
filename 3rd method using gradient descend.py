import numpy as np

# Define variables
R = np.array([0.2, 0.2, 0.3, 0.3]) # expected returns
C = np.array([[0.04, 0.01, 0.02, 0.03],
              [0.01, 0.09, 0.05, 0.02],
              [0.02, 0.05, 0.12, 0.06],
              [0.03, 0.02, 0.06, 0.08]]) # covariance matrix
gamma = 0.06 # risk tolerance
w = np.array([0.2, 0.2, 0.3, 0.3]) # initial weights

def obj_func(w):
    return w.dot(C).dot(w.T) - gamma * w.dot(R.T)

# gradient
def obj_grad(w):
    return 2 * C.dot(w) - gamma * R

# gradient descent
alpha = 0.01 # learning rate
max_iter = 10000 # number of iterations
for i in range(max_iter):
    grad = obj_grad(w)
    w = w - alpha * grad
    if np.linalg.norm(grad) < 1e-5:
        break

print("Final portfolio weights:", w)
print("Objective function value:", obj_func(w))
