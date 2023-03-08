import time
import numpy as np
from scipy.optimize import minimize

start_time = time.time()

# objective function
def objective(weights, returns, cov_matrix, target_return):
    portfolio_return = np.dot(weights, returns)
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    penalty = 2000 * abs(portfolio_return - target_return)
    return np.sqrt(portfolio_variance) + penalty

# constraints
def constraint(weights):
    return np.sum(weights) - 1

# optimizing function
def optimize_portfolio(returns, cov_matrix, target_return):
    num_assets = len(returns)
    bounds = tuple((0,1) for i in range(num_assets))
    constraints = ({'type': 'eq', 'fun': constraint})
    initial_weights = num_assets * [1./num_assets]

    opt_results = minimize(objective, initial_weights, args = (returns, cov_matrix, target_return), method = 'SLSQP', bounds = bounds, constraints = constraints)

    return opt_results.x

# inputs
returns = np.array([0.2, 0.2, 0.3, 0.3])
cov_matrix = np.array([[0.04, 0.01, 0.02, 0.03],
                       [0.01, 0.09, 0.05, 0.02],
                       [0.02, 0.05, 0.12, 0.06],
                       [0.03, 0.02, 0.06, 0.08]])
target_return = 0.06

# call the optimization function
weights = optimize_portfolio(returns, cov_matrix, target_return)


# results
print('Optimal weights:', weights)
print('Optimal portfolio return:', np.dot(weights, returns))
print('Optimal portfolio variance:', np.dot(weights.T, np.dot(cov_matrix, weights)))
print(f"Time taken: {(time.time() - start_time)*1000} ms")
