import cvxpy as cp
import numpy as np

for i in range(10):
    n = 100
    L = np.tril(np.random.uniform(low=0.00001, high=0.1, size=(n, n)), k=-1)
    Sigma = np.array(L + L.T + np.eye(n)/2)
    mu = [np.random.uniform(0, 1) for _ in range(n)]
    #print(mu)
    #print((L))
    #w = cp.Variable(n)
    #ret = np.array(mu).T @ w
    #risk = cp.quad_form(w, Sigma)

    #objective = cp.Minimize(risk)
    #constraints = [cp.sum(w) == 1, w >= 0, ret >= 0.2]
    #prob = cp.Problem(objective, constraints)
    #prob.solve()

    #print(f"Optimal portfolio weights: {w.value}")
    #print(f"Expected return: {ret.value}")
    #print(f"Minimum risk (variance): {prob.value}")
    try:
        with open("text.txt", "a")as file:
            for i in range(len(mu)):
                file.write(str(mu[i]))
                file.write("\n")
    except:
        print("error")
    try:
        with open("text.txt", "a")as file:
            for i in range(len(L)):
                file.write(str(L[i]))
                file.write("\n")
    except:
        print("error")
