import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def gmm(X, K, max_iter=20, smoothing=0.1):
    N, D = X.shape
    M = np.zeros((K, D))
    R = np.zeros((N, K))
    C = np.zeros((K, D, D))
    pi = np.ones(K) / K  # uniform

    # initialize M to random, initialize C to spherical with variance 1
    for k in range(K):
        M[k] = X[np.random.choice(N)]
        C[k] = np.eye(D)

    costs = np.zeros(max_iter)
    weighted_pdfs = np.zeros((N, K))  # we'll use these to store the PDF value of sample n and Gaussian k
    for i in range(max_iter):
        # step 1: determine assignments / resposibilities

        for k in range(K):
            weighted_pdfs[:, k] = pi[k] * multivariate_normal.pdf(X, M[k], C[k])
        R = weighted_pdfs / weighted_pdfs.sum(axis=1, keepdims=True)

        # step 2: recalculate params
        for k in range(K):
            Nk = R[:, k].sum()
            pi[k] = Nk / N
            M[k] = R[:, k].dot(X) / Nk

            delta = X - M[k]  # N x D
            Rdelta = np.expand_dims(R[:, k], -1) * delta  # multiplies R[:,k] by each col. of delta - N x D
            C[k] = Rdelta.T.dot(delta) / Nk + np.eye(D) * smoothing  # D x D

        costs[i] = np.log(weighted_pdfs.sum(axis=1)).sum()
        if i > 0:
            if np.abs(costs[i] - costs[i - 1]) < 0.1:
                break

    plt.subplot(312)
    plt.plot(costs)
    plt.title("Costs")

    plt.subplot(313)
    random_colors = np.random.random((K, 3))
    colors = R.dot(random_colors)
    plt.scatter(X[:, 0], X[:, 1], c=colors)
    plt.title("GMM result")
    plt.show()

    print("pi:", pi)
    print("means:", M)
    print("covariances:", C)
    return R
