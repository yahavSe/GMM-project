import numpy as np
import matplotlib.pyplot as plt
from function import gmm

def main():
    # assume 3 means
    D = 2 # so we can visualize it more easily
    s = 10 # separation so we can control how far apart the means are
    mu1 = np.array([0, 0])
    mu2 = np.array([s, s])
    mu3 = np.array([0, s])

    N = 2000 # number of samples
    X = np.zeros((N, D))
    X[:1200, :] = np.random.randn(1200, D)*2 + mu1
    X[1200:1800, :] = np.random.randn(600, D) + mu2
    X[1800:, :] = np.random.randn(200, D)*0.5 + mu3

    # what does it look like without clustering?
    plt.subplot(311)
    plt.scatter(X[:,0], X[:,1])
    plt.title('Original data')

    K = 3
    gmm(X, K)



if __name__ == '__main__':
    main()