import numpy as np

g = 4
h =4

def _toeplitz_rect(g, h, gamma):
    i = np.arange(g)[:, None]
    j = np.arange(h)[None, :]
    return gamma ** np.abs(i - j)

def sample_X(rng, rows, rho, gamma):
    print(rho, gamma)
    if np.isclose(rho, 0) and np.isclose(gamma,0):
        return rng.standard_normal((rows, g+h))
    Tg = _toeplitz_rect(g, g, gamma)
    Th = _toeplitz_rect(h, h, gamma)
    Tgh = _toeplitz_rect(g, h, gamma)
    S = np.block([[Tg, rho * Tgh], [rho * Tgh.T, Th]])

    L = np.linalg.cholesky(S)
    assert np.isfinite(L).all()

    Z = rng.standard_normal((rows, g + h))
    A = Z @ L.T
    assert np.isfinite(A).all()

    return A

rng = np.random.default_rng(4)

# sample_X(rng, 30, 0, 1) # 0 and 1 block
sample_X(rng, 10, 0, 0) # Id
sample_X(rng, 10, 0, .3) # diagonal toeplitz block
sample_X(rng, 10, 0.5, 0) # block
sample_X(rng, 10, 0.5, .5)
