import numpy as np

def complex_z(x_min = -2, x_max = 2, y_min = -2, y_max = 2, width = 800, height = 800, max_iter = 100):
    """
    Iterates z_{i+1} = z_i^2 + c for each point c = x + iy in the complex plane.

    Returns:
        C : 2D array of complex numbers (the grid of c values)
        iter_counts : 2D array of iteration counts at which each point diverged (max_iter for boundededness)
    """
    # Grid of c values
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    C = x[np.newaxis, :] + 1j * y[:, np.newaxis]

    Z = np.zeros_like(C)
    iter_counts = np.full(C.shape, max_iter, dtype=int)
    not_diverged = np.ones(C.shape, dtype=bool)

    for i in range(max_iter):
        Z[not_diverged] = Z[not_diverged]**2 + C[not_diverged]
        just_diverged   = not_diverged & (np.abs(Z) > 2)
        iter_counts[just_diverged] = i
        not_diverged[just_diverged] = False

    return C, iter_counts