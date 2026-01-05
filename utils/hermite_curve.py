import numpy as np

def hermite_curve(P0, P1, T0, T1, num_points=100):
    P0 = np.array(P0, dtype=float)
    P1 = np.array(P1, dtype=float)
    T0 = np.array(T0, dtype=float)
    T1 = np.array(T1, dtype=float)

    t = np.linspace(0.0, 1.0, num_points)

    h1 =  2 * t**3 - 3 * t**2 + 1
    h2 = -2 * t**3 + 3 * t**2
    h3 =      t**3 - 2 * t**2 + t
    h4 =      t**3 -     t**2

    points = (h1[:, None] * P0 +
              h2[:, None] * P1 +
              h3[:, None] * T0 +
              h4[:, None] * T1)

    return points 