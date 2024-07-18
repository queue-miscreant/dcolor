import numpy as np
from matplotlib.colors import hsv_to_rgb

def normalize(arr):
    """Used for normalizing data in array based on min/max values"""
    arrMin = np.min(arr)
    arrMax = np.max(arr)
    arr = arr - arrMin
    return arr / (arrMax - arrMin)

def magnitude_oscillating(zz):
    """
    Converts a complex 2D array `zz` to an RGB image with normal domain coloring.

    Hue is taken from the complex argument of `zz`.
    Saturation and value covary with the logarithm of the magnitude of `zz`,
    effectively giving logarithmic countours.
    """
    H = normalize(np.angle(zz) % (2.0 * np.pi))  # Hue determined by arg(z)
    r = np.log2(1.0 + np.abs(zz))
    S = (1.0 + np.abs(np.sin(2.0 * np.pi * r))) / 2.0
    V = (1.0 + np.abs(np.cos(2.0 * np.pi * r))) / 2.0

    return hsv_to_rgb(np.dstack((H, S, V)))

def raw_magnitude_oscillating(zz):
    """
    Converts a complex 2D array `zz` to an RGB image.
    Similar to `magnitude_oscillating`, but with a "rawer" conversion to RGB.

    Generally similar to `magnitude_oscillating`, with R as hue,
    G as saturation, and B as value.
    """
    h = normalize(np.angle(zz) % (2.0 * np.pi))  # Hue determined by arg(z)
    r = np.log2(1.0 + np.abs(zz))
    s = (1.0 + np.abs(np.sin(2.0 * np.pi * r))) / 2.0
    v = (1.0 + np.abs(np.cos(2.0 * np.pi * r))) / 2.0

    r = np.empty_like(h)
    g = np.empty_like(h)
    b = np.empty_like(h)

    i = (h * 6.0).astype(int)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    idx = i % 6 == 0
    r[idx] = v[idx]
    g[idx] = t[idx]
    b[idx] = p[idx]

    idx = i == 1
    r[idx] = q[idx]
    g[idx] = v[idx]
    b[idx] = p[idx]

    idx = i == 2
    r[idx] = p[idx]
    g[idx] = v[idx]
    b[idx] = t[idx]

    idx = i == 3
    r[idx] = p[idx]
    g[idx] = q[idx]
    b[idx] = v[idx]

    idx = i == 4
    r[idx] = t[idx]
    g[idx] = p[idx]
    b[idx] = v[idx]

    idx = i == 5
    r[idx] = v[idx]
    g[idx] = p[idx]
    b[idx] = q[idx]

    idx = s == 0
    r[idx] = v[idx]
    g[idx] = v[idx]
    b[idx] = v[idx]

    return np.stack([r, g, b], axis=-1)

def green_magnitude(zz, expand=1.0):
    """
    Converts a complex 2D array `zz` to an RGB image.

    Small magnitues are colored black, large ones are white, and values
    between appear green.
    The rate at which this occurs is controlled by `expand`.
    """
    absz = np.abs(zz)
    r = absz * 0.5 / expand
    g = absz * 1.00 / expand
    b = absz * 0.5 / expand
    r = np.clip(r, 0, 1)
    g = np.clip(g, 0, 1)
    b = np.clip(b, 0, 1)
    return np.dstack((r, g, b))
