from typing import Callable, Literal
from typing_extensions import TypeAliasType

from matplotlib import colormaps
from matplotlib.colors import hsv_to_rgb, Colormap as MPLColorMap
import numpy as np
import numpy.typing as npt

FloatArray = TypeAliasType(
    "FloatArray", npt.NDArray[np.floating]
)  # Array type after applying angle() or abs()
Image = TypeAliasType(
    "Image", np.ndarray[Literal[3], np.dtype[np.floating]]
)  # RGB image
ComplexColorMap = Callable[[FloatArray, FloatArray], Image]


def normalize(arr: npt.NDArray) -> npt.NDArray:
    """Used for normalizing data in array based on min/max values"""
    arrMin = np.min(arr)
    arrMax = np.max(arr)
    arr = arr - arrMin
    return arr / (arrMax - arrMin)


def smoothing(t: npt.NDArray, scale: float):
    """Nice bijection on [0,1] for weighing saturation and value contributions"""
    return np.log(1 + t**scale * (np.e - 1))


def domain_polezero(mag: FloatArray, arg: FloatArray, scale: float = 0.2) -> Image:
    """
    Converts a complex 2D array `zz` to an RGB image with normal domain coloring.

    Hue is taken from the complex argument of `zz`.
    Zeroes are colored black and poles are colored white.
    """
    low_magnitudes_black = smoothing(mag, scale=scale)
    high_magnitudes_white = smoothing(1 / (np.finfo(mag.dtype).eps + mag), scale=scale)

    H = arg / (2.0 * np.pi)
    S = high_magnitudes_white * (mag >= 1.0) + (mag < 1.0)
    V = low_magnitudes_black * (mag < 1.0) + (mag >= 1.0)

    return hsv_to_rgb(np.dstack((H, S, V)))


# def magnitude_oscillating(zz: ComplexPlane) -> Image:
def magnitude_oscillating(mag: FloatArray, arg: FloatArray) -> Image:
    """
    Converts a complex 2D array `zz` to an RGB image with normal domain coloring.

    Hue is taken from the complex argument of `zz`.
    Saturation and value covary with the logarithm of the magnitude of `zz`,
    effectively giving logarithmic countours.
    """
    H = arg / (2.0 * np.pi)  # Hue determined by arg(z)
    r = np.log2(1.0 + mag)
    S = (1.0 + np.abs(np.sin(2.0 * np.pi * r))) / 2.0
    V = (1.0 + np.abs(np.cos(2.0 * np.pi * r))) / 2.0

    return hsv_to_rgb(np.dstack((H, S, V)))


def raw_magnitude_oscillating(mag: FloatArray, arg: FloatArray) -> Image:
    """
    Converts a complex 2D array `zz` to an RGB image.
    Similar to `magnitude_oscillating`, but with a "rawer" conversion to RGB.

    Generally similar to `magnitude_oscillating`, with R as hue,
    G as saturation, and B as value.
    """
    h = arg / (2.0 * np.pi)  # Hue determined by arg(z)
    r = np.log2(1.0 + mag)
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


def green_magnitude(mag: FloatArray, arg: FloatArray, expand: float = 1.0) -> Image:
    """
    Converts a complex 2D array `zz` to an RGB image.

    Small magnitues are colored black, large ones are white, and values
    between appear green.
    The rate at which this occurs is controlled by `expand`.
    """
    r = mag * 0.5 / expand
    g = mag * 1.00 / expand
    b = mag * 0.5 / expand
    r = np.clip(r, 0, 1)
    g = np.clip(g, 0, 1)
    b = np.clip(b, 0, 1)
    return np.dstack((r, g, b))
