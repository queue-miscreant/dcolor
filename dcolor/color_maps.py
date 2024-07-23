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


def smoothing(t: npt.NDArray, scale: float):
    """Nice bijection on [0,1] for weighing saturation and value contributions"""
    return np.log(1 + t**scale * (np.e - 1))


def overlay(a: Image, b: Image) -> Image:
    """Multiply `a` for low `b` values and screen `a` for high `b` values"""
    multiply = 2*a*b
    screen = 1.0 - 2*(1.0 - b)*(1.0 - a)
    filter_ = (b <= 0.5).astype(multiply.dtype)

    return multiply * filter_ + screen * (1.0 - filter_)


class GenericColorMap:
    """
    Class for creating common domain coloring maps.

    Since magnitude ranges over [0, oo) and argument (angle) ranges over [0, 2pi),
    these values must be converted to range over [0, 1), the domain of Matplotlib
    colormaps.
    To do so, the functions specified by `mag_premap` and `arg_premap` are used.
    By default, these are the following maps:

    mag |-> 2*arctan(mag) / pi
    arg |-> arg / 2pi

    Finally, the values are translated to images using the colormaps provided,
    then blended together by [overlaying](https://en.wikipedia.org/wiki/Blend_modes#Overlay)
    the magnitude over the argument image.
    """
    def __init__(
        self,
        mag_colormap: MPLColorMap = colormaps["gray"],
        arg_colormap: MPLColorMap = colormaps["hsv"],
        mag_premap: Callable[[FloatArray], FloatArray] = lambda x: (2.0 * np.arctan(x) / np.pi),
        arg_premap: Callable[[FloatArray], FloatArray] = lambda x: (x / (2.0 * np.pi)),
    ):
        self.mag_colormap = mag_colormap
        self.arg_colormap = arg_colormap
        self.mag_premap = mag_premap
        self.arg_premap = arg_premap

    def __call__(self, mag: FloatArray, arg: FloatArray):
        """
        Convert the magnitude and argument arrays provided to a single RGB image
        """
        colored_arg = self.arg_colormap(self.arg_premap(arg))
        colored_mag = self.mag_colormap(self.mag_premap(mag))

        return overlay(colored_arg[:,:,0:3], colored_mag[:,:,0:3])  # discard alpha

class MagnitudeMap(GenericColorMap):
    """
    A GenericColormap which disregards its argument data
    """
    def __init__(
        self,
        mag_colormap: MPLColorMap = colormaps["inferno"],
        mag_premap: Callable[[FloatArray], FloatArray] = lambda x: (2.0 * np.arctan(x) / np.pi),
    ):
        super().__init__(
            mag_colormap = mag_colormap,
            arg_colormap = colormaps["Greys"],
            mag_premap = mag_premap,
            arg_premap = lambda x: np.ones_like(x) * 0.5,
        )


def _domain_polezero_mag_premap(x: FloatArray, scale: float):
    """Magnitude premap for `domain_polezero`"""
    low_magnitudes_black = smoothing(x, scale=scale)
    high_magnitudes_white = smoothing(1 / (np.finfo(x.dtype).eps + x), scale=scale)
    return (low_magnitudes_black / 2.0) * (x < 1.0) + (1.0 - high_magnitudes_white / 2.0) * (x >= 1.0)


def domain_polezero(mag: FloatArray, arg: FloatArray, scale: float = 0.2) -> Image:
    """
    Converts a complex 2D array `zz` to an RGB image with "typical" domain coloring.

    Hue is taken from the complex argument of `zz`.
    Values near zeroes are colored darker and values near poles are colored brighter.
    """
    return GenericColorMap(
        mag_premap=lambda x: _domain_polezero_mag_premap(x, scale=scale),
    )(mag, arg)


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
