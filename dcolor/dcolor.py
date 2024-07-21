#!/usr/bin/env python3
from typing import Callable
from typing_extensions import TypeAliasType

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

import dcolor.color_maps as color_maps

ComplexFunction = TypeAliasType(
    "ComplexFunction",
    Callable[[npt.NDArray[np.complexfloating]], npt.NDArray[np.complexfloating]],
)


class DColor:
    def __init__(self, samples=1000, xmin=-8.0, xmax=8.0, ymin=-8.0, ymax=8.0):
        # mpl.rcParams["toolbar"] = "None"
        self._samples = samples
        # axes
        self._xmin = xmin
        self._xmax = xmax
        self._ymin = ymin
        self._ymax = ymax

    def makeDomain(self) -> npt.NDArray[np.complexfloating]:
        """Create the domains for Real (x) and Imaginary (y) values respectively"""
        x = np.linspace(self._xmin, self._xmax, self._samples)
        y = np.linspace(self._ymin, self._ymax, self._samples)
        xx, yy = np.meshgrid(x, y)
        return xx + 1j * yy

    def plot(
        self,
        f: ComplexFunction,
        *,
        color_map: color_maps.ComplexColorMap = color_maps.magnitude_oscillating,
        grid: bool = True,
    ):
        """Plot a complex-valued function `f` on the current matplotlib axes.
        Note that the current figure's DPI settings will be used to draw the image.
        If this matters to you, call `Figure.set_dpi` beforehand.

        Arguments:
        `f` --  The complex-valued function to plot. A callable which accepts numpy array of complex values.

        Optional keyword arguments:
        `color_map` --  A function which converts a complex number to an RGB color
                        (or an array of the former to an array of the latter).
                        The default is `dcolor.color_maps.magnitude_oscillating`

        `grid` --       Whether or not to draw gridlines at the tick positions
        """
        zz = f(self.makeDomain())
        rgb = color_map(zz)

        ax = plt.gca()

        # Plot the image with y extents backwards
        ax.imshow(rgb, extent=(self._xmin, self._xmax, self._ymax, self._ymin))
        # make CCW orientation positive
        # subsequent plots on the same axes will not be affected
        ax.invert_yaxis()

        ax.set_xlabel("$\\Re$")
        ax.set_ylabel("$\\Im$")
        ax.axhline(y=0, color="k")
        ax.axvline(x=0, color="k")
        if grid:
            ax.grid(True, which="both", linestyle="dashed")
