#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

import dcolor.color_maps as color_maps

class DColor:
    def __init__(self, samples=1000, xmin=-8.0, xmax=8.0, ymin=-8.0, ymax=8.0):
        # mpl.rcParams["toolbar"] = "None"
        self._samples = samples
        # axes
        self._xmin = xmin
        self._xmax = xmax
        self._ymin = ymin
        self._ymax = ymax

    def makeDomain(self):
        """Create the domains for Real (x) and Imaginary (y) values respectively"""
        x = np.linspace(self._xmin, self._xmax, self._samples)
        y = np.linspace(self._ymin, self._ymax, self._samples)
        xx, yy = np.meshgrid(x, y)
        return xx + 1j * yy

    def plot(self, f, color_map=color_maps.magnitude_oscillating, xdim=8, ydim=8, plt_dpi=100, title=""):
        """Plot a complex-valued function
        Arguments:
        f -- a (preferably) lambda-function defining a complex-valued function
        Keyword Arguments:
        xdim -- x dimensions
        ydim -- y dimensions
        plt_dpi -- density of pixels per inch
        """
        zz = f(self.makeDomain())
        rgb = color_map(zz)

        fig = plt.figure(figsize=(xdim, ydim), dpi=plt_dpi)
        ax = fig.gca()
        ax.imshow(rgb, extent=(self._xmin, self._xmax, self._ymax, self._ymin))  # y extents backwards for inverted y axis
        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')
        ax.grid(True, which='both', linestyle="dashed")
        ax.invert_yaxis()  # make CCW orientation positive
        ax.get_xaxis().set_visible(True)
        ax.get_yaxis().set_visible(True)
        ax.set_title(title)
        plt.show()
