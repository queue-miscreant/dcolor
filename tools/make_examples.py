from typing import Dict

import numpy as np
from matplotlib import colormaps
import matplotlib.pyplot as plt

import dcolor

test_function = lambda z: ((z**2 - 1) * (z - 2 - 1j) ** 2) / (z**2 + 2 + 2j)

files_to_color_maps: Dict[str, dcolor.ComplexColorMap] = {
    # "dcolor": dcolor.magnitude_oscillating,
    # "hsvcolor": dcolor.raw_magnitude_oscillating,
    # "rgbcolor": dcolor.green_magnitude,
    # "domain_polezero": dcolor.domain_polezero,
    "generic": dcolor.GenericColorMap(),
    "magnitude_flag": dcolor.MagnitudeMap(
        mag_colormap=colormaps["flag"]
    ),
}

def generate_examples():
    dc = dcolor.DColor()
    for fname, color_map in files_to_color_maps.items():
        fig = plt.figure(figsize=(8,8), dpi=100)
        plt.title(r"$f(z) = \frac{(z^2 - 1)(z - 2 - i)^2}{z^2 + 2 + 2i}$")
        dc.plot(test_function, color_map=color_map)
        fig.savefig(f"images/{fname}.png")


def generate_colormaps():
    xmin, xmax = 0, 2.0*np.pi
    ymin, ymax = -3, 3
    arg = np.linspace(xmin, xmax, 1000)
    mag = np.logspace(ymin, ymax, 1000)
    arg_x, arg_y = np.meshgrid(arg, mag)

    for fname, color_map in files_to_color_maps.items():
        fig = plt.figure(figsize=(8,8), dpi=100)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.xlabel("$\\arg(z)$")
        plt.ylabel("$\\log_{10}|z|$")
        plt.xticks([*np.linspace(xmin, xmax, 7)], [
            "$0$",
            "$\\pi / 3$",
            "$2\\pi / 3$",
            "$\\pi$",
            "$4\\pi / 3$",
            "$5\\pi / 3$",
            "$2\\pi$",
        ])
        image = color_map(arg_y, arg_x)
        plt.imshow(image, origin="lower", extent=(xmin, xmax, ymin, ymax))
        plt.gca().set_aspect((xmax - xmin) / (ymax - ymin))
        fig.savefig(f"images/colormap_{fname}.png")

generate_examples()
generate_colormaps()
