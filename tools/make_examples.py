from typing import Dict

import matplotlib.pyplot as plt

import dcolor

test_function = lambda z: ((z**2 - 1) * (z - 2 - 1j) ** 2) / (z**2 + 2 + 2j)

files_to_color_maps: Dict[str, dcolor.ComplexColorMap] = {
    "dcolor": dcolor.magnitude_oscillating,
    "hsvcolor": dcolor.raw_magnitude_oscillating,
    "rgbcolor": dcolor.green_magnitude,
    "domain_polezero": dcolor.domain_polezero,
}

dc = dcolor.DColor()
for fname, color_map in files_to_color_maps.items():
    fig = plt.figure(figsize=(8,8), dpi=100)
    plt.title(r"$f(z) = \frac{(z^2 - 1)(z - 2 - i)^2}{z^2 + 2 + 2i}$")
    dc.plot(test_function, color_map=color_map)
    fig.savefig(f"images/{fname}.png")
