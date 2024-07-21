import matplotlib.pyplot as plt
import numpy as np

def dcplot(dc, f, title, color_map):
    dc.plot(f, color_map=color_map)
    plt.title(title)
    plt.show()

class example:
    def __init__(self, dc, color_map):
        dcplot(dc, lambda z: z, title="z", color_map=color_map)
        dcplot(dc, lambda z: 1/z, title="z", color_map=color_map)
        dcplot(dc, lambda z: z**z, title="z**z", color_map=color_map)
        dcplot(dc, lambda z: z + 4, title="z+4", color_map=color_map)
        dcplot(dc, lambda z: z - 4, title="z-4", color_map=color_map)
        dcplot(dc, lambda z: z + 4j, title="z+4j", color_map=color_map)
        dcplot(dc, lambda z: z - 4j, title="z-4j", color_map=color_map)
        dcplot(
            dc,
            lambda z: (((z + 4) * (z - 4) * (z + 4j) * (z - 4j)) ** (1 / 8)),
            title="(((z+4)*(z-4)*(z+4j)*(z-4j))**(1/8))",
            color_map=color_map,
        )
        dcplot(dc, lambda z: 1 / z, title="1/z", color_map=color_map)
        dcplot(dc, lambda z: (np.sin(1 / z)), title="sin(1/z)", color_map=color_map)
        dcplot(dc, lambda z: (np.cos(1 / z)), title="cos(1/z)", color_map=color_map)
        dcplot(dc, lambda z: np.sin(z), title="sin(z)", color_map=color_map)
        dcplot(dc, lambda z: np.cos(z), title="cos(z)", color_map=color_map)
        dcplot(dc, lambda z: z**3 - 1, title="z**3 -1", color_map=color_map)
        dcplot(dc, lambda z: z**2 + 4 + 4j, title="z**2+4+4j", color_map=color_map)
        dcplot(dc, lambda z: z**8 + 4 + 4j, title="z**8+4+4j", color_map=color_map)
        dcplot(dc, lambda z: z**16 + 4 + 4j, title="z**16+4+4j", color_map=color_map)
        dcplot(
            dc,
            lambda z: ((z**2 - 1) * (z - 2 - 1j) ** 2) / (z**2 + 2 + 2j),
            title="((z**2-1)*(z-2- 1j)**2)/(z**2 +2+ 2j)",
            color_map=color_map,
        )
