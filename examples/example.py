import numpy as np


class example:
    def __init__(self, dc):
        dc.plot(lambda z: z, title="z")
        dc.plot(lambda z: z**z, title="z**z")
        dc.plot(lambda z: z + 4, title="z+4")
        dc.plot(lambda z: z - 4, title="z-4")
        dc.plot(lambda z: z + 4j, title="z+4j")
        dc.plot(lambda z: z - 4j, title="z-4j")
        dc.plot(
            lambda z: (((z + 4) * (z - 4) * (z + 4j) * (z - 4j)) ** (1 / 8)),
            title="(((z+4)*(z-4)*(z+4j)*(z-4j))**(1/8))",
        )
        dc.plot(lambda z: 1 / z, title="1/z")
        dc.plot(lambda z: (np.sin(1 / z)), title="sin(1/z)")
        dc.plot(lambda z: (np.cos(1 / z)), title="cos(1/z)")
        dc.plot(lambda z: np.sin(z), title="sin(z)")
        dc.plot(lambda z: np.cos(z), title="cos(z)")
        dc.plot(lambda z: z**3 - 1, title="z**3 -1")
        dc.plot(lambda z: z**2 + 4 + 4j, title="z**2+4+4j")
        dc.plot(lambda z: z**8 + 4 + 4j, title="z**8+4+4j")
        dc.plot(lambda z: z**16 + 4 + 4j, title="z**16+4+4j")
        dc.plot(
            lambda z: ((z**2 - 1) * (z - 2 - 1j) ** 2) / (z**2 + 2 + 2j),
            title="((z**2-1)*(z-2- 1j)**2)/(z**2 +2+ 2j)",
        )
