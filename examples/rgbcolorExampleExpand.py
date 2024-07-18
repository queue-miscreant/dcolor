from functools import partial
from dcolor import DColor, green_magnitude

dc = DColor()
dc.plot(lambda z: z**2 + 4 + 4j, title="z**2+4+4j", color_map=partial(green_magnitude, expand=8))
dc.plot(lambda z: z**8 + 4 + 4j, title="z**8+4+4j expand 8", color_map=partial(green_magnitude, expand=8))
dc.plot(lambda z: z**16 + 4 + 4j, title="z**16+4+4j expand 16", color_map=partial(green_magnitude, expand=16))
