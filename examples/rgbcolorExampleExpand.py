from dcolor import DColorRGB

dc = DColorRGB()
dc.plot(lambda z : z**2+4+4j,title='z**2+4+4j')
dc.plot(lambda z : z**8+4+4j,title='z**8+4+4j expand 8', expand=8)
dc.plot(lambda z : z**16+4+4j,title='z**16+4+4j expand 16', expand=16)
