import numpy as np
import matplotlib.pyplot as plt
import json
import skimage.color

s = "let colormaps = {};\n"

for colormap in ["copper", "Spectral", "spring", "summer", "autumn", "winter", "ocean", "magma", "viridis", "RdBu", "PiYG", "PRGn", "BrBG", "PuOr", "RdGy"]:
    c = plt.get_cmap(colormap)
    C = c(np.array(np.round(np.linspace(0, 255, 100)), dtype=np.int32))
    C = C[:, 0:3]
    C = skimage.color.rgb2hsv(C)
    C[:, 2] = 0.3 + 0.4*C[:, 2]
    C = skimage.color.hsv2rgb(C)
    C = np.round(C, 5)
    C = C.tolist()
    s += "colormaps.{} = {};\n".format(colormap, C)
print(s)
