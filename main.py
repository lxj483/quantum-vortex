import cv2
import numpy as np
import glob
from vortex import Vortex


images = glob.glob("./Fieldcold/*.png")
vortex=Vortex(images[0],(18,18))
vortex.detect_vortices(1,threshold=150,inverse=True)
vortex.calc_pinning_force()
vortex.draw_pinning_force()
length_x=18
length_y=18