import glob
from vortex import Vortex


images = glob.glob("./Fieldcold/*.png")
vortex=Vortex(images[0],(20,20),lambda_=2)
vortex.detect_vortices(1,threshold=140,inverse=True,show=True)
# vortex.calc_pinning_force()
# vortex.analyze_vortex_neighbors()
# vortex.draw_pinning_force(text=True)