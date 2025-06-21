import glob
from vortex import Vortex


images = glob.glob("./Fieldcold/*.jpg")
vortex=Vortex(images[1],(20,20),lambda_=2)
vortex.detect_vortices(1,threshold=130,inverse=False,show=True)
vortex.calc_pinning_force()
vortex.analyze_vortex_neighbors()
vortex.draw_pinning_force(text=True)