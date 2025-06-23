import glob
from vortex import Vortex
import numpy as np

extensions=['jpg','png']
images=[]
for ext in extensions:
    images.extend(glob.glob(f"./Fieldcold/*.{ext}"))

vortex=Vortex(images[1],(20,20))

#检测涡旋位置，inverse颜色反转，show预览与手动调整
vortex.detect_vortices(1,threshold=130,inverse=False,show=True) 
#计算钉扎力，lambda_为T=Tf穿透深度，单位微米
# vortex.calc_pinning_force(lambda_=1)

#Delaunay 三角剖分与配位数计算
# vortex.analyze_vortex_neighbors()

#绘制钉扎力
# vortex.draw_pinning_force()

#保存涡旋位置
#np.savetxt('vortex_positions.txt', vortex.vortex_positions)

#保存配位数
#np.savetxt('coordination_numbers.txt', vortex.coordination_number)

#保存邻接关系
#np.savetxt('neighbor.txt', vortex.neighbor)
