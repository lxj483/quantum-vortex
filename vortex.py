import numpy as np
from scipy.ndimage import maximum_filter
import cv2
import glob
import matplotlib.pyplot as plt
from scipy.special import kn
from scipy.spatial.distance import pdist, squareform
import matplotlib
matplotlib.use('TkAgg')
class Vortex:
    def __init__(self,image_path, size, lambda_=1):
        self.size=np.array(size)
        self.image_path=image_path

        self.lambda_ = lambda_   # Penetration depth um
        self.vortex_positions=None
        self.pinning_forces=None
        self.pixel_size=np.array([1,1])


    def build_dog_pyramid(self,min_radius,scales_per_octave=5,sigma=1,inverse=False):
        #构建高斯金字塔
        image = cv2.imread(self.image_path)
        if image is None:
            print("Error: unable to load image.")
            return
        if len(image.shape)==3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            image=image.astype(np.float32)
        if inverse:
            image=255-image
        k = 2 ** (1.0 / scales_per_octave)  # 尺度倍增系数
        image=cv2.GaussianBlur(image,(0,0),sigmaX=min_radius*sigma)
        image=cv2.resize(image,(int(image.shape[1]/min_radius),int(image.shape[0]//min_radius)),interpolation=cv2.INTER_NEAREST)
        octave_images = np.empty((scales_per_octave+1,*image.shape), dtype=np.float32)
        sigma_total = sigma
        for s in range(scales_per_octave + 1):  # 每octave多生成3幅用于极值检测
            if  s == 0:
                octave_images[0]=(image.astype(np.float32))
            else:
                sigma_effective = sigma_total * np.sqrt(k**2 - 1)
                octave_images[s] = cv2.GaussianBlur(octave_images[s-1], (0, 0), sigmaX=sigma_effective)
            sigma_total *= k
        #构建高斯差分金字塔
        k=2**(1.0/scales_per_octave)
        dog_octave = np.zeros((scales_per_octave,*octave_images.shape[1:]), dtype=np.float32)
        for i in range(1, len(octave_images)):
            dog_octave[i-1] = octave_images[i] - octave_images[i-1]
        min_val=np.min(dog_octave)
        max_val=np.max(dog_octave)
        dog_octave=(dog_octave-min_val)/(max_val-min_val+1e-5)*255
        return dog_octave

    def detect_vortices(self,min_radius,threshold=100,inverse=False,show=False):
        image = cv2.imread(self.image_path)
        self.pixel_size=self.size/np.array(image.shape[:2])
        scales_per_octave=5
        sigma=1
        dog_octave = self.build_dog_pyramid(min_radius,scales_per_octave,sigma,inverse)

        # 在多尺度上检测涡旋
        neighborhood_max=maximum_filter(dog_octave,size=(2*scales_per_octave+1,5,5))
        local_max=((dog_octave==neighborhood_max)&(dog_octave>threshold))
        y_coords,x_coords=(np.array(np.where(local_max)[1:])*min_radius).astype(np.int16)
        vortices=np.array(list(zip(x_coords,y_coords)))
        self.vortex_positions=vortices
        if show:
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(image)
            scatter = ax.scatter(vortices[:, 0], vortices[:, 1], 
                               c='red', marker='o', s=20, 
                               edgecolors='white', linewidths=0.5)
            ax.set_title('Detected Vortices (Click to add/remove)')
            ax.axis('off')
            plt.tight_layout()
            def onclick(event):
                if event.inaxes != ax:
                    return                
                # 获取当前涡旋点
                current_vortices = self.vortex_positions.copy()                    
                # 查找最近的涡旋点
                if len(current_vortices) > 0:
                    distances = np.linalg.norm(current_vortices - [event.xdata, event.ydata], axis=1)
                    min_idx = np.argmin(distances)                        # 如果距离足够近则删除，否则添加
                    if distances[min_idx] < 2:  # 5像素阈值
                        current_vortices = np.delete(current_vortices, min_idx, axis=0)
                    else:
                        current_vortices = np.vstack([current_vortices, 
                                                     [event.xdata, event.ydata]])
                else:                        
                    current_vortices = np.array([[event.xdata, event.ydata]])
                    
                    # 更新显示和类属性
                scatter.set_offsets(current_vortices)
                self.vortex_positions = current_vortices.astype(np.int16)
                fig.canvas.draw()

            fig.canvas.mpl_connect('button_press_event', onclick)
            plt.show()
        return self.vortex_positions*self.pixel_size
        
    def calc_pinning_force(self):
        vortex_positions=self.vortex_positions*self.pixel_size

        # Load vortex positions data (in microns)
        # Define constants
        phi_0 = 2.07e-3  # Flux quantum in Wb*1e12 (weber)
        mu_0 = 4 * np.pi * 1e-7  # Vacuum permeability

        # Initialize arrays
        num_vortices = vortex_positions.shape[0]
        pinning_forces = np.zeros_like(vortex_positions)
        const_factor = phi_0**2 / (2 * np.pi * mu_0 * self.lambda_**3)#*1e-12

        # 计算所有点对之间的距离矩阵
        distance_matrix = squareform(pdist(vortex_positions))

        # 计算力向量 (向量化实现)
        r_vectors = vortex_positions[:, np.newaxis] - vortex_positions  # 所有相对位置向量
        r_norms = distance_matrix[..., np.newaxis]  # 所有距离
        r_norms[np.diag_indices(num_vortices)]=np.inf
        K1_values = kn(1, distance_matrix/self.lambda_)  # 所有K1值


        # 计算力贡献 (向量化)
        force_contributions = (const_factor * (r_vectors / r_norms) * K1_values[..., np.newaxis])
        # force_contributions[np.diag_indices(num_vortices)] = 0  # 排除自身贡献
        self.pinning_forces = np.sum(force_contributions, axis=1)
        return pinning_forces

    def draw_pinning_force(self,text=False):
        # Visualization
        plt.figure(figsize=(8, 6))
        pinning_force_magnitudes=np.linalg.norm(self.pinning_forces,axis=1)
        scatter = plt.scatter(self.vortex_positions[:, 0]*self.pixel_size[0] , 
                            self.vortex_positions[:, 1]*self.pixel_size[1] , 
                            c=pinning_force_magnitudes,
                            s=100)
        plt.xlabel('X Position (μm)')
        plt.ylabel('Y Position (μm)')
        plt.colorbar(scatter, label='Pinning Force Magnitude')
        plt.gca().set_aspect('equal')
        ax=plt.gca()
        ax.invert_yaxis()
        ax.xaxis.set_ticks_position('top')  # x轴移到顶部
        ax.yaxis.set_ticks_position('left')  # y轴保持在左侧
        # Optional: Add vortex numbers as text label
        if text:
            for i, (x, y) in enumerate(self.vortex_positions*self.pixel_size):
                plt.text(x, y, str(i+1), ha='center', va='center', color='black',fontsize=6)
        plt.tight_layout()
        plt.show()

    def analyze_vortex_neighbors(self):
        """分析涡旋点的邻居关系，计算配位数和最近邻距离"""
        # 获取去重后的涡旋位置
        points = self.vortex_positions
        
        # 计算Delaunay三角剖分
        from scipy.spatial import Delaunay
        tri = Delaunay(points)
        
        # 初始化配位数数组和最近邻距离数组
        coordination_number = np.zeros(len(points))
        vnv=tri.vertex_neighbor_vertices
        # 计算每个点的配位数和最近邻距离
        for i in range(len(points)):
            # 获取相邻三角形索引
            neighbors = vnv[1][vnv[0][i]:vnv[0][i+1]]
            coordination_number[i] = len(neighbors)
        
        # 定义边缘距离阈值
        edge_threshold = 0.5
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        
        # 确定边缘点
        is_edge_point = ((points[:, 0] - x_min) < edge_threshold) | \
                       ((x_max - points[:, 0]) < edge_threshold) | \
                       ((points[:, 1] - y_min) < edge_threshold) | \
                       ((y_max - points[:, 1]) < edge_threshold)
        
        # 根据配位数和是否为边缘点分类
        neighbor_points = {
            4: points[(coordination_number == 4) & ~is_edge_point],
            5: points[(coordination_number == 5) & ~is_edge_point],
            6: points[(coordination_number == 6) & ~is_edge_point],
            7: points[(coordination_number == 7) & ~is_edge_point],
            8: points[(coordination_number == 8) & ~is_edge_point]
        }
        
        # 绘图
        plt.figure()
        plt.triplot(points[:, 0], points[:, 1], tri.simplices, color='#0072BD', linewidth=1)
        
        markers = {
            4: ('s', 'k'),  # 正方形, 黑色
            5: ('p', 'r'),  # 五角星, 红色
            6: ('h', 'g'),  # 六角形, 绿色
            7: ('v', 'b'),  # 倒三角, 蓝色
            8: ('^', 'm')   # 正三角, 洋红
        }
        
        for num, (marker, color) in markers.items():
            if len(neighbor_points[num]) > 0:
                plt.scatter(neighbor_points[num][:, 0], neighbor_points[num][:, 1],
                          marker=marker, color=color, s=100)
        
        plt.axis('equal')
        plt.axis('off')
        plt.show()
        
        return {
            'coordination_numbers': coordination_number,
            'neighbor_points': neighbor_points
        } 

if __name__=='__main__':
    # 使用示例
    import glob
    from vortex import Vortex


    images = glob.glob("./Fieldcold/*.jpg")
    vortex=Vortex(images[1],(20,20),lambda_=2)
    vortex.detect_vortices(1,threshold=130,inverse=False,show=False)
    vortex.calc_pinning_force()
    vortex.analyze_vortex_neighbors()
    # vortex.draw_pinning_force(text=True)