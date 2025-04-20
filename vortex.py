import numpy as np
from sklearn.cluster import DBSCAN
import cv2
import glob



def local_contrast(img, window_size=10):
    """计算局部对比度"""
    if(len(img.shape)==3):
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32)
    min_img=cv2.erode(img,np.ones((window_size,window_size)))
    max_img=cv2.dilate(img,np.ones((window_size,window_size)))
    img=255*(img-min_img)/(max_img-min_img+1e-5)
    # img=img.astype(np.uint8)
    # cv2.imshow('img',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img


def build_gaussian_pyramid(image, num_octaves=4, scales_per_octave=5, sigma=1.6,local_contrast_flag=False):
    """构建高斯金字塔"""
    pyramid = []
    k = 2 ** (1.0 / scales_per_octave)  # 尺度倍增系数
    
    for octave in range(num_octaves):
        octave_images = []
        sigma_total = sigma
        if local_contrast_flag:
            local_contrast_image=local_contrast(image,window_size=int(sigma_total*5))
        else:
            local_contrast_image=image
        for s in range(scales_per_octave + 3):  # 每octave多生成3幅用于极值检测
            if  s == 0:
                octave_images.append(local_contrast_image.astype(np.float32))
            else:
                sigma_effective = sigma_total * np.sqrt(k**2 - 1)
                blurred = cv2.GaussianBlur(octave_images[-1], (0, 0), sigmaX=sigma_effective)
                octave_images.append(blurred)
            sigma_total *= k
        pyramid.append(octave_images)
        
        # 下采样准备下一octave
        if octave < num_octaves - 1:
            image = cv2.resize(octave_images[-3], 
                             (octave_images[-3].shape[1]//2, octave_images[-3].shape[0]//2),
                             interpolation=cv2.INTER_NEAREST)
    return pyramid

def build_dog_pyramid(gaussian_pyramid,scales_per_octave=5,sigma=1.6):
    """构建高斯差分金字塔"""
    k=2**(1.0/scales_per_octave)
    dog_pyramid = []
    for octave in gaussian_pyramid:
        dog_octave = []
        sigma_total=sigma*k
        for i in range(1, len(octave)):
            dog = octave[i] - octave[i-1]
            dog_octave.append(dog)
            sigma_total*=k
        dog_pyramid.append(dog_octave)
    return dog_pyramid


#gittest
# 在detect_vortices_by_convolution中替换create_vortex_kernel调用为：
def detect_vortices_by_convolution(image_path, min_radius=2, max_radius=7,color_threshold=0.5,split=0.7,more_precise=3,erosion=0,inverse=False,local_contrast=False,watershad=False):
    """
    使用卷积和聚类方法在图像中检测涡旋。

    参数:
        image_path (str): 输入图像的路径。
        min_radius (int, 可选): 检测到的涡旋的最小半径。默认为 20。
        max_radius (int, 可选): 检测到的涡旋的最大半径。默认为 50。
        color_threshold (float, 可选): 基于颜色过滤涡旋的阈值。默认为 0.5。
        split (float, 可选): 计算 DBSCAN 聚类半径的参数。默认为 0.7。
        more_precise (int, 可选): DBSCAN 聚类所需的最小样本数。默认为 7。
        inverse (bool, 可选): 是否反转阈值处理。默认为 False。

    返回:
        list: 一个元组列表，表示检测到的涡旋的坐标。
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Error: unable to load image.")
        return
    if len(image.shape)==3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray=image.astype(np.float32)
    if inverse:
        gray=255-gray
    # clahe=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # gray=clahe.apply(gray).astype(np.float32)
    # gray=local_contrast(gray,window_size=15)
    # cv2.imshow('img',gray.astype(np.uint8))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # 构建金字塔
    num_octave=int(round(np.log(min(gray.shape)) / np.log(2) - 1))
    scales_per_octave=3
    sigma=1.6
    if local_contrast:
        gaussian_pyramid = build_gaussian_pyramid(gray, num_octaves=num_octave,scales_per_octave=scales_per_octave,sigma=sigma,local_contrast_flag=True)
    else:
        gaussian_pyramid = build_gaussian_pyramid(gray, num_octaves=num_octave,scales_per_octave=scales_per_octave,sigma=sigma,local_contrast_flag=False)
    dog_pyramid = build_dog_pyramid(gaussian_pyramid)


    # 在多尺度上检测涡旋
    vortices = []
    for octave_idx, dog_octave in enumerate(dog_pyramid):
        scale = 2 ** octave_idx  # 当前octave的尺度因子
        for layer_idx, dog in enumerate(dog_octave):
            max_scale_radius=4*sigma*(2**((layer_idx+1)/scales_per_octave))
            # 归一化并二值化
            norm_dog = cv2.normalize(dog, None, 0,255, cv2.NORM_MINMAX).astype(np.uint8)
            threshold, binary = cv2.threshold(norm_dog,0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            dist_transform = cv2.distanceTransform(binary,cv2.DIST_L2,5)
            ret, binary = cv2.threshold(dist_transform,erosion/scale,255,0)
            binary = binary.astype(np.uint8)
            
            # 寻找连通区域
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours=list(contours)
            for contour in contours:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                if radius<1 or radius<min_radius/scale:
                    continue
                #分水岭算法分割涡旋簇
                elif watershad and radius>max_scale_radius and radius>max_radius/scale:
                    mask=np.zeros_like(norm_dog)
                    cv2.drawContours(mask,[contour],-1,255,-1)
                    mask[mask==255]=norm_dog[mask==255 ]
                    thresholds=np.linspace(threshold,255,40)
                    for thresh in thresholds[:10]:
                        _,binary2=cv2.threshold(mask,thresh,255,cv2.THRESH_BINARY)
                        new_contours, _ = cv2.findContours(binary2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if len(new_contours)>1:
                            contours.extend(list(new_contours))
                            contour=[]
                            break
                    continue

                # region=norm_dog[int(max(x-radius,0)):int(min(x+radius,norm_dog.shape[1])), int(max(y-radius,0)):int(min(y+radius,norm_dog.shape[0]))]
                # sig = np.max(region)
                x, y, radius = int(x*scale), int(y*scale), int(radius*scale)
                if min_radius < radius < max_radius and gray[y,x]<(np.max(gray)-np.min(gray))*color_threshold:  # 过滤不合理的大小
                    # cv2.circle(image, (x, y), radius, (0, int(sig), 0), 2)
                    vortices.append((x, y))

    
    if len(vortices) > 0:
        # 转换为numpy数组
        points = np.array(vortices)
        
        # 使用DBSCAN聚类，eps是邻域半径，min_samples是最小样本数
        clustering = DBSCAN(eps=(split*min_radius+(1-split)*2*max_radius), min_samples=more_precise).fit(points)
        
        # 计算每个簇的中心点
        clustered_vortices = []
        for label in set(clustering.labels_):
            if label != -1:  # 忽略噪声点
                cluster_points = points[clustering.labels_ == label]
                center = np.mean(cluster_points, axis=0).astype(int)
                clustered_vortices.append(tuple(center))
        
        # 用聚类后的中心点替换原始点
        vortices = clustered_vortices
        
        # 在图像上绘制聚类后的中心点
        for (x, y) in clustered_vortices:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)  # 用红色实心圆标记聚类中心

    
    # 显示结果
    # cv2.imshow("Detected Vortices", image)
    return vortices
    
if __name__=='__main__':
    # 使用示例
    images = glob.glob("./Fieldcold/*.png")
    detect_vortices_by_convolution(images[1],min_radius=2,max_radius=8,color_threshold=0.5,split=0.6 ,more_precise=1,erosion=0,inverse=True,watershad=True)  # 可调整阈值
    cv2.waitKey(0)
    cv2.destroyAllWindows()