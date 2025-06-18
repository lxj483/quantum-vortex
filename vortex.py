import numpy as np
from scipy.ndimage import maximum_filter
import cv2
import glob



def build_gaussian_pyramid(image, scales_per_octave=5, min_radius=5,sigma=1.6):
    """构建高斯金字塔"""
    k = 2 ** (1.0 / scales_per_octave)  # 尺度倍增系数
    image=cv2.GaussianBlur(image,(0,0),sigmaX=min_radius*sigma)
    image=cv2.resize(image,(image.shape[1]//min_radius,image.shape[0]//min_radius),interpolation=cv2.INTER_NEAREST)
    octave_images = np.empty((scales_per_octave+1,*image.shape), dtype=np.float32)
    sigma_total = sigma
    for s in range(scales_per_octave + 1):  # 每octave多生成3幅用于极值检测
        if  s == 0:
            octave_images[0]=(image.astype(np.float32))
        else:
            sigma_effective = sigma_total * np.sqrt(k**2 - 1)
            octave_images[s] = cv2.GaussianBlur(octave_images[s-1], (0, 0), sigmaX=sigma_effective)
        sigma_total *= k
        # 下采样准备下一octave
    return octave_images

def build_dog_pyramid(octave,scales_per_octave):
    """构建高斯差分金字塔"""
    k=2**(1.0/scales_per_octave)
    dog_octave = np.zeros((scales_per_octave,*octave.shape[1:]), dtype=np.float32)
    for i in range(1, len(octave)):
        dog_octave[i-1] = octave[i] - octave[i-1]
    min_val=np.min(dog_octave)
    max_val=np.max(dog_octave)
    dog_octave=(dog_octave-min_val)/(max_val-min_val+1e-5)*255
    return dog_octave

def detect_vortices_by_convolution(image_path, min_radius=2, inverse=False, threshold=100):
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
    # 构建金字塔
    scales_per_octave=5
    sigma=1
    gaussian_pyramid = build_gaussian_pyramid(gray, scales_per_octave=scales_per_octave,min_radius=min_radius,sigma=sigma)
    dog_octave = build_dog_pyramid(gaussian_pyramid,scales_per_octave=scales_per_octave)


    # 在多尺度上检测涡旋
    neighborhood_max=maximum_filter(dog_octave,size=(2*scales_per_octave+1,5,5))
    local_max=((dog_octave==neighborhood_max)&(dog_octave>threshold))
    y_coords,x_coords=np.array(np.where(local_max)[1:])*min_radius
    for vortice in zip(x_coords,y_coords):
        cv2.circle(image, vortice, 2, (0, 255, 0), -1)  # 用红色实心圆标记聚类中心
    cv2.imshow('img2',image)
    return 
    

    
if __name__=='__main__':
    # 使用示例
    images = glob.glob("./Fieldcold/*.png")
    detect_vortices_by_convolution(images[1],2,threshold=100,inverse=True)  # 可调整阈值
    cv2.waitKey(0)
    cv2.destroyAllWindows()