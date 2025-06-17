import numpy as np
import cv2

arr=np.array([1,2,3,4,5,6,7,8,9]).reshape((3,3))
arr[:]=arr.max(axis=0)
print(arr)