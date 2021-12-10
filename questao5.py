
import cv2
import numpy as np
from matplotlib import pyplot as plt
   
      
def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if( x>=0 and y >=0):
            if img[x][y] >= center:
                new_value = 1
        else:
            if(center == 0):
                new_value = 1
    except:
        if(center == 0):
            new_value = 1
    return new_value
   
# Function for calculating LBP
def lbp_calculated_pixel(img, x, y):
    
    center = img[x][y]
   
    val_ar = []
      
    # top_left
    val_ar.append(get_pixel(img, center, x-1, y-1))
      
    # top
    val_ar.append(get_pixel(img, center, x-1, y))
      
    # top_right
    val_ar.append(get_pixel(img, center, x-1, y + 1))
      
    # right
    val_ar.append(get_pixel(img, center, x, y + 1))
      
    # bottom_right
    val_ar.append(get_pixel(img, center, x + 1, y + 1))
      
    # bottom
    val_ar.append(get_pixel(img, center, x + 1, y))
      
    # bottom_left
    val_ar.append(get_pixel(img, center, x + 1, y-1))
      
    # left
    val_ar.append(get_pixel(img, center, x, y-1))
    power_val = [1, 2, 4, 16, 128, 64 ,32 ,8]
    val = 0
    if(x == 1079 and y == 0):
        print(val_ar)
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
          
    return val

img  = np.zeros((1080,1080))

for i in range(1080):
    for j in range(1080):
        if(i < 640 and j < 640 and i >=220 and j>=220):
            img[i][j] = 1

lbp = np.zeros((1080,1080))
corners = []
for i in range(1080):
    for j in range(1080):
        lbp[i, j] = lbp_calculated_pixel(img, i, j)
        if(lbp[i, j] == 248):
            print(lbp[i, j])
        if(lbp[i,j] == 22 or lbp[i,j] == 208 or lbp[i,j] == 104 or lbp[i,j] == 11):
            corners.append((i,j))



img = cv2.cvtColor(img.astype('float32'),cv2.COLOR_GRAY2RGB)

cv2.imshow('img', img)
cv2.imwrite('./results/questao5/img1.png', img*255)
cv2.waitKey(0)
cv2.destroyAllWindows()

for corner in corners:
    cv2.circle(img, corner, radius=3, color=(0, 255, 0), thickness=-1)

cv2.imshow('img', img)
cv2.imwrite('./results/questao5/img2.png', img*255)
cv2.waitKey(0)
cv2.destroyAllWindows()

