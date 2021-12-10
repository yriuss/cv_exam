import cv2
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, pi, cos, sin
from numpy.core.fromnumeric import shape
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter
from collections import defaultdict
from canny import canny_edge_detector
from PIL import Image, ImageDraw

ddept = cv2.CV_16S
img = cv2.imread('./imagens_teste/aaa.png')



def resize(img, scale_percent = 60):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

img1 = resize(img, 50)
cv2.imshow('img', img1)
cv2.imwrite('./results/questao2/input.png', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


blur_img = cv2.medianBlur(img,7)

red_img = np.zeros(shape=blur_img[:,:,2].shape)

for i in range(blur_img.shape[0]):
    for j in range(blur_img.shape[1]):
        if(int(blur_img[i,j,2]) - int(blur_img[i,j,0]) > 30 and int(blur_img[i,j,2]) - int(blur_img[i,j,1]) > 30):
            red_img[i,j] = blur_img[i,j,2]
red_img = resize(red_img, 50)
img = resize(img, 50)


cv2.imshow('img', red_img)
cv2.imwrite('./results/questao2/img2.png', red_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

red_img = np.uint8(red_img)
edges = cv2.Canny(red_img, 150, 200)


#ret,thresh = cv2.threshold(red_img,200,255,cv2.THRESH_BINARY)

rows = gray.shape[0]

ret,edges = cv2.threshold(edges,200,255,cv2.THRESH_BINARY)

cv2.imshow('img', edges)
cv2.imwrite('./results/questao2/img3.png', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp = 1, minDist=59, param1=100, param2=30, minRadius=30, maxRadius = 150)

#circles = detectCircles(edges, 400,  )

#hough_radii = np.arange(100, 200, 1)
#hough_res = hough_circle(edges, hough_radii)
#
#print(hough_res)
#accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                           #total_num_peaks=3)
#circles = []
#for i in range(len(cx)):
#    circles.append([cx,cy,radii])

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        # circle center
        cv2.circle(img, center, 1, (0, 100, 100), 3)
        # circle outline
        radius = i[2]
        cv2.circle(img, center, radius, (0, 255, 0), 3)

cv2.imshow('img', img)
cv2.imwrite('./results/questao2/img4.png', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.imread('./imagens_teste/aaa.png')
img = resize(img, 50)
max_idx = np.argmax(circles[0,:,2])
circles = circles[:,max_idx,:][0]
center = (circles[0], circles[1])
# circle center
cv2.circle(img, center, 1, (0, 100, 100), 3)
# circle outline
radius = circles[2]
cv2.circle(img, center, radius, (0, 255, 0), 3)


cv2.imshow('img', img)
cv2.imwrite('./results/questao2/img5.png', img)
cv2.waitKey(0)
cv2.destroyAllWindows()



#cv2.imshow('img', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#cv2.imwrite("input.png", red_img)
#
#input_image = Image.open("input.png")
## Find circles
#rmin = 100
#rmax = 202
#steps = 100
#threshold = 0.4
#
#acc = defaultdict(int)
#
#points = []
#for r in range(rmin, rmax + 1):
#    for t in range(steps):
#        points.append((r, int(r * cos(2 * pi * t / steps)), int(r * sin(2 * pi * t / steps))))
#
#
#for x in range(edges.shape[0]):
#    for y in range(edges.shape[1]):
#        if(edges[x,y] > 0):
#            for r, dx, dy in points:
#                a = x - dx
#                b = y - dy
#                acc[(a, b, r)] += 1
#
#
#
#circles = []
#for k, v in sorted(acc.items(), key=lambda i: -i[1]):
#    x, y, r = k
#    if v / steps >= threshold and all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc in circles):
#        circles.append((x, y, r))
#print(circles)
#if circles is not None:
#    for circle in circles:
#        center = (circle[0],circle[1])
#        print(center)
#        # circle center
#        cv2.circle(img, center, 1, (0, 100, 100), 3)
#        # circle outline
#        radius = circle[2]*2
#        print(radius)
#        cv2.circle(img, center, radius, (0, 255, 0), 3)
#    
#cv2.imshow('img', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()