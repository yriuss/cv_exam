from os import remove
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.linear_model import LinearRegression

ddept = cv2.CV_16S
img = cv2.imread('./imagens_teste/img2.png')

def calculate_pline_dist(x1, y1, a, b, c):
    return abs((a * x1 + b * y1 + c)) / (math.sqrt(a * a + b * b))

def get_line_parameters(x1,y1, x2,y2):
    m = (y2-y1)/(x2-x1)
    b = 1
    a = -m
    c = -(y1 - m*x1)
    return a, b, c


def filter_lines(lines):
    new_line = []
    max_line_dist = 0.0
    min_dist = 490
    theta_max = 0
    #PEGA INCLINAÇÃO DA MAIOR RETA
    for i in range(lines.shape[0]):
        x1, y1, x2, y2 = lines[i][0][0], lines[i][0][1], lines[i][0][2], lines[i][0][3]
        dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        if(max_line_dist < dist and dist < 2000):
            theta_max = math.asin((y2-y1)/np.sqrt((x2-x1)**2 + (y2-y1)**2))
            max_line_dist = max([max_line_dist, dist])
    
    #FILTRA AS RETAS COM BASE EM UMA DISTÂNCIA MÍNIMA E NA INCLINAÇÃO DA MAIOR RETA
    for i in range(lines.shape[0]):
        x1, y1, x2, y2 = lines[i][0][0], lines[i][0][1], lines[i][0][2], lines[i][0][3]
        theta = math.asin((y2-y1)/np.sqrt((x2-x1)**2 + (y2-y1)**2))
        dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        if(np.abs(theta - theta_max) < 0.3 and dist > min_dist):
            new_line.append(lines[i][0].tolist())
    
    new_line = np.array(new_line)
    new_line = new_line[new_line[:, 2].argsort()]
    
    last_analysis = []
    counter = 0
    for line1 in new_line:
        
        dist = []
        line_inc = []
        
        alone = True
        for i in range(new_line.shape[0]):
            x0, y0, x1, y1 = line1
            x2, y2, x3, y3 = new_line[i]
            a,b,c = get_line_parameters(x2,y2, x3,y3)
            d = calculate_pline_dist(x1, y1, a, b, c)
            if(d < 40):
                dist.append(np.sqrt((x3-x2)**2 + (y3-y2)**2))
                line_inc.append(new_line[i].tolist())
                alone = False
        counter+=1
        
        if(dist!=[] and (not alone)):
            idx = np.argmax(dist)
            last_analysis.append([line_inc[idx]])
        else:
            last_analysis.append([line1])
    
    counter = 1
    result = []
    for line1 in last_analysis:
        condition = True
        for i in range(len(last_analysis)):
            if(counter+i >= len(last_analysis)):
                break
            if(line1 == last_analysis[i+counter]):
                condition = False
        counter+=1
        if(condition == True):
            result.append(line1[0])
    arr = np.array(result)
    arr = arr[arr[:, 3].argsort()]
    print(arr)
    print("O número de andares é: ", count_floors(arr))
    return np.array(result)

def count_floors(lines):
    count = 0
    started = False
    stoped = False
    line_count = 0
    for line in lines:
        for i in range(len(lines)):
            x0, y0, x1, y1 = line
            x2, y2, x3, y3 = lines[i]
            a,b,c = get_line_parameters(x2,y2, x3,y3)
            d = calculate_pline_dist(x1, y1, a, b, c)

            if(d> 1 and d < 100 and started and not stoped and line_count > 1):
                stoped = True
                count +=1

            if(d < 0.0001 and started and not stoped and line_count > 1):
                count+=1
            
            if(d > 60 and d < 80 and line_count > 0):
                started = True
            
        line_count +=1
    return count

def resize(img, scale_percent = 60):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

resized1 = resize(img)
resized = resize(img)
cv2.imshow('img', resized)
cv2.imwrite('./results/questao1/input.png', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

x = cv2.Sobel(gray, ddept, 1,0, ksize=3, scale=0.9)
y = cv2.Sobel(gray, ddept, 0,1, ksize=3, scale=0.9)

absx= cv2.convertScaleAbs(x)
absy = cv2.convertScaleAbs(y)
edges = cv2.addWeighted(absx, 0.5, absy, 0.5,0)


ret,edges = cv2.threshold(edges,30,255,cv2.THRESH_BINARY)

#edges = cv2.Canny(gray, 100, 200)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 490, maxLineGap=30)


cv2.imshow('edge', edges)
cv2.imwrite('./results/questao1/img2.png', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(resized1, (x1, y1), (x2, y2), (0, 255, 0), 1)

cv2.imshow('edge', resized1)
cv2.imwrite('./results/questao1/img3.png', resized1)
cv2.waitKey(0)
cv2.destroyAllWindows()

lines = filter_lines(lines)



if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(resized, (x1, y1), (x2, y2), (0, 255, 0), 1)



cv2.imshow('edge', resized)
cv2.imwrite('./results/questao1/img4.png', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

#def convolution(image, kernel, average=False, verbose=False):
#    if len(image.shape) == 3:
#        print("Found 3 Channels : {}".format(image.shape))
#        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#        print("Converted to Gray Channel. Size : {}".format(image.shape))
#    else:
#        print("Image Shape : {}".format(image.shape))
# 
#    print("Kernel Shape : {}".format(kernel.shape))
# 
#    if verbose:
#        plt.imshow(image, cmap='gray')
#        plt.title("Image")
#        plt.show()
# 
#    image_row, image_col = image.shape
#    kernel_row, kernel_col = kernel.shape
# 
#    output = np.zeros(image.shape)
# 
#    pad_height = int((kernel_row - 1) / 2)
#    pad_width = int((kernel_col - 1) / 2)
# 
#    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
# 
#    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image
# 
#    if verbose:
#        plt.imshow(padded_image, cmap='gray')
#        plt.title("Padded Image")
#        plt.show()
# 
#    for row in range(image_row):
#        for col in range(image_col):
#            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
#            if average:
#                output[row, col] /= kernel.shape[0] * kernel.shape[1]
# 
#    print("Output Image size : {}".format(output.shape))
# 
#    if verbose:
#        plt.imshow(output, cmap='gray')
#        plt.title("Output Image using {}X{} Kernel".format(kernel_row, kernel_col))
#        plt.show()
# 
#    return output
#
#def sobel_edge_detection(image, filter, verbose=False):
#    new_image_x = convolution(image, filter, verbose)
# 
#    if verbose:
#        plt.imshow(new_image_x, cmap='gray')
#        plt.title("Horizontal Edge")
#        plt.show()
# 
#    new_image_y = convolution(image, np.flip(filter.T, axis=0), verbose)
# 
#    if verbose:
#        plt.imshow(new_image_y, cmap='gray')
#        plt.title("Vertical Edge")
#        plt.show()
# 
#    gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))
# 
#    gradient_magnitude *= 255.0 / gradient_magnitude.max()
# 
#    if verbose:
#        plt.imshow(gradient_magnitude, cmap='gray')
#        plt.title("Gradient Magnitude")
#        plt.show()
# 
#    return gradient_magnitude
# 
#
#def gaussian_kernel(size, sigma=1, verbose=False):
# 
#    kernel_1D = np.linspace(-(size // 2), size // 2, size)
#    for i in range(size):
#        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
#    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
# 
#    kernel_2D *= 1.0 / kernel_2D.max()
# 
#    if verbose:
#        plt.imshow(kernel_2D, interpolation='none',cmap='gray')
#        plt.title("Image")
#        plt.show()
# 
#    return kernel_2D
#def dnorm(x, mu, sd):
#    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)
#
#
#def gaussian_blur(image, kernel_size, verbose=False):
#    kernel = gaussian_kernel(kernel_size, sigma=math.sqrt(kernel_size), verbose=verbose)
#    return convolution(image, kernel, average=True, verbose=verbose)
#
#filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
#image = gaussian_blur(resized, 9, verbose=False)
#sobel_edge_detection(resized, filter, verbose=True)