import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

ddept = cv2.CV_16S
img = cv2.imread('./imagens_teste/paredes.png')
img1 = cv2.imread('./imagens_teste/paredes2.png')

def get_line_parameters(x1,y1, x2,y2):
    m = (y2-y1)/(x2-x1)
    b = 1
    a = -m
    c = -(y1 - m*x1)
    return a, b, c

def extract_wall(img_shape, lines):
    left_lines = []
    right_lines = []
    vertical_lines = []
    middle_lines = []

    lines = lines[lines[:, 0].argsort()]
    
    counter =0
    for line in lines:
        x0, y0, x1, y1 = line
        theta = math.asin((y1-y0)/np.sqrt((x1-x0)**2 + (y1-y0)**2))
        if(abs(abs(theta) - np.pi/2) < 0.1):
            vertical_lines.append(line)
        elif(x0 == 0):
            left_lines.append(line)
        elif(x1 == img_shape[1] - 1):
            right_lines.append(line)
        else:
            middle_lines.append(line)
        counter += 1
    

    middle_left_lines = middle_lines[0:2]

    if(len(middle_lines) > 2):
        middle_right_lines = middle_lines[0:-2]
    else:
        middle_right_lines = middle_left_lines
    
    middle_right_lines = np.array(middle_right_lines)
    middle_right_lines = middle_right_lines[middle_right_lines[:, 1].argsort()]
    middle_left_lines = np.array(middle_left_lines)
    middle_left_lines = middle_left_lines[middle_left_lines[:, 1].argsort()]

    left_lines = np.array(left_lines)
    left_lines = left_lines[left_lines[:, 1].argsort()]

    vertical_lines = np.array(vertical_lines)
    vertical_lines = vertical_lines[vertical_lines[:, 0].argsort()]

    right_lines = np.array(right_lines)
    right_lines = right_lines[right_lines[:, 1].argsort()]
    
    x0, y0, x1, y1 = left_lines[0]
    theta1 = math.asin((y1-y0)/np.sqrt((x1-x0)**2 + (y1-y0)**2))

    x2, y2, x3, y3 = right_lines[0]
    theta2 = math.asin((y3-y2)/np.sqrt((x3-x2)**2 + (y3-y2)**2))

    wall = np.zeros(img_shape)
    x4, y4, x5, y5 = right_lines[1]
    x6, y6, x7, y7 = left_lines[1]
    a1,_,c1 = get_line_parameters(x0,y0,x1,y1)
    a2,_,c2 = get_line_parameters(x2,y2,x3,y3)
    a3,_,c3 = get_line_parameters(x4,y4,x5,y5)
    a4,_,c4 = get_line_parameters(x6,y6,x7,y7)

    for y in range(wall.shape[0]):
        for x in range(wall.shape[1]):
            if(theta1 > 0):
                if(x < vertical_lines[0][0] and y > (-a1*x-c1)and y < (-a4*x-c4)):
                    wall[y][x] = 255
            
            if(theta2 < 0):
                if(x > vertical_lines[len(vertical_lines) - 1][0] and y > (-a2*x-c2) and y < (-a3*x-c3) ):
                    wall[y][x] = 255
    

    return wall
    
    


def calculate_pline_dist(x1, y1, a, b, c):
    return abs((a * x1 + b * y1 + c)) / (math.sqrt(a * a + b * b))

def get_line_parameters(x1,y1, x2,y2):
    if(abs(x2 - x1) < 1):
        m = 10000000
    else:
        m = (y2-y1)/(x2-x1)

    b = 1
    a = -m
    c = -(y1 - m*x1)
    return a, b, c

def not_past_line(past_lines, line):
    condition = True
    for i in range(len(past_lines)):
        if(np.array_equal(line, past_lines[i])):
            condition = False
    return condition

def filter_lines(lines):
    new_lines = []
    past_line = []
    count = 0
    for line1 in lines:
        candidate_lines = []
        
        candidate_lines_width = []
        x0, y0, x1, y1 = line1[0]
        
        for i in range(lines.shape[0]):
            theta1 = math.asin((y1-y0)/np.sqrt((x1-x0)**2 + (y1-y0)**2))
            x2, y2, x3, y3 = lines[i][0]
            theta2 = math.asin((y3-y2)/np.sqrt((x3-x2)**2 + (y3-y2)**2))
            a,b,c = get_line_parameters(x2,y2, x3,y3)
            
            width2 = np.sqrt((x3-x2)**2 + (y3-y2)**2)
            d = calculate_pline_dist(x1, y1, a, b, c)
            if(abs(theta2-theta1) < 0.2 and d < 40):
                if(not_past_line(past_line, lines[i][0])):
                    candidate_lines.append(lines[i][0])
                    past_line.append(lines[i][0])
                    candidate_lines_width.append(width2)
        if(candidate_lines_width != []):
            candidate_lines_width = np.array(candidate_lines_width)
            new_lines.append(candidate_lines[np.argmax(candidate_lines_width)])
    return np.array(new_lines)









def resize(img, scale_percent = 60):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

def question(img, parede):
    resized = resize(img)
    resized2 = resize(img)
    resized3 = resize(img)
    cv2.imshow('img', resized)
    cv2.imwrite('./results/questao4/img1'+parede+'.png', resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    x = cv2.Sobel(gray, ddept, 1,0, ksize=3, scale=1.5)
    y = cv2.Sobel(gray, ddept, 0,1, ksize=3, scale=1.5)


    absx= cv2.convertScaleAbs(x)
    absy = cv2.convertScaleAbs(y)
    edges = cv2.addWeighted(absx, 0.5, absy, 0.5,0)

    ret,edges = cv2.threshold(edges,30,255,cv2.THRESH_BINARY)


    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 200, maxLineGap=30)



    cv2.imshow('edge', edges)
    cv2.imwrite('./results/questao4/img2'+parede+'.png', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(resized, (x1, y1), (x2, y2), (0, 255, 0), 1)

    cv2.imshow('edge', resized)
    cv2.imwrite('./results/questao4/img3'+parede+'.png', resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    lines = filter_lines(lines)


    wall = extract_wall(gray.shape, lines)

    for i in range(resized.shape[0]):
        for j in range(resized.shape[1]):
            if(wall[i][j] == 255):
                resized3[i,j,1] = wall[i,j]
                resized3[i,j,0] = 0
                resized3[i,j,2] = 0





    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(resized2, (x1, y1), (x2, y2), (0, 255, 0), 1)

    cv2.imshow('edge', resized2)
    cv2.imwrite('./results/questao4/img4'+parede+'.png', resized2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    cv2.imshow('edge', resized3)
    cv2.imwrite('./results/questao4/img5'+parede+'.png', resized3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

question(img, "parede1")
question(img1, "parede2")



