import cv2
import numpy as np
from matplotlib import pyplot as plt
import queue
from numpy.core.fromnumeric import amax
from scipy import ndimage
import copy


img = cv2.imread('araras1.png')
another_img = cv2.imread('araras2.png')




def get_labels(img, ramo):
    #cv2.imshow('img_blue', img_blue)
    #cv2.waitKey(0)
    img_blue = img[:,:,0]
    img_red = img[:,:,2]
    #cv2.destroyAllWindows()
    #cv2.imshow('img_red', img_red)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    ret,thresh = cv2.threshold(img_gray,170,255,cv2.THRESH_BINARY)

    cv2.imshow('img', 255 - thresh)
    cv2.imwrite("./results/questao3/thresh.png", 255 -thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    thresh = np.abs(thresh - 255)


    #cv2.imshow('img', img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    num_labels, labels_im = cv2.connectedComponents(thresh, connectivity=8, ltype=cv2.CV_16U)

    N = 1000

    araras1 = np.zeros((labels_im.shape[0], labels_im.shape[1]))
    araras2 = np.zeros((labels_im.shape[0], labels_im.shape[1]))

    list_araras = [araras1, araras2]
    label_counter = 1

    for i in range(1, num_labels+1):
        pts =  np.where(labels_im == i)

        if len(pts[0]) < N:
            labels_im[pts] = 0
        else:
            labels_im[pts] = label_counter
            list_araras[label_counter - 1][pts] = 255
            label_counter+=1

    ##MOSTRANDO ARARAS DO LABEL 1
    cv2.imshow('img', list_araras[0])
    cv2.imwrite("./results/questao3/"+ramo+"label1.png", list_araras[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #
    #MOSTRANDO ARARAS DO LABEL 2
    cv2.imshow('img', list_araras[1])
    cv2.imwrite("./results/questao3/"+ramo+"label1.png", list_araras[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #TIRANDO A DIFERENÇA DAS ARARAS
    araras_azuis = list_araras[1] - np.uint8(img_blue)
    araras_vermelhas = list_araras[0] - np.uint8(img_red)
    blue = -1

    red = -1

    blue = []
    red = []
    for i in range(len(list_araras)):

        #cv2.imshow('img', list_araras[i])
        #
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        diff_blue = list_araras[i] - img_blue

        result_blue = np.mean(np.where(diff_blue >= 0, diff_blue, 0))
        diff_red = list_araras[i] - np.uint8(img_red)

        result_red = np.mean(np.where(diff_red >= 0, diff_red, 0))
        #print(result_blue,result_red)
        if(result_blue < result_red):
            blue.append(i)
        if(result_red < result_blue):
            red.append(i)
    img1 = copy.deepcopy(img)
    img2 = copy.deepcopy(img)

    for k in blue:
        for i in range(img1.shape[0]):
            for j in range(img1.shape[1]):
                if(list_araras[k][i][j] == 255):
                    img1[i,j,0] = 255
                    img1[i,j,1] = 0
                    img1[i,j,2] = 0
    
    for k in red:
        for i in range(img1.shape[0]):
            for j in range(img1.shape[1]):
                if(list_araras[k][i][j] == 255):
                    img2[i,j,0] = 0
                    img2[i,j,1] = 0
                    img2[i,j,2] = 255

    #MOSTRANDO ARARAS AZUIS
    cv2.imshow('img', img1)
    cv2.imwrite("./results/questao3/"+ramo+"azuis.png", img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #MOSTRANDO ARARAS VERMELHAS
    cv2.imshow('img', img2)
    cv2.imwrite("./results/questao3/"+ramo+"vermelhas.png", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

get_labels(img, "frame1")
get_labels(another_img, "frame2")