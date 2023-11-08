import cv2
import numpy as np
import os
#from matplotlib import pyplot as plt
'''
def fit(path):
    filelist = os.listdir(path)
    for item in filelist:
        if item.endswith('.jpg'):  #取出图片
            item=path+'/'+item
            img=cv2.imread(item)  #读取图片
            img_test = cv2.Canny(img, 200, 100, 1)  # Canny边缘检测

            ret, thresh = cv2.threshold(img_test, 127, 255, cv2.THRESH_BINARY)  # 图片二值化
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 返回contours
            for cnt in contours:
                if len(cnt) > 0.8:  # 检测长度是否大于一定值
                    ell = cv2.fitEllipse(cnt)  # 检测椭圆
                    img = cv2.ellipse(img, ell, (255, 255, 0), 2)  # 绘制
            cv2.imshow("img", img)  # 显示图片
            cv2.waitKey(0)

fit(r"./")
'''




img=cv2.imread(r".\image2.jpg") #导入图片
img_test=cv2.Canny(img,200,100,1) #Canny边缘检测

ret,thresh = cv2.threshold(img_test,127,255,cv2.THRESH_BINARY) #图片二值化
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)#返回contours
for cnt in contours:
    if len(cnt)>0.8: #检测长度是否大于一定值
        ell=cv2.fitEllipse(cnt) #检测椭圆
        img = cv2.ellipse(img, ell, (255, 255, 0), 2) #绘制
cv2.imshow("0",img) #显示函数
cv2.waitKey(0)
