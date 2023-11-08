import cv2
import numpy as np
import os

def normalize8(I): #转换成uint8图像
  mn = I.min()
  mx = I.max()
  mx -= mn
  I = ((I - mn)/mx) * 255
  return I.astype(np.uint8)

def preprocess(path):  #图像预处理
    T=np.zeros((92*112,40*8))
    file_num = 40
    train_num = 8
    count=0
    for i in range(1, file_num + 1):
        person_path = path + '/s' + str(i)
        for j in range(1, train_num + 1):
            face_path = person_path + '/' + str(j) + '.pgm'
            image=cv2.imread(face_path,0)
            image = cv2.equalizeHist(image)
            image=image.reshape(92*112,1)
            T[:,[count]]=image
            count=count+1
    return T

def mean_image(image): #去均值
    mean=np.mean(image, axis=1)
    cv2.imwrite(r'./model/meanface.pgm', mean)
    for i in range(40*8):
        image[:,[i]]=image[:,[i]]-mean[:, np.newaxis]
    return image

def get_eigen(T,energy_f): #计算协方差矩阵的特征值和特征向量，并根据能量百分比选取特征向量
    # 求协方差矩阵特征向量与特征值
    cov = np.dot(T.T,T)
    cov=cov/320
    eigen_value, eigen_vect = np.linalg.eig(cov)
    # 按照能量值进行排序
    eigen_vect=np.dot(eigen_vect,T.T).T
    #eigen_vect = np.dot(T,eigen_vect)
    sorted = np.argsort(eigen_value[::-1])
    eigen_value = eigen_value[sorted]
    eigen_vect = eigen_vect[:, sorted]
    # 根据能量值确定使用多少张特征脸
    energy_total = sum(eigen_value)
    energy_level = energy_f * energy_total
    energy = 0
    for i in range(eigen_value.shape[0]):
        energy = energy + eigen_value[i]
        if energy > energy_level:
            break;
    print("选取前", i,'特征向量')
    PC_num = i
    eigenface = eigen_vect[:, 0:PC_num]
    return eigenface,PC_num


T=preprocess(r'./archive')
train=mean_image(T)
power=float(input('请输入能量值(小数)：'))
eigenface,PC_num=get_eigen(train,power)

#将结果eigenface写入model文件中

result=normalize8(eigenface)
cv2.imwrite(r'./model/train_result.pgm',result)

#cv2.imwrite(r'./model/train_result.pgm',eigenface)

#显示前十张特征脸
show=eigenface[:,0]
show=show.reshape(112,92)
#show=np.uint8(show)
show=normalize8(show)
for i in range(1,10):
    train_show = eigenface[:,i]
    train_show = train_show.reshape(112, 92)
    #train_show = np.uint8(train_show)
    train_show=normalize8(train_show)
    show=np.hstack((show,train_show))

cv2.imshow('10 of eigenfaces',show)
cv2.waitKey(0)
cv2.destroyAllWindows()


