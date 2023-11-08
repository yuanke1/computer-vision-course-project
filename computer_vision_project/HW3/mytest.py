import cv2
import numpy as np
import os

def normalize8(I): #转换成uint8图像
  mn = I.min()
  mx = I.max()
  mx -= mn
  I = ((I - mn)/mx) * 255
  return I.astype(np.uint8)

trainresult=cv2.imread(r'./model/train_result.pgm',0) #读取训练结果
trainresult=trainresult.astype(np.float64)

people=input('请选择第几个人(1到40)：')

testimage=cv2.imread(r'./archive/s'+str(people)+'/9.pgm',0) #读取测试图
testimage=testimage.astype(np.float64)
a=testimage

meanface=cv2.imread(r'./model/meanface.pgm',0) #读取平均脸
meanface=meanface.astype(np.float64)

testimage=testimage.reshape(112*92,1)
testimage=testimage-meanface

y=np.dot(testimage.T,trainresult) #识别
testimage_f=(np.dot(y,trainresult.T)).T #重构
'''
y=np.dot(trainresult.T,testimage) #识别
testimage_f=(np.dot(trainresult,y)) #重构
'''
#显示重构后的人脸
testimage_f=testimage_f.reshape(112,92)
testimage_f=normalize8(testimage_f)
#testimage_f=normalize8(testimage_f+meanface.reshape(112,92))
a=normalize8(a)
testimage_add_testimagef=normalize8(testimage_f+a*0.9)
show=np.hstack((a,testimage_add_testimagef))
cv2.imshow('face after recognition',show)
cv2.waitKey(0)
cv2.destroyAllWindows()

#识别最相似图像

def distance(a,b,trainresult):
  return np.linalg.norm(np.dot(a.T,trainresult)-np.dot(b.T,trainresult))

testface=cv2.imread(r'./archive/s'+str(people)+'/9.pgm',0)
testface= cv2.equalizeHist(testface)
testface=testface.reshape(112*92,1)
testface=testface.astype(np.float64)

path=r'./archive'
file_num = 40
train_num = 8
file=1
num=1
min=1000000000
for i in range(1, file_num + 1):
  person_path = path + '/s' + str(i)
  for j in range(1, train_num + 1):
    face_path = person_path + '/' + str(j) + '.pgm'
    image = cv2.imread(face_path, 0)
    image = cv2.equalizeHist(image)
    image = image.reshape(92 * 112, 1)
    image=image.astype(np.float64)
    if(distance(testface,image,trainresult)<min):
      min=distance(testface,image,trainresult)
      file=i
      num=j


print('找到的人为第'+str(file)+'个人')

image_find=cv2.imread(r'./archive/s'+str(file)+'/'+str(num)+'.pgm',0)
show=np.hstack((normalize8(testface.reshape(112,92)),image_find))
cv2.imshow('face_finded',show)
cv2.waitKey(0)
cv2.destroyAllWindows()









'''
testface=np.dot((a.reshape(112*92,1)).T,trainresult)

def preprocess(path):  # 图像预处理
  T = np.zeros((92 * 112, 40 * 8))
  file_num = 40
  train_num = 8
  count = 0
  for i in range(1, file_num + 1):
    person_path = path + '/s' + str(i)
    for j in range(1, train_num + 1):
      face_path = person_path + '/' + str(j) + '.pgm'
      image = cv2.imread(face_path, 0)
      image = cv2.equalizeHist(image)
      image = image.reshape(92 * 112, 1)
      T[:, [count]] = image
      count = count + 1
  return T

def mean_image(image):  # 去均值
  mean = np.mean(image, axis=1)
  for i in range(40 * 8):
    image[:, [i]] = image[:, [i]] - mean[:, np.newaxis]
  return image

T=preprocess(r'./archive')
#train=mean_image(T)
train=T
train=train.astype(np.float64)

minDistance = np.linalg.norm(np.dot(train[:,[0]].T,trainresult)-testface,ord=1)

print(np.linalg.norm(np.dot(train[:,[20]].T,trainresult)-testface,ord=1))

file=1
num=1
for i in range(1,320):
  distance=np.linalg.norm(np.dot(train[:,[i]].T,trainresult)-testface,ord=1)
  if minDistance > distance:
    minDistance=distance
    file=i//10+1
    num=i%10+1

print(file)
print(num)

image_find=cv2.imread(r'./archive/s'+str(file)+'/'+str(num)+'.pgm',0)
show=np.hstack((normalize8(a.reshape(112,92)),image_find))
cv2.imshow('find face',show)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
