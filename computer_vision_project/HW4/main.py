import cv2
import numpy as np

img1_path='image.png'

img=cv2.imread(img1_path)
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
size = gray_img.shape[::-1]
print(gray_img.shape)
w1,h1=(9,6)
# w2,h2=(11,8)

cp_int = np.zeros((w1 * h1, 3), np.float32)
cp_int[:, :2] = np.mgrid[0:w1, 0:h1].T.reshape(-1, 2)
# cp_world: corner point in world space, save the coordinate of corner points in world space.
cp_world = cp_int * 0.02

ret, cp_img = cv2.findChessboardCorners(gray_img, (w1,h1), None)
print(ret,cp_img)
obj_points = []  # the points in world space
img_points = []  # the points in image space (relevant to obj_points)
obj_points.append(cp_world)
img_points.append(cp_img)
# view the corners
cv2.drawChessboardCorners(img, (w1,h1), cp_img, ret)
cv2.imshow('image_after_corner',img)
cv2.waitKey(0)
cv2.imwrite("image_after_corner.jpg",img)

ret,mat_inter,coff_dis,v_rot,v_trans = cv2.calibrateCamera(obj_points, img_points, gray_img.shape[::-1], None, None)

print("内参=",mat_inter)
print("畸变系数=",coff_dis)
print("旋转向量=",v_rot)
print("平移向量=",v_trans)

#矫正图像
newMatrix, roi = cv2.getOptimalNewCameraMatrix(mat_inter, coff_dis, (w1,h1), 1, (w1,h1)) #矫正图像
dst = cv2.undistort(img, mat_inter, coff_dis, None, newMatrix)
cv2.imshow('image_after_calibration',dst)
cv2.waitKey(0)
cv2.imwrite('./image_after_calibration.jpg',dst)

#生成鸟瞰图
point1 = np.array([cp_img[0,:], cp_img[8,:], cp_img[-9,:], cp_img[-1,:]], dtype=np.float32)
point2 = np.array([cp_int[0,:][:-1],cp_int[8,:][:-1],cp_int[-9,:][:-1],cp_int[-1,:][:-1]], dtype=np.float32)
point2[True] = point2[True]*20 + 400

M = cv2.getPerspectiveTransform(point1,point2)

out_img = cv2.warpPerspective(dst, M, size)

cv2.imshow('image_birdeye',out_img)
cv2.waitKey(0)
cv2.imwrite(r'./image_birdeye.jpg',out_img)
