import cv2
import numpy as np 

img = cv2.imread('train-10/Highway/image_0019.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
sift = cv2.xfeatures2d.SIFT_create()
kp,des = sift.detectAndCompute(gray,None) 
kp = kp[:30]
des = des[:30]
#print('kp: ',kp)
print('len(kp): ',len(kp))
#print('des: ',des)
print('des.shape: ',des.shape)

img=cv2.drawKeypoints(gray,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('sift_keypoints.jpg',img)
























































































































