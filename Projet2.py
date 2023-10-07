# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 20:48:38 2022

@author: Mounir
"""
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import toolkit


##############################  CALIBRAGE DE LA CAMERA ###################################
#Le code du TP 4 


# Defining the dimensions of checkerboard
CHECKERBOARD = (7,9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# max number of iterations=30

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = [] 
error =[]


# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

nx = 7
#Enter the number of inside corners in y
ny = 9
# Extracting path of individual image stored in a given directory
images = glob.glob('C:/Users/Mounir/Desktop/calibration_imagesM/*.jpeg')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
   
    if ret == True:
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
        
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("img", 600, 400)  
    cv2.imshow('img',img)
    cv2.waitKey(0)

cv2.destroyAllWindows()

h,w = img.shape[:2]
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("ret: \n")
print(ret)
print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)




###################      OUVERTURE DES DEUX IMAGES + MATCHES   #########################################



img1 = cv2.imread('C:/Users/Mounir/Desktop/PROJET 2 CV/3.jpeg')
img2 = cv2.imread('C:/Users/Mounir/Desktop/PROJET 2 CV/4.jpeg')

im1 = cv2.imread('C:/Users/Mounir/Desktop/PROJET 2 CV/3.jpeg', cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread('C:/Users/Mounir/Desktop/PROJET 2 CV/4.jpeg' , cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT_create()

gray_train = im1
gray_query = im2


plt.figure(figsize=(100,100))
fig, (ax1, ax2) = plt.subplots(1, 2)


ax1.imshow(gray_train,cmap='gray')
ax2.imshow(gray_query,cmap='gray')

kp1, desc1 = sift.detectAndCompute(gray_train, None)
kp2, desc2 = sift.detectAndCompute(gray_query, None)


result2 = cv2.drawKeypoints(gray_query, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
result1 = cv2.drawKeypoints(gray_train, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(result2,cmap='gray')
ax2.imshow(result1,cmap='gray')

for _,keypoint in enumerate(kp1[:4]):
  x = keypoint.pt[0]
  y = keypoint.pt[1]
  s = keypoint.size
  print("""*Keypoint {} :
        x = {:.2f},
        y  = {:.2f},
        kp_size = {:.2f}""".format(_,x,y,s))
  
        
##################### MATCHES ########################################

matches = cv2.BFMatcher().knnMatch(desc2,desc1, k=2)

good = []
pts1=[]
pts2=[]
for m,n in matches:
    if m.distance < 0.8*n.distance:
        good.append([m])

comparaison = cv2.drawMatchesKnn(gray_query,kp2,gray_train,kp1,good,None,flags=2)

plt.imshow(comparaison)
cv2.imwrite("C:/Users/Mounir/Desktop/PROJET 2 CV/comparaison.jpeg",comparaison)

#Sift et matching des points d'interets 
#Calcul des coordonnees homogennes (pour la matrice essentielle E)
pts1, pts2= toolkit.find_correspondence_points(img1, img2)

points1 = toolkit.cart2hom(pts1)
points2 = toolkit.cart2hom(pts2)


#Affichage des points d'interets sur les images
fig, ax = plt.subplots(1, 2)
ax[0].autoscale_view('tight')
ax[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
ax[0].plot(points1[0], points1[1], 'r.')
ax[1].autoscale_view('tight')
ax[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
ax[1].plot(points2[0], points2[1], 'r.')
fig.show()


intrinsic = np.array([ 
       [1176.27875 ,0 , 582.739112],
       [0, 1172.88712 ,790.434461],
       [0 ,0,1]])  
        


#######################  CALCUL DES POINTS 3D #################################



# 1) Calculer la matrice essentielle E avec les points 2D

# Premierement la normalisation des points
points1n = np.dot(np.linalg.inv(intrinsic), points1)
points2n = np.dot(np.linalg.inv(intrinsic), points2)
E = toolkit.compute_essential_normalized(points1n, points2n)


# On prend la camera 1 comme referentiel 
# On se retrouve avec 4 matrices 
P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
P2s = toolkit.compute_P_from_essential(E)

ind = -1
for i, P2 in enumerate(P2s):
    # Trouvons la bonne matrice parmi les 4
    d1 = toolkit.reconstruct_one_point(
        points1n[:, 0], points2n[:, 0], P1, P2)

    # Convertir les coordonnees de P2 en coordonnees du monde 
    P2_homogenous = np.linalg.inv(np.vstack([P2, [0, 0, 0, 1]]))
    d2 = np.dot(P2_homogenous[:3, :4], d1)

    if d1[2] > 0 and d2[2] > 0:
        ind = i

P2 = np.linalg.inv(np.vstack([P2s[ind], [0, 0, 0, 1]]))[:3, :4]
tripoints3d = toolkit.linear_triangulation(points1n, points2n, P1, P2)

fig = plt.figure()
fig.suptitle('3D reconstructed', fontsize=16)
ax = fig.gca(projection='3d')

ax.plot(tripoints3d[0], tripoints3d[1], tripoints3d[2], 'b.')

ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_zlabel('z axis') 

ax.view_init(elev=254, azim=95)

plt.show()


