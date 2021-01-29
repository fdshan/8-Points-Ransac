'''
Install opencv:
pip install opencv-python==3.4.2.16
pip install opencv-contrib-python==3.4.2.16
'''

import cv2
import numpy as np
import random as rm
from matplotlib import pyplot as plt
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--UseRANSAC", type=int, default=0 )

# data/921919841_a30df938f2_o.jpg
# data/4191453057_c86028ce1f_o.jpg

# data/7433804322_06c5620f13_o.jpg
# data/9193029855_2c85a50e91_o.jpg
parser.add_argument("--image1", type=str,  default='data/myleft.jpg' )
parser.add_argument("--image2", type=str,  default='data/myright.jpg' )
args = parser.parse_args()

print(args)

#reference: http://www.cs.cmu.edu/~16385/s17/Slides/12.4_8Point_Algorithm.pdf
def FM_by_normalized_8_point(pts1,  pts2):
    #F, _ = cv2.findFundamentalMat(pts1,pts2,  cv2.FM_8POINT )
    # comment out the above line of code. 
    
    # Your task is to implement the algorithm by yourself.
    # Do NOT copy&paste any online implementation. 
    # add one column: [x y] -> [x y 1]
    addone = np.ones(len(pts1))
    pts1 = np.c_[pts1, addone]
    pts2 = np.c_[pts2, addone]
    # F:  fundmental matrix
    #normalize the points
    #The origin of the new coordinate system should be centered (have its origin) at the centroid (center of gravity) of the image points
    x1 = np.sum(pts1[:,0])
    y1 = np.sum(pts1[:,1])
    x2 = np.sum(pts2[:,0])
    y2 = np.sum(pts2[:,1])
    
    # center1, center2 is the centroid of pts1, pts2
    center1 = [x1/len(pts1), y1/len(pts1)]
    center2 = [x2/len(pts2), y2/len(pts2)]
    
    # mean distance from centroid
    # pts1
    sum1 = 0
    for i in range(len(pts1)):
        sum1 += np.sqrt((pts1[i, 0] - center1[0]) ** 2 + (pts1[i, 1] - center1[1]) ** 2)

    mean_distance1 = sum1 / len(pts1)

    # pts2
    sum2 = 0
    for i in range(len(pts2)):
        sum2 += np.sqrt((pts2[i, 0] - center2[0]) ** 2 + (pts2[i, 1] - center2[1]) ** 2)

    mean_distance2 = sum2 / len(pts2)
    
    # construct a 3 * 3 matrix to translate the points
    # mean distance from the origin to a point equals sqrt(2)
    T1= np.array([[np.sqrt(2) / mean_distance1, 0, -center1[0] * (np.sqrt(2) / mean_distance1)], 
        [0, np.sqrt(2) / mean_distance1, -center1[1] * (np.sqrt(2) / mean_distance1)], 
        [0, 0, 1]])
    
    T2= np.array([[np.sqrt(2) / mean_distance2, 0, -center2[0] * (np.sqrt(2) / mean_distance2)], 
        [0, np.sqrt(2) / mean_distance2, -center2[1] * (np.sqrt(2) / mean_distance2)], 
        [0, 0, 1]])

    # normalize the points
    pts1 = T1 @ pts1.T
    pts2 = T2 @ pts2.T

    pts1 = pts1.T
    pts2 = pts2.T

    #construct matrix A
    A = np.ones((len(pts1), 9))
    
    index = 0
    for i in range(0, len(pts1)):
        #xi*xi'
        A[i][0] = pts1[index][0]*pts2[index][0]
        #xi*yi'
        A[i][1] = pts1[index][0]*pts2[index][1]
        #xi
        A[i][2] = pts1[index][0]
        #yi*xi'
        A[i][3] = pts1[index][1]*pts2[index][0]
        #yi*yi'
        A[i][4] = pts1[index][1]*pts2[index][1]
        #yi
        A[i][5] = pts1[index][1]
        #xi'
        A[i][6] = pts2[index][0]
        #yi'
        A[i][7] = pts2[index][1]

        index += 1

    #find the SVD of ATA
    #reference: https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
    # in 2D case, the rows of VH are the eigenvector of ATA
    U, S, V = np.linalg.svd(A)
    V = V.T

    #Entries of F are the elements of column of V corresponding to the least singular value
    F = V[:, 8]
    F = F.reshape(3, 3)
    F = F.T

    #Enforce rank 2 constraint on F
    U, S, V = np.linalg.svd(F)
    S = np.diag([S[0], S[1], 0])
    F = U @ S @ V
    
    #de-normalize F
    F = T2.T @ F @ T1

    # normalizes the matrix so that F[2,2] = 1
    F = F / F[2][2]

    return  F

def FM_by_RANSAC(pts1,  pts2):
    #F, mask = cv2.findFundamentalMat(pts1,pts2,  cv2.FM_RANSAC )	
    # comment out the above line of code. 
	
    # Your task is to implement the algorithm by yourself.
    # Do NOT copy&paste any online implementation. 
	# sample the number of points required to fit the model
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    
    n = 0
    cur_mask = np.zeros(155)
    for i in range(0, len(pts1)):
        # choose 8 pairs of matching points randomly
        num = rm.sample(range(0, len(pts1)), 8)
        pts1List = []
        for i in range(0, 8):
            pts1List.append(pts1[num[i]])
        
        # do the same thing for pts2, match points
        pts2List = []
        for i in range(0, 8):
            pts2List.append(pts2[num[i]])
        

        # Fi, fundamental matrix
        cur_F = FM_by_normalized_8_point(pts1List, pts2List)
        cur_num = 0

        # add one column: [x y] -> [x y 1]
        addone = np.ones(len(pts1))
        pts1 = np.c_[pts1, addone]
        pts2 = np.c_[pts2, addone]
        
        # compute the number of inliners ni, with respect to Fi
        for j in range(0, len(pts1)):
            # fitting line
            # line = [a b c]
            line = cur_F @ pts1[j].T
            a = line[0]
            b = line[1]
            c = line[2]
            # compute the distance between point and line
            dis = np.absolute(a * pts2[j, 0] + b * pts2[j, 1] + c) / np.sqrt(a ** 2 + b ** 2)

            # if the distance < threshold, treat as inlier, mask == 1
            threshold = 1.45
            if dis < threshold:
                cur_mask[j] = 1
                cur_num += 1
        #If ni > n
        #   n = ni
        #   F = Fi
        if cur_num > n:
            n = cur_num
            F = cur_F
            mask = cur_mask

        # delete one column: [x y 1] -> [x y]
        pts1 = np.delete(pts1, -1, axis = 1)
        pts2 = np.delete(pts2, -1, axis = 1)

    # F:  fundmental matrix
    # mask:   whetheter the points are inliers
    return  F, mask

	
img1 = cv2.imread(args.image1,0) 
img2 = cv2.imread(args.image2,0)  

sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

good = []
pts1 = []
pts2 = []

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)
		
		
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

F = None
if args.UseRANSAC:
    F,  mask = FM_by_RANSAC(pts1,  pts2)
    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]    
else:
    F = FM_by_normalized_8_point(pts1,  pts2)

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2
	
	
# Find epilines corresponding to points in second image,  and draw the lines on first image
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,  F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img6)
plt.show()

# Find epilines corresponding to points in first image, and draw the lines on second image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
plt.subplot(121),plt.imshow(img4)
plt.subplot(122),plt.imshow(img3)
plt.show()
