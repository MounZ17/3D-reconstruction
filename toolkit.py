# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 21:23:06 2022

@author: Mounir
"""
import cv2
import numpy as np


# Authors: alyssaq
#https://github.com/alyssaq/3Dreconstruction
#Code du tp 6

def find_correspondence_points(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()
   
    # REcherche des point d'interêt pour chaque image 
    kp1, des1 = sift.detectAndCompute(
        cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = sift.detectAndCompute(
        cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)
    
    
    # Matching
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.375 * n.distance:
            good.append(m)
   
    src_pts = np.asarray([kp1[m.queryIdx].pt for m in good])
    dst_pts = np.asarray([kp2[m.trainIdx].pt for m in good])

    
    retval, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 100.0)
    mask = mask.ravel()

    
    pts1 = src_pts[mask == 1]
    pts2 = dst_pts[mask == 1]
    
    
    return pts1.T, pts2.T 


##############################################################


def reconstruct_one_point(pt1, pt2, m1, m2):
    """
    produit en croix de pt1 et m1 * X = 0
      
    pt1 x m1 * X  =  pt2 x m2 * X  =  0
    """
    A = np.vstack([
        np.dot(skew(pt1), m1),
        np.dot(skew(pt2), m2)])
    
    U, S, V = np.linalg.svd(A)
    P = np.ravel(V[-1, :4])
    return P / P[3]


def linear_triangulation(p1, p2, m1, m2):
    num_points = p1.shape[1]
    res = np.ones((4, num_points))

    for i in range(num_points):
        A = np.asarray([
            (p1[0, i] * m1[2, :] - m1[0, :]),
            (p1[1, i] * m1[2, :] - m1[1, :]),
            (p2[0, i] * m2[2, :] - m2[0, :]),
            (p2[1, i] * m2[2, :] - m2[1, :])
        ])

        _, _, V = np.linalg.svd(A)
        X = V[-1, :4]
        res[:, i] = X / X[3]

    return res


def compute_epipole(F):
    """ Computes the (right) epipole from a
        fundamental matrix F.
        (Use with F.T for left epipole.)
    """
    # return null space of F (Fx=0)
    U, S, V = np.linalg.svd(F)
    e = V[-1]
    return e / e[2]


def Triangulation(P1, P2, pts1, pts2, K, repeat):
    if not repeat:
        points1 = np.transpose(pts1)
        points2 = np.transpose(pts2)
    else:
        points1 = pts1
        points2 = pts2

    cloud = cv2.triangulatePoints(P1, P2, points1, points2)
    cloud = cloud / cloud[3]

    return points1, points2, cloud


def skew(x):
    """ Create a skew symmetric matrix *A* from a 3d vector *x*.
        Property: np.cross(A, v) == np.dot(x, v)
    :param x: 3d vector
    :returns: 3 x 3 skew symmetric matrix from *x*
    """
    return np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ])


def compute_P_from_essential(E):
    """ Compute the second camera matrix (assuming P1 = [I 0])
        from an essential matrix. E = [t]R
    :returns: list of 4 possible camera matrices.
    """
    U, S, V = np.linalg.svd(E)

    # Ensure rotation matrix are right-handed with positive determinant
    if np.linalg.det(np.dot(U, V)) < 0:
        V = -V

    # create 4 possible camera matrices (Hartley p 258)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    P2s = [np.vstack((np.dot(U, np.dot(W, V)).T, U[:, 2])).T,
          np.vstack((np.dot(U, np.dot(W, V)).T, -U[:, 2])).T,
          np.vstack((np.dot(U, np.dot(W.T, V)).T, U[:, 2])).T,
          np.vstack((np.dot(U, np.dot(W.T, V)).T, -U[:, 2])).T]

    return P2s


def correspondence_matrix(p1, p2):
    p1x, p1y = p1[:2]
    p2x, p2y = p2[:2]

    return np.array([
        p1x * p2x, p1x * p2y, p1x,
        p1y * p2x, p1y * p2y, p1y,
        p2x, p2y, np.ones(len(p1x))
    ]).T

    return np.array([
        p2x * p1x, p2x * p1y, p2x,
        p2y * p1x, p2y * p1y, p2y,
        p1x, p1y, np.ones(len(p1x))
    ]).T


def compute_image_to_image_matrix(x1, x2, compute_essential=False):
    """ Compute the fundamental or essential matrix from corresponding points
        (x1, x2 3*n arrays) using the 8 point algorithm.
        Each row in the A matrix below is constructed as
        [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1]
    """
    A = correspondence_matrix(x1, x2)
    # compute linear least square solution
    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    # constrain F. Make rank 2 by zeroing out last singular value
    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    if compute_essential:
        S = [1, 1, 0] # Force rank 2 and equal eigenvalues
    F = np.dot(U, np.dot(np.diag(S), V))

    return F


def scale_and_translate_points(points):
    """ Scale and translate image points so that centroid of the points
        are at the origin and avg distance to the origin is equal to sqrt(2).
    :param points: array of homogenous point (3 x n)
    :returns: array of same input shape and its normalization matrix
    """
    x = points[0]
    y = points[1]
    center = points.mean(axis=1)  # mean of each row
    cx = x - center[0] # center the points
    cy = y - center[1]
    dist = np.sqrt(np.power(cx, 2) + np.power(cy, 2))
    scale = np.sqrt(2) / dist.mean()
    norm3d = np.array([
        [scale, 0, -scale * center[0]],
        [0, scale, -scale * center[1]],
        [0, 0, 1]
    ])

    return np.dot(norm3d, points), norm3d


def compute_normalized_image_to_image_matrix(p1, p2, compute_essential=False):
    """ Computes the fundamental or essential matrix from corresponding points
        using the normalized 8 point algorithm.
    :input p1, p2: corresponding points with shape 3 x n
    :returns: fundamental or essential matrix with shape 3 x 3
    """
    n = p1.shape[1]
    if p2.shape[1] != n:
        raise ValueError('Number of points do not match.')

    # preprocess image coordinates
    p1n, T1 = scale_and_translate_points(p1)
    p2n, T2 = scale_and_translate_points(p2)

    # compute F or E with the coordinates
    F = compute_image_to_image_matrix(p1n, p2n, compute_essential)

    # reverse preprocessing of coordinates
    # We know that P1' E P2 = 0
    F = np.dot(T1.T, np.dot(F, T2))

    return F / F[2, 2]


def compute_essential_normalized(p1, p2):
    return compute_normalized_image_to_image_matrix(p1, p2, compute_essential=True)




##################################################################################


def cart2hom(arr):
    """ covertir les coordonnees cartesienne en homogennes en ajoutant une ligne de 1
    """
    if arr.ndim == 1:
        return np.hstack([arr, 1])
    return np.asarray(np.vstack([arr, np.ones(arr.shape[1])]))

