import numpy as np
import cv2

def houghScoring(ground_image,line, alpha = 1.0):
    if np.median(ground_image[:10][:]) == 1:
        ground_image = (ground_image - 1)
    coordinates = calcLineParams(line)
    hough_image = np.zeros_like(ground_image)
    pts = np.array([[0,0],[ground_image.shape[1],0],(coordinates[2],coordinates[3]),(coordinates[0],coordinates[1])])
    _=cv2.drawContours(hough_image, np.int32([pts]),0, 1, -1)
    hough_image = (hough_image - 1)
    cost = calcCost(hough_image, ground_image, alpha)
    return coordinates, cost

def calcLineParams(line):
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = 0
    y1 = int(y0 + 1080*(a))
    x2 = 1920
    y2 = int(y0 - 1080*(a))
    return [x1, y1, x2, y2]

def calcCost(hough_image, ground_image, alpha):
    cost = np.sum(np.where(np.logical_xor(hough_image==0, ground_image==0), 1, 0))
    return cost

def calcEuclidianDistance(true_coords,pred_coords):
    eu_dist = np.linalg.norm(np.subtract(true_coords, pred_coords))
    return eu_dist

def printEvaluationMetrics(loss_list):
    print("Max Euclidean Distance: ", np.max(loss_list))
    print("Mean Euclidean Distance: ", np.mean(loss_list))
