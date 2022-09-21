import numpy as np
import cv2


# Calculate cost per Hough line
def houghScoring(ground_image,line, alpha = 1.0):
    # Check if ground image is land mask, if not, fix it
    if np.median(ground_image[:10][:]) == 1:
        ground_image = (ground_image - 1)
    
    # Calculate x,y coordinates of Hough Lines
    coordinates = calcLineParams(line, ground_image.shape)
    hough_image = np.zeros_like(ground_image)
    pts = np.array([[0,0],[ground_image.shape[1],0],(coordinates[2],coordinates[3]),(coordinates[0],coordinates[1])])
    _=cv2.drawContours(hough_image, np.int32([pts]),0, 1, -1)
    
    # Hough line as a binary image
    hough_image = (hough_image - 1)
    
    # Score Hough line
    cost = calcCost(hough_image, ground_image, alpha)
    return coordinates, cost

# Calculate x,y coordinates from polar coordinates of Hough Transform. Source: https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
def calcLineParams(line, image_shape):
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = 0
    y1 = int(y0 + image_shape[0]*(a))
    x2 = image_shape[1]
    y2 = int(y0 - image_shape[0]*(a))
    return [x1, y1, x2, y2]

# Source: McGee et al 2005: Obstacle Detection for Small Autonomous Aircraft using Sky Segmentation
def calcCost(hough_image, ground_image, alpha):
    cost = np.sum(np.where(np.logical_xor(hough_image==0, ground_image==0), 1, 0))
    return cost

# Score predicted horizon line against ground truth horizon line
def calcEuclidianDistance(true_coords,pred_coords):
    eu_dist = np.linalg.norm(np.subtract(true_coords, pred_coords))
    return eu_dist

# Print evaluation metrics to console
def printEvaluationMetrics(loss_list):
    print("Max Euclidean Distance: ", np.max(loss_list))
    print("Mean Euclidean Distance: ", np.mean(loss_list))

