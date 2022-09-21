import cv2
import numpy as np
import os
import glob
import json
import sys
import time
import metrics
    
# Return horizon contour line for input image
def getContourLine(img):
    # Apply blurring, thresholding and morphological transformations to identify land
    # Gaussian blur to smooth image. 
    img_gauss = cv2.GaussianBlur(img, (15,15), 9)  # HARDCODED
    
    # Otsu threshold to binarize image. We are using blue channel only
    thr, img_otsu = cv2.threshold(img_gauss[:,:,0], thresh=0, maxval=1,
        type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    land_mask = img_otsu-1
    
    # Morphological functions to reduce noise
    opening = cv2.morphologyEx(land_mask, cv2.MORPH_OPEN, (21,21)) # HARDCODED
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, (21,21)) # HARDCODED
    
    # Use land mask to detect horizon contour
    # Canny edge detection
    image_canny_closing = cv2.Canny(image=land_mask, threshold1=100, threshold2=200) # HARDCODED
    
    # Identify longest contour (land contour)
    contours, hierarchy = cv2.findContours(image_canny_closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key = len)
    #Source: https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
    arclen = cv2.arcLength(contour, True)
    # Smooth contour line
    cnt = cv2.approxPolyDP(contour, 0.002*arclen, False)
    # Create binary image with just contour line
    img_cont = np.zeros_like(image_canny_closing)
    img_cont = cv2.drawContours(img_cont,[cnt],0,255,2)
    return img_cont, land_mask


# Apply Hough Transform, score hough lines, and return best fitting horizon line
def getHorizonLineCoords(img_contours, land_mask):
    # Draw hough lines on smoothened horizon contour. Source: https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
    lines = cv2.HoughLines(img_contours,75,np.pi/180,3) # HARDCODED
    all_coordinates = []
    all_cost = []
    
    # Select the best two hough lines, score them to isolate the line that minimizes cost
    for i in range(2):
        coordinates, cost = metrics.houghScoring(land_mask, lines[i], alpha=1.0)
        all_coordinates.append(coordinates)
        all_cost.append(cost)
        
    x1, y1, x2, y2 = all_coordinates[np.argmin(all_cost)]
    return [x1, y1, x2, y2]


# Bring it all together
def detectHorizon(input_path, output_dir, label=None):
    '''
    Inputs
    input_path: path to single input image
    output_dir: directory where all output images are saved
    label(optional): ground truth in json format
    
    Outputs
    loss: list of Euclidean Distance per image
    Also saves a copy of the input image with the horizon line in the output directory
    '''
    
    img = cv2.imread(input_path)
    img_out = img.copy()
    img_name = os.path.basename(input_path)
    output_path = os.path.join(output_dir, img_name)
    
    # Apply spatial filters to make edge detection better
    # land_mask = applySpatialFilters(img)
    # Edge detection plus simplified land mask detection
    img_contours, land_mask = getContourLine(img)
    # Score and draw best fitting Hough line
    horizon_line_xy = getHorizonLineCoords(img_contours, land_mask)        
    
    # Save image to file
    cv2.line(img_out,(horizon_line_xy[0],horizon_line_xy[1]),(horizon_line_xy[2],horizon_line_xy[3]),(0,0,255),2)
    cv2.imwrite(output_path, img_out)
    
    # Return evaluation metrics if ground truth is found
    if label:
        true_coords = label[img_name]["left"] + label[img_name]["right"]
        loss = metrics.calcEuclidianDistance(true_coords, horizon_line_xy)
        return loss 


