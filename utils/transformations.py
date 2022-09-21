import cv2
import numpy as np
import os
import glob
import json
import sys
import time
sys.path.append("../utils/")
import metrics

def applySpatialFilters(img):
    img_gauss = cv2.GaussianBlur(img, (15,15),9)
    thr, img_otsu = cv2.threshold(img_gauss[:,:,0], thresh=0, maxval=1,
        type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    land_mask = img_otsu-1
    opening = cv2.morphologyEx(land_mask, cv2.MORPH_OPEN, (21,21))
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, (21,21))
    return closing

def getContourLine(land_mask):
    image_canny_closing = cv2.Canny(image=land_mask, threshold1=100, threshold2=200)
    contours, hierarchy = cv2.findContours(image_canny_closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key = len)
    arclen = cv2.arcLength(contour, True)
    cnt = cv2.approxPolyDP(contour, 0.002*arclen, False)
    img_cont = np.zeros_like(image_canny_closing)
    img_cont = cv2.drawContours(img_cont,[cnt],0,255,2)
    return img_cont, land_mask

def getHorizonLineCoords(img_contours, land_mask):
    lines = cv2.HoughLines(img_contours,75,np.pi/180,3)
    all_coordinates = []
    all_cost = [] 
    for i in range(2):
        coordinates, cost = metrics.houghScoring(land_mask, lines[i], alpha=1.0)
        all_coordinates.append(coordinates)
        all_cost.append(cost)
    x1, y1, x2, y2 = all_coordinates[np.argmin(all_cost)]
    return [x1, y1, x2, y2]
    
def detectHorizon(input_path, output_dir, label = None):
    img = cv2.imread(input_path)
    img_out = img.copy()
    img_name = os.path.basename(input_path)
    output_path = os.path.join(output_dir, img_name)
    
    #apply spatial filters to make edge detection better
    land_mask = applySpatialFilters(img)
    #edge detection plus simplified land mask detection
    img_contours, land_mask = getContourLine(land_mask)
    #score and draw best fitting Hough line
    horizon_line_xy = getHorizonLineCoords(img_contours, land_mask)        
    
    #save image to file
    cv2.line(img_out,(horizon_line_xy[0],horizon_line_xy[1]),(horizon_line_xy[2],horizon_line_xy[3]),(0,0,255),2)
    cv2.imwrite(output_path, img_out)
    
    #return evaluation metrics if ground truth is found
    if label:
        true_coords = label[img_name]["left"] + label[img_name]["right"]
        loss = metrics.calcEuclidianDistance(true_coords, horizon_line_xy)
        return loss 


