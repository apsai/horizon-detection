import cv2
import numpy as np
import os
import glob
import json
import sys
import time
sys.path.append("../utils/")
import metrics

def getContourLine(img):
    img_gauss = cv2.GaussianBlur(img, (15,15),9)
    thr, img_otsu = cv2.threshold(img_gauss[:,:,0], thresh=0, maxval=1,
        type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    land_mask = img_otsu-1
    opening = cv2.morphologyEx(land_mask, cv2.MORPH_OPEN, (21,21))
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, (21,21))
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
    
def drawHorizonLine(input_folder, output_folder):
    start_time = time.time()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    label_json_path = os.path.join(os.path.dirname(input_folder), "ground_truth.json")
    try:
        with open(label_json_path, "r") as f:
            label = json.load(f)
    except Exception as e:
        label = None
        print("Evaluation metrics cannot be calculated.")
    img_paths = glob.glob(input_folder + "/frame*")    

    loss = []
    
    for img_path in img_paths:
        img = cv2.imread(img_path)
        img_out = img.copy()
        img_name = os.path.basename(img_path)
        output_path = os.path.join(output_folder, img_name)
        label_name = os.path.basename(os.path.dirname(img_path))
        img_contours, land_mask = getContourLine(img)
        horizon_line_xy = getHorizonLineCoords(img_contours, land_mask)        
        if label:
            true_coords = label[img_name]["left"] + label[img_name]["right"]
            loss.append(metrics.calcEuclidianDistance(true_coords, horizon_line_xy))
        cv2.line(img_out,(horizon_line_xy[0],horizon_line_xy[1]),(horizon_line_xy[2],horizon_line_xy[3]),(0,0,255),2)
        cv2.imwrite(output_path, img_out)
    time_taken = time.time() - start_time
    print("Processing done")
    print(f'Time Taken to process {len(img_paths)} images is {time_taken}')    
    if label is not None:
        metrics.printEvaluationMetrics(loss)



