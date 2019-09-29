import numpy as np
import cv2
import sys
import os
import pandas as pd
from pathlib import Path

ImageFolder = "C:\\Users\Family\PycharmProjects\Hyundai\data\IMAGE"

THRESHOLD_CONTOUR_UPPER = 500
THRESHOLD_CONTOUR_MIN = 10
WIDTH_BG = 15
HEIGHT_BG = 7
WIDTH_D = 21
HEIGHT_D = 9
ORIGIN_POINT_X = None
ORIGIN_POINT_Y = None

def findDotPos(WIDTH, HEIGHT, file):
    TARGET_NUMBER = WIDTH * HEIGHT
    # 1. Load img
    img = cv2.imread(file)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imblur = cv2.GaussianBlur(imgray, (5, 5), 0)
    ret, threshold = cv2.threshold(imblur, 50, 255, cv2.THRESH_BINARY)

    # 2. Contour info.
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = np.array(contours)

    # 3. contour and position info.
    PositionList = []
    ContourList = []
    Num_Spot = 0

    # 4. Store appropriate contours and position info. of elements
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > THRESHOLD_CONTOUR_UPPER:
            continue
        elif area < THRESHOLD_CONTOUR_MIN:
            continue
        Num_Spot += 1
        M = cv2.moments(contour)
        X = int(M["m10"] / M["m00"])
        Y = int(M["m01"] / M["m00"])
        PositionList.append([X, Y])
        ContourList.append(contour)

    # Terminate if detection was unsuccessful
    if Num_Spot != TARGET_NUMBER:
        print("The spot detection was failed!!!!\n")
        print(file)
        #sys.exit(True)

    #########################################################################################
    # 5. point 보정
    # 5-1. 전체 점 정보 (PREPROCESSED - numpy the position info.)
    PositionList = np.array(PositionList)
    # 5-1. PROCESS the position info. w.r.t. the x-axis.
    Sorted_by_X = PositionList[np.lexsort((PositionList[:,1], PositionList[:,0]))]
    #print(Sorted_by_X)
    # 5-2. 가운데 열 좌표정보
    MID_COL = Sorted_by_X[int(HEIGHT*((WIDTH-1)/2)):int(HEIGHT*((WIDTH-1)/2))+HEIGHT, :]
    # 5-3. 가운데 높이 점을 원점으로.
    MID_COL_Sorted_Y = MID_COL[np.lexsort((MID_COL[:,0], MID_COL[:,1]))]
    ORIGIN_POINT = MID_COL_Sorted_Y[int((HEIGHT-1)/2), :]
    # 5-4. 원점 지정.
    ORIGIN_POINT_X = ORIGIN_POINT[0]
    ORIGIN_POINT_Y = ORIGIN_POINT[1]
    # 5-6. 순서보정 (x축 기준(Sorted_by_X)을 또 y축 기준으로)
    Sorted_by_X_Y = np.zeros(shape=(0, 2))
    for i in range(WIDTH):
        block = Sorted_by_X[i*HEIGHT:(i+1)*HEIGHT,:]
        block_by_Y = block[np.lexsort((block[:,0], block[:,1]))]
        block_by_Y = np.array(block_by_Y) # shape: (HEIGHT,2)
        for j in range(HEIGHT):
            Sorted_by_X_Y = np.vstack((Sorted_by_X_Y, block_by_Y[j,:]))
    #print(Sorted_by_X_Y)
    # Sorted_by_X_Y : X -> Y 순으로 정렬된 점들의 좌표 값.

    # 5-6-2. 가로로 순서 보정
    Reorder_X_wise = np.zeros(shape=(HEIGHT*WIDTH, 2))
    for i in range(HEIGHT*WIDTH):
        if i % HEIGHT == 0:
            quotient = int(i / HEIGHT)
            for j in range(i, i+HEIGHT):
                if j == i:
                    Reorder_X_wise[quotient] = Sorted_by_X_Y[i]
                elif j != i:
                    Reorder_X_wise[quotient+WIDTH*(j-i)] = Sorted_by_X_Y[j]
    #print("*"*50)
    #print(Reorder_X_wise)
    Correct_by_origin = np.zeros(shape=(HEIGHT*WIDTH,2))

    for p in range(WIDTH*HEIGHT):
        x = Reorder_X_wise[p][0] - ORIGIN_POINT_X
        y = Reorder_X_wise[p][1] - ORIGIN_POINT_Y
        Correct_by_origin[p] = [x, y]
    #print("*"*50)
    #print(Correct_by_origin) # 최종적으로 순서 및 원점이동 완료된 좌표값.
    #########################################################################################


    # Draw the contours.
    cv2.drawContours(img, ContourList, -1, (0, 255, 0), 1)
    # Draw the 'origin point'.
    cv2.circle(img, (ORIGIN_POINT[0], ORIGIN_POINT[1]), 3, (0, 0, 255), -1)
    # Show the result.
    cv2.imshow('Contours and origin point', img)
    cv2.waitKey(0)

    return Correct_by_origin

findDotPos(WIDTH_D, HEIGHT_D, "21.jpg")
def StoreCSV():
    for base, dirs, files in os.walk(ImageFolder):
        # B차종(15x7)
        if '_B' in base:
            for file in files:
                if "21.j" in file:
                    # Get prev. img info
                    B_prev_img_info = findDotPos(WIDTH_BG, HEIGHT_BG, os.path.join(base, "21.jpg"))
                    df = pd.DataFrame(B_prev_img_info)
                    with open("B_21.csv", 'a') as BP:
                        df.to_csv(BP, header=None, index=False, encoding='utf-8')
                elif "211.j" in file:
                    # Get ans. img info
                    print(dir)
                    B_aft_img_info = findDotPos(WIDTH_BG, HEIGHT_BG, os.path.join(base, "211.jpg"))
                    df = pd.DataFrame(B_aft_img_info)
                    with open("B_211.csv", 'a') as BA:
                        df.to_csv(BA, header=None, index=False, encoding='utf-8')
        # G차종(15x7)
        elif "_G" in base:
            for file in files:
                if "21.j" in file:
                    G_prev_img_info = findDotPos(WIDTH_BG, HEIGHT_BG, os.path.join(base, "21.jpg"))
                    df = pd.DataFrame(G_prev_img_info)
                    with open("G_21.csv", 'a') as GP:
                        df.to_csv(GP, header=None, index=False, encoding='utf-8')
                elif "211.j" in file:
                    G_aft_img_info = findDotPos(WIDTH_BG, HEIGHT_BG, os.path.join(base, "211.jpg"))
                    df = pd.DataFrame(G_aft_img_info)
                    with open("G_211.csv", 'a') as GA:
                        df.to_csv(GA, header=None, index=False, encoding='utf-8')
        # D차종(21x9)
        elif "_D" in base:
            for file in files:
                if "21.j" in file:
                    D_prev_img_info = findDotPos(WIDTH_D, HEIGHT_D, os.path.join(base, "21.jpg"))
                    df = pd.DataFrame(D_prev_img_info)
                    with open("D_21.csv", 'a') as DP:
                        df.to_csv(DP, header=None, index=False, encoding='utf-8')
                elif "211.j" in file:
                    D_aft_img_info = findDotPos(WIDTH_D, HEIGHT_D, os.path.join(base, "211.jpg"))
                    df = pd.DataFrame(D_aft_img_info)
                    with open("D_211.csv", 'a') as DA:
                        df.to_csv(DA, header=None, index=False, encoding='utf-8')

#if __name__ == "__main__":
#    StoreCSV()