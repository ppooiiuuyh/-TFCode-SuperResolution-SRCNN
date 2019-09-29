import cv2
import numpy as np

contour_list = []
pos_list = []
def find_dots(img, threshold = 127):
    """ binarization """
    img = ((img > threshold)*255).astype(np.uint8)

    """ find contures """
    contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = np.array(contours)

    """ choose active contures & find dot pos"""
    num_dots_contained_in_contour = []
    for c in contours:
        num_dots_contained_in_contour.append(c.shape[0])
    active_num_dots = np.median(num_dots_contained_in_contour)

    for c in contours:
        c = np.squeeze(c) #(None,1,2) -> (None,2)
        if len(c) > active_num_dots*0.6 and len(c) < active_num_dots*1.5:
            contour_list.append(c)
            pos_list.append( (int(np.mean(c[:,0])), int(np.mean(c[:,1]))))
    print(len(pos_list))

    """ find center pos """
    # there may exist one large conture
    center_pos = None
    for c in contours:
        c = np.squeeze(c) #(None,1,2) -> (None,2)
        if len(c) > active_num_dots*1.5:
            center_square_pos = (int(np.mean(c[:,0])), int(np.mean(c[:,1])))

    # 사각형의 중심점과 가장 가까운 점이 중심점일것
    min_temp = 9999
    for p in pos_list:
        dist = np.sqrt(np.sum((np.array(p) - np.array(center_square_pos)) ** 2))
        if dist < min_temp :
            min_temp = dist
            center_pos = p

    """ visualize for evalution """
    bg = (img * 0.2).astype(np.uint8)
    for ds,p in zip(contour_list,pos_list):
        for d in ds:
            bg[d[1],d[0]] = 255
        bg[p[1],p[0]]=255
        bg[center_pos[1],center_pos[0]]=255
        cv2.imshow("image",bg)
        cv2.waitKey(10)

    return pos_list, center_pos, contour_list

img = cv2.imread("./211.jpg", cv2.IMREAD_GRAYSCALE)
print(len(pos_list))
find_dots(img)