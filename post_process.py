import cv2
import numpy as np

filename = './predicted_masks/ISIC_0000319_linear_segmentation.png'

def post_process(input_mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    opening = cv2.morphologyEx(input_mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    ret,thresh = cv2.threshold(closing,127,255,0)
    im2,contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt_max_area = 0
    subsidiary_index = -1
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        print(area)
        if area>cnt_max_area:
            cnt_max_area = area
            max_index = i
        elif area>0.3*cnt_max_area:
            subsidiary_index = i
    img = np.zeros(input_mask.shape,dtype=np.uint8)
    if subsidiary_index==-1:
        output_mask = cv2.drawContours(img, [contours[max_index]], 0,(255,255,255), -1)
    else:
        output_mask = cv2.drawContours(img, [contours[max_index],contours[subsidiary_index]], 0,(255,255,255), -1)
    return output_mask


input_mask = cv2.imread(filename)
input_mask = cv2.cvtColor(input_mask,cv2.COLOR_BGR2GRAY)
print(input_mask.shape)
output_mask = post_process(input_mask)
cv2.imshow('frame',output_mask)
cv2.waitKey(1000)

cv2.destroyAllWindows()