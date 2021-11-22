# %% [markdown]
# # Official Implement of Canny detector with openCV
# 
# - [OpenCV doc](https://docs.opencv.org/4.5.1/da/d5c/tutorial_canny_detector.html)
# 
# - [Github code](https://github.com/opencv/opencv/blob/master/samples/python/tutorial_code/ImgTrans/canny_detector/CannyDetector_Demo.py)

# %%
from __future__ import print_function
import cv2 as cv
import argparse
from matplotlib import pyplot as plt
max_lowThreshold = 100
window_name = 'Edge Map'
title_trackbar = 'Min Threshold:'
ratio = 3
kernel_size = 3

def CannyThreshold(val):
    low_threshold = val
    # img_blur = cv.blur(src_gray, (3,3))
    # img_blur = cv.blur(src_gray, (21,21))
    img_blur=cv.GaussianBlur(src_gray,(51,51),10)
    detected_edges = cv.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
    mask = detected_edges != 0
    dst = src * (mask[:,:,None].astype(src.dtype))
    cv.imshow(window_name, dst)
    # cv.imshow(window_name,mask[:,:,None].astype(src.dtype))
    # plt.imshow(mask)

parser = argparse.ArgumentParser(description='Code for Canny Edge Detector tutorial.')
# parser.add_argument('--input', help='Path to input image.', default='fruits.jpg')
parser.add_argument('--input', help='Path to input image.', default='LenaSoderberg.jpg')
args = parser.parse_args()

src = cv.imread(cv.samples.findFile(args.input))
if src is None:
    print('Could not open or find the image: ', args.input)
    exit(0)

src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

cv.namedWindow(window_name)
cv.createTrackbar(title_trackbar, window_name , 0, max_lowThreshold, CannyThreshold)

CannyThreshold(0)
cv.waitKey()

# %%



