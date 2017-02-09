import os
import numpy as np
import cv2
#import glob
import pickle
import matplotlib.pyplot as plt


mtx = None
dist = None

#
# Load pickled camera matrix and distortion coefficients
# Returns: camera matrix 'mtx' and distortion coefficients 'dist'.
#
def camera_calibration_params():
    with open('camera_dist_pickle.p', mode='rb') as f:
        dist_pickle = pickle.load(f)
        mtx = dist_pickle["mtx"]
        dist = dist_pickle["dist"]
    return mtx, dist


#
# Undistort input image, using camera matrix and distortion coeffs.
# Param: img - source image.
# Returns: undistorted image
#
def undistort_img(img):
    assert(mtx is not None)
    assert(dist is not None)
    return cv2.undistort(img, mtx, dist, None, mtx)


#
# Image binarization.
# Param: img - source RGB image
# Returns: binary image 
#
def binarize(img,
             s_thresh=(90, 255),
             l_thresh=(40, 255),
             sx_thresh=(20, 100), ksize_sx=3#11
            ):    
    # Convert to HLS color space and separate the L & S channels
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=ksize_sx) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    # Threshold lightness
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1
    
    binary = np.zeros_like(l_binary)
    binary[(l_binary == 1) & (s_binary == 1) | (sxbinary == 1)] = 1
    
    #kernel = np.ones((3, 3), binary.dtype)
    # remove white blobs
    #binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    # fill black holes
    #binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
#    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    color_binary = np.dstack((l_binary, sxbinary, s_binary))
    binary = np.dstack(( binary, binary, binary))
    
    return color_binary, binary  


def init():
    global mtx, dist
    mtx, dist = camera_calibration_params()