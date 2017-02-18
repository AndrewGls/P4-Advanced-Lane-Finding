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
# Returns source points for perspective transformation:
# 4 corners used to transform to bird view.
# Returns: numpy array of 4 2D points.
def birdview_corners():
    # for straight_lines1.jpg
    corners = np.float32([[190,720],   # bottom-left
                          [592,450],   # top-left
                          [689,450],   # top-right
                          [1120,720]]) # bottom-right
    
    # for straight_lines2.jpg
#    corners = np.float32([[190,720],   # bottom-left
#                          [583,456],   # top-left
#                          [698,456],   # top-right
#                          [1120,720]]) # bottom-right
    return corners

    
    
#
# Warps image from camera view to bird view or back to camera view.
# Params: img - source image
# Returns wrapped image and perspective transformation matrix M.
#
def warp_img_M(img, tobird = True):
    src = birdview_corners() # trapezoid in camera space
    
    offset = (140,0) # left/right offset
    
    dst_top_left  = np.array([src[0,0], 0])
    dst_top_right = np.array([src[3,0], 0])
    
    # rectangle in bird-view space
    dst = np.float32([src[0]+offset,        # bottom-left 
                      dst_top_left+offset,  # top-left
                      dst_top_right-offset, # top-right
                      src[3]-offset])       # bottom-right    
    if tobird:
        M = cv2.getPerspectiveTransform(src, dst)
    else:
        M = cv2.getPerspectiveTransform(dst, src)
    
    size_wh = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, size_wh , flags=cv2.INTER_LINEAR)
    
    return warped, M
    
    
#
# Warps image from camera view to bird view or back to camera view.
# Params: img - source image
# Returns wrapped image.
#
def warp_img(img, tobird = True):
    warp,_ = warp_img_M(img, tobird)
    return warp

    
#
# Image binarization.
# Param: img - source RGB image
# Returns: binary image (Gray,Gray,Gray) and channels for visualization 
#
def binarize(img,
             s_thresh=(90, 255),
             l_thresh=(40, 255),#l_thresh=(60, 255),#l_thresh=(40, 255),
#             sx_thresh=(20, 100),
             sx_thresh=(30, 100),
             ksize_sx=3#11
            ):    
    # Convert to HLS color space and separate the L & S channels
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float)
#    s_channel = hsv[:,:,1]
    v_channel = hsv[:,:,2]
    
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
#    binary[(l_binary == 1) & (s_binary == 1) | (sxbinary == 1)] = 1
    binary[(l_binary == 1) & (s_binary == 1) | (sxbinary == 1) | (v_channel > 220)] = 1
    
    #kernel = np.ones((3, 3), binary.dtype)
    # remove white blobs
    #binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    # fill black holes
    #binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack((l_binary, sxbinary, s_binary))
    binary = (np.dstack(( binary, binary, binary))*255.).astype('uint8')
    
    return binary, color_binary

    
#
# Image binarization.
# Params: source RGB image.
# Returns: binarized image as (Gray,Gray,Gray)
#
def binarize_img(img):
    binary,_ = binarize(img)
    return binary

    

#
# Applies an image mask to source image.
# Returns: masked image.
#
def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """    
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

    
    
#
# Applies ROI mask to source image to mask noise
#  at the left and right sides of frame.
# Params: img - source image.
# Returns: masked image
#   
def ROI(img):
    shape = img.shape
    vertices = np.array([[(0,0),(shape[1],0),(shape[1],0),(7*shape[1]/8,shape[0]),
                      (shape[1]/8,shape[0]), (0,0)]],dtype=np.int32)
    mask = region_of_interest(img, vertices)
    return mask    
    
    
#
# Binarize source RGB image.
# Returns: undistorted binarized image with 1 channel, macked by ROI.
#
def binarize_pipeline(img):
    img = undistort_img(img)
    binary = binarize_img(img)
    binary = warp_img(binary)
    binary = ROI(binary)
    binary = binary[:,:,0]

#    img = undistort_img(img)
#    binary = warp_img(img)
#    binary = binarize_img(binary)
#    binary = ROI(binary)
#    binary = binary[:,:,0]

    return binary
    
def binarize_pipeline_ex(img):
    img = undistort_img(img)
    binary = binarize_img(img)
    warped = warp_img(binary)
    warped = ROI(warped)
    warped = warped[:,:,0]
    return binary, warped

#    img = undistort_img(img)
#    warped = warp_img(img)
#    binary = binarize_img(warped)
#    binary = ROI(binary)
#    binary = binary[:,:,0]

#    return warped, binary
    
#
# Initialization: loads camera calibration parameters.
#
def init():
    global mtx, dist
    mtx, dist = camera_calibration_params()
