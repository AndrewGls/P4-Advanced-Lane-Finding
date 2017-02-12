#import os
#import tools
#import glob
import cv2
import numpy as np
#import pickle
#import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import scipy
#from scipy import signal



#
# Finding position of left/right lines at the bottom of binary
# warped image via dinding peakcs in hystogram.
# Params: binary - binary warped image.
#         thresh = noise thresholding for detected peacks.
#         centerX_base - expected position of line at the bottom of binary image
#         thresh_dist - expected distance between detected peak and expected position of line centerX_base.
def find_peaks(binary, 
               centerX_base,
               thresh=3000,
               thresh_dist=150,
               sigma=20,
               verbose=False):
 
    # Assuming you have created a warped binary image called "bynary"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary[binary.shape[0]/2:,:], axis=0)
    filtered = scipy.ndimage.filters.gaussian_filter1d(histogram, sigma)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    
    if centerX_base < midpoint:
        # detect peak for left line
        peak_ind = np.argmax(filtered[:midpoint])
        peak_x = peak_ind
    else:
        # detect peak for right line
        peak_ind = np.argmax(filtered[midpoint:])
        peak_x = peak_ind + midpoint

    if verbose:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
        f.tight_layout()

        ax1.set_title('Histrogram of binarized warped image')
        ax1.plot(range(len(histogram)), histogram, 'b', range(len(filtered)), filtered, 'r')
        print('pick value: ', filtered[peak_ind])
        if centerX_base < midpoint:
            print('peak before sigma: ', np.argmax(histogram[:midpoint]))
            ax2.set_title('Histrogram of Left line')
            ax2.plot(histogram[:midpoint], 'r')
        else:
            print('peak before sigma: ', (np.argmax(histogram[midpoint:]) + midpoint))
            ax2.set_title('Histrogram of Right line - midpoint')
            ax2.plot(range(binary.shape[1]-midpoint), histogram[midpoint:] + midpoint, 'g')
        print('peak_x after sigma: ', peak_x)
        plt.show()
    
    if filtered[peak_ind] < thresh:
        peak_x = 0
        if verbose: print('noise: peak value < ', thresh)
    elif np.abs(peak_x - centerX_base) > thresh_dist:
        peak_x = 0
        if verbose: print('too large dist: abs(peak_x - center_x):', np.abs(peak_x-centerX_base))
        
    if verbose: print('*** final peak_x: ', peak_x)
    return peak_x


#
# Finds left or right line using window area, sliding along Y direction.
# Params: binary - binary warped image.
#         x_base - expected center of window for line searching at the bottom of image.
#         margin - sets the width of the windows +/- margin
# Returns: line as two arrays of X coords and Y coords,
#          line_fit as np.array, list of window rectangles if verbose=True.
#
def detect_line(binary, x_base, margin=100, verbose=False):

    # These will be the starting point for the left and right lines
    #midpoint = np.int(binary.shape[1]/2)

    # Visualization of window rectangles
    win_rects = []
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    x_current = x_base
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary.shape[0] - (window+1)*window_height
        win_y_high = binary.shape[0] - window*window_height
        win_x_low = x_current - margin
        win_x_high = x_current + margin
        if verbose:
            win_rects.append(((win_x_low,win_y_low),(win_x_high,win_y_high)))
        # Identify the nonzero pixels in x and y within the window
        good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
        # Append these indices to the lists
        lane_inds.append(good_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_inds) > minpix:
            x_current = np.int(np.mean(nonzerox[good_inds]))

    # Concatenate the arrays of indices
    lane_inds = np.concatenate(lane_inds)

    # Extract left and right line pixel positions
    lanex = nonzerox[lane_inds]
    laney = nonzeroy[lane_inds]
    
    # Fit a second order polynomial to each
    lane_fit = np.empty(shape=(0,0))
    if len(laney):
        lane_fit = np.polyfit(laney, lanex, 2)

    return (lanex, laney), lane_fit, win_rects



#
# Draws detected lanes to an image.
# Params: binary - binary warped image (1 channel).
#         left_linex - x points of left lane in binary image.
#         left_liney - y points of left lane in binary image.
#         right_linex - x points of right lane in binary image.
#         right_liney - y points of right lane in binary image.
#         left_winds - windows for left lane.
#         left_winds - windows for right lane.
# Returns: RGB image, left_fitx, right_fitx, ploty.
#
def draw_lanes_with_windows(binary,
                            left_linex, left_liney, left_fit,
                            right_linex, right_liney, right_fit,
                            left_winds, right_winds):
    
    #print(type(left_fit))
    
    out_img = np.dstack((binary, binary, binary))*255

    left_fitx = []
    right_fitx = []
    ploty = []
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, out_img.shape[0]-1, out_img.shape[0] )
    if len(left_linex):
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    if len(right_linex):
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Draw green window rectangles
    for wl, wr in zip(left_winds, right_winds):
        cv2.rectangle(out_img,wl[0],wl[1],(0,255,0), 2)
        cv2.rectangle(out_img,wr[0],wr[1],(0,255,0), 2)

    if len(left_linex):
        out_img[left_liney, left_linex] = [255,0,0]
    if len(right_linex):
        out_img[right_liney, right_linex] = [0,0,255]
        
    # Cast the x and y points into format compatible with cv2.fillPoly()
    left_pts = None
    right_pts = None
    if len(left_linex):
        left_fitx_pts1 = np.array([np.transpose(np.vstack([left_fitx-2, ploty]))])
        left_fitx_pts2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+2, ploty])))])
        left_pts = np.hstack((left_fitx_pts1, left_fitx_pts2))
    if len(right_linex):
        right_fitx_pts1 = np.array([np.transpose(np.vstack([right_fitx-2, ploty]))])
        right_fitx_pts2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+2, ploty])))])
        right_pts = np.hstack((right_fitx_pts1, right_fitx_pts2))
    # Draw yellow lanes in output image
    if len(left_linex) or len(right_linex):
        window_img = np.zeros_like(out_img)
    if len(left_linex):
        cv2.fillPoly(window_img, np.int_([left_pts]), (255,255, 0))
    if len(right_linex):
        cv2.fillPoly(window_img, np.int_([right_pts]), (255,255, 0))
    if len(left_linex) or len(right_linex):
        out_img = cv2.addWeighted(out_img, 1, window_img, 1., 0)
            
    return out_img
    
    
    
# Line detection in ROI.
# Assume you now have a new warped binary image 
# from the next frame of video (also called "binary_warped")
# It's now much easier to find line pixels!
def detect_line_in_roi(binary, line_fit, margin=100):
    nonzero = binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    line_lane_inds = []
    
    if len(line_fit):
        line_lane_inds = ((nonzerox > (line_fit[0]*(nonzeroy**2) + line_fit[1]*nonzeroy + line_fit[2] - margin)) & (nonzerox < (line_fit[0]*(nonzeroy**2) + line_fit[1]*nonzeroy + line_fit[2] + margin))) 

    # Again, extract left and right line pixel positions
    linex = nonzerox[line_lane_inds]
    liney = nonzeroy[line_lane_inds] 
    # Fit a second order polynomial to each
    if len(linex) and len(liney):
        line_fit = np.polyfit(liney, linex, 2)
    else:
        line_fit = np.empty(shape=(0,0))
    # Generate x and y values for plotting
#    ploty = np.linspace(0, binary.shape[0]-1, binary.shape[0] )
#    line_fitx = line_fit[0]*ploty**2 + line_fit[1]*ploty + line_fit[2]

    return (linex, liney), line_fit


#
# Assume you now have a new warped binary image 
# from the next frame of video (also called "binary_warped")
# It's now much easier to find line pixels!
#
def draw_detect_line_in_roi(binary_warped,
                            left_fit, leftx, lefty,
                            right_fit, rightx, righty,
                            margin=100):
    
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    
    left_fitx = []
    right_fitx = []
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    if len(leftx):
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    if len(rightx): 
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_pts = []
    right_line_pts = []
    if len(leftx):
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
    if len(rightx):
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
    
    # Draw the lane onto the warped blank image
    if len(left_line_pts):
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    if len(right_line_pts):
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    # Draw yellow lanes...
    
    # Cast the x and y points into format compatible with cv2.fillPoly()
    left_pts = []
    right_pts = []
    if len(left_fitx):
        left_fitx_pts1 = np.array([np.transpose(np.vstack([left_fitx-2, ploty]))])
        left_fitx_pts2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+2, ploty])))])
        left_pts = np.hstack((left_fitx_pts1, left_fitx_pts2))
    if len(right_fitx):
        right_fitx_pts1 = np.array([np.transpose(np.vstack([right_fitx-2, ploty]))])
        right_fitx_pts2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+2, ploty])))])
        right_pts = np.hstack((right_fitx_pts1, right_fitx_pts2))
    # Draw yellow lanes in output image
    window_img = np.zeros_like(out_img)
    if len(left_pts):
        cv2.fillPoly(window_img, np.int_([left_pts]), (255,255, 0))
    if len(right_pts):
        cv2.fillPoly(window_img, np.int_([right_pts]), (255,255, 0))
    out_img = cv2.addWeighted(out_img, 1, window_img, 1., 0)
    
    return out_img    
