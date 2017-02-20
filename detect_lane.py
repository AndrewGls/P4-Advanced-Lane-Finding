import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
from collections import deque
from tools import binarize_pipeline, binarize_pipeline_ex, undistort_img, warp_img_M, warp_img




#
# Finding position of left/right lines at the bottom of binary
# warped image via detection peakcs in histogram.
# Params: binary - binary warped image.
#         thresh = noise thresholding for detected peacks.
#         centerX_base - expected position of line at the bottom of binary image
#         thresh_dist - expected distance between detected peak and expected position of line centerX_base.
def find_peaks(binary, 
               centerX_base,
               thresh=11,
               thresh_dist=150,
               sigma=20,
               verbose=False):
 
    # Assuming you have created a warped binary image called "bynary"
    # Take a histogram of the bottom half of the image
    binary = ((binary).astype('float64')) / 255.
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
        print('pick value: ', filtered[peak_x])
        if centerX_base < midpoint:
            print('peak before sigma: ', np.argmax(histogram[:midpoint]))
            ax2.set_title('Histrogram of Left line')
            ax2.plot(histogram[:midpoint], 'r')
        else:
            print('peak before sigma: ', (np.argmax(histogram[midpoint:]) + midpoint))
            ax2.set_title('Histrogram of Right line minus ' + str(midpoint))
            ax2.plot(range(binary.shape[1]-midpoint), histogram[midpoint:], 'g')
        print('peak_x after sigma: ', peak_x)
        plt.show()
    
    if filtered[peak_x] < thresh:
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

    binary = ((binary).astype('float64')) / 255.
    
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
    
    out_img = np.dstack((binary, binary, binary))

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
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
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


#
# Define a class to receive the characteristics of each line detection
#
class Line():
    def __init__(self, def_line_pos, img_width, n = 5):
        # size of queue to store data for last delected n frames
        self.n = n
        # number of fitted lines in buffer
        self.n_buffered = 0
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = deque([],maxlen=n)#[] 
        # fit-coeffs of the last n fits of the line
        self.recent_fit_coeffs = deque([],maxlen=n)  
        # average x values of the fitted line over the last n iterations
        self.bestx = None     
        # polynomial coefficients averaged over the last n iterations
        self.avg_fit_coeffs = None  
        # radius of curvature of the line in some units
        self.radius_of_curvature = None 
        # distance in meters of vehicle center from the line
        self.line_base_pos = None 
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        # x values for detected line pixels
        self.allx = None  
        # y values for detected line pixels
        self.ally = None
        # position in pixels of fitted line at the bottom of image
        self.line_pos = None
        # polynomial coefficients of the most recent fit
        self.current_fit_coeffs = [np.array([False])]
        # x values of the most recent fit
        self.current_fit_x_vals = [np.array([False])]
        # y values for line fit: const y-grid for image
        self.fit_y_vals = np.linspace(0, 100, num=101) * 7.2
        # center of image (center of car) in pixels along x is used as base pos. 
        self.center_of_car = img_width/2 + 14 # 14 is systematic error
        # default position of line
        self.def_line_pos = def_line_pos
        
    def set_current_fit_x_vals(self):
        yvals = self.fit_y_vals
        if len(self.current_fit_coeffs):
            self.current_fit_x_vals = self.current_fit_coeffs[0]*yvals**2 + self.current_fit_coeffs[1]*yvals + self.current_fit_coeffs[2]
        else:
            self.current_fit_x_vals = np.empty(shape=(0,0))
        pass
    
    def set_line_base_pos(self):
        if len(self.current_fit_coeffs):
            y_eval = max(self.fit_y_vals)
            self.line_pos = self.current_fit_coeffs[0]*y_eval**2 + self.current_fit_coeffs[1]*y_eval + self.current_fit_coeffs[2]
            self.line_base_pos = (self.line_pos - self.center_of_car)*3.7/611. # 3.7 meters is ~611 pixels along x direction            
  
    def calc_diffs(self):
        if len(self.current_fit_coeffs):
            if self.n_buffered > 0:
                self.diffs = self.current_fit_coeffs - self.avg_fit_coeffs
            else:
                self.diffs = np.array([0,0,0], dtype='float')
        else:
            self.diffs = np.array([0,0,0], dtype='float')
        pass
            
    def set_radius_of_curvature(self):
        # Define y-value where we want radius of curvature (choose bottom of the image)
        y_eval = max(self.fit_y_vals)
        if self.avg_fit_coeffs is not None:
            self.radius_of_curvature = ((1 + (2*self.avg_fit_coeffs[0]*y_eval + self.avg_fit_coeffs[1])**2)**1.5) \
                             /np.absolute(2*self.avg_fit_coeffs[0])
            
    def push_data(self):
        self.recent_xfitted.appendleft(self.current_fit_x_vals)
        self.recent_fit_coeffs.appendleft(self.current_fit_coeffs)
        assert(len(self.recent_xfitted)==len(self.recent_fit_coeffs))
        self.n_buffered = len(self.recent_xfitted)
        
    def pop_data(self):        
        if self.n_buffered > 0:
            self.recent_xfitted.pop()
            self.recent_fit_coeffs.pop()
            assert(len(self.recent_xfitted)==len(self.recent_fit_coeffs))
            self.n_buffered = len(self.recent_xfitted)
            
    def set_avgx(self):
        fits = self.recent_xfitted
        if len(fits):
            av = 0
            for fit in fits:
                av += np.array(fit)
            av = av / len(fits)
            self.avgx = av
            
    def set_avgcoeffs(self):
        coeffs = self.recent_fit_coeffs
        if len(coeffs):
            av = 0
            for coeff in coeffs:
                av += np.array(coeff)
            av = av / len(coeffs)
            self.avg_fit_coeffs = av
            
    # here come sanity checks of the computed metrics
    def accept_lane(self):
        if len(self.current_fit_coeffs) == 0:
            print('lane is not detected: empty current_fit_coeffs!')
            return False
            pass
        flag = True
        maxdist = 2.8  # distance in meters from the lane
        if(abs(self.line_base_pos) > maxdist ):
            print('lane too far away')
            flag  = False   
        # Uncomment for challenge_video.mp4 sample, needs to fix bug with left line for this sample.
        #if self.n_buffered:
        #    relative_delta = self.diffs / self.avg_fit_coeffs
        #    # allow maximally this percentage of variation in the fit coefficients from frame to frame
        #    if not (abs(relative_delta) < np.array([0.7,0.5,0.15])).all():
        #        print('fit coeffs too far off [%]',relative_delta)
        #        flag=False
            
        return flag
        
    def update(self, line_x, line_y, line_fit, verbose=False):
        self.allx = line_x
        self.ally = line_y
        self.current_fit_coeffs = line_fit
        self.set_current_fit_x_vals()
        self.set_line_base_pos()
        self.calc_diffs()
        if self.accept_lane():
            self.detected=True
            self.push_data()
            self.set_avgx()
            self.set_avgcoeffs()            
        else:
            self.detected=False            
            self.pop_data()
            if self.n_buffered>0:
                self.set_avgx()
                self.set_avgcoeffs()
                
        self.set_radius_of_curvature()
               
        return self.detected,self.n_buffered


#
# Extracts lane from binary image: line_xvals, line_yvals, line_fit_coeff
# Returns: line_xvals, line_yvals, line_fit_coeff
#
def get_line_from_image(binary, line, verbose=False):
    failedPeak = False

    # detect line in binary image
    linex = np.empty(shape=(0,0))
    liney = np.empty(shape=(0,0))
    line_fit = np.empty(shape=(0,0))
    if line.detected:
        (linex, liney), line_fit = detect_line_in_roi(binary, line.current_fit_coeffs)
    else:
        line_pos = find_peaks(binary, line.def_line_pos, verbose=verbose)
        if line_pos == 0:
            failedPeak = True
            line_pos = line.def_line_pos
        (linex, liney), line_fit, _ = detect_line(binary, line_pos)

    return (linex, liney), line_fit, failedPeak


#
# Projects the lane onto the road.
#
def project_lanes_onto_road(img, left_fitx, right_fitx, yvals):
    
    # Create an image to draw the lines on
    color_warp = np.zeros_like(img).astype(np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    undist = undistort_img(img)    
    unwarp,Minv = warp_img_M(img,tobird=False)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    
    return result

    
#
# Detects left and right lanes in binary image.
#
def process_image_ex(img, leftL, rightL, frame_ind=0, verbose=False):
    not_warped = None
    
    if verbose:
        not_warped, binary = binarize_pipeline_ex(img)
    else:
        binary = binarize_pipeline(img)
        
    verbose_peak = (verbose and frame_ind == 0)

    # detect left line
    (leftx, lefty), left_fit, failedPeak = get_line_from_image(binary, leftL, verbose=verbose_peak)
    if failedPeak:
        print('Failed left line detection!')
        plt.imsave(str(frame_ind)+'_L_trouble_image.jpg',img)
    leftL.update(leftx, lefty, left_fit)
    #print('left_fit_coeff::', leftL.current_fit_coeffs)
        
    # detect right line
    (rightx, righty), right_fit, failedPeak = get_line_from_image(binary, rightL, verbose=verbose_peak)
    if failedPeak:
        print('Failed right line detection!')
        plt.imsave(str(frame_ind)+'_R_trouble_image.jpg',img)
    rightL.update(rightx, righty, right_fit)
    #print('right_fit_coeff::', rightL.current_fit_coeffs)
        
    
    result = project_lanes_onto_road(img, leftL.avgx, rightL.avgx, leftL.fit_y_vals)

    ymax = 0
    
    # Draw debug board with Binarization-View, Lane-Detesction-View
    if verbose:
        board_ratio = 0.25
        board_x = 20
        board_y = 20
        board_h = int(img.shape[0] * board_ratio)
        board_w = int(img.shape[1] * board_ratio)
            
        ymin = board_y
        ymax = board_h + board_y
        xmin = board_x
        xmax = board_x + board_w
    
        offset_x = board_x + board_w
        
        # draw binary image
        board_img = cv2.resize(not_warped, dsize=(board_w, board_h), interpolation=cv2.INTER_LINEAR)
        result[ymin:ymax, xmin:xmax, :] = board_img
        
        # draw warped binary image
        xmin += offset_x
        xmax += offset_x
        img_warped = warp_img(undistort_img(img))
        board_img = cv2.resize(img_warped, dsize=(board_w, board_h), interpolation=cv2.INTER_LINEAR)
        result[ymin:ymax, xmin:xmax, :] = board_img
        
        # draw warped binary image
        xmin += offset_x
        xmax += offset_x
        board_img = draw_detect_line_in_roi(binary, left_fit, leftx, lefty, right_fit, rightx, righty)
        board_img = cv2.resize(board_img, dsize=(board_w, board_h), interpolation=cv2.INTER_LINEAR)
        result[ymin:ymax, xmin:xmax, :] = board_img

        # add frame_index text at the bottom of board
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result, 'frame {:d}'.format(frame_ind), (xmax + 20, 60), font, 0.9, (255, 0, 0), 2, cv2.LINE_AA)

                
    text_pos_x = 20
    text_pos_y = ymax + 40

    lane_dist = 3.7 / 2 # distance from the center of car to the lane is 3.7/2 meters
    off_center = round( ( (abs(leftL.line_base_pos)-lane_dist) + (lane_dist - rightL.line_base_pos) )/2, 2 )    
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    if off_center < 0:
        text = str('Vehicle is ' + str(abs(off_center)) + 'm of left of center')
    else:
        text = str('Vehicle is ' + str(off_center) + 'm of right of center')       
    cv2.putText(result, text, (text_pos_x, text_pos_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    if leftL.radius_of_curvature and rightL.radius_of_curvature:
#        curvature = round((leftL.radius_of_curvature + rightL.radius_of_curvature) / 2./1000., 1)
        # Takes min curvature because more robust detected lane usually has less curvature coeff A in f(y)=A^2*y + B*y + C.
        curvatureL = round(leftL.radius_of_curvature/1000., 1)
        curvatureR = round(rightL.radius_of_curvature/1000., 1)
        curvature = min(curvatureL, curvatureR) 
        text = str('Radius of Curvature: ' + str(curvature) + '(km)')
        cv2.putText(result, text, (text_pos_x,text_pos_y+40), font, 1, (255, 255, 255), 2, cv2.LINE_AA)    
        
    return result
    