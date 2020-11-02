# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 18:12:28 2020

@author: xinwei
"""

from __future__ import print_function
import cv2
import argparse
import numpy as np
import math
import time
from PIL import Image
import pyautogui
import keyboard
#from pynput.mouse import Button, Controller

#import serial

max_value = 255
max_type = 4
max_binary_value = 255
trackbar_type = 'Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted'
trackbar_value = 'Value'
trackbar_blur = 'Blur kernel size'
window_name = 'Threshold Demo'
isColor = False
#myPic = cv2.imread("E:\cs297\hw4\test.jpg")
#myImg = Image.open(r"E:\cs297\hw4\test.jpg")
font = cv2.FONT_HERSHEY_SIMPLEX
c = 0
x0 =300
y0= 200
width = 800
height = 1000


#for part 4
screenShot = 0
storedAngle = 0

def nothing(x):
    pass

#for HSV skinmask
def calculateSkinMask(frame):
    lower_HSV = np.array([0, 40, 00], dtype = "uint8" )
    upper_HSV = np.array([17, 170, 255], dtype = "uint8" )
    convertedHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMaskHSV = cv2.inRange(convertedHSV, lower_HSV, upper_HSV)
    
    lower_YCrCb = np.array((0, 138, 67), dtype = "uint8" )
    upper_YCrCb = np.array((255, 173, 133), dtype = "uint8" )
    convertedYCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    skinMaskYCrCb = cv2.inRange(convertedYCrCb, lower_YCrCb, upper_YCrCb)
    skinMask = cv2.add(skinMaskHSV,skinMaskYCrCb)
    
    

    
    '''
    #this is too slow!
    skinCrCbHist = np.zeros((256,256), dtype= np.uint8)
    cv2.ellipse(skinCrCbHist, (113,155),(23,25), 43, 0, 360, (255,255,255), -1)
    YCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    (y,Cr,Cb) = cv2.split(YCrCb) 
    skinMaskYCrCb = np.zeros(Cr.shape, dtype = np.uint8)
    (x,y) = Cr.shape
    for i in range(0, x):
        for j in range(0, y):
                if skinCrCbHist [Cr[i][j], Cb[i][j]] > 0: #if not in the ellipse
                    skinMaskYCrCb[i][j] = 255
    '''         
    
    
    return skinMask

def calculateSkinMask2(frame):
    YCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    (y,cr,cb) = cv2.split(YCrCb) 
    cr1 = cv2.GaussianBlur(cr, (5,5), 0)
    _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #Ostu处理
    #res = cv2.bitwise_and(frame,frame, mask = skin)
    return skin

#for part 4 def position down
def isDecreased(current, previous, thres):
    if ((current - previous)<thres):
        return True
    else:
        return False
    
def isIncreased(current, previous, thres):
    if ((current - previous)>thres):
        return True
    else:
        return False


def fourierDesciptor(res):
    #Laplacian算子进行八邻域检测
    #gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    gray = res
    dst = cv2.Laplacian(gray, cv2.CV_16S, ksize = 3)
    Laplacian = cv2.convertScaleAbs(dst)
    contour = find_contours(Laplacian)#提取轮廓点坐标
    contour_array = contour[0][:, 0, :]#注意这里只保留区域面积最大的轮廓点坐标
    ret_np = np.ones(dst.shape, np.uint8) #创建黑色幕布
    ret = cv2.drawContours(ret_np,contour[0],-1,(255,255,255),1) #绘制白色轮廓
    contours_complex = np.empty(contour_array.shape[:-1], dtype=complex)
    contours_complex.real = contour_array[:,0]#横坐标作为实数部分
    contours_complex.imag = contour_array[:,1]#纵坐标作为虚数部分
    fourier_result = np.fft.fft(contours_complex)#进行傅里叶变换
    #fourier_result = np.fft.fftshift(fourier_result)
    descirptor_in_use = truncate_descriptor(fourier_result)#截短傅里叶描述子
    #reconstruct(ret, descirptor_in_use)
    return ret, descirptor_in_use

def find_contours(Laplacian):
    #binaryimg = cv2.Canny(res, 50, 200) #二值化，canny检测
    h = cv2.findContours(Laplacian,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) #寻找轮廓
    contour = h[1]
    contour = sorted(contour, key = cv2.contourArea, reverse=True)#对一系列轮廓点坐标按它们围成的区域面积进行排序
    return contour
    
#fourier result
def truncate_descriptor(fourier_result):
    MIN_DESCRIPTOR = 32
    descriptors_in_use = np.fft.fftshift(fourier_result)
    
    #use the middle MIN_DESCRIPTOR
    center_index = int(len(descriptors_in_use) / 2)
    low, high = center_index - int(MIN_DESCRIPTOR / 2), center_index + int(MIN_DESCRIPTOR / 2)
    descriptors_in_use = descriptors_in_use[low:high]
    
    descriptors_in_use = np.fft.ifftshift(descriptors_in_use)
    
    return descriptors_in_use

def reconstruct(img, descirptor_in_use):
    #descirptor_in_use = truncate_descriptor(fourier_result, degree)
    #descirptor_in_use = np.fft.ifftshift(fourier_result)
    #descirptor_in_use = truncate_descriptor(fourier_result)
    #print(descirptor_in_use)
    contour_reconstruct = np.fft.ifft(descirptor_in_use)
    contour_reconstruct = np.array([contour_reconstruct.real,
                                    contour_reconstruct.imag])
    contour_reconstruct = np.transpose(contour_reconstruct)
    contour_reconstruct = np.expand_dims(contour_reconstruct, axis = 1)
    if contour_reconstruct.min() < 0:
        contour_reconstruct -= contour_reconstruct.min()
    contour_reconstruct *= img.shape[0] / contour_reconstruct.max()
    contour_reconstruct = contour_reconstruct.astype(np.int32, copy = False)
 
    black_np = np.ones(img.shape, np.uint8) #创建黑色幕布
    black = cv2.drawContours(black_np,contour_reconstruct,-1,(255,255,255),1) #绘制白色轮廓
    cv2.imshow("contour_reconstruct", black)
    #cv2.imwrite('recover.png',black)

    return black

def _get_eucledian_distance(vect1, vect2):
    distant = vect1[0] - vect2[0]
    dist = np.sqrt(np.sum(np.square(distant)))
    # 或者用numpy内建方法
    # vect1 = list(vect1)
    # vect2 = list(vect2)
    # dist = np.linalg.norm(vect1 - vect2)
    return dist

    

cam = cv2.VideoCapture(0)
#cv2.namedWindow(window_name)

'''
# Create a Trackbar to choose a value for a parameter    
cv2.createTrackbar(parameter_value_name, window_name , 
                   parameter_min_value, parameter_max_value, nothing)  
'''
'''
cv2.createTrackbar(trackbar_type, window_name , 3, max_type, nothing)
# Create Trackbar to choose Threshold value
cv2.createTrackbar(trackbar_value, window_name , 0, max_value, nothing)
# Call the function to initialize
cv2.createTrackbar(trackbar_blur, window_name , 1, 20, nothing)
# create switch for ON/OFF functionality
color_switch = 'Color'
cv2.createTrackbar(color_switch, window_name,0,1,nothing)
cv2.createTrackbar('Contours', window_name,0,1,nothing)
'''

#for gesture part 4
multiFrameScroll = []
multiFrameOK = []
multiFrameAngle = []
multiFrameCount = 0

while True:
    ret, frame = cam.read()
    if not ret:
        break
    
    #0: Binary
    #1: Binary Inverted
    #2: Threshold Truncated
    #3: Threshold to Zero
    #4: Threshold to Zero Inverted
    '''
    threshold_type = cv2.getTrackbarPos(trackbar_type, window_name)
    threshold_value = cv2.getTrackbarPos(trackbar_value, window_name)
    blur_value = cv2.getTrackbarPos(trackbar_blur, window_name)
    blur_value = blur_value+ (  blur_value%2==0)
    isColor = (cv2.getTrackbarPos(color_switch, window_name) == 1)
    findContours = (cv2.getTrackbarPos('Contours', window_name) == 1)
   
    isColor ==False
    #convert to grayscale
    if isColor == False:
        src_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, dst = cv2.threshold(src_gray, threshold_value, max_binary_value, threshold_type )
        blur = cv2.GaussianBlur(dst,(blur_value,blur_value),0)
        if findContours:
            _, contours, hierarchy = cv2.findContours( blur, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE )
            blur = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)  #add this line

            output = cv2.drawContours(blur, contours, -1, (0, 255, 0), 1)
            print(str(len(contours))+"\n")
        else:
            output = blur
        
        
    else:
        _, dst = cv2.threshold(frame, threshold_value, max_binary_value, threshold_type )
        blur = cv2.GaussianBlur(dst,(blur_value,blur_value),0)
        output = blur
     
    cv2.imshow(window_name, output)
    '''
    '''
    ########################################################################
    PART 1
    ########################################################################
    '''
    #PART 1 a: Extracting Hand from the feed
    
    cv2.rectangle(frame, (x0,y0),(x0+width, y0+height),(0,255,0))
    frame= frame[y0:y0+height, x0:x0+width]
    #cv2.imshow("roi", tmpROI)
    skinMask = calculateSkinMask(frame)
    
    skinMask2 = calculateSkinMask2(frame)
    
    #res = cv2.bitwise_and(frame,frame, mask = skinMask) 
    
    #Part 1 b: erosion & dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    #kernel = np.ones((3,3), np.uint8)
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
    
    skinMask2 = cv2.erode(skinMask2, kernel, iterations = 2)
    skinMask2 = cv2.dilate(skinMask2, kernel, iterations = 2)
    
    #Gaussian blur
    skinMask = cv2.GaussianBlur(skinMask, (3,3), 0)
    skinMask2 = cv2.GaussianBlur(skinMask2, (3,3), 0)
    #_, skinTmp = cv2.threshold(skinMaskBlur, 0, max_binary_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
   
    skin = cv2.bitwise_and(frame, frame, mask = skinMask)
    skin2 = cv2.bitwise_and(frame,frame, mask = skinMask2)
    #cv2.imshow("dst_demo", skin)
    
    #output for part 1
    part1 = cv2.cvtColor(skin,cv2.COLOR_BGR2GRAY)
    part1_2 = cv2.cvtColor(skin2,cv2.COLOR_BGR2GRAY)
    cv2.imshow("part 1 output", part1)
    cv2.imshow("method 2", part1_2)
    '''
    ########################################################################
    PART 2
    ########################################################################
    '''
    
    
    #PART 2
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #gray = part1
    _, thresh = cv2.threshold(gray, 0, max_binary_value, cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU)
    
    ret, markers, stats, centroids = cv2.connectedComponentsWithStats(thresh,ltype=cv2.CV_16U)
    markers = np.array(markers, dtype=np.uint8)
    label_hue = np.uint8(179*markers/np.max(markers))
    blank_ch = 255*np.ones_like(label_hue)
    #marks = np.uint8(markers / np.max(markers) * 255)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img,cv2.COLOR_HSV2BGR)
    labeled_img[label_hue==0] = 0
    
    
    
    statsSortedByArea = stats[np.argsort(stats[:, 4])]
    #roi = statsSortedByArea[-3][0:4]
    #x, y, w, h = roi
    #subImg = labeled_img[y:y+h, x:x+w]
    for i in range (-min(3,len(statsSortedByArea)),0):
        cv2.putText(labeled_img, "%d" %(i),
                    (round(centroids[i,0]),round(centroids[i,1])),font, 1.2, (255, 255, 255), 2)
    
    #output for part 2_a
    #cv2.imshow("previous", labeled_img)
    
    if (ret>2):
        try :
            roi = statsSortedByArea[-3][0:4]
            x, y, w, h = roi
            subImg = labeled_img[y:y+h, x:x+w]
            subImg = cv2.cvtColor(subImg, cv2.COLOR_BGR2GRAY);
            _, contours, _ = cv2.findContours(subImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            maxCntLength = 0
            for i in range(0,len(contours)):
                cntLength = len(contours[i])
                if (cntLength>maxCntLength):
                    cnt = contours[i]
                    maxCntLength = cntLength
            if (maxCntLength>=5):
                ellipseParam = cv2.fitEllipse(cnt)
                subImg = cv2.cvtColor(subImg, cv2.COLOR_GRAY2RGB);
                subImg = cv2.ellipse(subImg,ellipseParam,(0,255,0),2)
                
            subImg = cv2.resize(subImg, (0,0), fx=3, fy=3)
            (x,y),(MA,ma),angle = cv2.fitEllipse(cnt)
            
            '''
            print ('------------------------')
            print ("central points: (%f, %f)" %(x,y))
            print ("Major axis = %f, minor axis = %f" %(MA, ma))
            print ("rotation angle is %f degree" %(angle))
            '''
            
            #output for part 2_b
            #cv2.imshow( "ROI " +str(2), subImg)
            cv2.waitKey(5)
        except :
            print ( "No hand found" )

    
    '''
    ########################################################################
    PART 3
    ########################################################################
    '''
    
    #PART 3
    
    # a
    
    
    #thresholdedHandImage = cv2.cvtColor(gray,cv2.COLOR_BGR2GRAY)
    
    '''
    #FOR forier
    
    
    binaryimg = cv2.Canny(frame, 50, 200) #canny detection
    h, contours, _ = cv2.findContours(binaryimg, 
                                      cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours=sorted(contours,key=cv2.contourArea,reverse=True)
    h = sorted(contours,key=cv2.contourArea,reverse=True)
    drawcontour = h
    ret = np.ones(gray.shape, np.uint8)
    cv2.drawContours(ret,drawcontour,-1,(255,255,255),1) #draw contour
    #cv2.imshow("contours", ret)
    
    
    rettmp, descirptor_in_use = fourierDesciptor(ret)
    thresholdedHandImage = reconstruct(frame, descirptor_in_use)
    #output for part 2
    '''
    
    #redue the noise of thresholdedHandImage
    thresholdedHandImage = frame
    fgbg = cv2.createBackgroundSubtractorMOG2()  
    fgmask = fgbg.apply(thresholdedHandImage)
    kernel2 = np.ones((5, 5), np.uint8)
    fgmask = cv2.erode(fgmask, kernel2, iterations=1)  # 膨胀
    res2 = cv2.bitwise_and(thresholdedHandImage, thresholdedHandImage, mask=fgmask)
    ycrcb = cv2.cvtColor(res2, cv2.COLOR_BGR2YCrCb)  # divide in to YUV image, then get CR number
    (_, cr, _) = cv2.split(ycrcb)
    cr1 = cv2.GaussianBlur(cr, (5, 5), 0)  # GaussianBlur
    _, skin2 = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  

    
    _, contours, _ = cv2.findContours(skin2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)       
    contours=sorted(contours,key=cv2.contourArea,reverse=True)       
    thresholdedHandImgBefore = thresholdedHandImage
    fingerCount = 0
    if len(contours)>1:
       largestContour = contours[0]
       #hull = cv2.convexHull(largestContour, returnPoints = False)
       
       hull = cv2.convexHull(largestContour)
       handArea = cv2.contourArea(largestContour)
       hullArea = cv2.contourArea(hull)
       hull = cv2.convexHull(largestContour, returnPoints = False)
       ratio = (hullArea - handArea/handArea)/100
       
       #for part 4
       
       for cnt in contours[:1]:
           defects = cv2.convexityDefects(cnt,hull)
           
           if ( not isinstance(defects,type(None))):
               for i in range(defects.shape[0]):
                   s,e,f,d = defects[i,0]
                   start = tuple(cnt[s][0])
                   end = tuple(cnt[e][0])
                   far = tuple(cnt[f][0])
                   
                   #before heuristic filtering
                   cv2.line(thresholdedHandImgBefore,start,end,[0,255,0])
                   cv2.circle(thresholdedHandImgBefore,far,5,[0,255,0])
                   
                   #3b heuristic filtering
                   c_squared = (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2
                   a_squared = (far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2
                   b_squared = (end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2
                   
                   c = np.sqrt(c_squared)
                   b = np.sqrt(b_squared)
                   a = np.sqrt(a_squared)
                   s = (a+b+c)/2
                   ar = np.sqrt(s*(s-a)*(s-b)*(s-c))
                   d = (2*ar)/c
                   
                   angle = np.arccos((a_squared + b_squared - c_squared ) / (2 * np.sqrt(a_squared * b_squared)))
                   
                   
                   if (angle <= np.pi / 3 and d > 30):
                       fingerCount += 1
                       cv2.line(thresholdedHandImage,start,end,[0,0,255],2) #is this required?
                       cv2.circle(thresholdedHandImage, far, 4, [0, 0, 255], -1)
                       storedAngle = angle
                   
                   M = cv2.moments(largestContour)
                   #cX = int(M[ "m10" ] / M[ "m00" ])
                   #cY = int(M[ "m01" ] / M[ "m00" ])
             
    #cv2.imshow("labeled image", labeled_img)
    #cv2.imshow("before heuristic", thresholdedHandImgBefore)
    
    if(fingerCount !=0):
        fingerCount +=1
        cv2.putText(thresholdedHandImage, "finger count = %d" %(fingerCount), 
                                    (50,100), font, 1.0, (255, 255, 255), 2)
        
    else:
        cv2.putText(thresholdedHandImage, "no finger detected",
                    (50,100), font, 1.0, (255, 255, 255), 2)
    
    #cv2.imshow("after heuristic", thresholdedHandImage)
    

    
    '''
    ########################################################################
    PART 4  
    ########################################################################
    '''
    
    #reduce noise
    # 1. smoothing it with a blurring operator 
    # 2. threshold 
    # 3. obtain a binary mask again.
    
    final = thresholdedHandImage
    
    
    #part 4
    
    #simple gestures
    frameThres = 20
    
    
    
    #1. Xinwei: if we have a OK gesture, left click the mouse
    if (len(multiFrameOK) < frameThres):
        if(fingerCount ==2):
            if (len(cnt)>5):
                (x,y),(MA,ma),angle = cv2.fitEllipse(cnt)
                if(min(MA,ma)!=0):
                    #cv2.imshow( "ROI " +str(2), subImg)
                    average = max(MA,ma)/min(MA,ma)
                    multiFrameOK.append(average)
    elif (len(multiFrameOK) == frameThres):
        if(fingerCount ==2):
            newAverage =np.average(average)
            print(newAverage)
            if (newAverage<2.0):
                pyautogui.leftClick()
                cv2.putText(final, "OK gesture for right click" , 
                                            (50,130), font, 1.0, (255, 255, 255), 1)
    
    largestContour = contours[0] 
    M = cv2.moments(largestContour)
    offsetX = 700
    offsetY = 300
    scaleX = 1
    scaleY = 1
    cX = offsetX + scaleX *int(M["m10"] / M["m00"])  
    cY = offsetY + scaleY *int(M["m01"] / M["m00"])  
    
    '''
    #2. Beth: navigating mouse accourding to hand 
    
    offsetX = 700
    offsetY = 300
    scaleX = 1
    scaleY = 1
    cX = offsetX + scaleX *int(M["m10"] / M["m00"])  
    cY = offsetY + scaleY *int(M["m01"] / M["m00"])  
    pyautogui.moveTo(cX, cY, duration=0.02, tween=pyautogui.easeInOutQuad)
 '''
    
    spacePressed = keyboard.is_pressed('space')
    if(fingerCount == 1 and not(spacePressed)):       
        pyautogui.press('space')  
        spacePressed = True      
    else:  
        spacePressed = False  
    
    
    '''
    #3. Trista: 
    # simple gestue, use the vertical four gesture to trigger the screenshot
    
    (x,y),(MA,ma),angle = cv2.fitEllipse(cnt)  
    for angle in (113,118):
        if fingerCount == 4:
            im1 = pyautogui.screenshot('my_screenshot1.png')
            cv2.putText(thresholdedHandImage, "Picture captured", \
                        (50,300), font, 0.8, (255, 255, 0), 2)  
    '''
    
    # 3. Trista
    # simple gestue, use the vertical four gesture to trigger the screenshot
    (x,y),(MA,ma),angle = cv2.fitEllipse(cnt)  
    for angle in (113,118):
        if fingerCount == 4:
            screenShot += 1
            im1 = pyautogui.screenshot('my_screenshot_%d.png'%(screenShot))
            cv2.putText(final, "Picture captured", \
                        (50,300), font, 0.8, (255, 255, 0), 2)    
            
    
    '''
    ##################
    #complex gestures#
    ##################
    '''
    # Complex gestures
    # 1. Xinwei
    # if finger is 2, and the position of the finger goes down
    # scroll down the screen
    # if the ellipse' y value goes down, scroll down
    t = 50
    
    '''
    if (len(multiFrameScroll) < frameThres):
        if (fingerCount ==4):
            multiFrameScroll.append(cY)
    elif (len(multiFrameScroll) == frameThres):
        if (fingerCount ==4):
            pos = cY
            averagePos =np.average(multiFrameScroll)
            if(isDecreased(pos, averagePos , t)):
                pyautogui.scroll(30)
                cv2.putText(final, "4 fingers for scroll up" , 
                                        (50,150), font, 1.0, (255, 255, 255), 1)
            elif(isIncreased(pos, averagePos, t)):
                pyautogui.scroll(-30)
                cv2.putText(final, "4 fingers for scroll down" , 
                                        (50,150), font, 1.0, (255, 255, 255), 1)
     '''   
        
    #2. Xinwei
    # if finger is two and ange angle between two fingers increases
    # mouse: continuous pressed left 
    # else: release mouse
    tAngle = 0.2
    if (len(multiFrameAngle) < frameThres):
        if (fingerCount ==2):
            multiFrameAngle.append(storedAngle)
    elif (len(multiFrameAngle) == frameThres):
        if (fingerCount ==2):
            currentAngle = storedAngle
            averageAngle =np.average(multiFrameAngle)
            print("current angle: %f"%(currentAngle))
            print("averageAngle: %f" %(multiFrameAngle[-3] ))
            print("====================")
            
            multiFrameCount +=1
            
            
            if(isDecreased(currentAngle, multiFrameAngle[-3] , tAngle)):
                pyautogui.mouseUp()
                cv2.putText(final, "release mouse" , 
                                        (50,150), font, 1.0, (255, 255, 255), 1)
            elif(isIncreased(currentAngle, multiFrameAngle[-3], tAngle)):
                pyautogui.mouseDown()
                cv2.putText(final, "press mouse" , 
                                        (50,150), font, 1.0, (255, 255, 255), 1)
            if (multiFrameCount==frameThres):
                multiFrameCount = 0
            else:
                multiFrameAngle[multiFrameCount]=currentAngle
            
    '''
    # 3. Trista
    # I start from the use "yeah" gesture. \
    # When I rotate my fingers left with two fingers attached, then turn the volume up
    # When I rotate my fingers right with two fingers attached, then turn the volume down
    
    (x,y),(MA,ma),angle = cv2.fitEllipse(cnt)  
    preangle = 0
    preMA = 0
    prema = 0
    multiFrameAngle2 = []
    frameThres = 20

    tAngle = 0.2
    if (len(multiFrameAngle2) < frameThres):
        if (fingerCount ==2):
            multiFrameAngle2.append(angle)
    elif (len(multiFrameAngle2) == frameThres):
        if (fingerCount ==2):
            preangle = angle
            preMA = MA
            prema = ma
            averageAngle =np.average(multiFrameAngle2)
   
        if preangle in (14,17) and preMA/prema in(0.7,0.75):
            if angle in (80,85) and MA/ma in(0.55,0.6):
                pyautogui.press('F12')  
                cv2.putText(final, "Volume Up" , 
                                (50,300), font, 0.8, (255, 255, 0), 2)
            
            elif angle in (25,35) and MA/ma in(0.3,0.35):
                pyautogui.hotkey('F11')  
                cv2.putText(final, "Volume Down" , 
                                (50,300), font, 0.8, (255, 255, 0), 2)
    
    '''
    
    #4. Bethany
    
    
    
    c = c+1
    cv2.imshow("final", final)
    k = cv2.waitKey(1) #k is the key pressed
    if k == 27 or k==113:  #27, 113 are ascii for escape and q respectively
        #exit
        cv2.destroyAllWindows()
        cam.release()
        break



