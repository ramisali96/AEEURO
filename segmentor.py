# import the necessary packages
import numpy as np
import cv2
import imutils
import os
import glob


def order_points(pts):

    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]


    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def transformFourPoints(image, pts):

    rect = order_points(pts)
    (tl, tr, br, bl) = rect


    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))


    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))


    dst = np.array([[0, 0],    [maxWidth - 1, 0],    [maxWidth - 1, maxHeight - 1],    [0, maxHeight - 1]], dtype="float32")


    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def get_initials(path_to_exam,ID):
 
    
 data_path1 = os.path.join(path_to_exam,'*g')
 files12 = glob.glob(data_path1)
 scale_percent = 2500 # percent of original size
 for f12 in files12:
  if "exam_"+str(ID)+"_1" in f12:
   image = cv2.imread(os.path.abspath(f12))
   print(os.path.abspath(f12))
   ratio = image.shape[0] / 500.0
   orig = image.copy()
   image = imutils.resize(image, height = 500)
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   gray = cv2.GaussianBlur(gray, (5, 5), 0)
   edged = cv2.Canny(gray, 75, 200)
#   cv2.imshow("Step 1", edged)
#   cv2.waitKey(0)

   print("STEP 1: Edge Detection")
   cv2.destroyAllWindows()

   cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
   cnts = cnts[0] #if imutils.is_cv2() else cnts[1]
   cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

   for c in cnts:

       peri = cv2.arcLength(c, True)
       approx = cv2.approxPolyDP(c, 0.02 * peri, True)
       
       if len(approx) == 4:
           screenCnt = approx
           break

   print("STEP 2: Finding contours of paper")
   cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
   cv2.imshow("Outline", image) 
   cv2.waitKey(0)
   cv2.destroyAllWindows()

   warped = transformFourPoints(orig, screenCnt.reshape(4, 2) * ratio)

   print("STEP 3: Applying perspective transform")
   img= imutils.resize(warped, height = 650)
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   gray = cv2.GaussianBlur(gray, (3, 3), 1.1)
   edged1 = cv2.Canny(gray, 75, 100)
   cv2.imshow("Segmented Page", edged1)
   cv2.waitKey(0)
#Line Segmentation

 #find contours
   ctrs, hier = cv2.findContours(edged1,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

 #sort contours
   sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
   
 count=0

 for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)
    if w<100 and h<150:
      continue
    # Getting ROI
    roi = img[y+5:y+h-5, x+5:x+w-5]
    
    # show ROI
#    cv2.imshow('line no:'+str(count),roi)
    if count!=0:                              #name line
     cv2.imwrite(os.path.join(path_to_exam + '\line_out' , 'exam_'+str(ID)+'_name_line_'+str(count-1)+'.png'),roi)
     cv2.imshow("Segmented Line", roi)
     cv2.waitKey(0)
   
    if count==0:                              #reg no. line
     cv2.imwrite(os.path.join(path_to_exam + '\line_out' , 'exam_'+str(ID)+'_reg_line_'+str(count)+'.png'),roi)
     cv2.imshow("Segmented Line", roi)
     cv2.waitKey(0)
   
       
    cv2.rectangle(img,(x,y),( x + w, y + h ),(90,0,255),2)
    cv2.imshow("Contours", img)
    cv2.waitKey(0)
    count=count+1

#Word Segmentation
 kernel = np.ones((1,6), np.uint8)
 scale_percent = 2500 # percent of original size
 
 #first process reg no. from line to words
 data_path = os.path.join(path_to_exam+"\line_out",'*g')
 files1 = glob.glob(data_path)
 
 for f11 in files1:
  if "exam_"+str(ID)+"_reg_line_" in f11:
   img = cv2.imread(os.path.abspath(f11))
   
 cv2.destroyAllWindows()   
 width = int(img.shape[1] * scale_percent / 100)
 height = int(img.shape[0] * scale_percent / 100)
 dim = (width, height)
# resize image
 resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
 resized=img
 im2=cv2.fastNlMeansDenoising(resized,None,10,7,27)
 gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
 ret, thresh1 = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
# cv2.imshow("Threshold Image", thresh)
 sx = cv2.Sobel(thresh1,cv2.CV_32F,1,0)
 sy = cv2.Sobel(thresh1,cv2.CV_32F,0,1)
 m1 = cv2.magnitude(sx,sy)
 m1 = cv2.normalize(m1,None,0.,255.,cv2.NORM_MINMAX,cv2.CV_8U)
 cv2.imshow("Post Sobel + Adp Thresh", m1)
 cv2.waitKey(0)
   
 #dilation
 img_b1=cv2.GaussianBlur(m1, (3,3),1) 
 img_dil1 = cv2.dilate(img_b1, kernel, iterations=1)
 ret, img_dil1 = cv2.threshold(img_dil1,0,255,cv2.THRESH_BINARY)
 cv2.imshow("Post Dilation + Blur", img_dil1)
 cv2.waitKey(0)
   
#find contours
 ctrs, hier = cv2.findContours(img_dil1.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#sort contours
 sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
 count=0
 for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
  x, y, w, h = cv2.boundingRect(ctr)
    # Getting ROI
  roi = img[y:y+h, x:x+w]
  roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
  ret, roi = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY)
  cv2.imwrite(os.path.join(path_to_exam + '/word_out' , 'exam_'+str(ID)+'_reg_word_'+str(count)+'.png'),roi)
  cv2.imshow("Segmented Word", roi)
  cv2.waitKey(0)

  cv2.rectangle(img,(x,y),( x + w, y + h ),(90,0,255),2)
  count=count+1
  cv2.waitKey(0)
   
#then process can. name from line to words
 data_path = os.path.join(path_to_exam+"\line_out",'*g')
 files2 = glob.glob(data_path)
 for f12 in files2:
  if "exam_"+str(ID)+"_name_line_" in f12:   #pick name line
   img = cv2.imread(os.path.abspath(f12))
  
   cv2.destroyAllWindows()   
   width = int(img.shape[1] * scale_percent / 100)
   height = int(img.shape[0] * scale_percent / 100)
   dim = (width, height)
# resize image
   resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
   resized=img
   im2=cv2.fastNlMeansDenoising(resized,None,10,7,27)
   gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
   ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
#  cv2.imshow("Threshold Image", thresh)
   sx = cv2.Sobel(thresh,cv2.CV_32F,1,0)
   sy = cv2.Sobel(thresh,cv2.CV_32F,0,1)
   m = cv2.magnitude(sx,sy)
   m = cv2.normalize(m,None,0.,255.,cv2.NORM_MINMAX,cv2.CV_8U)
#  cv2.imshow("Image", m)
 #dilation
   img_b=cv2.GaussianBlur(m, (3,3),1) 
   img_dil = cv2.dilate(img_b, kernel, iterations=1)
   ret, img_dil = cv2.threshold(img_dil,0,255,cv2.THRESH_BINARY)
#  cv2.imshow("Dilated image", img_dil )
   cv2.waitKey(0)
#find contours
   ctrs, hier = cv2.findContours(img_dil.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#sort contours
   sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
   count=0
   for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)
    # Getting ROI
    roi = img[y:y+h, x:x+w]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    ret, roi = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(path_to_exam + '\word_out' , 'exam_'+str(ID)+'_name_word_'+str(count)+'.png'),roi)
    cv2.imshow("Segmented Word", roi)
    cv2.waitKey(0)
    cv2.rectangle(img,(x,y),( x + w, y + h ),(90,0,255),2)
    count=count+1
    cv2.waitKey(0)



def get_AS(path_to_exam, total_qs,ID):
 
#get exam answer sheet
#    
# data_path = os.path.join(path_to_exam,'*g')
# files = glob.glob(data_path)
# for f1 in files:
#  if "exam_"+str(ID)+"_2" in f1:
#   image = cv2.imread(os.path.abspath(f1))
#   
# ratio = image.shape[0] / 500.0
# orig = image.copy()
# image = imutils.resize(image, height = 500)
#
#
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# gray = cv2.GaussianBlur(gray, (5, 5), 0)
# edged = cv2.Canny(gray, 75, 200)
#
#
# print("STEP 1: Edge Detection")
#
# cv2.destroyAllWindows()
#
# cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] #if imutils.is_cv2() else cnts[1]
# cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
#
#
# for c in cnts:
#
#     peri = cv2.arcLength(c, True)
#     approx = cv2.approxPolyDP(c, 0.02 * peri, True)
#
#     if len(approx) == 4:
#         screenCnt = approx
#         break
#
# print("STEP 2: Finding contours of paper")
# cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
# cv2.imshow("Outline", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# warped = transformFourPoints(orig, screenCnt.reshape(4, 2) * ratio)
#
# print("STEP 3: Applying perspective transform")
# img= imutils.resize(warped, height = 650)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = cv2.GaussianBlur(gray, (3, 3), 0.8)
# edged = cv2.Canny(gray, 30, 100)
# y=10
# x=10
# w=800
# h=650
# crop_img_1 = edged[y:y+h, x:x+w]
# img = img[y:y+h, x:x+w]
# cv2.imshow("cropped", crop_img_1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
##Line Segmentation
#
# #find contours
# ctrs, hier = cv2.findContours(crop_img_1,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# #sort contours
# sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
# count=0
# 
# for i, ctr in enumerate(sorted_ctrs):
#    # Get bounding box
#    x, y, w, h = cv2.boundingRect(ctr)
#    if w<100 and h<150:
#      continue
#    # Getting ROI
#    roi = img[y+5:y+h-5, x+5:x+w-5]
#    cv2.imwrite(os.path.join(path_to_exam + '\line_out' , 'exam_'+str(ID)+'_ans_line_'+str(total_qs-count)+'.png'),roi)
#    cv2.rectangle(img,(x,y),( x + w, y + h ),(90,0,255),2)
#    count=count+1
#    cv2.imshow("line",img)
#    cv2.waitKey(0)
 #Word Segmentation of answer lines
 scale_percent = 2500 # percent of original size
 
 #from line to words
 data_path = os.path.join(path_to_exam+"\line_out",'*g')
 files3 = glob.glob(data_path)
 count=0
 for f3 in files3:
  if "exam_"+str(ID)+"_ans_" in f3:
   count=count+1
   img = cv2.imread(os.path.abspath(f3))
   cv2.destroyAllWindows()   
   width = int(img.shape[1] * scale_percent / 100)
   height = int(img.shape[0] * scale_percent / 100)
   dim = (width, height)
# resize image
   resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
   resized=img
   im2=cv2.fastNlMeansDenoising(resized,None,10,7,27)
   gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
   ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
#   cv2.imshow("Threshold Image", thresh)
   sx = cv2.Sobel(thresh,cv2.CV_32F,1,0)
   sy = cv2.Sobel(thresh,cv2.CV_32F,0,1)
   m = cv2.magnitude(sx,sy)
   m = cv2.normalize(m,None,0.,255.,cv2.NORM_MINMAX,cv2.CV_8U)
#   cv2.imshow("Image", m)
 #dilation
   img_b=cv2.GaussianBlur(m, (3,3),1) 
#  img_dil = cv2.dilate(img_b, kernel, iterations=1)
   ret, img_dil = cv2.threshold(img_b,0,255,cv2.THRESH_BINARY)
#   cv2.imshow("Dilated image", img_dil )
   cv2.waitKey(0)
#find contours
   ctrs, hier = cv2.findContours(img_dil.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#sort contours
   sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
   for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
     x, y, w, h = cv2.boundingRect(ctr)
    # Getting ROI
     roi = img[y:y+h, x:x+w]
     roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
     ret, roi = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY)
     cv2.imwrite(os.path.join(path_to_exam + '/word_out' , 'exam_'+str(ID)+'_ans_'+str(count)+'_word_'+str(i)+'.png'),roi)
     cv2.rectangle(img,(x,y),( x + w, y + h ),(90,0,255),2)
     cv2.waitKey(0)
#    
    
#
path_to_exam=r"C:\Users\Ramis\Documents\FYP"
#get_AS(path_to_exam,4,1)

#path_to_exam=r"C:\Users\Ramis\Documents\SEECS\FYP"
#get_initials(path_to_exam)
