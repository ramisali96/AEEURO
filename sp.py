# import the necessary packages
import numpy as np
import cv2
import imutils
import os
import glob



def get_AS(path_to_exam, total_qs,ID):
 

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
     
path_to_exam=r"C:\Users\Ramis\Downloads\FYP\Data\Teachers\178416\Industrial Process Control\Quiz1"
get_AS(path_to_exam,4,1)

from PIL import Image, ImageDraw, ImageFont

    # create Image object with the input image
os.chdir(path_to_exam)
img = Image.open("exam_"+"1"+"_2.jpg")
img  = img.transpose(Image.ROTATE_90)
img  = img.transpose(Image.ROTATE_90)
img  = img.transpose(Image.ROTATE_90)
    # initialise the drawing context with
    # the image object as background
draw = ImageDraw.Draw(img)

    # create font object with the font file and specify
    # desired size
    # create font object with the font file and specify
    # desired size
font = ImageFont.truetype('arial.ttf', size=150)

    # starting position of the message
(x, y) = (450, 100)
message = "Obtained Marks= 10/10" 
color = 'rgb(255, 0, 0)'  # black color

    # draw the message on the background
draw.text((x, y), message, fill=color, font=font)
   

    # save the edited image
path_to_marked_exam = os.path.join(path_to_exam+'\marked_exam','295')
try:
 os.mkdir(path_to_marked_exam)
except OSError:
 print ("Creation of the directory %s failed" % path_to_marked_exam)
else:
 print ("Successfully created the directory %s " % path_to_marked_exam)
 os.chdir(path_to_marked_exam)
 img.save("marked_exam_"+"1"+".jpg")


