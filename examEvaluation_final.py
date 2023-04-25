import ast 
import cv2
import os
import glob
from main import infer
from segmentor import get_initials
from segmentor import get_AS
import math
from fuzzywuzzy import fuzz

def listToString(s):
    # initialize an empty string
    str1 = ""

    # traverse in the string
    for ele in s:
        str1 += ele

        # return string
    return str1

def sequ(s1, s2):
    
    # matcher = difflib.SequenceMatcher(a=words1, b=words2)
    score = fuzz.token_set_ratio(s1, s2)
    return (score)


def extract_student_info(path_to_exam,ID):
   ####EXTRACT INFO ABOUT REG NUMBER AND NAME HERE####
    #read all images from a folder
    
   get_initials(path_to_exam,ID) #calls segmentation upon the exam cover page

   data_path = os.path.join(path_to_exam+'\word_out','*g')
   files = glob.glob(data_path)
   name_list=[]

   for f1 in files:
    if "exam_"+str(ID)+"_reg_word" in f1:
     student_reg_number=infer(os.path.abspath(f1))
    if "exam_"+str(ID)+"_name_word" in f1:
     name_list.append(infer(os.path.abspath(f1)))
   
   name=' '.join(name_list)            #joins all words of name
   return student_reg_number, name

    
def make_AS(path_to_exam, tqs, regno,ID):
    get_AS(path_to_exam, tqs,ID)  #5 is number of qs total
    
    data_path = os.path.join(path_to_exam + '\word_out','*g')
    files = glob.glob(data_path)
             
    file_as = open(os.path.join(path_to_exam, "AS_"+str(regno)+".txt"), "w")
#logic for combining words
    fnames_list=[]
    sp=[]
    for z1 in files:
     if "exam_"+str(ID)+"_ans_" in z1:
      fnames_list.append(os.path.basename(os.path.abspath(z1)))
      sp.append(os.path.abspath(z1))
    
    x=0
    
    while x<len(fnames_list):
    
     t1=fnames_list[x]
     if x==len(fnames_list)-1:
      t2=fnames_list[0]
     else:
      t2= fnames_list[x+1]
     if t1[:12]==t2[:12]:
      file_as.write(infer(sp[x]))
      file_as.write(" ")  
      file_as.write(infer(sp[x+1])+"\n")  
      x=x+1
     else:
      file_as.write(infer(sp[x])+"\n")
     x=x+1
    file_as.close()
    
def check_exam(path_to_exam,student_reg_number,total_marks,ID):
    file2= open(path_to_exam+"\AS_"+str(student_reg_number)+".txt", encoding="utf8")
    file1 = open(path_to_exam+"\MS.txt", encoding="utf8")
    lines1 = file1.readlines()
    lines2 = file2.readlines()

    count = 0
    with open(path_to_exam+"\out_"+str(student_reg_number)+".txt", 'w') as file_out:
        for i in range(len(lines1)):
            txt1 = str(lines1[i])
            txt2 = str(lines2[i])
            x = (sequ(txt1, txt2))
            # print(x)
            file_out.write(str(x))
            file_out.write('\n')
            count += x
    score = round(count / len(lines1), 2)
    # print('Score: ', score)
    with open(path_to_exam+"\score_"+str(student_reg_number)+".txt", 'w') as file_out:
        file_out.write(str(score))

    student_marks = str(score)

    # import required classes
    from PIL import Image, ImageDraw, ImageFont

    # create Image object with the input image
    os.chdir(path_to_exam)
    img = Image.open("exam_"+str(ID)+"_2.jpg")
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
    message = "Obtained Marks= " + str(math.ceil((score/100)*total_marks))+"/"+str(total_marks)
    color = 'rgb(255, 0, 0)'  # black color

    # draw the message on the background
    draw.text((x, y), message, fill=color, font=font)
   

    # save the edited image
    path_to_marked_exam = os.path.join(path_to_exam+'\marked_exam', str(student_reg_number))
    try:
     os.mkdir(path_to_marked_exam)
    except OSError:
     print ("Creation of the directory %s failed" % path_to_marked_exam)
    else:
     print ("Successfully created the directory %s " % path_to_marked_exam)
    os.chdir(path_to_marked_exam)
    img.save("marked_exam_"+str(ID)+".jpg")

    return student_marks, path_to_marked_exam

def evaluate_exam(path_to_exam, total_marks):
    total_exams=1
    regno=[]
    student_name=[]
    student_marks=[]
    path_to_marked_exam=[]
    
    mypath=path_to_exam
    with open(path_to_exam+"\details.txt", 'r') as f:
      mydict = ast.literal_eval(f.read())
    
    #recieve ms from details.txt and make ms.txt
    ms = mydict["Answers"].split(",")
    file_ms = open(os.path.join(path_to_exam, "MS" +".txt"), "w")
    for y in ms:
     file_ms.write(y+"\n")
    file_ms.close()
    
    for i in range (total_exams):
    #get student reg no. and Name from cover page
     reg, name= extract_student_info(mypath,i+1)
     regno.append(reg)
     student_name.append(name)
     make_AS(path_to_exam,len(ms), regno[i],i+1)  #this takes path of folder containing exam sheets and gives back a .txt file after OCR
     marks, pm=check_exam(path_to_exam,regno[i],total_marks,i+1) #checks exxam with ms and returns score
     student_marks.append(marks)
     path_to_marked_exam.append(pm)
#     print("Student Name: ", student_name[i])
#     print("Reg. No.: ", regno[i])
#     print("Score: ", student_marks[i])
#     print("Path: ", path_to_marked_exam[i])


    return regno,student_name,student_marks,path_to_marked_exam



regno,student_name,student_marks,path_to_marked_exam=evaluate_exam(r"C:\Users\Ramis\Downloads\FYP\Data\Teachers\178416\Industrial Process Control\Quiz1",10)

print("Name "+str(student_name[0]))
print("Reg no. "+str(regno[0]))
print("Marks "+str(student_marks[0]))
print("Path to marked exam: "+str(path_to_marked_exam[0]))

print("Name "+str(student_name[1]))
print("Reg no. "+str(regno[1]))
print("Marks "+str(student_marks[1]))
print("Path to marked exam: "+str(path_to_marked_exam[1]))
    