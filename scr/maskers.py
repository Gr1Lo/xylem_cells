
import shutil
import numpy as np
import cv2
import os
from google.colab.patches import cv2_imshow

def inverte(imagem):
    im_n = (255-imagem)
    return im_n

def masker3(directory_in_str, direct_train_dir):
  directory = os.fsencode(directory_in_str)
  directory_d = os.fsencode(direct_train_dir + "/image")
  #foder_making
  n_st = 0        
  for file_d in os.listdir(directory_d):
          filename_d = os.fsdecode(file_d)
          n_st = n_st + 1


  kernel = np.ones((3,3),np.uint8)
  for file in os.listdir(directory):
      filename = os.fsdecode(file)
      if filename.endswith("c.png"): 
          _filename_ = filename.replace('c', '')
          shutil.copyfile(directory_in_str + str(_filename_), 
                          direct_train_dir + "/image/" + str(n_st) + '.png')

          im = cv2.imread(directory_in_str + str(filename))
          b,g,r = cv2.split(im)

          #red
          ret,th_b0 = cv2.threshold(b,15,127,cv2.THRESH_BINARY)
          ret,th_b1 = cv2.threshold(b,30,127,cv2.THRESH_BINARY_INV)
          img_bb = cv2.bitwise_and(th_b0,th_b0,mask = th_b1)

          ret,img_gg = cv2.threshold(g,5,127,cv2.THRESH_BINARY_INV)

          ret,th_r0 = cv2.threshold(r,125,127,cv2.THRESH_BINARY)
          ret,th_r1 = cv2.threshold(r,145,127,cv2.THRESH_BINARY_INV)
          img_rr = cv2.bitwise_and(th_r0,th_r0,mask = th_r1)

          img_bg = cv2.bitwise_and(img_bb,img_bb,mask = img_gg)
          img_bgr = cv2.bitwise_and(img_rr,img_rr,mask = img_bg)

          closing_r = cv2.morphologyEx(img_bgr, cv2.MORPH_CLOSE, kernel)
          cv2_imshow(closing_r)

          #black
          ret,img_bb1 = cv2.threshold(b,2,255,cv2.THRESH_BINARY_INV)
          ret,img_gg1 = cv2.threshold(g,2,255,cv2.THRESH_BINARY_INV)
          ret,img_rr1 = cv2.threshold(r,2,255,cv2.THRESH_BINARY_INV)

          img_bg1 = cv2.bitwise_and(img_bb1,img_bb1,mask = img_gg1)
          img_bgr1 = cv2.bitwise_and(img_rr1,img_rr1,mask = img_bg1)

          closing_b = cv2.morphologyEx(img_bgr1, cv2.MORPH_CLOSE, kernel)
          #closing_b = cv2.dilate(closing_b,kernel,iterations = 1)
          cv2_imshow(closing_b)

          #sum
          res = cv2.add(closing_r,closing_b)
          cv2_imshow(res)
          cv2.imwrite(direct_train_dir + '/label/' + str(n_st) + '.png', res)
          n_st = n_st + 1




def masker2(directory_in_str, direct_train_dir):
  directory = os.fsencode(directory_in_str)
  directory_d = os.fsencode(direct_train_dir + "/image")
  #foder_making
  n_st = 0        
  for file_d in os.listdir(directory_d):
          filename_d = os.fsdecode(file_d)
          n_st = n_st + 1


  kernel = np.ones((3,3),np.uint8)
  for file in os.listdir(directory):
      filename = os.fsdecode(file)
      if filename.endswith("c.png"): 
          _filename_ = filename.replace('c', '')
          shutil.copyfile(directory_in_str + str(_filename_), 
                          direct_train_dir + "/image/" + str(n_st) + '.png')

          im = cv2.imread(directory_in_str + str(filename))
          b,g,r = cv2.split(im)

          #red
          ret,th_b0 = cv2.threshold(b,15,255,cv2.THRESH_BINARY)
          ret,th_b1 = cv2.threshold(b,30,255,cv2.THRESH_BINARY_INV)
          img_bb = cv2.bitwise_and(th_b0,th_b0,mask = th_b1)

          ret,img_gg = cv2.threshold(g,5,255,cv2.THRESH_BINARY_INV)

          ret,th_r0 = cv2.threshold(r,125,255,cv2.THRESH_BINARY)
          ret,th_r1 = cv2.threshold(r,145,255,cv2.THRESH_BINARY_INV)
          img_rr = cv2.bitwise_and(th_r0,th_r0,mask = th_r1)

          img_bg = cv2.bitwise_and(img_bb,img_bb,mask = img_gg)
          img_bgr = cv2.bitwise_and(img_rr,img_rr,mask = img_bg)

          closing_r = cv2.morphologyEx(img_bgr, cv2.MORPH_CLOSE, kernel)
          cv2_imshow(closing_r)

          #black
          '''ret,img_bb1 = cv2.threshold(b,2,255,cv2.THRESH_BINARY_INV)
          ret,img_gg1 = cv2.threshold(g,2,255,cv2.THRESH_BINARY_INV)
          ret,img_rr1 = cv2.threshold(r,2,255,cv2.THRESH_BINARY_INV)'''

          ret,img_bb1 = cv2.threshold(b,2,255,cv2.THRESH_BINARY_INV)
          ret,img_gg1 = cv2.threshold(g,2,255,cv2.THRESH_BINARY_INV)
          ret,img_rr1 = cv2.threshold(r,2,255,cv2.THRESH_BINARY_INV)

          img_bg1 = cv2.bitwise_and(img_bb1,img_bb1,mask = img_gg1)
          img_bgr1 = cv2.bitwise_and(img_rr1,img_rr1,mask = img_bg1)

          closing_b = cv2.morphologyEx(img_bgr1, cv2.MORPH_CLOSE, kernel, iterations = 2)
          #closing_b = cv2.dilate(closing_b,kernel,iterations = 3)
          cv2_imshow(closing_b)

          #sum
          res = cv2.add(closing_r,closing_b)
          closing_res0 = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)
          closing_res = inverte(closing_res0)
          cv2_imshow(closing_res)
          #print(closing_res[10:20,10:20])
          cv2.imwrite(direct_train_dir + '/label/' + str(n_st) + '.png', closing_res)
          n_st = n_st + 1











