from scr.model import *
from scr.data import *
from scr.unet_modified import *
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2
import os
import numpy as np
from shutil import rmtree
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint


def crop_img(train_v, inp_img, out_fold):

  font                   = cv2.FONT_HERSHEY_SIMPLEX
  fontScale              = 5
  fontColor              = (255,0,0)
  lineType               = 2

  img0 = cv2.imread(inp_img,0)
  cv2.imwrite('imm0.png', img0)
  img = cv2.imread('imm0.png')
  img2 = cv2.imread('imm0.png')
  height, width, channels = img.shape
  print(height, width, channels)
  n_num = 0
  i_b = 0
  for i in range((height//train_v)):
    j_b = 0
    for j in range((width//train_v)):
      px = img[i_b:(i+1)*train_v, j_b:(j+1)*train_v]
      cv2.imwrite(out_fold + '/' +
                  str(n_num) + '.png', px)
      
      bottomLeftCornerOfText = ((i+1)*train_v,j_b)
      j_b = j_b + train_v

      cv2.putText(img2,str(n_num), 
                  bottomLeftCornerOfText, 
                  font, 
                  fontScale,
                  fontColor,
                  lineType)

      n_num = n_num + 1

    i_b = i_b + train_v

  cv2.imwrite('imm2.png', img2)
  return n_num



def sew_img(train_v, inp_img, cop_img, out_fold):

  img = cv2.imread(inp_img)
  cv2.imwrite('cop_img.png', img)
  c_m = cv2.imread(cop_img)


  height, width, channels = img.shape
  n_num = 0
  i_b = 0
  for i in range(height//train_v):
    j_b = 0
    for j in range(width//train_v):

      img_ii = cv2.imread(out_fold + '/' + str(n_num) + '_predict.png')
      c_m[i_b:(i+1)*train_v, j_b:(j+1)*train_v] = img_ii

      j_b = j_b + train_v
      n_num = n_num + 1

    i_b = i_b + train_v

  cv2.imwrite('_2_1_' + str(cop_img), c_m)
def crop_img1(train_v, inp_img, out_fold, s_v):

  r_v = train_v-s_v*2
  img0 = cv2.imread(inp_img,0)
  img0 = cv2.copyMakeBorder(img0, 512, 512, 512, 512, borderType=cv2.BORDER_CONSTANT, value = 0)
  cv2.imwrite('imm0.png', img0)
  img = cv2.imread('imm0.png')
  height, width, channels = img.shape
  n_num = 0
  i_b = 0
  for i in range((height//r_v)):
    j_b = 0
    for j in range((width//r_v)):
      px0 = img[i_b:(i+1)*r_v, j_b:(j+1)*r_v]
      px = cv2.copyMakeBorder(px0, s_v, s_v, s_v, s_v, borderType=cv2.BORDER_REPLICATE)
      cv2.imwrite(out_fold + '/' +
                  str(n_num) + '.png', px)

      j_b = j_b + r_v

      n_num = n_num + 1

    i_b = i_b + r_v

  return n_num



def sew_img1(train_v, inp_img, inp_fold, out_img, NN, s_v=0):

  r_v = train_v-s_v*2

  img = cv2.imread(inp_img)
  height0, width0, channels0 = img.shape
  img = cv2.copyMakeBorder(img, train_v, train_v, train_v, train_v, borderType=cv2.BORDER_CONSTANT, value = 0)
  cv2.imwrite('cop_img.png', img)
  c_m = cv2.imread('cop_img.png')

  height, width, channels = img.shape
  n_num = 0
  i_b = 0
  for i in range(height//r_v):
    j_b = 0
    for j in range(width//r_v):
      print(n_num)
      if n_num < NN:
        img_ii = cv2.imread(inp_fold + '/' + str(n_num) + '_predict.png')
        height1, width1, channels1 = img_ii.shape
        c_m[i_b:(i+1)*r_v, j_b:(j+1)*r_v] = img_ii[s_v:height1-s_v, s_v:width1-s_v]

        j_b = j_b + r_v
      n_num = n_num + 1

    i_b = i_b + r_v

  c_m1 = c_m[train_v-1:train_v-1+height0, train_v-1:train_v-1+width0]
  cv2.imwrite(out_img, c_m1)

def sta(ep_num, step_num, weights='imagenet'):
  if os.path.exists("t_train"):
    rmtree('t_train') 

  os.mkdir('t_train')

  #NN = crop_img(512, 'membrane/membrane/im1.png', 'membrane/membrane/g_train')
  #NN = crop_img1(512, 'membrane/membrane/im1.png', 'membrane/membrane/g_train',3)

  '''data_gen_args = dict(rotation_range=0.2,
                      width_shift_range=0.05,
                      height_shift_range=0.05,
                      shear_range=0.05,
                      zoom_range=0.01,
                      horizontal_flip=True,
                      fill_mode='nearest')'''
  
  data_gen_args = dict(rotation_range=180,
                      zoom_range=0.1,
                      horizontal_flip=True,
                      vertical_flip=True,
                      fill_mode='reflect')
  myGene = trainGenerator(2,'tr/tr_data/sour','image','label',data_gen_args,save_to_dir = "t_train", target_size = (512,512))

  #model = unet()
  model = get_unet_inception_resnet_v2(input_shape=(512,512,3),weights=weights)
  print("Compiling Model")
  model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
  model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
  callbacks = [
      EarlyStopping(min_delta=0.005, monitor='accuracy', patience=4, verbose=0),
      ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
  ]
  model.fit_generator(myGene,steps_per_epoch=step_num,epochs=ep_num,callbacks=callbacks)

  
  return model
