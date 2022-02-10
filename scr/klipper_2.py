from shutil import rmtree
import cv2
import os

if os.path.exists("membrane/membrane/g2_train"):
  rmtree('membrane/membrane/g2_train') 

os.mkdir('membrane/membrane/g2_train')

def crop_img(train_v, inp_img, out_fold, addd):

  font                   = cv2.FONT_HERSHEY_SIMPLEX
  fontScale              = 3
  fontColor              = (255,0,0)
  linetype               = 2

  if addd == '':
    img0 = cv2.imread(inp_img,0)
  else:
    img0 = cv2.imread(inp_img)
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
      if n_num in [0,2,4,8]:
        cv2.imwrite(out_fold + '/' +
                    str(n_num) + addd + '.png', px)
      
      #bottomLeftCornerOfText = ((i+1)*train_v,j_b)
      bottomLeftCornerOfText = (j_b, (i+1)*train_v)
      j_b = j_b + train_v

      cv2.putText(img2,str(n_num), 
                  bottomLeftCornerOfText, 
                  font, 
                  fontScale,
                  fontColor,
                  linetype)

      n_num = n_num + 1

    i_b = i_b + train_v

  cv2.imwrite('imm2_2.png', img2)
  return n_num

