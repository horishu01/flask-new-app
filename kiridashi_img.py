import shutil
import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob
import os
#元画像を取り出して顔部分を正方形で囲み、64×64pにリサイズ、別のファイルにどんどん入れてく

in_dir = "./origin"
in_dir_str = f"{in_dir}/*"
out_dir = "./face_image"
in_folder=glob.glob(in_dir_str)
# in_fileName=os.listdir("./origin_image/")


for folder in in_folder:
  for filename in glob.glob(folder+"/*"):
    image=cv2.imread(filename)
    if image is None:
      print("Not open:",image)
      continue
    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    # 顔認識の実行
    face_list=cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=2,minSize=(100,100))
    print(face_list)
    #顔が１つ以上検出された時
    if len(face_list) > 0:
        #for rect in face_list:
           #x,y,width,height=rect
            #image_face = image[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
        for x, y, w, h in face_list:
            image_face = image[y: y + h , x:x + w]
            if image_face.shape[0]<100:
                continue
            image_face = cv2.resize(image_face,(100,100))
            #fileName=out_dir+f"{x}{y}{w}{h}"+filename.replace(in_dir, "")
            #cv2.imwrite(fileName,image_face)
    #顔が検出されなかった時
    else:
        print("no face")
        continue
        print(image_face.shape)
    #print(image_face)
    #print(filename)
    #保存
    #fileName=os.path.join(out_dir, filename.replace(in_dir, ""))
    fileName=out_dir+filename.replace(in_dir, "")
    #print(fileName)
    cv2.imwrite(fileName,image_face)
