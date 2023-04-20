#左右反転、閾値処理、ぼかしで８倍の水増し
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
def scratch_image(img, flip=True, thr=True, filt=True):
    # 水増しの手法を配列にまとめる
    methods = [flip, thr, filt]
    # ぼかしに使うフィルターの作成
    filter1 = np.ones((3, 3))
    # オリジナルの画像データを配列に格納
    images = [img]
    # 手法に用いる関数
    scratch = np.array([
    lambda x: cv2.flip(x, 1),
    lambda x: cv2.threshold(x, 100, 255, cv2.THRESH_TOZERO)[1],
    lambda x: cv2.GaussianBlur(x, (5, 5), 0),
    ])
    # 加工した画像を元と合わせて水増し
    doubling_images = lambda f, imag: np.r_[imag, [f(i) for i in imag]]
    for func in scratch[methods]:
        images = doubling_images(func, images)
    return images
# 画像の読み込み
name_l=["bakarizumu","hamada_masatoshi","kojima_yoshio","tsuchida_teruyuki"]
for name_geinin in name_l:
    in_dir = f"./face_image/{name_geinin}/*"
    in_jpg=glob.glob(in_dir)
    img_file_name_list=os.listdir(f"./face_image/{name_geinin}/")
    for i in range(len(in_jpg)):
        print(str(in_jpg[i]))
        img = cv2.imread(str(in_jpg[i]))
        #img = cv2.imread(str(."face_image"))
        scratch_face_images = scratch_image(img)
        for num, im in enumerate(scratch_face_images):
            fn, ext = os.path.splitext(img_file_name_list[i])
            file_name=os.path.join(f"./face_scratch_image/{name_geinin}",str(fn+"_"+str(num)+".jpg"))
            cv2.imwrite(str(file_name) ,im)
