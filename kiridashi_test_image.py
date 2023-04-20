import shutil
import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob
import os
import random
name_l=["bakarizumu","hamada_masatoshi","kojima_yoshio","tsuchida_teruyuki"]
for name_geinin in name_l:
    in_dir="./face_image"
    in_dir = f"{in_dir}/{name_geinin}"
    os.makedirs(f"./test_image/{name_geinin}", exist_ok=True) 
    #img_file_name_listをシャッフル、そのうち2割をtest_imageディテクトリに入れる

    in_jpg = glob.glob(in_dir+"/*")
    random.shuffle(in_jpg)
    num = len(in_jpg)//5
    for i in range(num):
        shutil.copy(in_jpg[i], f"./test_image/{name_geinin}")  