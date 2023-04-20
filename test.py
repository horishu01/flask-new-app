import glob
import os
import shutil

filenames = glob.glob("origin_image/*")

for idex, path_ in enumerate(filenames):
   names = path_.split("\\")
   print(names)
   names = names[1].split("_")
   print(names)
   if not os.path.exists(f"origin/{names[0]}"):
    # ディレクトリが存在しない場合、ディレクトリを作成する
    os.makedirs(f"origin/{names[0]}")
   shutil.copy(path_, f"origin/{names[0]}/{idex}.jpg")