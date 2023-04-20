import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras import optimizers


# お使いの仮想環境のディレクトリ構造等によってファイルパスは異なります。
path_bakarizumu = os.listdir("\\Users\\s.hori\\Desktop\\new_app\\test_image\\bakarizumu")
path_hamada_masatoshi = os.listdir("\\Users\\s.hori\\Desktop\\new_app\\test_image\\hamada_masatoshi")
path_kojima_yoshio = os.listdir("\\Users\\s.hori\\Desktop\\new_app\\test_image\\kojima_yoshio")
path_tsuchida_teruyuki = os.listdir("\\Users\\s.hori\\Desktop\\new_app\\test_image\\tsuchida_teruyuki")


img_bakarizumu = []
img_hamada_masatoshi = []
img_kojima_yoshio = []
img_tsuchida_teruyuki = []

for i in range(len(path_bakarizumu)):
    img = cv2.imread("\\Users\\s.hori\\Desktop\\new_app\\test_image\\bakarizumu\\" + path_bakarizumu[i]) #ディレクトリ名語尾に /
    if (img is None): 
        continue
    b,g,r = cv2.split(img)
    img = cv2.merge([r,g,b])
    img = cv2.resize(img, (50,50))
    img_bakarizumu.append(img)

for i in range(len(path_hamada_masatoshi)):
    img = cv2.imread("\\Users\\s.hori\\Desktop\\new_app\\test_image\\hamada_masatoshi\\" + path_hamada_masatoshi[i])#ディレクトリ名語尾に/
    if (img is None):
        continue
    b,g,r = cv2.split(img)
    img = cv2.merge([r,g,b])
    img = cv2.resize(img, (50,50))
    img_hamada_masatoshi.append(img)

for i in range(len(path_kojima_yoshio)):
    img = cv2.imread("\\Users\\s.hori\\Desktop\\new_app\\test_image\\kojima_yoshio\\" + path_kojima_yoshio[i]) #ディレクトリ名語尾に /
    if (img is None):
        continue
    b,g,r = cv2.split(img)
    img = cv2.merge([r,g,b])
    img = cv2.resize(img, (50,50))
    img_kojima_yoshio.append(img)

for i in range(len(path_tsuchida_teruyuki)):
    img = cv2.imread("\\Users\\s.hori\\Desktop\\new_app\\test_image\\tsuchida_teruyuki\\" + path_tsuchida_teruyuki[i]) #ディレクトリ名語尾に /
    if (img is None):
        continue
    b,g,r = cv2.split(img)
    img = cv2.merge([r,g,b])
    img = cv2.resize(img, (50,50))
    img_tsuchida_teruyuki.append(img)

X = np.array(img_bakarizumu + img_hamada_masatoshi + img_kojima_yoshio + img_tsuchida_teruyuki)
y =  np.array([0]*len(img_bakarizumu) + [1]*len(img_hamada_masatoshi) + [2]*len(img_kojima_yoshio) + [3]*len(img_tsuchida_teruyuki) )#ディレクトリ名語尾に/

rand_index = np.random.permutation(np.arange(len(X)))
X = X[rand_index]
y = y[rand_index]
y = to_categorical(y)
reconstructed_model = load_model("my_model")
scores = reconstructed_model.evaluate(X, y, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
def separately(img):
    img = cv2.resize(img, (50, 50))
    pred = np.argmax(reconstructed_model.predict(np.array([img])))
    if pred == 0:
        return 'bakarizumu'
    elif pred == 1:
        return 'hamada_masatoshi'
    elif pred == 2:
        return 'kojima_yoshio'
    elif pred == 3:
        return "tsuchida_teruyuki"
# pred_gender関数に写真を渡して分類を予測します
img = cv2.imread("\\Users\\s.hori\\Desktop\\new_app\\test_image\\bakarizumu\\" + path_bakarizumu[10]) #ディレクトリ名 ＋語尾に/
b,g,r = cv2.split(img)
print("\\Users\\s.hori\\Desktop\\new_app\\test_image\\bakarizumu\\" + path_bakarizumu[10])
img1 = cv2.merge([r,g,b])
#plt.imshow(img1)
#plt.show()
print(separately(img))
