import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import optimizers
from keras.callbacks import EarlyStopping


# お使いの仮想環境のディレクトリ構造等によってファイルパスは異なります。
path_bakarizumu = os.listdir("\\Users\\s.hori\Desktop\\new_app\\face_scratch_image\\bakarizumu")
path_hamada_masatoshi = os.listdir("\\Users\\s.hori\\Desktop\\new_app\\face_scratch_image\\hamada_masatoshi")
path_kojima_yoshio = os.listdir("\\Users\\s.hori\\Desktop\\new_app\\face_scratch_image\\kojima_yoshio")
path_tsuchida_teruyuki = os.listdir("\\Users\\s.hori\\Desktop\\new_app\\face_scratch_image\\tsuchida_teruyuki")


img_bakarizumu = []
img_hamada_masatoshi = []
img_kojima_yoshio = []
img_tsuchida_teruyuki = []

for i in range(len(path_bakarizumu)):
    img = cv2.imread("\\Users\\s.hori\\Desktop\\new_app\\face_scratch_image\\bakarizumu\\" + path_bakarizumu[i]) #ディレクトリ名語尾に /
    if (img is None): 
        continue
    b,g,r = cv2.split(img)
    img = cv2.merge([r,g,b])
    img = cv2.resize(img, (50,50))
    img_bakarizumu.append(img)

for i in range(len(path_hamada_masatoshi)):
    img = cv2.imread("\\Users\\s.hori\\Desktop\\new_app\\face_scratch_image\\hamada_masatoshi\\" + path_hamada_masatoshi[i])#ディレクトリ名語尾に/
    if (img is None):
        continue
    b,g,r = cv2.split(img)
    img = cv2.merge([r,g,b])
    img = cv2.resize(img, (50,50))
    img_hamada_masatoshi.append(img)

for i in range(len(path_kojima_yoshio)):
    img = cv2.imread("\\Users\\s.hori\\Desktop\\new_app\\face_scratch_image\\kojima_yoshio\\" + path_kojima_yoshio[i]) #ディレクトリ名語尾に /
    if (img is None):
        continue
    b,g,r = cv2.split(img)
    img = cv2.merge([r,g,b])
    img = cv2.resize(img, (50,50))
    img_kojima_yoshio.append(img)

for i in range(len(path_tsuchida_teruyuki)):
    img = cv2.imread("\\Users\\s.hori\\Desktop\\new_app\\face_scratch_image\\tsuchida_teruyuki\\" + path_tsuchida_teruyuki[i]) #ディレクトリ名語尾に /
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

# データの分割
X_train = X[:int(len(X)*0.8)]
y_train = y[:int(len(y)*0.8)]
X_test = X[int(len(X)*0.8):]
y_test = y[int(len(y)*0.8):]

# 正解ラベルをone-hotの形にします
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# モデルにvggを使います
input_tensor = Input(shape=(50, 50, 3))
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

# vggのoutputを受け取り、4クラス分類する層を定義します
# その際中間層を下のようにいくつか入れると精度が上がります
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(4, activation='softmax')) #Dense 
# vggと、top_modelを連結します
model = Model(vgg16.inputs, top_model(vgg16.output))

# vggの層の重みを変更不能にします
for layer in model.layers[:19]:
    layer.trainable = False

# コンパイルします
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# EaelyStoppingの設定
early_stopping =  EarlyStopping(
                            monitor='val_loss',
                            min_delta=0.0,
                            patience=2,
)

# 学習を行います
history = model.fit(X_train, y_train, batch_size=100, epochs=100, validation_data=(X_test, y_test),
callbacks=[early_stopping] # CallBacksに設定
)
model.save("my_model")
# 画像を一枚受け取り、芸人を判別する。
def separately(img):
    img = cv2.resize(img, (50, 50))
    pred = np.argmax(model.predict(np.array([img])))
    if pred == 0:
        return 'bakarizumu'
    elif pred == 1:
        return 'hamada_masatoshi'
    elif pred == 2:
        return 'kojima_yoshio'
    elif pred == 3:
        return "tsuchida_teruyuki"
    
# 精度の評価（適切なモデル名に変えて、コメントアウトを外してください）
scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# pred_gender関数に写真を渡して分類を予測します
img = cv2.imread("\\Users\\s.hori\\Desktop\\new_app\\face_scratch_image\\bakarizumu\\" + path_bakarizumu[10]) #ディレクトリ名 ＋語尾に/
b,g,r = cv2.split(img)
img1 = cv2.merge([r,g,b])
plt.imshow(img1)
plt.show()
print(separately(img))