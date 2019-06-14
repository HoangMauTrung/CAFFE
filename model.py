import tensorflow 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
import load_data
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
import numpy as np

(X, y) = load_data.load()
gle = LabelEncoder()
labels = gle.fit_transform(y)
mappings = { index: label for index, label in enumerate(gle.classes_)}
print(mappings)

label_binary = LabelBinarizer()
img_label = label_binary.fit_transform(labels)

X = np.asanyarray(X)
img_label = np.asanyarray(img_label)

from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X, img_label, test_size= 0.2,random_state=42)

print('Data train :' ,X_train.shape)
print('Data test : ',X_test.shape)


model = Sequential()

inputShape = (224, 224, 3)
model.add( Conv2D( 16, (3,3), padding= 'same',input_shape = inputShape ))
model.add(Activation("relu"))
model.add(BatchNormalization(axis = -1))
model.add(MaxPool2D(pool_size=(3,3)))
model.add(Dropout(0.25))

model.add(Conv2D(32,(5,5), padding = 'same'))
model.add(Activation("relu"))
model.add(BatchNormalization(axis = -1))
model.add(MaxPool2D( pool_size = (3,3)))
model.add(Dropout(0.25))


model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(BatchNormalization(axis = -1))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(28))

model.add(Activation('sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

print("[INFO] : trainning network .......")

model.fit(X_train, y_train, batch_size= 100, epochs= 15, validation_data= (X_test,y_test))

model.save('model_predict.h5')
