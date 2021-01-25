from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential

import numpy as np
import matplotlib.pyplot as plt

# load data
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

SIZE = 64

images = []
for i in range(1,101):
    img = load_img(
        r'D:/Atik/pythonScripts/WCNN/Dataset/cnnData/Encoder/fakeFIG_{}.png'.format(i),
               grayscale=False, color_mode="rgb")
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    images.append(img)
   
x_train_noisy = np.vstack(images)
noisy_train = x_train_noisy / 255.

images = []
for i in range(1,101):
    img = load_img(
        r'D:/Atik/pythonScripts/WCNN/Dataset/cnnData/Encoder/FIG{}.png'.format(i),
               grayscale=False, color_mode="rgb")
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    images.append(img)
   
x_train = np.vstack(images)
clean_train = x_train / 255.


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(SIZE, SIZE, 3)))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
 
model.add(MaxPooling2D((2, 2), padding='same'))
     
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

model.summary()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(noisy_train, clean_train, 
                                                    test_size = 0.20, random_state = 0)


model.fit(x_train, y_train, epochs=5000, batch_size=16, shuffle=True, verbose = 1,
          validation_split = 0.1)


print("Test_Accuracy: {:.2f}%".format(model.evaluate(np.array(x_test), np.array(y_test))[1]*100))

no_noise_img = model.predict(x_train)

plt.figure(figsize=(40, 4))
for i in range(3):
    # display original
    ax = plt.subplot(3, 20, i + 1)
    plt.imshow(y_test[i])
    
    # display reconstructed (after noise removed) image
    ax = plt.subplot(3, 20, 40 +i+ 1)
    plt.imshow(no_noise_img[i])
plt.show()