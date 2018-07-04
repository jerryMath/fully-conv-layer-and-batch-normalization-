import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization

# training X shape (60000, 28x28), Y shape (60000, ). test X shape (10000, 28x28), Y shape (10000, )
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# data pre-processing
x_train = x_train.reshape(-1, 1,28, 28)/255.
x_test = x_test.reshape(-1, 1,28, 28)/255.
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# build model
model = Sequential()

# conv 1
model.add(Convolution2D(
          batch_input_shape=(None, 1, 28, 28),
          filters=32,
          kernel_size=5,
          strides=1,
          padding='same',    
          data_format='channels_first',))#(32, 28, 28)
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))#(32, 14, 14)

# conv 2
model.add(Convolution2D(64, 5, strides=1, padding='same', data_format='channels_first'))#(64, 14, 14)
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))#(64, 7, 7)

# fully convolutional layers
model.add(Convolution2D(128, 7, strides=1, padding='valid', data_format='channels_first'))#(128, 1, 1)
model.add(Dropout(0.5))
model.add(Convolution2D(10, 1, strides=1, padding='same', data_format='channels_first'))#(10, 1, 1)
model.add(Flatten())
model.add(Activation('softmax'))

# compile
model.compile(optimizer=Adam(lr=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# train
print('training ------------')
model.fit(x_train, y_train, epochs=1, batch_size=64,)

# evaluate
print('testing ------------')
loss, accuracy = model.evaluate(x_test, y_test)
print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)

# show layers
print(model.summary())

