# A very simple perceptron for classifying american sign language letters
import signdata
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from keras.utils import np_utils
import wandb
from wandb.keras import WandbCallback
from keras.optimizers import RMSprop

# logging code
run = wandb.init()
config = run.config
config.team_name = "teambob"
config.loss = "categorical_crossentropy"
config.optimizer = "adam"
config.epochs = 10
input_shape = (28, 28, 1)

if (config.team_name == 'default'):
    raise ValueError("Please set config.team_name to be your team name")

# load data
(X_test, y_test) = signdata.load_test_data()
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
(X_train, y_train) = signdata.load_train_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

img_width = X_test.shape[1]
img_height = X_test.shape[2]

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_train.shape[1]

# you may want to normalize the data here..

# create model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=config.epochs, validation_data=(X_test, y_test),
                    callbacks=[WandbCallback(data_type="image", labels=signdata.letters)])