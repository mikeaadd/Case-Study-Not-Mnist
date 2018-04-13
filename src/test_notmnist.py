import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import h5py # need this to read weights saved in .h5 files
import numpy as np
import matplotlib.pyplot as plt

# define paths from current working dir (same as training)
current = os.getcwd()
data_dir = os.path.dirname(''.join([current,'/../data/']))
trainpath = data_dir + '/cat_dog/train/' # must match dirs in loop below
testpath = data_dir + '/cat_dog/validation/'

# parameterize and create an image generator to read in test data
test_datagen = ImageDataGenerator(rescale=1. / 255) # if 'predict' doesn't work try not using rescale
img_width, img_height = 150, 150
batch_size = 16
nb_validation_samples = 72 # total number of validation images

validation_generator = test_datagen.flow_from_directory(
    testpath,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# make sure input shape is right
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# build a model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# load saved weights
model.load_weights('cnn_10epochsbatch4.h5')

# compile model
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy']) # might be able to add more metrics here?

# test model
score = model.evaluate_generator(validation_generator, nb_validation_samples // batch_size + 1) # read about size thing on stack exchange
metrics = model.metrics_names
# model.metrics_names to get score labels
print('{} = {}'.format(metrics[0],score[0]))
print('{} = {}'.format(metrics[1],score[1]))
y_pred = model.predict_generator(validation_generator, nb_validation_samples // batch_size + 1, verbose = 1)

# plot predictions (probability of class0, class1)
fig, ax = plt.subplots(1,2,figsize=(8,4))
ax[0].hist(y_pred)
ax[0].set_title('histogram of predictions')
ax[1].plot(range(len(y_pred)),y_pred,'og')
ax[1].set_title('Predictions')
plt.savefig('model_figs/predictions_plot.png',dpi=250)
plt.close()
