# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import skimage.io
import matplotlib.pyplot as plt
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.optimizers import RMSprop
# from sklearn.model_selection import train_test_split
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D
# from keras.layers import Activation, Dropout, Flatten, Dense
#import sys
def loss_curve_plot(history, fname='loss_curves'):
    ''' inputs:
        history is a model object from
            history = model.fit_generator( *args)
        fname is a string with relative path
    '''
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    ax.plot(history.history['loss'],'r',linewidth=3.0)
    ax.plot(history.history['val_loss'],'b',linewidth=3.0)
    ax.legend(['Training loss', 'Validation Loss'],fontsize=18)
    ax.set_xlabel('Epochs ',fontsize=16)
    ax.set_ylabel('Loss',fontsize=16)
    ax.set_title('Loss Curves',fontsize=16)
    plt.savefig(fname + '.png',dpi=250)
    plt.close()

# plot the accuracy curve
def accuracy_curve_plot(history, fname='accuracy_curves'):
    ''' inputs:
        history is a model object from
            history = model.fit_generator( *args)
        fname is a string with relative path to figs dir if you want
    '''
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    ax.plot(history.history['acc'],'r',linewidth=3.0)
    ax.plot(history.history['val_acc'],'b',linewidth=3.0)
    ax.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
    ax.set_xlabel('Epochs ',fontsize=16)
    ax.set_ylabel('Accuracy',fontsize=16)
    ax.set_title('Accuracy Curves',fontsize=16)
    plt.savefig(fname + '.png',dpi=250)
    plt.close()

'''
"Building powerful image classification models using very little data"
from blog.keras.io.
use this:
https://keras.io/preprocessing/image/

SETUP:
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:
data/
    train/
        A/
            dog001.jpg
            dog002.jpg
            ...
        B/
            cat001.jpg
            cat002.jpg
            ...
    test/
        A/
            dog001.jpg
            dog002.jpg
            ...
        B/
            cat001.jpg
            cat002.jpg
            ...
'''


if __name__=='__main__':

    # define path from current working directory
    current = os.getcwd()
    data_dir = os.path.dirname(''.join([current,'/../data/']))
    train_data_dir = data_dir + '/train/' # must match dirs in loop below
    validation_data_dir = data_dir + '/test/'

    ''' input parameters '''
    # setup parameters
    img_width, img_height = 28,28
    batch_size = 200 # number of pictures in each batch
    n_epochs = 15
    n_classes = 10
    pool_size = (2,2)
    nb_train_samples = 15365
    nb_validation_samples = 3363
    save_fname = 'nadam'

    ''' end input parameters '''

    # flexibility for (l,w,3) vs (3,l,w)
    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_width, img_height) # 3 if RGB
    else:
        input_shape = (img_width, img_height, 1)

    ''' build model '''
    model = Sequential()

    # convolution layers
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))

    # flatten to conventional neural net
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    #rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss='categorical_crossentropy', # binary_crossentropy
                  optimizer='nadam',
                  metrics=['accuracy'])

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255)#,
        #shear_range=0.2,
        #zoom_range=0.2,
        #horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    ''' create the generators'''
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        color_mode='grayscale',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        color_mode='grayscale',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    ''' fit model '''
    summary = model.summary() # I think this does something cool

    history = model.fit_generator(
        train_generator,
        epochs=n_epochs,#,
        #steps_per_epoch=nb_train_samples // batch_size,
        validation_data=validation_generator)
        #validation_steps=nb_validation_samples // batch_size)

    # save model
    model.save(save_fname + '.h5')

    # plot the loss curve
    loss_curve_plot(history, fname='figs/' + save_fname + '_loss_curves')
    accuracy_curve_plot(history, fname='figs/' + save_fname + '_accuracy_curves')
