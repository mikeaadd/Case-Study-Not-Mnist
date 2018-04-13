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
import sys
from sklearn.metrics import classification_report
import pdb
import pandas as pd

def classifaction_report_csv(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv('classification_report.csv', index = False)
    return dataframe

# def plot_predictions(y_pred,filename):
#     xx = range(y_pred.shape[0])
#     fig, ax = plt.subplots(1,1,figsize=(8,4))
#     for i in range(y_pred.shape[1]):
#         ax.plot(xx,y_pred[:,i])
#         ax.set_title('histogram of predictions')

def plot_predictions(y_pred,filename):
    sums = np.sum(y_pred, axis=0)
    labels = ['A','B','C','D','E','F','G','H','I','J']
    plt.bar(labels,sums)
    plt.title('predictions')
    plt.savefig(filename +'.png',dpi=250)
    plt.close()

if __name__=='__main__':
    # define path from current working directory
    current = os.getcwd()
    data_dir = os.path.dirname(''.join([current,'/../data/']))
    train_data_dir = data_dir + '/train/' # must match dirs in loop below
    test_data_dir = data_dir + '/test/'

    # parameterize and create an image generator to read in test data
    test_datagen = ImageDataGenerator(rescale=1. / 255) # if 'predict' doesn't work try not using rescale
    img_width, img_height = 28, 28
    batch_size = 200
    nb_validation_samples = 72 # total number of validation images
    savename = 'test1'

    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical')

    # make sure input shape is right
    # if K.image_data_format() == 'channels_first':
    #     input_shape = (3, img_width, img_height)
    # else:
    #     input_shape = (img_width, img_height, 3)

    # build a model
    # model = Sequential()
    # model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Conv2D(32, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Conv2D(64, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Flatten())
    # model.add(Dense(64))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1))
    # model.add(Activation('sigmoid'))

    # load saved weights
    from keras.models import load_model
    model = load_model(current + '/' + sys.argv[1])
    print(model.summary())  # As a reminder.

    # compile model
    model.compile(loss='categorical_crossentropy', # binary_crossentropy
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # test model
    score = model.evaluate_generator(test_generator, nb_validation_samples // batch_size + 1) # read about size thing on stack exchange
    metrics = model.metrics_names
    # model.metrics_names to get score labels
    print('{} = {}'.format(metrics[0],score[0]))
    print('{} = {}'.format(metrics[1],score[1]))
    y_pred = model.predict_generator(test_generator, verbose = 1)

    y_test = np.argmax(y_pred, axis=1) # Convert one-hot to index
    report = classification_report(test_generator.classes[test_generator.index_array], y_test)
    report_df = classifaction_report_csv(report)

    # plot predictions (probability of class0, class1)
    plot_predictions(y_pred,'figs/' + savename + '_predictions')
