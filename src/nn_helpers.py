import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import keras
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import pdb
from keras import models

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
    plt.savefig('model_figs/accuracy_curves.png',dpi=250)
    plt.close()

def activation_visualizations(model, img_tensor):

    # Extracts the outputs of the top 8 layers:
    layer_outputs = [layer.output for layer in model.layers[:9]]
    # Creates a model that will return these outputs, given the model input:
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

    # This will return a list of 5 Numpy arrays:
    # one array per layer activation
    activations = activation_model.predict(img_tensor)

    # These are the names of the layers, so can have them as part of our plot
    layer_names = []
    for layer in model.layers[:9]:
        layer_names.append(layer.name)

    images_per_row = 16

    i = 1

    # Now let's display our feature maps
    for layer_name, layer_activation in zip(layer_names, activations):

        # This is the number of features in the feature map
        n_features = layer_activation.shape[-1]

        # The feature map has shape (1, size, size, n_features)
        size = layer_activation.shape[1]

        # We will tile the activation channels in this matrix
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))

        # We'll tile each filter into this big horizontal grid
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                                                 :, :,
                                                 col * images_per_row + row]
                # Post-process the feature to make it visually palatable
                # pdb.set_trace()
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                if i < 3:
                    channel_image *= 32
                else:
                    channel_image += 64
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size,
                             row * size : (row + 1) * size] = channel_image
        i +=1

        # Display the grid
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')

    # plt.show()

if __name__ == '__main__':
    plt.close("all")
    model = load_model('/Users/sec/galvanize/case_studies/Case-Study-Not-Mnist/src/15epochs.h5')
    print(model.summary())  # As a reminder.
    img_path = '/Users/sec/galvanize/case_studies/Case-Study-Not-Mnist/data/train/G/SVRDIEN1c2hpbmcgSGVhdnkucGZi.png'

    # Import image and conver to tensor
    img = image.load_img(img_path, grayscale=True, target_size=(28, 28))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)

    # The model was trained on inputs that were preprocessed in the following way:
    img_tensor /= 255.

    # Show layer filters
    activation_visualizations(model, img_tensor)

    # Show Image
    # plt.imshow(img)
    # plt.show()
