import matplotlib.pylab as plt
import numpy as np
import pandas as pd

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
