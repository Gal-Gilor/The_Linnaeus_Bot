import os
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib as jb


from typing import Dict, Tuple, List, Union, Optional

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score

import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Input, UpSampling2D
from keras.models import Sequential, Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False


def annotate(axes:Union[np.ndarray, plt.Axes]):
    '''
    annotate a singular ax or a matrix of subplots ax
    inputs: 
        axes: singular matplotlib.pyplot ax or a matrix or ax
    '''
    def _annotate(ax: plt.Axes):
        for p in ax.patches:

            # establish text position
            _horizontal_pos = p.get_x() + p.get_width() / 2
            _vertical_pos = p.get_y() + p.get_height() / 2

            # define the value to add to the graph
            value = f'{int(p.get_height())}' 

            # add text to graph
            ax.text(_horizontal_pos, _vertical_pos, value,  ha="center", fontsize=16) 
            pass
    
    # check whether if axes is a singular ax or a matrix of ax
    if isinstance(axes, np.ndarray):
        for _, ax in np.ndenumerate(axes):
            _annotate(ax)
    else:
        _annotate(axes)
    pass

# Image visualisation



def create_unsupervised_model(weights_path=None, weights_name=None, shape=(256, 256, 1)):
    '''
    This function create a keras Model with input shape of (256 256, 1)
    create_unsupervised_model(weights_path=None, weights_name=None, shape=(256, 256, 1)):
    Input:
        weights_path: The full path to the model weights directory. default=None
        weights_name: Weights file name. default=None
        shape: input shape. default=(256, 256, 1)
    Returns:
        This model returns a compiled Model with input shape (256 256, 1)
        and output shape of 1.
    '''
    input_dim = Input(shape=shape)

    # encoded representation of the input
    encoded = Conv2D(48, (3, 3), activation='relu', padding='same')(input_dim)
    encoded = MaxPool2D((2, 2), padding='same')(encoded)

    encoded = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    encoded = MaxPool2D((2, 2), padding='same')(encoded)

    encoded = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    encoded = MaxPool2D((2, 2), padding='same')(encoded)

    encoded = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    encoded = MaxPool2D((2, 2), padding='same')(encoded)

    # reconstruction of the input
    decoded = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    decoded = UpSampling2D((2, 2))(decoded)

    decoded = Conv2D(16, (3, 3), activation='relu', padding='same')(decoded)
    decoded = UpSampling2D((2, 2))(decoded)

    decoded = Conv2D(32, (3, 3), activation='relu', padding='same')(decoded)
    decoded = UpSampling2D((2, 2))(decoded)

    decoded = Conv2D(48, (3, 3), activation='relu', padding='same')(decoded)
    decoded = UpSampling2D((2, 2))(decoded)

    decoded = Conv2D(1, (3, 3), padding='same')(decoded)

    # define model input and output
    model = Model(input_dim, decoded)

    # compile
    model.compile(optimizer='Adam', loss='mse')
    # optional: loading previous weights
    if weights_path and weights_name:
        model.load_weights('{}{}'.format(weights_path, weights_name))

    return model


def plot_intermediate_activation(model, images, bottom=0, top=3, save=False):
    '''
    This function plots the intermidiate interpartation of an image
    plot_intermediate_activation(model, images, bottom=0, top=3, save=False):
    Inputs:
        model: Compiled keras model object
        images: Numpy array containing image you want to disply
        bottom: Which layer to begin showing
        top: Which layer to stop showing
        save: Optional, saving as .png image object. default=False
    Returns:
         Images displaying what each convolution layer detects        
    '''
    images = np.expand_dims(images, axis=-1)

    # extract layer's outputs
    layer_outputs = [layer.output for layer in model.layers[bottom:top]]

    # Creates keras Model
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(images)
    layer_names = [layer.name for layer in model.layers[bottom:top]]

    # Define how the amount of images per row
    images_per_row = 16

    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]  # number of features
        size = layer_activation.shape[1]  # feature map
        n_cols = n_features // images_per_row  # tiles the activation channels
        display_grid = np.zeros((size * n_cols, images_per_row * size))

        for col in range(n_cols):  # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                # post-processes the features to visualize the activation layers
                channel_image = layer_activation[0, :, :,
                                                 col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size,
                             row * size: (row + 1) * size] = channel_image

        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        if save:
            plt.savefig('activation_{}_vis.jpg'.format(save))
            save += 1
    return

def plot_confusion_matrix(y_test, y_pred, class_names, save=None):
    '''
    plot_confusion_matrix(y_test, y_pred, class_names, save=False, name='name')
    Params:
        y_test: list. The true labels
        y_pred: list. The model's labels
        class_names: list, the classes names'
        save: Default=None. Saves the image as png

    Returns:
        Returns confusion matrix plot
    '''
    plt.rcParams["axes.grid"] = False
    plt.rcParams['figure.figsize'] = 10, 10
    plt.rcParams['axes.spines.right'] = True
    plt.rcParams['axes.spines.top'] = True

    matrix = confusion_matrix(y_test, y_pred)
    plt.matshow(matrix, cmap=plt.cm.Blues, aspect=1.2,
                alpha=0.6)

    # add title and Axis Labels
    plt.title('Confusion Matrix', fontsize=20)
    plt.ylabel('Actual', fontsize=16)
    plt.xlabel('Predicted', fontsize=16)

    # append text
    for i, j in itertools.product(range(matrix.shape[0]),
                                  range(matrix.shape[1])):
        plt.text(j, i, matrix[i, j],
                 horizontalalignment="center",
                 color="black")

    if save:
        plt.tight_layout()
        plt.savefig('{}_cm.png'.format(save))

    # add color bar
    plt.colorbar()
    return

def save_predict_class(model, images, labels, path):
    '''
    This function compiles and saves different prediction metrics
    save_predict_class(model, images, labels):
    Input:
        model: Model object
        images: NumPy array shape=(256, 256, 1)
        labels: Actual labels
    Returns:
        A dictionary with the raw images, number of images, model accuracy, model loss,
        model predictions,  predicted class names, actual labels, and 
        misclassified images
    '''
    dictionary = {}

    # predict
    labels_cat = to_categorical(labels)
    preds = model.predict(images)
    classes = model.predict_classes(images)
    loss, acc = model.evaluate(images, labels_cat)
    misclassified = find_wrong_classification(images, labels, classes)

    # compile the results
    dictionary['raw'] = images
    dictionary['n_images'] = len(labels)
    dictionary['acc'] = acc
    dictionary['loss'] = loss
    dictionary['pred_weights'] = preds
    dictionary['pred_class'] = classes
    dictionary['actual'] = labels
    dictionary['misclassified'] = misclassified

    # save the condensed results
    jb.dump(dictionary, path)
    print('Prediction Accuracy: {}%'.format(acc * 100))
    return dictionary
