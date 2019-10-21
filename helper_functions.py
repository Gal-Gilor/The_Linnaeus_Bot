from PIL import Image
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import plotly.graph_objects as go
import joblib as jb
import itertools
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
import pydot
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Input, MaxPool2D, UpSampling2D
from keras.models import Sequential, load_model, Model
from keras.utils import to_categorical, plot_model, vis_utils
from keras.preprocessing import image
from keras import backend as k

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

## Image processing


def process_images(inpath, outpath, dim_tuple, start=1, extension='jpg'):
    '''
    process_images(inpath, outpath, dim_tuple, extension, start=1):
    This function creates new images, reshapes, grayscale, and save the images in a desired location
    Input:
        inpath: Path to the image files
        outpath: Path to where the images should be saved
        dim_tuple: Desired image size
        extension: The image file type. default='jpg'
        start: Part saved image name. default=1
    Returns:
        Returns a folder containing double amount of images
    '''
    #open images
    for file in tqdm(glob(f'{inpath}*.{extension}')):
        with Image.open(file) as img:
            #rotate image
            rotated_images = rotate_images(img)
            
            # resize images
            resized = resizing(rotated_images, dim_tuple)
            
            #grayscale images
            gray_images = grayscale(resized)
            
            
            # save the images
            save_preprocessed_images(gray_images, outpath, extension, start)
            start += 2                      
    return


def rotate_images(image):
    '''
    rotate_images(image):
    This function rotates an image on it's center 7 times (45, 90, 135, 180, 225, 270, and mirror image)
    Input:
        image: One image file
    Returns:
        A list of images containing the original image and the rotated one
    '''
    rotated_images = []
    
    chirl_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    
    rotated_images.extend([image, chirl_image])  
    return rotated_images


def resizing(images, dim_tuple):
    '''
    resizing(images, dim_tuple):
    This function resizes a list of images
    Input:
        images: List of images
        dim_tuple: Tuple containing the desired hight and the width
    Returns:
        List of resized images        
    '''
    resized = [image.resize(dim_tuple) for image in images]
    return resized


def grayscale(images):
    '''
    grayscale(images):
    This transforms RGB images to grayscale images
    Input:
        images:List of RBG images
    Returns:
        List of grayscale images
    '''
    gray_images = [image.convert(mode='L') for image in images]
    return gray_images


# save images in a different path
def save_preprocessed_images(processed_images, outpath, start, extension='jpg'):
    '''
    save_preprocessed_images(processed_images, outpath, extension, start)
    This function saves any type of image in a specific directory
    Input:
        processed_images: List of images
        outpath: Where you want the files to be saved
    Returns:
        Does not return a variable. Creates image files 
    '''
    [image.save(f'{outpath}\\image{i}.{extension}') for i, image in enumerate(processed_images, start)]
    return


def save_train_test(inpath, outpath, n, j, extension='jpg'):
    '''
    save_train_test(inpath, outpath, extension, n, label):
    This function converts an image file to a numpy array, normalizes the the pixel values and saves
    Input:
        outpath: The path where you want to save the file
        n: int. slicing the matrix
        j: int. slicing the matrix
            
    '''
    images = []
    for file in tqdm(glob(f'{inpath}*.{extension}')[n:j]):
        with Image.open(file) as img:
            np_image = np.array(img) / 255
            np_image = np_image.expand_dims(np_image, axis=0)
            images.append(np_image)
    
    images = np.asarray(images)   
    np.save(outpath, images)
    del images
    return


## Image classification


def load_data(damselpath, dragonpath, start, end):
    '''
    This function loads a pre-saved numpy arrays with their labels
    load_data(damselpath, dragonpath, start, end):
    Input:
        damselpath: Damselflies numpy location
        dragonpath: Dragonflies numpy location
        start: Integer for slicing the arrays
        end: Integer for slicing the arrays
    Returns:
        Retuns a concatenated numpy array with dragonfly and damselfly images (dragons first),
        and a vector the size of the image vector where dragons value is 1 and damsel 0
    '''
    data = np.concatenate((np.load(dragonpath)[start:end], np.load(damselpath)[start:end]), axis=0)
    dragon_images = len(np.load(dragonpath)[start:end])
    labels = labels_vector(data, dragon_images)                           
    return data, labels


def labels_vector(data, dragon_images):
    '''
    This function creates a labels vector
    labels_vector(data, dragon_images):
    Input:
        data: List containing the images in numpy array format
        dragon_images: The the amount of dragonfly images in the array
    Returns:
        This function returns a binary vector the length of the given data vector.
        The function assumes the vector is order dragonfly images come first
    '''
    labels = np.zeros(len(data))
    labels[:dragon_images] = 1
    return labels


def create_model(weights_path=None, weights_name=None):
    '''
    This function create a sequantial model with input shape of (256 256, 1)
    create_model(weights_path=None, weights_name=None):
    Input:
        weights_path: The full path to the model weights directory. default=None
        weights_name: Weights file name. default=None
    Returns:
        This model returns a compiled sequantial model with input shape (256 256, 1)
        and output shape of 2.
    '''
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='tanh',
                     input_shape=(256, 256, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))

    model.add(Dense(2, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
    # optional: loading previous weights
    if weights_path and weights_name:
        model.load_weights(f'{weights_path}{weights_name}')
    return model


def create_train_validation(training_data, labels):
    '''
    This function seprates the images and labels to training and validation sets
    create_train_validation(training_data, labels):
    Input:
        training_data: Images in numpy array format
        labels: Vector with the image labels
    Returns:
        Shuffled, reshaped arrays with matching labels
    '''
    # split to sets
    x_train, x_val, y_train, y_val = train_test_split(training_data, labels, test_size=0.05)
    
    # adjust variables dimensions to fit into the cnn    
    x_train = np.expand_dims(x_train, axis=-1)
    x_val = np.expand_dims(x_val, axis=-1)
    
    y_train_cat = to_categorical(y_train)
    y_val_cat = to_categorical(y_val)
    return x_train, x_val, y_train, y_val, y_train_cat, y_val_cat


## Image visualisation


def load_model_stats(path):
    '''
    This function extracts the train history from keras train history object
    load_model_stats(path):
    Input:
        path: Path to folder where the history objects are located
    Returns:
        Merged training/validation accuracy, training/validation loss for the entire
        classifaction model process
    '''
    # load the files
    history = [jb.load(file) for file in glob(path)]
    
    # combine the accuracy and loss changes throughout the entire model training
    # training accuracy
    train_acc = [keras_object.history['acc'] for keras_object in history]
    train_acc = list(itertools.chain.from_iterable(train_acc))
    
    # validation accuracy
    val_acc = [keras_object.history['val_acc'] for keras_object in history]
    val_acc = list(itertools.chain.from_iterable(val_acc))
    
    # training loss
    train_loss = [keras_object.history['loss'] for keras_object in history]
    train_loss = list(itertools.chain.from_iterable(train_loss))
    
    # validation loss
    val_loss = [keras_object.history['val_loss'] for keras_object in history]
    val_loss = list(itertools.chain.from_iterable(val_loss))
    return [train_acc, val_acc], [train_loss, val_loss]



def plot_train_history(data, title, ytitle, xtitle, save=False):
    '''
    This function creates a time series plot
    load_model_stats(path):
    Input:
        data: List of lists Training data in the first cell and validation data in the second
        title: Plot title
        ytitle: Y axis label
        xtitle: x axis label
        save: Save name (optional)      
    Returns:
        Time series plot showing the change in accuracy or loss throughout the model training
    '''
    fig = go.Figure()
    
    # add traces
    fig.add_trace(go.Scatter(y=data[0],
                  mode='lines',
                  name=f'Training {ytitle}'))
    
    fig.add_trace(go.Scatter(y=data[1],
                  mode='lines',
                  name=f'Validation {ytitle}'))
        

    # add labels, change figure/font sizes, and remove grid lines
    fig.update_layout(autosize=False,
                      width=600,
                      height=550,
                       
                      xaxis=go.layout.XAxis(
                          title=xtitle,
                          titlefont=dict(size=18),
                          showgrid=False),
                      
                      yaxis=go.layout.YAxis(
                          title=ytitle,
                          titlefont=dict(size=18),
                          showgrid=False),

                      title=go.layout.Title(
                          text=title,
                          font=dict(size=22),
                          xref='paper')
                      )

    fig.update_yaxes(automargin=True)
    # optional: saving image to a specific location
    if save:
        fig.write_image(f'./Images/model_{save}.png')
    fig.show()
    return


def plot_bar_graph(x, y, title, ytitle, xtitle, save=False):
    '''
    This function creates a bar graph
    load_model_stats(path):
    Input:
        x: X axis data
        y: Y axis data
        title: Plot title
        ytitle: Y axis label
        xtitle: x axis label
        save: Save name (optional)       
    Returns:
        Bar graph showing the amount of images in each class
    '''
    fig = go.Figure()
    
    # add trace
    fig.add_trace(go.Bar(x = x,
                         y = y)
                 )
    
    # add labels, change figure/font sizes, and remove grid lines
    fig.update_layout(autosize=False,
                      width=600,
                      height=550,
                      
                      xaxis=go.layout.XAxis(
                          title_=xtitle,
                          titlefont=dict(size=18),
                          showgrid=False),

                      yaxis=go.layout.YAxis(
                          title_text=ytitle,
                          titlefont=dict(size=18),
                          showgrid=False),

                      title=go.layout.Title(
                          text=title,
                          font=dict(size=22),
                          xref='paper')
                      )

    fig.update_yaxes(automargin=True)
    # optional: saving image to a specific location
    if save:
        fig.write_image(f'./Images/model_{save}.png')
    fig.show()
    return


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
    
    #compile
    model.compile(optimizer='Adam', loss='mse')
    # optional: loading previous weights
    if weights_path and weights_name:
        model.load_weights(f'{weights_path}{weights_name}')
        
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
        n_features = layer_activation.shape[-1] # number of features
        size = layer_activation.shape[1] # feature map 
        n_cols = n_features // images_per_row # tiles the activation channels 
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        
        for col in range(n_cols): # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                # post-processes the features to visualize the activation layers
                channel_image = layer_activation[0, :, :,
                                                 col * images_per_row + row]
                channel_image -= channel_image.mean() 
                channel_image /= channel_image.std()  
                channel_image *= 64                   
                channel_image += 128                
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size, 
                             row * size : (row + 1) * size] = channel_image
    
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        if save:
            plt.savefig(f'activation_{save}_vis.jpg')
            save += 1
    return


def print_metrics(labels, predictions, print_score=True):
    '''
    This function prints model metrics
    print_metrics(labels, predictions, print_score=None):
    Input:
        labels: True labels
        predictions: Model predictions
        print_score: Optional, printing the scores
    Returns:
        2 variables, one for recall score and one for accuracy score
    This function receives model predictions along with the actual labels
        and returns the precision score, recall, accuracy and F1'''

    recall = round(recall_score(labels, predictions)*100, 2)
    acc = round(accuracy_score(labels, predictions)*100, 2)
   
    if print_score:
        print(f"Recall: {recall}")
        print(f"Accuracy: {acc}")

    return recall, acc


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
    matfig = plt.figure(figsize=(11,10))
    matrix = hlf.confusion_matrix(y_test, y_pred)
    plt.matshow(matrix, cmap=plt.cm.Purples, aspect=1.2, 
                alpha=0.6, fignum=matfig.number)
    plt.grid(b=None)
    # add color bar
    plt.colorbar()

    # add title and Axis Labels
    plt.title('Confusion Matrix', fontsize=20)
    plt.ylabel('Actual', fontsize=16)
    plt.xlabel('Predicted', fontsize=16)

    # add appropriate Axis Scales
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.grid(b=None)

    # add Labels to Each Cell
    thresh = matrix.max() / 2.  

    # iterate through the confusion matrix and append the labels
    for i, j in itertools.product(range(matrix.shape[0]), 
                                  range(matrix.shape[1])):
        plt.text(j, i, matrix[i, j], 
                 horizontalalignment="center",
                 color="black")

    plt.grid(b=None)
    if save:
        plt.grid(b=None)
        plt.tight_layout()
        plt.savefig(f'{save}_cm.png')
    return


def find_wrong_classification(images, labels, predictions):
    '''
    This function finds misclassifcation
    find_wrong_classification(images, labels, predictions):
    Input:
        images: Images
        labels: Actual labels
        prediction: Predicted labels
    Returns:
        All the misclassified images
    '''
    indices =  np.where(labels != predictions)
    return images[indices]