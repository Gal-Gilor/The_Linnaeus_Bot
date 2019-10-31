# The_Linnaeus_Bot

## Introduction

This project utilizes deep learning neural networks to classify images of damselflies and dragonflies and to generate images through image de-noising techniques (auto-encoders).

_For those interested in a shorter recap:_ [_Presentation Slides_](https://docs.google.com/presentation/d/1xrlXFUVkA1hmsYD6TaPTXyuSGk6ACH_ZHcb40gw1cX0/edit?usp=sharing "Presentation")

### Table of Contents

   - [Tech Stack](#tech-stack)
  
   - [Process](#process)
  
   - [Data and EDA](#data-and-eda)
       
   - [Supervised Model](#supervised-model)

   - [Unsupervised Model](#unsupervised-model)

   - [Future Improvements](#future-improvements)

## Tech Stack

- Python libraries
  
    - [NumPy](https://www.numpy.org/ "Numpy")

    - [Pillow](https://pillow.readthedocs.io/en/stable/ "Pillow")
    - [Keras](https://keras.io/ "Keras")

    - [Scikit-learn](https://scikit-learn.org/stable/ "Sklearn")
    - [Matplotlib](https://matplotlib.org/ "Matplotlib")
    - [Plotly](https://plot.ly/ "Plotly")

## Process

For this project, I used part of a google competition dataset; [iNat Challange 2019](https://sites.google.com/view/fgvc6/competitions/inaturalist-2019/ "iNat Challange 2019"). The dataset contained 8462 damselfly images (1.72 GB) and 9197 dragonfly images (1.76 GB). All the images were resized to have a maximum dimension of 800 pixels and saved as JPEG. I then process the images and train a convolutional neural network (CNN) to distinguish between a dragonfly and a damselfly. Additionally, I experimented with image de-noising techniques using CNN's to generate images of dragonflies for classification purposes.
  
## Data and EDA

The original dataset contained 82 GB of images for various living organisms. Due to time constraint, I focused on the 8462 damselfly, and 9197 dragonfly images.

<img src=Images/image_pie.png alt="Classes pie chart" width="400"/>

As part of the image processing stage, I resize every image to 256 by 256 pixels, grayscale, convert the image to a Numpy array, and normalized the pixel values by dividing every pixel by 255. Additionally, I augmented the data and created the mirror image to doubled the number of available images.

<img src=Images/class_balance.png alt="Final image count" width="400"/>

### Creating the Test Set

After preparing the images for analysis I had 16924 (215 MB) damselfly images, and 18394 (237 MB) dragonfly images. The training set comprised of 12694 damselfly and 13797 dragonfly images (26491‬ images; 75% of the data). The test set comprised of  4230 damselfly and 4597 dragonfly images (8827‬ images; 25% of the data). Due to limited computational power, I saved the training and testing sets for damselflies and dragonflies separately in 4 different .npy files.

## Supervised model

#### Model Architecture

![CNN Architecture](Images/CNN_arch.png)

To complete the task of training the CNN, batching the data was necessary. Every batch consists of 4000 images, of which 5% reserved for validation purposes.

  * Train on 4000 Images
  * Save Weights
  * Clear Cache
  * Reload Weights
  * Retrain on 4000 New Images
  * Repeat 6 Times (24000 images total)

<img src=Images/model_accuracy.png alt="Training accuracy history" width="400"/> <img src=Images/model_accuracy.png alt="Training loss history" width="400"/>

After training the model on 24000 images the model achieves 85%~ accuracy on the testing set (8827‬ images)

![Confusion matrix](Images/supervised_cm_label.png)

## Unsupervised models

#### Model Architecture

![Autoencoder Architecture](Images/autoencoder_sum.png)

### Topic Modeling with LDA

Before running the model we noticed additional    processing is needed. We began by removing single character words and all the stand-alone digits. Unsure about the pros and cons of the different libraries for NLP, we utilized both Gensim and scikit-learn to run LDA models for topic modeling.

- scikit-learn
  1) We chose 14 as the number of topics (Amazon electronics department is made out of 14 sections). Additionally, we filtered out words that appeared in more than 50% of the reviews and words that appeared in less than 10 reviews. Looking at topics, we noticed that some words appear on several topics, meaning the topics are not independent of each other.

  2) We lowered the number of topics to 10 filtered words that appeared in more than 50% of reviews, and words that appeared in less than 15 reviews. Lowering the number of topics helped address the dependency problem between topics.

![Sckit-learn 14 topics LDA model](Images/sklearn_lda_14topics.PNG)
_14 topics LDA model_

![Sckit-learn 10 topics  bi-gram LDA model](Images/sklearn_lda_bi_10topics.PNG)
_10 topics LDA model_

- Gensim
   1) Lowering the number of topics and filtering words that appeared in over 50% of documents seemed to work well, thus we reused those hyperparameters. Also, we filtered the words that appeared in less than 10 reviews.

Every model returned slightly different results. The gensim LDA model created the most distinguishable topics in our eyes. Unfortunately, due to a lack of computing power and time, we were unable to use topic modeling for classification purposes.

![Gensim 10 topics wordclouds](Images/all_topic_wordclouds.PNG)

## Future Improvements

1. Optimize text cleaning process
    - Given this was our first time working with NLP techniques, we did not create an optimal pipeline for NLP pre-processing. We tokenized and lemmatized our text before realizing that NLTK's vectorizers take in a corpus of documents, rather than a list of tokens, to create vectors.

2. Use topics derived from LDA in supervised classification algorithms
    - We would have liked to have used the topics derived from the unsupervised learning algorithm, LDA, as classes in a supervised classification model.
