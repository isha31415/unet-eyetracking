import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint


#-------------------------------------------------------
# PARSE Command Line Arguments
#-------------------------------------------------------
corneal_opt=0 # For Corneal Data
pupil_opt = 0 # For Pupil Data
train_opt=0 # For Training UNET Model
verbose_opt=0 # For Verbose predict on entire dataset in the end
prediction_opt=0 # For prediction on provided file only
prediction_file_name = './images-3.jpeg'

#python UnetTrainTest.py pupil -f file_name
#python UnetTrainTest.py train pupil
#python UnetTrainTest.py corneal -f file_name
#python UnetTrainTest.py train corneal

i=1
while i < len(sys.argv):
    if sys.argv[i] == 'train':
        train_opt = 1
    elif sys.argv[i] == 'corneal':
        corneal_opt = 1
    elif sys.argv[i] == 'pupil':
        pupil_opt = 1
    elif sys.argv[i] == 'verbose':
        verbose_opt = 1
    elif sys.argv[i] == '-f':
        prediction_opt = 1
        #next field is the image name for prediction, skip an index
        prediction_file_name = sys.argv[i+1]
        i=i+1
    else:
        print("Argument ", sys.argv[i], " Not Recognized!")
    i=i+1


#-------------------------------------------------------
# Load and Prep Train and Evaluate Data (images)
#-------------------------------------------------------
if corneal_opt :
    print("Preparing Corneal Data.....")
    IMAGE_LIB = './datasets/corneal/img/'
    MASK_LIB = './datasets/corneal/masks/'
    weights_file = 'original_corneal.h5'
elif pupil_opt :
    print("Preparing Pupil Data.....")
    IMAGE_LIB = './datasets/pupil/img/'
    MASK_LIB = './datasets/pupil/masks/'
    weights_file = 'original_pupil.h5'


IMG_HEIGHT, IMG_WIDTH = 288, 432
SEED=42

all_images = [x for x in sorted(os.listdir(IMAGE_LIB)) if x[-1] == 'g']

x_data = np.empty((len(all_images), IMG_HEIGHT, IMG_WIDTH), dtype='float32')
for i, name in enumerate(all_images):
    im = cv2.imread(IMAGE_LIB + name, 0).astype("int16").astype('float32')
    im = cv2.resize(im, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
    im = (im - np.min(im)) / (np.max(im) - np.min(im))
    x_data[i] = im

y_data = np.empty((len(all_images), IMG_HEIGHT, IMG_WIDTH), dtype='float32')
for i, name in enumerate(all_images):
    im = cv2.imread(MASK_LIB + name, 0).astype('float32')/255.
    im = cv2.resize(im, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
    y_data[i] = im

x_data = x_data[:,:,:,np.newaxis]
y_data = y_data[:,:,:,np.newaxis]
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size = 0.5)

#-------------------------------------------------------
# Define UNET model
#-------------------------------------------------------
print("Compiling UNET Model.....")

#Optimization Cost Metrics
def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return ( (2. * intersection + smooth) /
             (K.sum(y_true_f) + K.sum(y_pred_f) + smooth) )

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

'''
#Worked only for Pupil but not for Corneal Reflection
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())
'''

input_layer = Input(shape=x_train.shape[1:])
c1 = Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same')(input_layer)

l = MaxPool2D(strides=(2,2))(c1)
c2 = Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same')(l)

l = MaxPool2D(strides=(2,2))(c2)
c3 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(l)

l = MaxPool2D(strides=(2,2))(c3)
c4 = Conv2D(filters=32, kernel_size=(1,1), activation='relu', padding='same')(l)

l = concatenate([UpSampling2D(size=(2,2))(c4), c3], axis=-1)
l = Conv2D(filters=32, kernel_size=(2,2), activation='relu', padding='same')(l)

l = concatenate([UpSampling2D(size=(2,2))(l), c2], axis=-1)
l = Conv2D(filters=24, kernel_size=(2,2), activation='relu', padding='same')(l)

l = concatenate([UpSampling2D(size=(2,2))(l), c1], axis=-1)
l = Conv2D(filters=16, kernel_size=(2,2), activation='relu', padding='same')(l)

l = Conv2D(filters=64, kernel_size=(1,1), activation='relu')(l)
l = Dropout(0.5)(l)

output_layer = Conv2D(filters=1, kernel_size=(1,1), activation='sigmoid')(l)

model = Model(input_layer, output_layer)

def my_generator(x_train, y_train, batch_size):
    data_generator = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        rotation_range=10,
        zoom_range=0.1,
        fill_mode='constant',
        cval=0.1).flow(x_train, x_train, batch_size, seed=SEED)
    mask_generator = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        rotation_range=10,
        zoom_range=0.1,
        fill_mode='constant',
        cval=0.0).flow(y_train, y_train, batch_size, seed=SEED)
    while True:
        x_batch, _ = data_generator.next()
        y_batch, _ = mask_generator.next()
        yield x_batch, y_batch


# By using the same RNG seed in both calls to ImageDataGenerator,
#we should get images and masks that correspond to each other.
#Let's check this, to be safe.

#-------------------------------------------------------
# Train UNET MOdel
#-------------------------------------------------------
if train_opt:
    print("Training UNET Model.....")
    #Changed the loss function to new one dice_coef_loss
    #model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=[dice_coef])
    model.compile(optimizer=Adam(1e-4), loss=dice_coef_loss, metrics=[dice_coef])

    weight_saver = ModelCheckpoint('original_pupil.h5', monitor='val_dice_coef',
                                save_best_only=True, save_weights_only=True)

    annealer = LearningRateScheduler(lambda x: 1e-3 * 0.8 ** x)

    hist = model.fit_generator(my_generator(x_train, y_train, 8),
                            steps_per_epoch = 200,
                            validation_data = (x_val, y_val),
                            epochs=2, verbose=1,
                            callbacks = [weight_saver, annealer])

#-------------------------------------------------------
# Evaluate UNET Models
#-------------------------------------------------------
print("Loading Weights file: ", weights_file)
model.load_weights(weights_file)


if prediction_opt :
    test_image_input = np.empty((1, IMG_HEIGHT, IMG_WIDTH), dtype='float32')
    im = cv2.imread(prediction_file_name, 0).astype("int16").astype('float32')
    im = cv2.resize(im, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
    test_image_input = (im - np.min(im)) / (np.max(im) - np.min(im))
    test_image_output = model.predict(test_image_input.reshape(1,IMG_HEIGHT, IMG_WIDTH, 1))[0,:,:,0]
    fig, ax = plt.subplots(1,2,figsize=(12,6))
    ax[0].imshow(test_image_input, cmap='gray')
    ax[1].imshow(test_image_output, cmap='gray')
    plt.show()

if verbose_opt:
    print("Predicting Results.....")

    y_hat_train = model.predict(x_train)
    #print Train image, corresponding mask image and also what model prediction does on them
    rows = len(x_train)
    for i in range(rows):
        fig, ax = plt.subplots(1,2,figsize=(12,6))
        print("Train Image number: ", i+1)
        ax[0].imshow(x_train[i,:,:,0], cmap='gray')
        ax[1].imshow(y_hat_train[i,:,:,0], cmap='gray')
        plt.show()

    #print Validate image, corresponding mask image, and also what model prediction does on them
    y_hat_val = model.predict(x_val)

    rows = len(x_val)
    for i in range(rows):
        fig, ax = plt.subplots(1,2,figsize=(12,6))
        print("Validage Image number: ", i+1)
        ax[0].imshow(x_val[i,:,:,0], cmap='gray')
        ax[1].imshow(y_hat_val[i,:,:,0], cmap='gray')
        plt.show()
