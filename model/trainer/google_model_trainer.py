import numpy as np
np.random.seed(1336) # for reproducibility

from sklearn.model_selection import train_test_split
from data_utility import  DataUtility
from keras import backend as K
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
# from datetime import datetime
# import pickle
# import itertools
import models
from time import time
# import sys

# Note:  The data needs to be in the following format...
# data/
#     train/
#         cateogry1/
#             img001.jpg
#             img002.jpg
#             ...
#         category2/
#             img001.jpg
#             img002.jpg
#             ...
#     validation/
#         category1/
#             img001.jpg
#             img002.jpg
#             ...
#         category2/
#             img001.jpg
#             img002.jpg
#             ...

# dimensions of our images.
# Default image size is 99x26

start_time = time()
img_width, img_height = 26, 99
saved_file_name = 'av_5_deep_full_words'
training_categories = ['on', 'off', 'yes', 'no', 'stop', 'go', 'up', 'down', 'left', 'right']
train_data_dir = '/Users/milesporter/Desktop/Kaggle Voice Challenge/model/data/preprocessed/train'
validation_data_dir = '/Users/milesporter/Desktop/Kaggle Voice Challenge/model/data/preprocessed/validation'
nb_train_samples = 1000 #49700
nb_validation_samples = 100 #2000
epochs = 50
batch_size = 32  # Note:  Must be less than or equal to the nb_validation_samples size.
display_points = int(nb_train_samples/800)
if display_points < 100:
    display_points = 100

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)

m = models.Models()
#model = m.get_cifar_model(input_shape, 10)
#model = m.get_cifar_model_2(input_shape, 10)
model = m.get_av_blog_model_4(input_shape, 10)
du = DataUtility(bucket_id='kaggle_voice_data', root_folder='/')

#X, Y = du.load_data_local('/Users/milesporter/Desktop/Kaggle Voice Challenge/data/npz')
X, Y = du.load_cloud_data()

x_train, y_train, x_test, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# x_train -> Training data to feed the net
# x_test ->  Training data for evaluation
# y_train -> VALIDATION data for net input
# y_test -> Expected Validation output
#
# Train the network with x_train and x_test
# Evaluate the network with y_train and y_test
# So x_test and y_test should be categorical

x_test = np_utils.to_categorical(x_test, 10)  # Shouldn't hard code this.
y_test = np_utils.to_categorical(y_test, 10)    # Shouldn't hard code this.

x_train = np.expand_dims(x_train, axis=3)
y_train = np.expand_dims(y_train, axis=3)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=0,
    height_shift_range=0.2,
    horizontal_flip=False
)

model.fit_generator(datagen.flow(x_train, x_test, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=epochs)

# # here's a more "manual" example
# for e in range(epochs):
#     print('Epoch', e)
#     batches = 0
#     for x_batch, y_batch in datagen.flow(x_train, x_test, batch_size=32):
#         model.fit(x_batch, y_batch, verbose=0)
#         batches += 1
#         if batches >= len(x_train) / 32:
#             # we need to break the loop by hand because
#             # the generator loops indefinitely
#             break

stop_time=time()
print("Total training time:  {0} seconds.".format(int(stop_time-start_time)))

du.save_model('local', model)
print("Model saved.")

du.save_categories("local_categories")
