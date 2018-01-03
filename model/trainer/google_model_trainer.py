import numpy as np
np.random.seed(1336) # for reproducibility

from sklearn.model_selection import train_test_split
from data_utility import DataUtility
from keras import callbacks as CB
from keras import backend as K
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
# from datetime import datetime
# import pickle
# import itertools
import models
from time import time
# import sys

start_time = time()
mg_width, img_height = 26, 99
saved_file_name = 'av_5_deep_full_words'
training_categories = ['on', 'off', 'yes', 'no', 'stop', 'go', 'up', 'down', 'left', 'right']
other_categories = ['four','three','bed','tree','bird','happy','one','two','cat','house','dog','left','seven','wow','marvin','sheila','eight','nine','six','zero','five']
train_data_dir = '../../data/preprocessed/train'
validation_data_dir = '../../data/preprocessed/validation'
nb_train_samples = 49700
nb_validation_samples = 2000
epochs = 10
batch_size = 32  # Note:  Must be less than or equal to the nb_validation_samples size.
img_width, img_height = 26, 99

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)

m = models.Models()
#model = m.get_cifar_model(input_shape, 10)
#model = m.get_cifar_model_2(input_shape, 10)
model = m.get_covn2d_six_layer_model(input_shape, len(training_categories)+1)
du = DataUtility(bucket_id='kaggle_voice_data', root_folder='/')

X, Y = du.load_data_local('../../data/npz', training_categories, other_categories)
#X, Y = du.du.load_local_binary_data('../../data/npz', target)

x_train, y_train, x_test, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# x_train -> Training data to feed the net
# x_test ->  Training data for evaluation
# y_train -> VALIDATION data for net input
# y_test -> Expected Validation output
#
# Train the network with x_train and x_test
# Evaluate the network with y_train and y_test
# So x_test and y_test should be categorical

x_test = np_utils.to_categorical(x_test, len(training_categories)+1)  # Note that the last category will contain the "Other" stuff.
y_test = np_utils.to_categorical(y_test, len(training_categories)+1)

x_train = np.expand_dims(x_train, axis=3)
y_train = np.expand_dims(y_train, axis=3)

datagen = ImageDataGenerator(
    featurewise_std_normalization=True,
    rotation_range=0,
    height_shift_range=0.2,
    horizontal_flip=False
)

#  Fit the data generator to the test data for featurewise_std.
datagen.fit(x_train)

model.fit_generator(datagen.flow(x_train, x_test, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=epochs)

# here's a more "manual" example
# for e in range(epochs):
#     print('Epoch', e)
#     batches = 0
#     for x_batch, y_batch in datagen.flow(x_train, x_test, batch_size=32):
#         model.fit(x_batch, y_batch, verbose=0, callbacks=[tb_callback])
#         batches += 1
#         if batches >= len(x_train) / 32:
#             # we need to break the loop by hand because
#             # the generator loops indefinitely
#             break
#  Just not using the damn generator.
# model.fit(x=x_train, y=x_test, validation_data=(y_train, y_test), batch_size=batch_size, epochs=epochs, verbose=1)

stop_time = time()
print("Total training time:  {0} seconds.".format(int(stop_time-start_time)))
model.save("./local_big_training")
#du.save_model('red_one_google', model)

print("Model saved.")

#du.save_categories("text_full_data_150_epochs_training")
