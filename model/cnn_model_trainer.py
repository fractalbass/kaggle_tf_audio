import numpy as np
np.random.seed(1336) # for reproducibility

from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from datetime import datetime
import pickle
from matplotlib import pyplot as plt
import itertools
import models
from time import time
import sys

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

train_data_dir = '/Users/milesporter/Desktop/Kaggle Voice Challenge/model/data/preprocessed/train'
validation_data_dir = '/Users/milesporter/Desktop/Kaggle Voice Challenge/model/data/preprocessed/validation'
nb_train_samples = 1000 #49700
nb_validation_samples = 100 #2000
epochs = 5
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
model = m.get_av_blog_model_4(input_shape, 30)

train_datagen = ImageDataGenerator(rescale=1. / 255,  height_shift_range=0.2)
#train_datagen = ImageDataGenerator(rescale=1.0/255)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)
#test_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical')

history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

stop_time=time()
print("Total training time:  {0} seconds.".format(int(stop_time-start_time)))

if sys.version_info[0] < 3:
    ts = time()
else:
    ts = int(datetime.timestamp(datetime.now()))

model.save('./saved_models/{0}_{1}.h5'.format(saved_file_name, ts))

# Save the class indicies:
pickle.dump(train_generator.class_indices, open("./saved_models/{0}_{1}.p".format(saved_file_name, ts), "wb"))
s = len(history.history['acc'])
st = 1
if s > display_points:
    st = int(s/display_points)

acc = list(itertools.islice(history.history['acc'],0,s,st))
val_acc = list(itertools.islice(history.history['val_acc'],0,s,st))
loss = list(itertools.islice(history.history['loss'],0,s,st))
val_loss = list(itertools.islice(history.history['val_loss'],0,s,st))
print(history.history.keys())
# summarize history for accuracy
plt.figure(1)
plt.subplot(211)
plt.plot(acc)
plt.plot(val_acc)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.legend(['train', 'test'], loc='upper left')
# summarize history for loss
plt.subplot(212)
plt.plot(loss)
plt.plot(val_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower left')
plt.show()
