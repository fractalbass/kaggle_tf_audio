from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from datetime import datetime
import pickle
from keras.utils import plot_model
from matplotlib import pyplot as plt
import itertools
import models
from python_speech_features import mfcc

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

img_width, img_height = 26, 99

train_data_dir = '/Users/milesporter/Desktop/Kaggle Voice Challenge/model/data/preprocessed/train'
validation_data_dir = '/Users/milesporter/Desktop/Kaggle Voice Challenge/model/data/preprocessed/validation'
nb_train_samples = 2000
nb_validation_samples = 100
epochs = 50
batch_size = 2  # Note:  Must be less than or equal to the nb_validation_samples size.
display_points = 200

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)

m = models.Models()
#model = m.get_cifar_model(input_shape, 10)
model = m.get_cifar_model_2(input_shape, 10)
train_datagen = ImageDataGenerator(rescale=1. / 255)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

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

ts = int(datetime.timestamp(datetime.now()))
model.save('./saved_models/kvc_model_{0}.h5'.format(ts))

# Save the class indicies:
pickle.dump(train_generator.class_indices, open("./saved_models/kvc_classes_{0}.p".format(ts), "wb"))
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
