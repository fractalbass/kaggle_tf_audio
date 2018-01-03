import sys
import numpy as np
np.random.seed(4655) # for reproducibility (1776 for stop, 1336 for others.)
from data_utility import DataUtility
from time import time
import models
from keras import callbacks as CB
from keras import backend as K
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

class MultiTrainer:

    save_dir = "../../saved_models"

    def train(self, target):
        start_time = time()
        img_width, img_height = 26, 99
        epochs = 20
        batch_size = 32
        tb_callback = CB.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=1, write_graph=True,
                                     write_grads=True, write_images=True, embeddings_freq=0,
                                     embeddings_layer_names=None, embeddings_metadata=None)

        m = models.Models()
        print('Training with target "{0}".'.format(target))
        du = DataUtility(bucket_id='kaggle_voice_data', root_folder='/')
        if K.image_data_format() == 'channels_first':
            input_shape = (1, img_width, img_height)
        else:
            input_shape = (img_width, img_height, 1)

        model = m.get_covn2d_six_layer_model(input_shape, 1)

        X, Y = du.load_local_binary_data('../../data/npz', target)
        # X, Y = du.load_cloud_binary_data(target)
        x_train, y_train, x_test, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

        # x_train -> Training data to feed the net
        # x_test ->  Training data for evaluation
        # y_train -> VALIDATION data for net input
        # y_test -> Expected Validation output
        #
        # Train the network with x_train and x_test
        # Evaluate the network with y_train and y_test
        # x_test = np_utils.to_categorical(x_test, 2)
        # y_test = np_utils.to_categorical(y_test, 2)

        new_x_train = np.expand_dims(x_train, axis=3)
        new_y_train = np.expand_dims(y_train, axis=3)

        # datagen = ImageDataGenerator(
        #     featurewise_std_normalization=True,
        #     rotation_range=0,
        #     height_shift_range=0.2,
        #     horizontal_flip=False
        # )

        #  Fit the data generator to the test data for featurewise_std.
        #datagen.fit(new_x_train)

        # x_train = x_train[0:nb_train_samples]
        # x_test = x_test[0:nb_train_samples]
        # y_train = y_train[0:nb_validation_samples]
        # y_test = y_test[0:nb_validation_samples]

        #model.fit_generator(datagen.flow(new_x_train, x_test, batch_size=batch_size),
        #                   steps_per_epoch=len(x_train) / batch_size, epochs=epochs, validation_data=(new_y_train, y_test))

        history = model.fit(x=new_x_train, y=x_test, validation_data=(new_y_train, y_test), batch_size=batch_size, epochs=epochs,
                   verbose=0, callbacks=[tb_callback])

        stop_time = time()
        print("Total training time:  {0} seconds.".format(int(stop_time - start_time)))
        # model.save("./local_big_training")
        du.save_multi_model(self.save_dir, '{0}'.format(target), model)
        print("Model saved as {0}.h5".format(target))
        return {"name": target, "accuracy": history.history['acc']}

if __name__ == '__main__':
    start_time = time()
    results = list()
    #target = sys.argv[1]
    mt = MultiTrainer()
    for t in ['on', 'off', 'yes', 'no', 'stop', 'go', 'up', 'down', 'left', 'right']:
    #for t in ['no']:
        results.append(mt.train(t))

    print("\n\nAccuracy summary:")
    for result in results:
        print("{0}: {1}".format(result["name"], result["accuracy"][-1]))
    stop_time=time()
    print("\n\nTotal training time: {0}".format(stop_time-start_time))