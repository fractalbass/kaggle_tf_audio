from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers
from keras.utils import plot_model
from keras.optimizers import SGD
from keras import backend as K

class Models():

    def get_cifar_model(self, input_shape, output_length):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same',
                         input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(output_length))
        model.add(Activation('softmax'))

        # initiate RMSprop optimizer
        opt = optimizers.rmsprop(lr=0.0001, decay=1e-6)

        # Let's train the model using RMSprop
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

        return model

    def get_cifar_model_2(self, input_shape, output_length):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(128, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(output_length))
        model.add(Activation('softmax'))

        #sgd = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
        #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # initiate RMSprop optimizer
        opt = optimizers.rmsprop(lr=0.00005, decay=1e-6)

        # Let's train the model using RMSprop
        model.compile(loss='categorical_crossentropy',
                     optimizer=opt,
                     metrics=['accuracy'])

        return model


    #  Taken from:
    #  https://www.analyticsvidhya.com/blog/2016/10/tutorial-optimizing-neural-networks-using-keras-with-image-recognition-case-study/#five
    #
    def get_av_blog_model(self, input_shape, output_length):

        model = Sequential()
        model.add(Conv2D(25, (5, 5), padding='same', input_shape=input_shape, use_bias=False))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(25, (5, 5), padding='same', use_bias=False))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(25, (4, 4), padding='same', use_bias=False))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())

        model.add(Dense(50))

        model.add(Activation('relu'))
        model.add(Dense(output_length))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    #  Taken from:
    #  https://www.analyticsvidhya.com/blog/2016/10/tutorial-optimizing-neural-networks-using-keras-with-image-recognition-case-study/#five
    #
    def get_av_blog_sigmoid_model(self, input_shape, output_length):

        model = Sequential()
        model.add(Conv2D(25, (5, 5), padding='same', input_shape=input_shape, use_bias=False))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(25, (5, 5), padding='same', use_bias=False))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(25, (4, 4), padding='same', use_bias=False))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())

        model.add(Dense(50))

        model.add(Activation('relu'))
        model.add(Dense(output_length))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

        #  Taken from:
        #  https://www.analyticsvidhya.com/blog/2016/10/tutorial-optimizing-neural-networks-using-keras-with-image-recognition-case-study/#five
        #

    def get_sigmoid_model_simple(self, input_shape, output_length):
        model = Sequential()
        model.add(Conv2D(25, (3, 3), padding='same', input_shape=input_shape, use_bias=False))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(25, (3, 3), padding='same', use_bias=False))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(25, (3, 3), padding='same', input_shape=input_shape, use_bias=False))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(25, (3, 3), padding='same', use_bias=False))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())

        model.add(Dense(150))

        model.add(Activation('relu'))
        model.add(Dense(output_length))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model


    def get_av_blog_model_2(self, input_shape, output_length):

        model = Sequential()
        model.add(Conv2D(25, (7, 7), padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(25, (5, 5), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(25, (5, 5), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(25, (4, 4), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())

        model.add(Dense(250))

        model.add(Activation('relu'))
        model.add(Dense(output_length))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def get_av_blog_model_3(self, input_shape, output_length):

        model = Sequential()
        model.add(Conv2D(32, (7, 7), padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (5, 5), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (4, 4), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())

        model.add(Dense(250))

        model.add(Activation('relu'))
        model.add(Dense(output_length))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def get_av_blog_model_4(self, input_shape, output_length):

        model = Sequential()
        model.add(Conv2D(32, (7, 7), padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(32, (6, 6), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(32, (5, 5), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(32, (4, 4), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dropout(0.25))

        model.add(Flatten())

        model.add(Dense(250))

        model.add(Activation('relu'))
        model.add(Dense(output_length))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

if __name__ == "__main__":

    print("You have chosen to turn the models class.  This will only output images of the various networks implemented here.")
    m = Models()
    img_width, img_height = 26, 99
    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 1)

    model1 = m.get_cifar_model(input_shape, 10)
    plot_model(model1, to_file='cifar_model_1.png')

    model2 = m.get_cifar_model_2(input_shape, 10)
    plot_model(model2, to_file='cifar_model_2.png')

    model3 = m.get_av_blog_model(input_shape, 10)
    plot_model(model3, to_file='av_model.png')

