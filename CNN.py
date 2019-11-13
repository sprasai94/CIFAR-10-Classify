import sys
from pathlib import Path
import pickle
import numpy as np
from matplotlib import pyplot as plt
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.layers import Dropout
from Fisher_LDF import find_accuracy, find_confusion_matrix, plotConfusionMatrix


# Read batch image file
def load_dataset():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # convert from integers to floats
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # normalize to range 0-1
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # encoding target values to categorical
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return x_train, y_train, x_test, y_test


# define cnn model
def define_model(filter_size, pool_size, stride):
    model = Sequential()
    # add convolution layer with given parameters
    model.add(
        Conv2D(32, filter_size, activation='relu', kernel_initializer='he_uniform', strides=(1, 1), padding='same',
               input_shape=(32, 32, 3)))
    # add pooling layer
    model.add(MaxPooling2D(pool_size=pool_size, strides=stride))

    model.add(Conv2D(32, filter_size, activation='relu', kernel_initializer='he_uniform', strides=(1, 1),
                     padding='same'))
    model.add(MaxPooling2D(pool_size=pool_size, strides=stride))

    model.add(Conv2D(64, filter_size, activation='relu', kernel_initializer='he_uniform', strides=(1, 1),
                     padding='same'))
    model.add(MaxPooling2D(pool_size=pool_size, strides=2))

    model.add(Flatten())

    # add fully connected layer
    model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))

    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(lr=0.0001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# plot accuracy and loss vs epochs
def plot_learning_curves(history):
    # plot loss vs epochs
    plt.subplot(211)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.6)
    plt.title('Cross Entropy Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')

    # plot accuracy vs epochs
    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='test')
    plt.legend()

    plt.show()
    plt.close()


# performs prediction on gives test sample
# Finds accuracy of classification
# Finds and plot the confusion matrix
def prediction(x_test, y_test, model):
    predictions = model.predict(x_test)
    y_prediction = np.argmax(predictions, axis=1)
    y_test = np.argmax(y_test, axis=1)
    accuracy = find_accuracy(y_test, y_prediction)
    print("Accuracy:", accuracy)
    confusion_matrix, error = find_confusion_matrix(y_prediction, y_test)
    print("Error:", error)
    plotConfusionMatrix(y_test, y_prediction, confusion_matrix,
                        normalize=True,
                        title=None,
                        cmap=None, plot=True)


def main():
    # load dataset
    x_train, y_train, x_test, y_test = load_dataset()
    print('1. Load dataset')

    # Taking Reduced Dataset
    x_train, y_train = x_train[:10000], y_train[:10000]
    x_test, y_test = x_test[:1000], y_test[:1000]

    # Create CNN Model
    model = define_model(filter_size=(5, 5), pool_size=(3, 3), stride=2)
    print("2. Create model")

    # fit model
    history = model.fit(x_train, y_train, epochs=2, batch_size=64, validation_data=(x_test, y_test), verbose=2)
    print("3. Fit model")

    # Plot accuracy and loss vs epochs during fitting
    plot_learning_curves(history)

    print('4. Predction')
    print('Testing on test set')
    prediction(x_test, y_test, model)
    print('Testing on training set')
    prediction(x_train, y_train, model)


if __name__ == '__main__':
    main()
