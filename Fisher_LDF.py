import pickle
import os
import numpy as np
from sklearn.utils import shuffle
from statistics import mean
from numpy.linalg import inv
from numpy import linalg as LA
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# Make a dictionary with key as classes name and value as list of instances of that class
def group_by_class(x_train, y_train):
    dict_class = {}
    for i in range(x_train.shape[0]):
        row = x_train[i]
        if y_train[i] not in dict_class:
            dict_class[y_train[i]] = []
        dict_class[y_train[i]].append(row)
    return dict_class


# For each class find the mean each feature
# Also finds the overall mean of each feature
def compute_mean(dict_class):
    dict_mean = {}
    for target, features in dict_class.items():
        dict_mean[target] = [(mean(attributes)) for attributes in zip(*features)]
    numbers = [dict_mean[key] for key in dict_mean]
    overall_mean = np.mean(numbers, axis=0)
    return dict_mean, overall_mean


# computes between class scatter matrix
def find_betweenclass_scatter(mean_class, mean_overall):
    x_overall = np.array([])
    num_of_features = mean_overall.shape[0]
    num_of_labels = len(mean_class.keys())
    b = np.matrix(np.zeros(num_of_features * num_of_features).reshape(num_of_features, num_of_features))
    for label in range(num_of_labels):
        x = mean_class[label] - mean_overall
        x_overall = np.append(x_overall, x)
    x = np.matrix(x_overall).reshape(num_of_labels, num_of_features)
    for i in range(num_of_labels):
        b = b + x[i].T * x[i]
    return b


# computes covariance matrix for each class
def find_covariance_matrix(dict_class, mean_class):
    num_of_features = len(dict_class[0][0])
    num_of_labels = len(dict_class.keys())
    # print(num_instances_class, num_of_features, num_of_labels)
    b = np.matrix(np.zeros(num_of_features * num_of_features).reshape(num_of_features, num_of_features))
    cov_matrices = []
    new_matrix = np.matrix([])
    # finds covariance for each class
    for label in range(len(dict_class)):
        num_instances_class = len(dict_class[label])
        b = np.matrix(np.zeros(num_of_features * num_of_features).reshape(num_of_features, num_of_features))
        # x = np.array(dict_class[label][:num_instances_class]) - np.array(mean_class[label])
        x = np.array(dict_class[label]) - np.array(mean_class[label])
        x = np.matrix(x).reshape(num_instances_class, num_of_features)
        for i in range(num_instances_class):
            b = b + x[i].T * x[i]
        cov_matrices.append(b * 1/num_instances_class)
    sum_cov = np.matrix(np.zeros(num_of_features * num_of_features).reshape(num_of_features, num_of_features))
    # summation of all the covariance matrix
    for i in range(num_of_labels):
        sum_cov = sum_cov + cov_matrices[i]
    return cov_matrices, sum_cov


# Reads the image data from batch file
def read_file(file):
    data_dir = Path("../data/cifar-10-python/cifar-10-batches-py/")
    images, labels = [], []
    for batch in data_dir.glob(file):
        batch_data = unpickle(batch)
        for i, flat_im in enumerate(batch_data[b"data"]):
            im_channels = []
            # Each image is flattened, with channels in order of R, G, B
            for j in range(3):
                im_channels.append(
                    flat_im[j * 1024: (j + 1) * 1024].reshape((32, 32))
                )
            # Reconstruct the original image
            images.append(np.dstack((im_channels)))
            # Save the label
            labels.append(batch_data[b"labels"][i])
    images = np.average(images, axis=3)
    x_train = []
    # flattens the image to 2d array
    for i in range(images.shape[0]):
        x_train.append(images[i].flatten())
    # Normalize data
    x_train = ((np.asarray(x_train)).astype('float32')) / 255
    labels = np.asarray(labels)
    return x_train, labels


def unpickle(file):
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d


# computes H matrix
def compute_H(a, b):
    # a = [[10, 2, 0.], [2, 4, 5], [2, 2, 5]]
    # finds eigen value and vector
    w, v = LA.eig(inv(a) * b)
    eig_pair = [(w[i], v[:, i]) for i in range(len(w))]

    # sort the eigvals in decreasing order
    eig_pair = sorted(eig_pair, key=lambda x: x[0], reverse=True)

    # take the first num_dims eigvectors
    w = (np.array([eig_pair[i][1] for i in range(9)])).reshape(9, -1)

    return np.transpose(w)


# Transform the class means and covariance matrices to the Fisher LDF space
def transformation(mean_class, h, cov_matrices):
    dict_mean = {}
    cov_matrices_tx = []
    for label in range(len(mean_class)):
        m = (np.array(mean_class[label])).reshape(1, 1024)
        dict_mean[label] = np.matmul(np.transpose(h), np.transpose(m))
        x = np.matrix(h).reshape(1024, 9)
        cov = x.T * cov_matrices[label] * x
        cov_matrices_tx.append(cov)
    return dict_mean, cov_matrices_tx


# computes mahalanobis distance on test sample
# returns the predicted value
def compute_d(h, x, mean_class, cov_matrices):
    h_mat = np.matrix(h).reshape(1024, 9)
    x_mat = np.matrix(x).reshape(1, 1024)
    f = h_mat.T * x_mat.T
    d_all = []
    # computes mahalanobis distance of test case with each class
    for i in range(len(mean_class)):
        m = np.matrix(mean_class[i]).reshape(9, 1)
        d = (f - m).T * inv(cov_matrices[i]) * (f - m)
        d_all.append(d)
    # make prediction with the least distance class
    prediction = np.argmin(d_all)
    return prediction


def rgb_to_gray(data):
    num_training_instance = data.shape[0]
    num_feature = int(data.shape[1] / 3)
    print(num_training_instance, num_feature)
    data_new = np.zeros((num_training_instance, num_feature))

    for i in range(num_training_instance):
        for j in range(num_feature):
            data_new[i][j] = int(np.mean([data[i][j], data[i][j+1024], data[i][j+2048]]))
    return data_new


# computes the accuracy of the classifier results
def find_accuracy(actual, predictions):
    correct = 0
    # comparing each values in list if they are similar
    for x, y in zip(actual, predictions):
        if x == y:
            correct += 1
    accuracy = (correct/float(len(actual))) * 100
    return accuracy


# function to plot the confusion matrix
def plotConfusionMatrix(test_set, y_pred, cm,  normalize=True, title=None, cmap = None, plot = True):

    # Compute confusion matrix
    # Find out the unique classes
    classes = list(np.unique(list(test_set)))
    if cmap is None:
        cmap = plt.cm.Blues
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Predicted label',
           xlabel='Actual label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if plot:
        plt.show()


# Generate a confusion matrix from predicted and actual classification
def find_confusion_matrix(p, a):
    num_of_classes = len(set(a))
    confusion_matrix = np.zeros((num_of_classes, num_of_classes))
    for i in range(len(a)):
        confusion_matrix[p[i]][a[i]] += 1
    # print('confusion matrix:',confusion_matrix)
    total = np.sum(confusion_matrix, axis=0)
    error = []
    for i in range(num_of_classes):
        error.append(((total[i] - confusion_matrix[i][i]) / total[i]) * 100)
    return confusion_matrix, error


# performs prediction on gives test sample
# Finds accuracy of classification
# Finds and plot the confusion matrix
def prediction(x_test, y_test, mean_class_tx, cov_matrices_tx, h):
    y_prediction = []
    for i in range(len(x_test)):
        predict = compute_d(h, x_test[i], mean_class_tx, cov_matrices_tx)
        y_prediction.append(predict)
    accuracy = find_accuracy(y_test, y_prediction)
    print("Accuracy:", accuracy)
    confusion_matrixx, error = find_confusion_matrix(y_prediction, y_test)
    print("Error:", error)
    # print(confusion_matrixx)
    plotConfusionMatrix(y_test, y_prediction, confusion_matrixx,
                        normalize=True,
                        title='Confusion matrix',
                        cmap=None, plot=True)


def main():
    # load training sample
    x_train, y_train = read_file('data_batch_1')
    # load test sample
    x_test, y_test = read_file('test_batch')
    print("1. DAtaSet generate.......")
    # Reduced dataset
    x_test, y_test = x_test[:1000], y_test[:1000]

    dict_class = group_by_class(x_train, y_train)
    print("2. Group by class")
    mean_class, mean_overall = compute_mean(dict_class)
    print("3. Compute mean")

    b = find_betweenclass_scatter(mean_class, mean_overall)
    print("4. Find B")
    cov_matrices, sum_cov = find_covariance_matrix(dict_class, mean_class)
    print("5. Find A")

    h = compute_H(sum_cov, b)
    print("6. Find H")
    mean_class_tx, cov_matrices_tx = transformation(mean_class, h, cov_matrices)
    print('7. Transformations')

    print('8. Prediction starts')
    print('Prediction for test sample')
    prediction(x_test, y_test, mean_class_tx, cov_matrices_tx, h)
    print('Prediction for training sample')
    prediction(x_train, y_train, mean_class_tx, cov_matrices_tx, h)


if __name__ == '__main__':
    main()



