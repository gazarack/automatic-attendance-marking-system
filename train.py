import time
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import os
import cv2
from numpy import argmax
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from tensorflow.keras import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Conv2D
from keras.models import Sequential
from keras.layers import Conv2D
from keras.utils import np_utils




def path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


def TrainImages():
    path = 'TrainingImages/'
    df = pd.read_csv("StudentDetails.csv")
    names = df["Name"]
    rolls = df["RollNo"]
    minh = minw = 1000
    shape0 = []  # height of image
    shape1 = []  # width of image

    for category in names:
        for files in os.listdir(path):
            shape0.append(plt.imread(path + files).shape[0])
            shape1.append(plt.imread(path + files).shape[1])
            # print(path + files)
        print(category, ' => height min : ', min(shape0), 'width min : ', min(shape1))
        print(category, ' => height max : ', max(shape0), 'width max : ', max(shape1))
        if minh > min(shape0):
            minh = min(shape0)
        if minw > min(shape1):
            minw = min(shape1)
        shape0 = []
        shape1 = []

    file1 = open("dimensions.txt", "w")
    file1.write(str(minh))
    file1.write("\n")
    file1.write(str(minw))
    file1.close()
        # initialize the data and labels
    data = []  # append all images (resize)
    labels = []  # append the category /label of image
    imagePaths = []  # append the path of each image
    HEIGHT = minh
    WIDTH = minw
    N_CHANNELS = 3

    # grab the image paths and randomly shuffle them
    for k, category in enumerate(names):
        for f in os.listdir(path):
            imagePaths.append([path + category + "." + str(rolls[k]) + '.' + str(k + 2) + ".jpg",
                               k])  # k=0 : 'dogs', k=1 : 'panda', k=2 : 'cats'

    print(imagePaths[:10])
    import random
    random.shuffle(imagePaths)
    print(imagePaths[:10])

    # loop over the input images
    for imagePath in imagePaths:
        # load the image, resize the image to be HEIGHT * WIDTH pixels (ignoring
        # aspect ratio) and store the image in the data list
        image = cv2.imread(imagePath[0])
        image = cv2.resize(image, (WIDTH, HEIGHT))  # .flatten()
        # print(imagePath[0], image.shape)
        data.append(image)

        # extract the class label from the image path and update the
        # labels list
        label = imagePath[1]
        labels.append(label)

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data) / 255.0  # independent features
    labels = np.array(labels)  # dependent features
    # labels = labels.resize(1, 1)
    # Let's check everything is ok
    # plt.subplots(3, 4)
    # for i in range(12):
    #    plt.subplot(3, 4, i + 1)
    #    plt.imshow(data[i])
    #    plt.axis('off')
    #    plt.title(names[labels[i]])
    # plt.show()

    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)
    # Preprocess class labels
    print(trainY)
    # img_rows, img_cols = 100, 100

    # trainX = trainX.reshape(trainX.shape[0], trainX.shape[1], trainX.shape[2],2)
    # testX = testX.reshape(testX.shape[0], testX.shape[1], testX.shape[2],2)

    trainY = np_utils.to_categorical(trainY, len(df.index))  # actual y

    print(trainX.shape)
    print(testX.shape)
    print(trainY.shape)
    print(testY.shape)

    model = Sequential()

    model.add(Convolution2D(76, (2, 2), activation='relu', input_shape=(HEIGHT, WIDTH, N_CHANNELS)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(76, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # max value access from 2,2 matrix
    model.add(Dropout(0.25))
    model.add(Flatten())  # to convert array of image  into 1D
    model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.5))
    model.add(Dense(len(df.index), activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())

    model.fit(trainX, trainY, batch_size=32, epochs=25, verbose=1)
    pred = model.predict(testX)
    predictions = argmax(pred, axis=1)
    accuracy = accuracy_score(testY, predictions)
    print("Accuracy : %.2f%%" % (accuracy * 100.0))

    model.save("models/attendancemodel.pickle")

    # return accuracy * 100.0
