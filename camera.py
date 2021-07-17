import os
import cv2
import numpy as np
import pandas as pd
from numpy import argmax
from sklearn.metrics import accuracy_score
from tensorflow import keras
import datetime

from imutils.video import WebcamVideoStream
from train import TrainImages


# collect image from webcam
class VideoCamera(object):
    def __init__(self):
        self.stream = WebcamVideoStream(0).start()

        # load model

    def __del__(self):
        self.stream.stop()

    # predict function

    # Check if path exists
    def path_exists(self, path):
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)

    def TakeImages(self, user_name, user_roll, sampleNum):

        self.path_exists("TrainingImages/")
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)

        image = self.stream.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if len(faces) != 0:
            sampleNum = sampleNum + 1
            cv2.imwrite("TrainingImages/" + user_name + "." + str(user_roll) + '.' + str(sampleNum) + ".jpg",
                        gray[y:y + h, x:x + w])
        # sampleNum = sampleNum + 1
        # print(check)
        ret, jpeg = cv2.imencode('.jpg', gray)
        data = []
        data.append(jpeg.tobytes())
        return data, sampleNum
        # time.sleep(2)
        # video.release()
        # cv2.destroyAllWindows()

    # we convert to frame to sent to web page
    def get_frame(self):
        df = pd.read_csv("StudentDetails.csv")
        names = df["Name"]
        rolls = df["RollNo"]

        file1 = open("dimensions.txt", "r")
        dim = file1.read().split("\n")
        minh = int(dim[0])
        minw = int(dim[1])
        file1.close()

        # collected image is in image
        model = keras.models.load_model('models/attendancemodel.pickle')
        detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        image = self.stream.read()
        # used to detect faces in image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(image, 1.3, 5)

        # to create rectangle on face and they will be present along the image
        for (x, y, h, w) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0), 2)
        if len(faces) != 0:
            gray = [cv2.resize(image[y:y + h, x:x + w], (minh, minw))]
            gray = np.array(gray) / 255.0
            pred_y = model.predict(gray)
            predictions = argmax(pred_y, axis=1)

            print(names[predictions[0]])
            font = cv2.FONT_HERSHEY_SIMPLEX
            image = cv2.putText(image, names[predictions[0]], (x, y), font, 1, (255, 255, 255), 2)

            exists = os.path.isfile("Attendance.csv")
            if exists:
                af = pd.read_csv("Attendance.csv")
            else:
                af = pd.DataFrame([], columns=['Name', 'RollNo', 'Time'])
            af.loc[len(df.index)] = [names[predictions[0]], rolls[predictions[0]], datetime.datetime.now()]
            af.to_csv("Attendance.csv", index=False)

        ret, jpeg = cv2.imencode('.jpg', image)  # we convert to array
        data = []
        data.append(jpeg.tobytes())
        return data
