from flask import Flask, render_template, Response, request
import cv2
import os
import pandas as pd

from camera import VideoCamera
from train import TrainImages

app = Flask(__name__)

camera = cv2.VideoCapture(0)  # use 0 for web camera


# for local webcam use cv2.VideoCapture(0)

def student_details(user_name, user_roll):
    exists = os.path.isfile("StudentDetails.csv")
    if exists:
        df = pd.read_csv("StudentDetails.csv")
        print(exists)
    else:
        df = pd.DataFrame([], columns=['Sl_No', 'Name', 'RollNo'])
    # data = [df.index+1, user_name, user_roll]
    # df = df.append(data)
    print(df)
    df.loc[len(df.index)] = [len(df.index) + 1, user_name, user_roll]

    if exists:
        os.remove("StudentDetails.csv")
    df.to_csv("StudentDetails.csv", index=False)


def gen_live(camera):  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame

        data = camera.get_frame()  # read the camera frame

        frame = data[0]
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def gen_take(camera, user_name, user_roll, i, j):  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        if j == 1:
            student_details(user_name, user_roll)

        if 0 < i < 200:
            data, i = camera.TakeImages(user_name, user_roll, i)

        j = j + 1

        frame = data[0]
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/save_feed', methods=("POST", "GET"))
def save_feed():
    i = j = 0
    user_name = request.form['user_name']
    user_roll = request.form['user_roll']
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_take(VideoCamera(), user_name, user_roll, i + 1, j + 1),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_live(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/details_feed')
def details_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return render_template('details.html')


@app.route('/train_feed')
def train_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    accuracy = TrainImages()
    return render_template('index.html')


@app.route('/student_list_feed')
def table_feed():
    exists = os.path.isfile("StudentDetails.csv")
    if exists:
        df = pd.read_csv("StudentDetails.csv")
        print(exists)
    else:
        df = pd.DataFrame([], columns=['Sl_No', 'Name', 'RollNo'])

    columns = df.columns.values

    return render_template('table.html', tables=[df.to_html(classes='data')], titles=columns)


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
