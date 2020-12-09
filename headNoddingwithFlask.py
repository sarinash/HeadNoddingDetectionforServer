import time
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


# save data temporary as the measures of the face
def gen():
    gesturetoJson = 0
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('tempraryOutput.avi', fourcc, 20.0, (640, 480))

    def distance(x, y):
        import math
        return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)

    """Video streaming generator function."""
    cap = cv2.VideoCapture(0)
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    face_cascade = cv2.CascadeClassifier('C:\\Users\\sarina\\Documents\\haarcascade_frontalface_default.xml')

    def get_coords(p1):
        try:
            return int(p1[0][0][0]), int(p1[0][0][1])
        except:
            return int(p1[0][0]), int(p1[0][1])

    # define font and text color
    font = cv2.FONT_HERSHEY_SIMPLEX
    # define movement threshodls
    gesture_threshold = 70
    # find the face in the image
    face_found = False
    frame_num = 0

    # Read until video is completed
    while (frame_num < 30):
        # Capture frame-by-frame
        frame_num += 1
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 125, 125), 2)
            face_found = True
        cv2.imshow('image', frame)
        out.write(frame)
        cv2.waitKey(1)

    face_center = x + w / 2, y + h / 3
    p0 = np.array([[face_center]], np.float32)

    gesture = False
    x_movement = 0
    y_movement = 0
    gesture_show = 60  # number of frames a gesture is shown
    while True:
        ret, frame = cap.read()
        old_gray = frame_gray.copy()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        cv2.circle(frame, get_coords(p1), 4, (0, 0, 255), -1)
        cv2.circle(frame, get_coords(p0), 4, (255, 0, 0))

        # get the xy coordinates for points p0 and p1
        a, b = get_coords(p0), get_coords(p1)
        x_movement += abs(a[0] - b[0])
        y_movement += abs(a[1] - b[1])

        text = 'x_movement: ' + str(x_movement)
        if not gesture: cv2.putText(frame, text, (50, 50), font, 0.4, (0, 255, 255), 1)
        text = 'y_movement: ' + str(y_movement)
        if not gesture: cv2.putText(frame, text, (50, 100), font, 0.4, (0, 255, 255), 1)

        if y_movement > gesture_threshold:
            gesture = 'Yes'
        if gesture and gesture_show > 0:
            gesturetoJson = 1
            cv2.putText(frame, 'Gesture Detected: ' + gesture, (50, 50), font, 0.8, (0, 255, 255), 1)
            gesture_show -= 1
        if gesture_show == 0:
            gesturetoJson = 0
            gesture = False
            x_movement = 0
            y_movement = 0
            gesture_show = 60  # number of frames a gesture is shown

        # print distance(get_coords(p0), get_coords(p1))
        p0 = p1

        cv2.imshow('image', frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        print(gesturetoJson)

    cap.release()
    cv2.destroyAllWindows()


gen()


@app.route('/<epoch_time>', methods=['GET', 'POST'])
def video_feed():
    epoch_time = int(time.time())
    my_details = {

        'name': 'Sarina',
        'request_time': epoch_time,
        'gesture': B
    }

    return jsonify(my_details)


if __name__ == "__main__":
    app.run()

