from flask import Flask, render_template, Response
import cv2
import numpy as np  
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import imutils

app = Flask(__name__)
camera = cv2.VideoCapture(-1)
face_cascade = cv2.CascadeClassifier('harcascades/haarcascade_frontalface_default.xml')
model = load_model('lenet.hdf5')

def generate_frames():
    while True:

        # read frames by camera
        success, frame = camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)
            font = cv2.FONT_HERSHEY_SIMPLEX
            for (x, y, w, h) in faces:
                roi = gray[y:y+h, x:x+w]
                roi = cv2.resize(roi, (28, 28))
                roi = roi.astype("float")/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                
                (notSmiling, smiling) = model.predict(roi)[0]
                label = "Smiling :)" if smiling > notSmiling else "Not Smiling :("
                color = (0, 255, 0) if smiling > notSmiling else (0, 0, 255)
                cv2.putText(frame,label,(10,200), font, 2, color, 3, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run()
