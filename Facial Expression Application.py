import cv2
from keras.models import model_from_json
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

model = model_from_json(open('cnn_model.json', 'r').read())
model.load_weights('face_weights.hdf5')
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, 1.1,7)
    for (x,y,w,h) in face:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        face_ = gray[y:y+h, x:x+w]
        face_ = cv2.resize(face_, (48,48))
        face_ = face_.reshape(1,48,48,1)/255
        pred = model.predict(face_)
        index = np.argmax(pred[0])
        emotion = labels[index]
        cv2.putText(frame, emotion, (x,y), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))
    cv2.imshow('Facial Expression Detection', frame)
    k = cv2.waitKey(1) and 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()

