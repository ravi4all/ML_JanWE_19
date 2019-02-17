import cv2
import numpy as np

dataset = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
capture = cv2.VideoCapture(0)

faceData = []

while True:
    ret, img = capture.read()
    # print(img)
    if ret:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = dataset.detectMultiScale(gray)
        for x,y,w,h in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,0),3)

            face = img[y:y+h, x:x+w, :]
            face = cv2.resize(face, (50,50))

            if len(faceData) < 200:
                faceData.append(face)
            print(len(faceData))

        cv2.imshow('result', img)
        if cv2.waitKey(1) == 27 or len(faceData) >= 200:
            break
    else:
        print("Camera not working")

faceData = np.asarray(faceData)
np.save('face_2.npy',faceData)

capture.release()
cv2.destroyAllWindows()
