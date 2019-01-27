'''A simple face and gender recognition program '''
import random
import numpy as np
import cv2
import cvlib as cv

face_detector = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
colors = [(229, 212, 255), (234, 255, 212), (255, 203, 238), (163, 255, 254), (255, 220, 219), (186, 179, 255), (186, 223, 255)]
stroke = 2

#To draw a fancy rectangle when a face is detected
def drawBorder(img, point1, point2, color, stroke, r, d):
    x1, y1 = point1
    x2, y2 = point2

    #Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, stroke)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, stroke)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, stroke)

    #Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, stroke)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, stroke)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, stroke)

    #Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, stroke)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, stroke)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, stroke)

    #Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, stroke)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, stroke)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, stroke)

#Random color picked from the color palette
color = colors.pop(random.randrange(len(colors)))
cap = cv2.VideoCapture(0)

while True:
    status, frame = cap.read()
    #Face detection using OpenCV's Haar Cascades
    faces = face_detector.detectMultiScale(frame, scaleFactor=1.7, minNeighbors=5)
    for (x, y, w, h) in faces:
        roi = frame[y:y+h, x:x+w]
        drawBorder(frame, (x, y), (x+w, y+h), color, stroke, 5, 5)
        face = np.copy(roi)
        #Gender detection with cvlib
        (label, confidence) = cv.detect_gender(face)
        idx = np.argmax(confidence)
        label = label[idx]
        label = "{}: {:.2f}%".format(label.capitalize(), confidence[idx] * 100)
        cv2.putText(frame, label, (x+7, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.imshow('Gender detector', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()