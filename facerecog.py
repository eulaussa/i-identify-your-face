'''A simple face recognition program '''
import cv2

def drawBorder(img, point1, point2, color, stroke, r, d):
    x1,y1 = point1
    x2,y2 = point2

    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, stroke)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, stroke)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, stroke)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, stroke)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, stroke)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, stroke)

    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, stroke)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, stroke)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, stroke)

    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, stroke)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, stroke)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, stroke)

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayscale, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        roi_gray = grayscale[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        color = (232, 155, 218)
        stroke = 2
        drawBorder(frame, (x,y), (x+w, y+h), color, stroke, 5, 5)

    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
