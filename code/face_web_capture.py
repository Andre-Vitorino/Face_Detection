import cv2

video = cv2.VideoCapture(0)
face_classifier = cv2.CascadeClassifier('/home/andre/Projects/Face_Detection/cascades/haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('/home/andre/Projects/Face_Detection/cascades/haarcascade_eye.xml')

while True:
    connected, frame = video.read()

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_detection = face_classifier.detectMultiScale(frame_gray, minSize=(30,30), minNeighbors=10)
    for (x, y, w, h) in face_detection:
        image = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)


        eyes_area = image[y:y + h, x:x + w]
        eyes_area_gray = cv2.cvtColor(eyes_area, cv2.COLOR_BGR2GRAY)
        eye_detection = eye_classifier.detectMultiScale(eyes_area_gray, minSize=(3,3), minNeighbors=10)

        for (xe, ye, we, he) in eye_detection:
            cv2.rectangle(eyes_area, (xe, ye), (xe + we, ye + he), (255, 0, 255), 2)


    cv2.imshow('Video WebCam', frame)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()

cv2.destroyAllWindows()
