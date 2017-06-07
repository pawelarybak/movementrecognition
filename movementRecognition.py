import cv2

video = cv2.VideoCapture('campus4-c0.avi')
subtractor = cv2.createBackgroundSubtractorMOG2(varThreshold=200, detectShadows=True)

minSize = 20
fps = 30
x_blur = 21
y_blur = 51

while video.isOpened():
    ret, frame = video.read()
    mask = subtractor.apply(frame)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.GaussianBlur(mask, (x_blur, y_blur), 0)
    _, contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > minSize and h > minSize:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    action = cv2.waitKey(1000//fps) & 0xFF
    if action == 27:
        break

video.release()
cv2.destroyAllWindows()
