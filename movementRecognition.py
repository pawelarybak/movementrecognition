import cv2
import sys


MIN_SIZE = 20
FPS = 30
X_BLUR = 21
Y_BLUR = 51


if __name__ == '__main__':
    if len(sys.argv) > 1:
        video = cv2.VideoCapture(sys.argv[1])
    else:
        video = cv2.VideoCapture()

    subtractor = cv2.createBackgroundSubtractorMOG2(varThreshold=200, detectShadows=True)

    while video.isOpened():
        ret, frame = video.read()
        mask = subtractor.apply(frame)  # subtract background from picture
        mask = cv2.medianBlur(mask, 5)  # median blur removes noise from mask
        mask = cv2.GaussianBlur(mask, (X_BLUR, Y_BLUR), 0)  # gaussian blur merge parts of moving object into whole
        _, contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w > MIN_SIZE and h > MIN_SIZE:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        cv2.imshow('mask', mask)
        action = cv2.waitKey(1000 // FPS) & 0xFF
        if action == 27:  # exit on escape
            break

    video.release()
    cv2.destroyAllWindows()
