import cv2
import numpy as np

video = cv2.VideoCapture('/home/someoddperson/Projects/movementRecognition/campus4-c0.avi')
ret, frame1 = video.read()

prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

while video.isOpened():
    ret, frame2 = video.read()
    nxt = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow('frame2', bgr)
    action = cv2.waitKey(1) & 0xFF
    if action == 27:
        break

    prvs = nxt

video.release()
cv2.destroyAllWindows()