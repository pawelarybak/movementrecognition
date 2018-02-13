import cv2
import sys
import numpy as np


MIN_SIZE = 40


if __name__ == '__main__':
    if len(sys.argv) > 1:
        video = cv2.VideoCapture(sys.argv[1])
    else:
        video = cv2.VideoCapture()

    ret, frame = video.read()

    prvs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(frame)

    while video.isOpened():
        ret, frame = video.read()
        nxt = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        pts, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        pts *= 100
        mask[..., 0] = pts
        mask[..., 1] = pts
        mask[..., 2] = pts
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        _, mask_gray = cv2.threshold(mask_gray, 130, 255, cv2.THRESH_BINARY)
        mask_gray = cv2.medianBlur(mask_gray, 5)
        mask_gray = cv2.GaussianBlur(mask_gray, (31, 31), 0)

        _, contours, _ = cv2.findContours(mask_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w > MIN_SIZE and h > MIN_SIZE:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('mask', mask_gray)
        cv2.imshow('frame', frame)
        action = cv2.waitKey(1) & 0xFF
        if action == 27:
            break

        prvs = nxt

    video.release()
    cv2.destroyAllWindows()