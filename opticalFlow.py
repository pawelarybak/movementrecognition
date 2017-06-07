import cv2
import numpy as np

video = cv2.VideoCapture('campus4-c0.avi')

min_dist = 0.8
feature_params = {
    'maxCorners': 1000,
    'qualityLevel': 0.1,
    'minDistance': 20,
    'blockSize': 7,
}

lk_params = {
    'winSize': (15, 15),
    'maxLevel': 2,
    'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
}

color = np.random.randint(0, 255, (100, 3))

ret, old_frame = video.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, **feature_params)

mask = np.zeros_like(old_frame)

while video.isOpened():
    ret, frame = video.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    good_new = p1[st == 1]
    good_old = p0[st == 1]

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        dist = cv2.norm(new - old)
        # mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        if(dist > min_dist):
            mask = cv2.circle(mask, (a, b), 5, color[i].tolist(), -1)

    img = cv2.add(frame, mask)

    cv2.imshow('mask', mask)
    cv2.imshow('frame', img)

    action = cv2.waitKey(30) & 0xFF
    if action == 27:
        break

    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
    mask = np.zeros_like(old_frame)

video.release()
cv2.destroyAllWindows()