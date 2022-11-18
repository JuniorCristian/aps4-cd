import cv2
import numpy as np
import sys
from skimage.transform import swirl
from time import sleep
from PIL import Image


(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
if __name__ == '__main__':
    # Set up tracker.
    # Instead of CSRT, you can also use
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[7]
    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        elif tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        elif tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        elif tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        elif tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        elif tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        elif tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        elif tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()

# Read video
video = cv2.VideoCapture("assets/vid.mkv")
# video = cv2.VideoCapture(0) # for using CAM
# Exit if video not opened.
if not video.isOpened():
    print("Could not open video")
    sys.exit()
# Read first frame.
ok, frame = video.read()
if not ok:
    print('Cannot read video file')
    sys.exit()

# Define an initial bounding box
# bbox = (287, 23, 86, 320)
# Uncomment the line below to select a different bounding box
bbox = cv2.selectROI(frame, False)
while True:
    ok, frame = video.read()
    bbox = cv2.selectROI(frame, False)
    while cv2.waitKey(1):
        if 0xFF == ord('r'):
            break
        if 0xFF == ord('q'):
            exit()
        if 0xFF == ord(' '):
            pass
# Initialize tracker with first frame and bounding box
ok = tracker.init(frame, bbox)

while True:
    ok, frame = video.read()
    if not ok:
        break
    timer = cv2.getTickCount()
    if bbox != (0, 0, 0, 0):
        ok, bbox = tracker.update(frame)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    if ok:
        if bbox[0] > 0 and bbox[1] > 0 and bbox[2] > 0 and bbox[3] > 0:
            crop_img = frame[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
            crop_img = cv2.resize(crop_img, (0, 0), fx=0.1, fy=0.1)
            crop_img = cv2.resize(crop_img, (0, 0), fx=10, fy=10)
            pil_image = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(pil_image)
            pil_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(pil_frame)
            pil_frame.paste(pil_image, (int(bbox[0]), int(bbox[1])))
            frame = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)
    else:
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    # Display tracker type on frame
    cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);
    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);
    # Display result
    cv2.imshow("Tracking", frame)
    # Exit if ESC pressed
    if cv2.waitKey(1) & 0xFF == ord(' '):
        bbox = cv2.selectROI(frame, False)
        while bbox == (0, 0, 0, 0):
            ok, frame = video.read()
            bbox = cv2.selectROI(frame, False)
        # Initialize tracker with first frame and bounding box
        ok = tracker.init(frame, bbox)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # if press SPACE bar
        break
video.release()
cv2.destroyAllWindows()

# while True:
#     ok, frame = video.read()
#     if not ok:
#         break
#     img_rgb = frame
#     img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
#     template = cv2.imread('assets/banana2.png', 0)
#     w, h = template.shape[::-1]
#     res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
#     threshold = 0.8
#     loc = np.where(res >= threshold)
#     for pt in zip(*loc[::-1]):
#         cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
#     cv2.imshow("app", img_rgb)
#     if cv2.waitKey(1) & 0xFF == ord('q'):  # if press SPACE bar
#         break
# video.release()
# cv2.destroyAllWindows()
