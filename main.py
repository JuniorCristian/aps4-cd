import cv2
import numpy as np
import sys
from skimage.transform import swirl
from time import sleep
import DuckDuckGoImages as ddg

filtro = "banana"
destino = "./assets/bananas/"

print("Iniciando download de imagens")
ddg.download('kittens')

# video = cv2.VideoCapture("assets/vid.mkv")
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
