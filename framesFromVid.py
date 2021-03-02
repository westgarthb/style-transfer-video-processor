import cv2
import numpy as np
from PIL import Image


# _____________
# configuration
# -------------
# defines the maximum height dimension in pixels. Used for down-sampling the video frames
max_height = 864
# defines the rate at which you want to capture frames from the input video
desiredFR = 29
# input video path
video_path = 'input_vid.MOV'

vid_obj = cv2.VideoCapture(video_path)
frame_interval = np.floor((1.0 / desiredFR) * 1000)
success, image = vid_obj.read()
img = Image.fromarray(image[:, :, 0:3])
scale_const = (max_height / image.shape[0])
max_width = int(image.shape[1] * scale_const)
cv2.imwrite('vid/000.png', np.asarray(img.resize((max_width, max_height))).astype(np.uint8))

count = 1
while success:
    msec_timestamp = count * frame_interval
    vid_obj.set(cv2.CAP_PROP_POS_MSEC, msec_timestamp)
    success, image = vid_obj.read()
    if not success:
        break
    img = Image.fromarray(image[:, :, 0:3])
    cv2.imwrite('vid/{:0>3d}.png'.format(count), np.asarray(img.resize((max_width, max_height))).astype(np.uint8))
    count += 1

