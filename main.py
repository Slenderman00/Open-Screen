from transformers import pipeline
import os
from PIL import Image
import cv2
import numpy as np

# Set environment variable
os.environ['QT_QPA_PLATFORM'] = 'xcb'

# Load depth estimation model
checkpoint = "vinvino02/glpn-nyu"
depth_estimator = pipeline("depth-estimation", model=checkpoint)

vid = cv2.VideoCapture(0)

while True:
    # Capture the video frame
    ret, frame = vid.read()
    frame = cv2.flipND(frame, 1)
    original_frame = frame.copy()
    frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    predictions = depth_estimator(pil_image)

    depth_map = predictions["depth"]
    threshold_depth = np.percentile(depth_map, 25)
    binary_mask = np.where(depth_map <= threshold_depth, 255, 0).astype(np.uint8)

    binary_mask_resized = cv2.resize(binary_mask, (original_frame.shape[1], original_frame.shape[0]))
    masked_frame = cv2.bitwise_and(original_frame, original_frame, mask=binary_mask_resized)

    cv2.imshow('masked frame', masked_frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture object and close all windows
vid.release()
cv2.destroyAllWindows()
