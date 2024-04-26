from transformers import pipeline
import os
from PIL import Image
import cv2
import numpy as np
import mediapipe as mp
import math

# Set environment variable
os.environ['QT_QPA_PLATFORM'] = 'xcb'

# pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

# Load depth estimation model
checkpoint = "vinvino02/glpn-nyu"
depth_estimator = pipeline("depth-estimation", model=checkpoint)

vid = cv2.VideoCapture(0)

width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
small_width = math.floor(width * 0.25)
small_height = math.floor(height * 0.25)

while True:
    # Capture the video frame
    ret, frame = vid.read()
    frame = cv2.flipND(frame, 1)
    original_frame = frame.copy()
    frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    predictions = depth_estimator(pil_image)

    depth_map = predictions["depth"]

    pose_res = pose.process(original_frame)
    depth_values = []
    for landmark in pose_res.pose_landmarks.landmark:
        x = math.floor(landmark.x * small_width)
        y = math.floor(landmark.y * small_height)

        print(depth_map.size)

        if 0 <= x < small_width and 0 <= y < small_height:
            print(x, y, small_width, small_height)

            depth_value = depth_map.getpixel((x, y))
            depth_values.append(depth_value)

    threshold_depth = np.average(depth_values) + 5
    print({'treshold_depth': threshold_depth})
    binary_mask = np.where(depth_map <= threshold_depth, 255, 0).astype(np.uint8)

    binary_mask_resized = cv2.resize(binary_mask, (original_frame.shape[1], original_frame.shape[0]))
    masked_frame = cv2.bitwise_and(original_frame, original_frame, mask=binary_mask_resized)

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing.draw_landmarks(masked_frame, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('masked frame', masked_frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture object and close all windows
vid.release()
cv2.destroyAllWindows()
