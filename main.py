from transformers import pipeline
import os
import cv2
import mediapipe as mp
import pyvirtualcam


from depthMap import generateDepthMap

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

generator = generateDepthMap()

with pyvirtualcam.Camera(width=int(width), height=int(height), fps=20) as cam:
    while True:
        # Capture the video frame
        ret, frame = vid.read()
        frame = cv2.flipND(frame, 1)

        generator.set_frame(frame)

        binary_mask = generator.get_mask()
        masked_frame = cv2.bitwise_and(frame, frame, mask=binary_mask)

        pose_res = generator.get_pose_res()

        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(masked_frame, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        # cv2.imshow('masked frame', masked_frame)

        masked_frame_rgb = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
        cam.send(masked_frame_rgb)
        cam.sleep_until_next_frame()

        # Exit loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture object and close all windows
    vid.release()
    cv2.destroyAllWindows()
