from transformers import pipeline
from PIL import Image
import cv2
import numpy as np
import mediapipe as mp
import math


class generateDepthMap():
    def __init__(self):
        self.frame = None
        self.mask = None
        self.running = False
        self.scale = 0.3
        self.pose_res = None

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False)

        # Load depth estimation model
        self.checkpoint = "vinvino02/glpn-nyu"
        self.depth_estimator = pipeline("depth-estimation", model=self.checkpoint)

    def set_frame(self, frame):
        self.frame = frame
        self.process()

    def get_mask(self):
        return self.mask

    def get_pose_res(self):
        return self.pose_res

    def get_ready(self):
        return self.mask is not None and self.pose_res is not None and self.running

    def process(self):
        if self.frame is not None:
            frame = self.frame
            original_frame = frame.copy()
            frame = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale)
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            predictions = self.depth_estimator(pil_image)
            depth_map = predictions["depth"]

            self.pose_res = self.pose.process(original_frame)

            threshold_depth = self.get_threshold_depth(depth_map, self.pose_res)
            binary_mask = np.where(depth_map <= threshold_depth, 255, 0).astype(np.uint8)
            binary_mask_resized = cv2.resize(binary_mask, (original_frame.shape[1], original_frame.shape[0]))

            self.mask = binary_mask_resized

    def get_threshold_depth(self, depth_map, pose_res):
        depth_map_width = depth_map.size[0]
        depth_map_height = depth_map.size[1]

        depth_values = []
        for landmark in pose_res.pose_landmarks.landmark:
            x = math.floor(landmark.x * depth_map_width)
            y = math.floor(landmark.y * depth_map_height)

            if 0 <= x < depth_map_width and 0 <= y < depth_map_height:

                depth_value = depth_map.getpixel((x, y))
                depth_values.append(depth_value)

        threshold_depth = np.average(depth_values) + 4

        return threshold_depth
