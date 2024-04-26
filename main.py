from transformers import pipeline
import os
import cv2
import mediapipe as mp
import pyvirtualcam
import numpy as np

from depthMap import generateDepthMap
from utils import is_cam_used
import threading
import time 

# Set environment variable
os.environ['QT_QPA_PLATFORM'] = 'xcb'


class CameraProcess:
    def __init__(self):
        self.vid = None
        self.mount_camera(0)
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.release_camera()
        self.cam_num = 2
        self.status_checker = None
        self.generator = generateDepthMap()
        self.running = True

    def check_cam(self):
        while self.running:
            if is_cam_used(self.cam_num):
                if self.vid is None:
                    print("mounting")
                    self.mount_camera(0)
            else:
                if self.vid is not None:
                    print("unmounting")
                    self.release_camera()
            time.sleep(1)

    def start_checker(self):
        self.status_checker = threading.Thread(target=self.check_cam)
        self.status_checker.start()

    def mount_camera(self, camera):
        self.vid = cv2.VideoCapture(camera)

    def release_camera(self):
        self.vid.release()
        self.vid = None

    def run(self):
        with pyvirtualcam.Camera(width=int(self.width), height=int(self.height), fps=20, device='/dev/video2') as camera:
            self.start_checker()
            while self.running:
                frame = np.zeros((int(self.height), int(self.width), 3), np.uint8)
                if self.vid is not None:
                    ret, frame = self.vid.read()
                    frame = cv2.flipND(frame, 1)

                    self.generator.set_frame(frame)

                    binary_mask = self.generator.get_mask()
                    masked_frame = cv2.bitwise_and(frame, frame, mask=binary_mask)

                    # pose_res = generator.get_pose_res()

                    frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)

                camera.send(frame)
                camera.sleep_until_next_frame()


if __name__ == "__main__":
    camera_process = CameraProcess()
    camera_process.run()
