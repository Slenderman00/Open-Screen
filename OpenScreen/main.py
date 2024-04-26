import os
import cv2
import pyvirtualcam
import numpy as np

from OpenScreen.depthMap import generateDepthMap
from OpenScreen.utils import is_cam_used
from OpenScreen.settings import create_settings, load_settings, settings_exist
import threading
import time

# Set environment variable
os.environ['QT_QPA_PLATFORM'] = 'xcb'


class CameraProcess:
    def __init__(self):
        # Create settings if they havent been created
        if not settings_exist():
            create_settings()

        self.settings = load_settings()
        self.vid = None
        self.real_camera = int(self.settings["general"]["real_camera"])
        self.mount_camera(self.real_camera)
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.release_camera()
        self.fake_camera = int(self.settings["general"]["fake_camera"])
        self.status_checker = None
        self.generator = generateDepthMap()
        self.running = True

    def check_cam(self):
        while self.running:
            if is_cam_used(self.fake_camera):
                if self.vid is None:
                    self.mount_camera(self.real_camera)
            else:
                if self.vid is not None:
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
        with pyvirtualcam.Camera(width=int(self.width), height=int(self.height), fps=20, device=f'/dev/video{self.fake_camera}') as camera:
            self.start_checker()
            while self.running:
                frame = np.zeros((int(self.height), int(self.width), 3), np.uint8)
                if self.vid is not None:
                    ret, frame = self.vid.read()

                    if self.settings["general"]["flip_image"]:
                        frame = cv2.flipND(frame, 1)

                    self.generator.set_frame(frame)

                    binary_mask = self.generator.get_mask()
                    masked_frame = cv2.bitwise_and(frame, frame, mask=binary_mask)

                    # pose_res = generator.get_pose_res()

                    frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)

                camera.send(frame)
                camera.sleep_until_next_frame()


def main():
    camera_process = CameraProcess()
    camera_process.run()


if __name__ == "__main__":
    main()
