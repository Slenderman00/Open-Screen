import cv2
import pyvirtualcam
import numpy as np

from OpenScreen.utils import is_cam_used, cam_exists
from OpenScreen.settings import create_settings, load_settings, settings_exist, edit_settings
import threading
import time
import argparse
from OpenScreen.backgroundReplacement import GenerateBackgroundReplacement

import openscreen_cpp


class CameraProcess:
    def __init__(self):
        self.settings = load_settings()

        self.vid = None
        self.real_camera = int(self.settings["general"]["real_camera"])
        self.mount_camera(self.real_camera)
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.release_camera()
        self.fake_camera = int(self.settings["general"]["fake_camera"])
        self.status_checker = None
        self.generator = GenerateBackgroundReplacement()
        self.running = True
        background_path = str(self.settings["general"]["background"])
        self.background = cv2.imread(background_path)
        if self.background is None:
            raise FileNotFoundError(f"Unable to load background image from path: {background_path}")

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
        # Check if virtual  and physical cam exists
        if not cam_exists(self.settings["general"]["fake_camera"]):
            print(f"Could not find virtual camera: /dev/video{self.settings['general']['fake_camera']}")
            quit()

        if not cam_exists(self.settings["general"]["real_camera"]):
            print(f"Could not find camera: /dev/video{self.settings['general']['real_camera']}")
            quit()

        self.background = cv2.resize(self.background, (int(self.width), int(self.height)))

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

                    # Replace black pixels with background pixels
                    black_pixels = np.all(masked_frame == [0, 0, 0], axis=2)
                    masked_frame[black_pixels] = self.background[black_pixels]

                    frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)

                camera.send(frame)
                camera.sleep_until_next_frame()


def main():
    if not settings_exist():
        create_settings()

    parser = argparse.ArgumentParser(description="OpenScreen")

    parser.add_argument(
        "--settings",
        "-s",
        action="store_true",
        help="Edit the settings",
    )

    args = parser.parse_args()

    if args.settings:
        edit_settings()
        return

    # camera_process = CameraProcess()
    # camera_process.run()

    openscreen_cpp.start("path/to/model.pt", 224, "path/to/background.jpg")


if __name__ == "__main__":
    main()
