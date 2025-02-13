import subprocess
import os
from OpenScreen.settings import create_settings, load_settings, settings_exist, edit_settings


def is_cam_used(cam):
    if isinstance(cam, int):
        process = subprocess.Popen(['fuser', f'/dev/video{cam}'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, _ = process.communicate()
        processes = len(stdout.decode().split())
        return processes > 1
    return False


def cam_exists(cam):
    return os.path.lexists(f'/dev/video{cam}')
    # return True