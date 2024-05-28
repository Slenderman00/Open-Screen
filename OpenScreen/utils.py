import subprocess
import os


def is_cam_used(cam):
    if isinstance(cam, int):
        process = subprocess.Popen(['fuser', f'/dev/video{cam}'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, _ = process.communicate()
        processes = len(stdout.decode().split())
        return processes > 1
    return False


def cam_exists(cam):
    #return os.path.isfile(f'/dev/video{cam}')
    return True
    