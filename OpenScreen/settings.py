import toml
import os
from pathlib import Path

own_dir = Path(__file__).parent.absolute()

settings = {
    "general": {
        "flip_image": False,
        "real_camera": 0,
        "fake_camera": 2,
        "background": str(own_dir / 'static/alley.png'),
        "flip_background": False,
        "blur_mask": False,
    },
    "model": {
        "model": str(own_dir / 'static/model.pth'),
        "image_resolution": 256,
        "depth": 5,
        "init_features": 32,
    },
    "debug": {
    }
}


def create_settings():
    # create the settings.toml file
    with open(own_dir / "settings.toml", "w") as f:
        toml.dump(settings, f)


def select_editor():
    # Use editor env var if it is set
    if 'EDITOR' in os.environ:
        return os.environ['EDITOR']

    editors = ["vim", "nano", "emacs", "code"]
    for i, editor in enumerate(editors):
        print(f"{i}: {editor}")

    while True:
        try:
            choice = int(input("Select an editor: "))
            if choice < 0 or choice >= len(editors):
                raise ValueError
            break
        except ValueError:
            print("Invalid choice")

    return editors[choice]


def edit_settings():
    os.system(f"{select_editor()} {own_dir / 'settings.toml'}")


def settings_exist():
    return (own_dir / "settings.toml").exists()


def load_settings():
    # check if settings.toml exists
    if not settings_exist():
        create_settings()
        edit_settings()
        return load_settings()

    with open(own_dir / "settings.toml", "r") as f:
        settings = toml.load(f)

    return settings
