from setuptools import setup

setup(
    name="OpenScreen",
    version="0.1.2",
    packages=["OpenScreen"],
    include_package_data=True,
    install_requires=[
        'numpy>=1.17.0',
        'Pillow==10.3.0',
        'pyvirtualcam==0.11.1',
        'toml==0.10.2',
        'torch==2.3.1',
        'torchvision==0.18.1',
        'opencv-python==4.11.0.86'
    ],
    package_data={"OpenScreen": ["static/*"]},
    entry_points="""
        [console_scripts]
        openscreen=OpenScreen.main:main
    """,
)
