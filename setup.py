from setuptools import setup, find_packages

setup(
    name='OpenScreen',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy==2.0.0',
        'Pillow==10.3.0',
        'pyvirtualcam==0.11.1',
        'toml==0.10.2',
        'torch==2.3.1',
        'torchvision==0.18.1',
        'opencv-python'
    ],
    package_data={
        'OpenScreen': ['static/*']
    },
    entry_points='''
        [console_scripts]
        openscreen=OpenScreen.main:main
    ''',
)
