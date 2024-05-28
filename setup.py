from setuptools import setup, find_packages

setup(
    name='OpenScreen',
    version='0.0.3',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'mediapipe',
        'numpy==1.26.4',
        'Pillow==10.3.0',
        'pyvirtualcam==0.11.1',
        'transformers==4.37.1',
        'toml',
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
