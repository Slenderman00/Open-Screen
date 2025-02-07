from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, include_paths, library_paths
import subprocess

torch_base_path = include_paths()[0].split("include")[0]

opencv_cflags = subprocess.check_output(['pkg-config', '--cflags', 'opencv4']).decode().split()
opencv_libs = subprocess.check_output(['pkg-config', '--libs', 'opencv4']).decode().split()
openscreen_cpp = CppExtension(
    "openscreen_cpp",
    sources=["OpenScreen/OpenScreenCpp/openScreen.cpp"],
    include_dirs=include_paths() + ["/usr/include/opencv4", torch_base_path],
    library_dirs=library_paths() + ["/usr/lib"],
    libraries=[
        "opencv_core",
        "opencv_highgui",
        "opencv_imgproc",
        "opencv_videoio",
        "opencv_imgcodecs",
        "gomp"
    ],
    extra_compile_args=["-std=c++17"] + opencv_cflags,
    extra_link_args=opencv_libs,
)

setup(
    name="OpenScreen",
    version="0.1.1",
    packages=["OpenScreen", "OpenScreen.OpenScreenCpp"],
    include_package_data=True,
    install_requires=[
        "numpy<2",
        "Pillow==10.3.0",
        "pyvirtualcam==0.11.1",
        "toml==0.10.2",
        "torch",
        "torchvision",
        "opencv-python"
    ],
    package_data={"OpenScreen": ["static/*"]},
    entry_points="""
        [console_scripts]
        openscreen=OpenScreen.main:main
    """,
    ext_modules=[openscreen_cpp],
    cmdclass={
        'build_ext': BuildExtension
    },
)
