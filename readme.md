![Open Screen](https://github.com/Slenderman00/open_screen/blob/master/media/banner.png?raw=true)

Are you tired of cleaning your room?  
Do you want to flex on your colleagues that you're on holiday in the Maldives?  
Well, fear no more because Open Screen is the solution.
---
With the power of modern technology Open Screen replaces your boring background with whatever you need!

---
## Usage
Open screen relies on [v4l2loopback-dkms](https://github.com/umlaeute/v4l2loopback) to create the virtual camera 
```
sudo modprobe v4l2loopback devices=1
```

To start open screen run the followin command. 

```
openscreen
```
Settings can be edited by running 
```
openscreen --settings
```
The following settings are available
```
[general]
flip_image = false
real_camera = 0
fake_camera = 2
depth_scale = 0.3
threshold_offset = 4
background = "/home/.../stock.png"
background_own_thread = false

[debug]
show_pose = true

```
## Installation
```
pip install git+https://github.com/Slenderman00/open_screen.git#egg=openscreen 
```

## Demo
![Demo](https://github.com/Slenderman00/open_screen/blob/master/media/openScreen.png?raw=true)

credits:
- https://www.flaticon.com/free-icon/green-screen_2059843
- https://www.freepik.com/icon/desk-chair_3484724#fromView=search&page=1&position=50&uuid=8ba6c8f3-cf4e-4fa7-930f-af0b9dca557e