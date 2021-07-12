# Neural Style Transfer Transition Video Processing
By Brycen Westgarth and Tristan Jogminas

## Description
This code extends the [neural style transfer](https://www.tensorflow.org/tutorials/generative/style_transfer) 
image processing technique to video
by generating smooth transitions between a sequence of 
reference style images across video frames. The generated output 
video is a highly altered, artistic representation of the input
video consisting of constantly changing abstract patterns and colors
that emulate the original content of the video. The user's choice
of style reference images, style sequence order, and style sequence
length allow for infinite user experimentation and the creation of 
an endless range of artistically interesting videos.


## System Requirements
This algorithm is computationally intensive so I highly 
recommend optimizing its performance by installing drivers for 
[Tensorflow GPU support](https://www.tensorflow.org/install/gpu)
if you have access to a CUDA compatible GPU. Alternatively, you can
take advantage of the free GPU resources available through Google Colab Notebooks. 
Even with GPU acceleration, the program may take several minutes to render a video. 

[Colab Notebook Version](https://colab.research.google.com/drive/1ZjSvUv0Wqib6khaiqcBvRrI5GeSjFcOV?usp=sharing)

## Configuration
All configuration of the video properties and input/output file
locations can be set by the user in config.py 

Configurable Variable in config.py			         | Description
------------------------|------------
ROOT_PATH     	| Path to input/output directory
FRAME_HEIGHT    | Sets height dimension in pixels to resize the output video to. Video width will be calculated automatically to preserve aspect ratio. Low values will speed up processing time but reduce output video quality 
INPUT_FPS 			    | Defines the rate at which frames are captured from the input video
INPUT_VIDEO_NAME     	| Filename of input video
STYLE_SEQUENCE     	| List that contains the indices corresponding to the image files in the 'style_ref' folder. Defines the reference style image transition sequence. Can be arbitrary length, the rate at which the video transitions between styles will be adjusted to fit the video
OUTPUT_FPS		    | Defines the frame rate of the output video
OUTPUT_VIDEO_NAME   | Filename of output video to be created
GHOST_FRAME_TRANSPARENCY | Proportional feedback constant for frame generation. Should be a value between 0 and 1. Affects the amount change that can occur between frames and the smoothness of the transitions. 
CLEAR_INPUT_FRAME_CACHE  | If True the program clears the captured input frames each run. If False, you can run multiple style sequences without having to recapture video frames
PRESERVE_COLORS      | If True the output video will preserve the colors of the input video. If  False the program will perform standard style transfer

**The user must find and place their own style reference images in the `style_ref` directory. 
 Style reference images can be
arbitrary size. For best results, try to use style reference images with similar dimensions
and aspect ratios. Three example style reference images are given.**<br/>
<br/>
Minor video time effects can be created by setting INPUT_FPS and OUTPUT_FPS to different relative values<br/>
- INPUT_FPS > OUTPUT_FPS creates a slowed time effect
- INPUT_FPS = OUTPUT_FPS creates no time effect
- INPUT_FPS < OUTPUT_FPS creates a timelapse effect


## Usage
```
$ python3 -m venv env
$ source env/bin/activate
$ pip3 install -r requirements.txt
$ python3 style_frames.py
```

## Examples
### Input Video
![file](/examples/reference.gif)
### Example 1
##### Reference Style Image Transition Sequence
![file](/examples/example1_style_sequence.png)
##### Output Video
![file](/examples/example1.gif)
##### Output Video with Preserved Colors
![file](/examples/example3.gif)
### Example 2
##### Reference Style Image Transition Sequence
![file](/examples/example2_style_sequence.png)
##### Output Video
![file](/examples/example2.gif)

##### [Example Video made using this program](https://youtu.be/vgl83UTciD8) 
