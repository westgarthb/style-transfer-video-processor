# Neural Style Transfer Transition Video Processing
This code extends the [neural style transfer](https://www.tensorflow.org/tutorials/generative/style_transfer) 
image processing technique to video by generating smooth transitions 
between a sequence of reference style images which are then applied to 
video frames.

## System Requirements

This algorithm is very computationally intensive and I highly 
recommended that you install drivers for [Tensorflow GPU support](https://www.tensorflow.org/install/gpu)
if you have access to a CUDA compatable GPU. Even with GPU acceleration
the program may take several minutes to render a video. 

#### Dependencies
*TODO...*

## Configuration and Usage
*TODO...*

## Examples
### Input Video
![file](/examples/reference.gif)
### Example 1
##### Style Transition Sequence
![file](/examples/example1_style_sequence.png)
##### Output Video
![file](/examples/example1.gif)
### Example 2
##### Style Transition Sequence
![file](/examples/example2_style_sequence.png)
##### Output Video
![file](/examples/example2.gif)