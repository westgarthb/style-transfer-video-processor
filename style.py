import tensorflow_hub as hub
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import glob
import imageio
import matplotlib.pylab as plt


os.environ['TFHUB_CACHE_DIR'] = '/Path/To/A/Standard/Local/Directory'

hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
hub_module = hub.load(hub_handle)

# _____________
# configuration
# -------------
# defines the reference style image transition sequence. Values correspond to indices of style_ref folder
style_sequence = [0, 1, 2, 3]
# frame rate of video reconstructed from styled images
reconstructionFR = 20
# filename of output video
output_name = 'output.mp4'


ref_length = len(style_sequence)
frame_length = len(glob.glob('vid/*'))
style_ref = np.zeros((len(glob.glob('style_ref/*')), 320, 512, 3))
t_const = np.ceil(frame_length / (ref_length-1))
count = 0


for filename in glob.glob('style_ref/*'):
    style_ref[count] = np.asarray(Image.open(filename)) / 255.0
    count += 1

transition_seq = np.zeros((ref_length, 320, 512, 3))
for i in range(ref_length):
    transition_seq[i] = style_ref[style_sequence[i]]

count = 0
ghostFrame = np.zeros((864, 486, 3))
for filename in glob.glob('vid/*'):
    content_img = np.asarray(Image.open(filename)) / 255.0
    if count > 0:
        content_img = (0.9 * content_img) + (0.1 * ghostFrame)
    content_img = tf.expand_dims(tf.cast(tf.convert_to_tensor(content_img), tf.float32), axis=0)
    new_img = ((t_const-1-(count%t_const))/t_const * transition_seq[int(count/t_const)]) + (
            (count % t_const) / t_const * transition_seq[(int(count / t_const) + 1) % ref_length])
    style_img = tf.cast(tf.convert_to_tensor(new_img), tf.float32)
    style_img = tf.expand_dims(style_img, axis=0)

    outputs = hub_module(tf.constant(content_img), tf.constant(style_img))
    stylized_img = outputs[0]
    ghostFrame = np.asarray(stylized_img[0])
    plt.imsave('Images/{:0>3d}.png'.format(count), np.asarray(stylized_img[0]))
    count += 1

writer = imageio.get_writer(output_name, format='mp4', mode='I', fps=reconstructionFR)

for filename in glob.glob('Images/*.png'):
    img = Image.open(filename)
    writer.append_data(np.asarray(img))

writer.close()

