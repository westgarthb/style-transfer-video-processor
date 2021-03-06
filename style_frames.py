# Brycen Westgarth and Tristan Jogminas
# March 5, 2021

import tensorflow_hub as hub
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import glob
import imageio
import matplotlib.pylab as plt
import cv2
from config import Config as config

os.environ['TFHUB_CACHE_DIR'] = config.TENSORFLOW_CACHE_DIR
hub_module = hub.load(config.TENSORFLOW_HUB_HANDLE)

class StyleFrame:

    MAX_CHANNEL_INTENSITY = 255.0

    def __init__(self):
        self.input_frame_directory = glob.glob(f'{config.INPUT_FRAME_DIRECTORY}/*')
        self.output_frame_directory = glob.glob(f'{config.OUTPUT_FRAME_DIRECTORY}/*')
        self.style_directory = glob.glob(f'{config.STYLE_REF_DIRECTORY}/*')
        for file in self.input_frame_directory + self.output_frame_directory:
            os.remove(file)

    def get_input_frames(self):
        vid_obj = cv2.VideoCapture(config.INPUT_VIDEO_PATH)
        frame_interval = np.floor((1.0 / config.INPUT_FPS) * 1000)
        success, image = vid_obj.read()
        img = Image.fromarray(image[:, :, 0:3])
        scale_constant = (config.FRAME_HEIGHT / image.shape[0])
        self.frame_width = int(image.shape[1] * scale_constant)
        img = img.resize((self.frame_width, config.FRAME_HEIGHT))
        cv2.imwrite(config.INPUT_FRAME_PATH.format(0), np.asarray(img).astype(np.uint8))

        count = 1
        while success:
            print(f"Input frame: {count}")
            msec_timestamp = count * frame_interval
            vid_obj.set(cv2.CAP_PROP_POS_MSEC, msec_timestamp)
            success, image = vid_obj.read()
            if not success:
                break
            img = Image.fromarray(image[:, :, 0:3])
            img = img.resize((self.frame_width, config.FRAME_HEIGHT))
            cv2.imwrite(config.INPUT_FRAME_PATH.format(count), np.asarray(img).astype(np.uint8))
            count += 1
        self.input_frame_directory = glob.glob(f'{config.INPUT_FRAME_DIRECTORY}/*')

    def get_style_info(self):
        ref_count = len(config.STYLE_SEQUENCE)
        frame_length = len(self.input_frame_directory)
        style_refs = list()
        self.t_const = np.ceil(frame_length / (ref_count - 1))

        for filename in sorted(self.style_directory):
            style_ref_img = Image.open(filename)
            scale_constant = (config.FRAME_HEIGHT / style_ref_img.height)
            style_ref_width = int(style_ref_img.width * scale_constant)
            style_ref_img = style_ref_img.resize((style_ref_width, config.FRAME_HEIGHT))
            style_refs.append(np.asarray(style_ref_img)[:, :, 0:3] / self.MAX_CHANNEL_INTENSITY)

        self.transition_style_seq = list()
        for i in range(ref_count):
            self.transition_style_seq.append(style_refs[config.STYLE_SEQUENCE[i]])

    def get_output_frames(self):
        self.input_frame_directory = glob.glob(f'{config.INPUT_FRAME_DIRECTORY}/*')
        ghost_frame = np.zeros((config.FRAME_HEIGHT, self.frame_width, 3))
        for count, filename in enumerate(sorted(self.input_frame_directory)):
            print(f"Output frame: {count+1}/{len(self.input_frame_directory)}")
            content_img = np.asarray(Image.open(filename)) / self.MAX_CHANNEL_INTENSITY
            if count > 0:
                content_img = ((1 - config.GHOST_FRAME_TRANSPARENCY) * content_img) + (config.GHOST_FRAME_TRANSPARENCY * ghost_frame)
            content_img = tf.expand_dims(tf.cast(tf.convert_to_tensor(content_img), tf.float32), axis=0)

            curr_style_img_index = int(count / self.t_const)
            prev_to_next_ratio = 1 - ((count % self.t_const) / self.t_const)
            prev_style = prev_to_next_ratio * self.transition_style_seq[curr_style_img_index]
            next_style = (1 - prev_to_next_ratio) * self.transition_style_seq[curr_style_img_index + 1]
            blended_img = prev_style + next_style

            blended_img = tf.cast(tf.convert_to_tensor(blended_img), tf.float32)
            blended_img = tf.expand_dims(blended_img, axis=0)

            stylized_img = hub_module(tf.constant(content_img), tf.constant(blended_img)).pop()
            stylized_img = tf.squeeze(stylized_img)

            ghost_frame = np.asarray(stylized_img)[:config.FRAME_HEIGHT, :self.frame_width]
            plt.imsave(config.OUTPUT_FRAME_PATH.format(count), np.asarray(stylized_img))
        self.output_frame_directory = glob.glob(f'{config.OUTPUT_FRAME_DIRECTORY}/*')

    def create_video(self):
        self.output_frame_directory = glob.glob(f'{config.OUTPUT_FRAME_DIRECTORY}/*')
        writer = imageio.get_writer(config.OUTPUT_VIDEO_PATH, format='mp4', mode='I', fps=config.OUTPUT_FPS)

        for count, filename in enumerate(sorted(self.output_frame_directory)):
            print(f"Saving frame: {count+1}/{len(self.output_frame_directory)}")
            img = Image.open(filename)
            writer.append_data(np.asarray(img))

        writer.close()

if __name__ == "__main__":
    sf = StyleFrame()
    print("Getting input frames")
    sf.get_input_frames()
    print("Getting style info")
    sf.get_style_info()
    print("Getting output frames")
    sf.get_output_frames()
    print("Saving video")
    sf.create_video()
