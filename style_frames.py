# Brycen Westgarth and Tristan Jogminas
# March 5, 2021
import os
import tensorflow_hub as hub
import numpy as np
import tensorflow as tf
from PIL import Image
import glob
import imageio
import matplotlib.pylab as plt
import cv2
import logging
from config import Config as config

logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TFHUB_CACHE_DIR'] = config.TENSORFLOW_CACHE_DIRECTORY
hub_module = hub.load(config.TENSORFLOW_HUB_HANDLE)

class StyleFrame:

    MAX_CHANNEL_INTENSITY = 255.0

    def __init__(self):
        self.input_frame_directory = glob.glob(f'{config.INPUT_FRAME_DIRECTORY}/*')
        self.output_frame_directory = glob.glob(f'{config.OUTPUT_FRAME_DIRECTORY}/*')
        self.style_directory = glob.glob(f'{config.STYLE_REF_DIRECTORY}/*')
        self.ref_count = len(config.STYLE_SEQUENCE)
        self.use_input_frame_cache = False

        if len(self.input_frame_directory):
            self.use_input_frame_cache = True
            # Retrieve an image in the input frame dir to get the width
            self.frame_width, _height = Image.open(self.input_frame_directory[0]).size

        files_to_be_cleared = self.output_frame_directory
        if config.CLEAR_INPUT_FRAME_CACHE:
            files_to_be_cleared += self.input_frame_directory
            self.use_input_frame_cache = False
        
        for file in files_to_be_cleared:
            os.remove(file)

    def get_input_frames(self):
        if self.use_input_frame_cache:
            print("Using cached input frames")
            return
        vid_obj = cv2.VideoCapture(config.INPUT_VIDEO_PATH)
        frame_interval = np.floor((1.0 / config.INPUT_FPS) * 1000)
        success, image = vid_obj.read()
        if image is None:
            raise ValueError(f"ERROR: Please provide missing video: {config.INPUT_VIDEO_PATH}")
        img = Image.fromarray(image[:, :, 0:3])
        scale_constant = (config.FRAME_HEIGHT / image.shape[0])
        self.frame_width = int(image.shape[1] * scale_constant)
        img = img.resize((self.frame_width, config.FRAME_HEIGHT))
        cv2.imwrite(config.INPUT_FRAME_PATH.format(0), np.asarray(img).astype(np.uint8))

        count = 1
        while success:
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
        frame_length = len(self.input_frame_directory)
        style_refs = list()
        resized_ref = False
        style_files = sorted(self.style_directory)
        self.t_const = frame_length if self.ref_count == 1 else np.ceil(frame_length / (self.ref_count - 1))

        # Open first style ref and force all other style refs to match size
        first_style_ref = Image.open(style_files.pop(0))
        first_style_width, first_style_height = first_style_ref.size
        np_first_style_ref =  np.asarray(first_style_ref)[:, :, 0:3]
        style_refs.append(np_first_style_ref / self.MAX_CHANNEL_INTENSITY)

        for filename in style_files:
            style_ref = Image.open(filename)
            style_ref_width, style_ref_height = style_ref.size
            # Resize all style_ref images to match first style_ref dimensions
            if style_ref_width != first_style_width or style_ref_height != first_style_height:
                resized_ref = True
                style_ref = style_ref.resize((first_style_width, first_style_height))
            np_style_ref =  np.asarray(style_ref)[:, :, 0:3]
            style_refs.append(np_style_ref / self.MAX_CHANNEL_INTENSITY)

        if resized_ref:
            print("WARNING: Resizing style images which may cause distortion. To avoid this, please provide style images with the same dimensions")

        self.transition_style_seq = list()
        for i in range(self.ref_count):
            if config.STYLE_SEQUENCE[i] is None:
                self.transition_style_seq.append(None)
            else:
                self.transition_style_seq.append(style_refs[config.STYLE_SEQUENCE[i]])

    def _trim_img(self, img):
        return img[:config.FRAME_HEIGHT, :self.frame_width]

    def get_output_frames(self):
        self.input_frame_directory = glob.glob(f'{config.INPUT_FRAME_DIRECTORY}/*')
        ghost_frame = None
        for count, filename in enumerate(sorted(self.input_frame_directory)):
            if count % 10 == 0:
                print(f"Output frame: {(count/len(self.input_frame_directory)):.0%}")
            content_img = np.asarray(Image.open(filename)) / self.MAX_CHANNEL_INTENSITY
            curr_style_img_index = int(count / self.t_const)
            mix_ratio = 1 - ((count % self.t_const) / self.t_const)
            inv_mix_ratio = 1 - mix_ratio

            prev_image = self.transition_style_seq[curr_style_img_index]
            next_image = self.transition_style_seq[curr_style_img_index + 1]
            
            prev_is_content_img = False
            next_is_content_img = False
            if prev_image is None:
                prev_image = content_img
                prev_is_content_img = True
            if next_image is None:
                next_image = content_img
                next_is_content_img = True
            # If both, don't need to apply style transfer
            if prev_is_content_img and next_is_content_img:
                ghost_frame = content_img
                plt.imsave(config.OUTPUT_FRAME_PATH.format(count), ghost_frame)
                continue
            
            if count > 0:
                content_img = ((1 - config.GHOST_FRAME_TRANSPARENCY) * content_img) + (config.GHOST_FRAME_TRANSPARENCY * ghost_frame)
            content_img = tf.cast(tf.convert_to_tensor(content_img), tf.float32)

            if prev_is_content_img:
                blended_img = next_image
            elif next_is_content_img:
                blended_img = prev_image
            else:
                prev_style = mix_ratio * prev_image
                next_style = inv_mix_ratio * next_image
                blended_img = prev_style + next_style

            blended_img = tf.cast(tf.convert_to_tensor(blended_img), tf.float32)
            expanded_blended_img = tf.constant(tf.expand_dims(blended_img, axis=0))
            expanded_content_img = tf.constant(tf.expand_dims(content_img, axis=0))
            # Apply style transfer
            stylized_img = hub_module(expanded_content_img, expanded_blended_img).pop()
            stylized_img = tf.squeeze(stylized_img)

            # Re-blend
            if prev_is_content_img:
                prev_style = mix_ratio * content_img
                next_style = inv_mix_ratio * stylized_img
            if next_is_content_img:
                prev_style = mix_ratio * stylized_img
                next_style = inv_mix_ratio * content_img
            if prev_is_content_img or next_is_content_img:
                stylized_img = self._trim_img(prev_style) + self._trim_img(next_style)

            if config.PRESERVE_COLORS:
                stylized_img = self._color_correct_to_input(content_img, stylized_img)
            
            ghost_frame = np.asarray(self._trim_img(stylized_img))

            plt.imsave(config.OUTPUT_FRAME_PATH.format(count), ghost_frame)
        self.output_frame_directory = glob.glob(f'{config.OUTPUT_FRAME_DIRECTORY}/*')

    def _color_correct_to_input(self, content, generated):
        # image manipulations for compatibility with PILLOW
        content = np.array((content * self.MAX_CHANNEL_INTENSITY)).astype(np.uint8)
        content = Image.fromarray(content).convert('YCbCr')
        generated = np.array((generated * self.MAX_CHANNEL_INTENSITY)).astype(np.uint8)
        generated = Image.fromarray(generated).convert('YCbCr')
        generated = generated.resize((self.frame_width, config.FRAME_HEIGHT))
        # extract channels
        _, cb, cr = content.split()
        y, _, _ = generated.split()
        # merge intensity and color spaces
        color_corrected = Image.merge('YCbCr', (y, cb, cr))
        return np.asarray(color_corrected.convert('RGB')) / self.MAX_CHANNEL_INTENSITY


    def create_video(self):
        self.output_frame_directory = glob.glob(f'{config.OUTPUT_FRAME_DIRECTORY}/*')
        writer = imageio.get_writer(config.OUTPUT_VIDEO_PATH, format='mp4', mode='I', fps=config.OUTPUT_FPS)

        for count, filename in enumerate(sorted(self.output_frame_directory)):
            if count % 10 == 0:
                print(f"Saving frame: {(count/len(self.output_frame_directory)):.0%}")
            img = Image.open(filename)
            writer.append_data(np.asarray(img))

        writer.close()
        print(f"Style transfer complete! Output at {config.OUTPUT_VIDEO_PATH}")

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
