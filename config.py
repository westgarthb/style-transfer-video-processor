# Brycen Westgarth and Tristan Jogminas
# March 5, 2021

class Config:
    ROOT_PATH = '.'
    # defines the maximum height dimension in pixels. Used for down-sampling the video frames
    FRAME_HEIGHT = 864
    CLEAR_INPUT_FRAME_CACHE = True
    # defines the rate at which you want to capture frames from the input video
    INPUT_FPS = 20
    INPUT_VIDEO_NAME = 'input_vid.MOV'
    INPUT_VIDEO_PATH = f'{ROOT_PATH}/{INPUT_VIDEO_NAME}'
    INPUT_FRAME_DIRECTORY = f'{ROOT_PATH}/input_frames'
    INPUT_FRAME_FILE = '{:0>4d}_frame.png'
    INPUT_FRAME_PATH = f'{INPUT_FRAME_DIRECTORY}/{INPUT_FRAME_FILE}'

    STYLE_REF_DIRECTORY = f'{ROOT_PATH}/style_ref'
    # defines the reference style image transition sequence. Values correspond to indices in STYLE_REF_DIRECTORY
    STYLE_SEQUENCE = [0 ,1, 2]

    OUTPUT_FPS = 20
    OUTPUT_VIDEO_NAME = 'output_video1.mp4'
    OUTPUT_VIDEO_PATH = f'{ROOT_PATH}/{OUTPUT_VIDEO_NAME}'
    OUTPUT_FRAME_DIRECTORY = f'{ROOT_PATH}/output_frames'
    OUTPUT_FRAME_FILE = '{:0>4d}_frame.png'
    OUTPUT_FRAME_PATH = f'{OUTPUT_FRAME_DIRECTORY}/{OUTPUT_FRAME_FILE}'

    GHOST_FRAME_TRANSPARENCY = 0.1

    TENSORFLOW_CACHE_DIRECTORY = f'{ROOT_PATH}/tensorflow_cache'
    TENSORFLOW_HUB_HANDLE = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
