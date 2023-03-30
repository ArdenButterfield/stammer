#!/usr/bin/env python3

from argparse import ArgumentParser
from typing import List
import numpy as np
from scipy.io import wavfile
from pathlib import Path
import shutil
import subprocess
import sys
from PIL import Image
import tempfile
import logging

import image_tiling
import fraction_bits
from audio_matching import BasicAudioMatcher, CombinedFrameAudioMatcher, UniqueAudioMatcher

TEMP_DIR = Path('temp')

MAX_BASIS_WIDTH = 6
MAX_TESSELLATION_COUNT = 9
DEFAULT_FRAME_LENGTH = 1/25 # Seconds

BAND_WIDTH = 1.2
INTERNAL_SAMPLERATE = 44100 # Hz

def test_command(cmd):
    try:
        subprocess.run(cmd, capture_output=True)
    except FileNotFoundError as error:
        logging.error(f"ERROR: '{cmd[0]}' not found. Please install it.")
        raise error

def file_type(path):
    # is the file at path an audio file, video file, or neither?
    return subprocess.run(
        [
            'ffprobe',
            '-loglevel', 'error',
            '-show_entries', 'stream=codec_type',
            '-of', 'csv=p=0',
            str(path)
        ],
        capture_output=True,
        check=True,
        text=True
    ).stdout

def get_duration(path):
    return subprocess.run(
            [
                'ffprobe',
                '-i', str(path),
                '-show_entries', 'format=duration',
                '-v', 'quiet',
                '-of', 'csv=p=0'
            ],
            capture_output=True,
            check=True,
            text=True
        ).stdout

def get_framecount(path):
    return subprocess.run(
            [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-count_frames',
                '-show_entries', 'stream=nb_read_frames',
                '-print_format', 'csv=p=0',
                str(path)
            ],
            capture_output=True,
            check=True,
            text=True
        ).stdout



def build_output_video(frames_dir, outframes_dir, matcher, video_frame_length, audio_frame_length, output_path):
    logging.info("building output video")
    
    def tesselate_composite(match_row, basis_coefficients, i):
        tiles: List[Image.Image] = []
        bits: List[List[int]] = []
        used_coeffs = [(j, coefficient) for j, coefficient in enumerate(basis_coefficients) if coefficient != 0]
        for k, coeff in used_coeffs:
            tiles.append(Image.open(frames_dir / f'frame{match_row[k]+1:06d}.png'))
            hot_bits,_ = fraction_bits.as_array(coeff)
            bits.append(hot_bits)
        tesselation = image_tiling.Tiling(height=tiles[0].height,width=tiles[0].width)
        output_frame = Image.new('RGB',(tiles[0].width, tiles[0].height))
        for m in np.arange(1,MAX_TESSELLATION_COUNT):
            first_hot = next(((offset, x) for offset, x in enumerate(bits) if x[m]), None)
            if first_hot is not None:
                do_tile = tesselation.needs_tiling
                tb = tiles[first_hot[0]].copy()
                x0, y0, w, h = tesselation.get_image_placement()
                tb.thumbnail((w,h))
                output_frame.paste(tb, (x0,y0))
                if do_tile:
                    output_frame.paste(tb,(x0, y0 + tb.height))
        output_frame.save(outframes_dir / f'frame{i:06d}.png')

    if type(matcher) in (BasicAudioMatcher, UniqueAudioMatcher):
        for video_frame_i in range(int(len(matcher.get_best_matches()) * audio_frame_length / video_frame_length)):
            elapsed_time = video_frame_i * video_frame_length
            audio_frame_i = int(elapsed_time / audio_frame_length)
            time_past_start_of_audio_frame = elapsed_time - (audio_frame_i * audio_frame_length)
            match_num = matcher.get_best_matches()[audio_frame_i]
            elapsed_time_in_carrier = match_num * audio_frame_length + time_past_start_of_audio_frame
            carrier_video_frame = int(elapsed_time_in_carrier / video_frame_length)
            shutil.copy(frames_dir / f'frame{carrier_video_frame+1:06d}.png', outframes_dir / f'frame{video_frame_i:06d}.png')
    elif type(matcher) == CombinedFrameAudioMatcher:
        best_matches = matcher.get_best_matches()
        basis_coefficients = matcher.get_basis_coefficients()
        for video_frame_i in range(int(len(matcher.get_best_matches()) * audio_frame_length / video_frame_length)):
            elapsed_time = video_frame_i * video_frame_length
            audio_frame_i = int(elapsed_time / audio_frame_length)
            time_past_start_of_audio_frame = elapsed_time - (audio_frame_i * audio_frame_length)
            match_row = matcher.get_best_matches()[audio_frame_i]
            match_row = [int((i * audio_frame_length + time_past_start_of_audio_frame)/video_frame_length) for i in match_row]
            tesselate_composite(match_row=match_row, basis_coefficients=basis_coefficients[audio_frame_i], i=video_frame_i)

    subprocess.run(
        [
            'ffmpeg',
            '-hide_banner',
            '-loglevel', 'error',
            '-y',
            '-framerate', str(1/video_frame_length),
            '-i', str(outframes_dir / 'frame%06d.png'),
            '-i', str(TEMP_DIR / 'out.wav'),
            '-c:a', 'aac',
            '-shortest',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            str(output_path)
        ],
        check=True
    )

COMMON_AUDIO_EXTS = [
    "wav",
    "wv",
    "mp3",
    "m4a",
    "aac",
    "ogg",
    "opus",
]

def is_audio_filename(name):
    import os.path
    return os.path.splitext(name)[1][1:] in COMMON_AUDIO_EXTS

def get_audio_as_wav_bytes(path):
    import io

    ff_out = bytearray(subprocess.check_output(
        [
            'ffmpeg',
            '-hide_banner',
            '-loglevel', 'error',
            '-i', str(path),
            '-vn', '-map', '0:a:0',
            '-ac', '1',
            '-ar', str(INTERNAL_SAMPLERATE),
            '-c:a', 'pcm_s16le',
            '-f', 'wav', '-'
        ]
    ))

    # fix file size in header length
    actual_data_len = len(ff_out)-44
    ff_out[4:8] = (actual_data_len).to_bytes(4,byteorder="little")

    return io.BytesIO(bytes(ff_out))

def process(carrier_path, modulator_path, output_path, custom_frame_length, mode):
    if not carrier_path.is_file():
        raise FileNotFoundError(f"Carrier file {carrier_path} not found.")
    if not modulator_path.is_file():
        raise FileNotFoundError(f"Modulator file {modulator_path} not found.")
    carrier_type = file_type(carrier_path)
    modulator_type = file_type(modulator_path)

    if 'video' in carrier_type:
        output_is_audio = is_audio_filename(output_path)
        carrier_is_video = not output_is_audio

        logging.info("Calculating video length")
        carrier_duration = float(get_duration(carrier_path))
        carrier_framecount = float(get_framecount(carrier_path))
        real_frame_length = carrier_duration / carrier_framecount
        if custom_frame_length is None:
            frame_length = real_frame_length
        else:
            frame_length = float(custom_frame_length)

        if not output_is_audio:
            logging.info("Separating video frames")
            frames_dir = TEMP_DIR / 'frames'
            frames_dir.mkdir()
            subprocess.run(
                [
                    'ffmpeg',
                    '-loglevel', 'error',
                    '-i', str(carrier_path),
                    str(frames_dir / 'frame%06d.png')
                ],
                check=True
            )

    elif 'audio' in carrier_type:
        carrier_is_video = False
        if custom_frame_length is None:
            frame_length = DEFAULT_FRAME_LENGTH
        else:
            frame_length = float(custom_frame_length)
    else:
        logging.error(f"Unrecognized file type: {carrier_path}. Should be audio or video")
        return

    if not (('video' in modulator_type) or ('audio' in modulator_type)):
        logging.error(f"Unrecognized file type: {modulator_path}. Should be audio or video")
        return

    logging.info("reading audio")
    _, carrier_audio = wavfile.read(get_audio_as_wav_bytes(carrier_path))
    _, modulator_audio = wavfile.read(get_audio_as_wav_bytes(modulator_path))


    if mode == "basic":
        matcher = BasicAudioMatcher(carrier_audio, modulator_audio, INTERNAL_SAMPLERATE, frame_length)
    elif mode == "combination":
        matcher = CombinedFrameAudioMatcher(carrier_audio, modulator_audio, INTERNAL_SAMPLERATE, frame_length)
    elif mode == "unique":
        matcher = UniqueAudioMatcher(carrier_audio, modulator_audio, INTERNAL_SAMPLERATE, frame_length)

    logging.info("creating output audio")
    matcher.make_output_audio(TEMP_DIR / 'out.wav')
    

    if carrier_is_video:
        outframes_dir = TEMP_DIR / 'outframes'
        outframes_dir.mkdir()
        build_output_video(frames_dir, outframes_dir, matcher, real_frame_length, frame_length, output_path)
    else:
        subprocess.run(
            [
                'ffmpeg',
                '-loglevel', 'error',
                '-y', '-i', str(TEMP_DIR / 'out.wav'),
                str(output_path)
            ],
            check=True
        )

def main():
    logging.basicConfig(format='%(message)s', level=logging.INFO)

    # check required command line tools
    test_command(['ffmpeg', '-version'])
    test_command(['ffprobe', '-version'])
    
    parser = ArgumentParser()
    parser.add_argument('carrier_path', type=Path, metavar='carrier_track', help='path to an audio or video file that frames will be taken from')
    parser.add_argument('modulator_path', type=Path, metavar='modulator_track', help='path to an audio or video file that will be reconstructed using the carrier track')
    parser.add_argument('output_path', type=Path, metavar='output_file', help='path to file that will be written to; should have an audio or video file extension (such as .wav, .mp3, .mp4, etc.)')
    parser.add_argument('--custom-frame-length', '-f', help='uses this number as frame length, in seconds. defaults to 0.04 seconds (1/25th of a second) for audio, or the real frame rate for video')
    parser.add_argument('-m', '--mode', choices=('basic', 'combination', 'unique'), default='basic', help="""Which algorithm Stammer will use.
        basic: replace each frame in the modulator with the most similar frame in the carrier.
        combination: replace each frame in the modulator with a linear combination of several frames in the carrier, to more closely approximte it.
        unique: limit each carrier frame to only appear once. If the carrier is longer than the modulator, some carrier frames will not be played, if it is shorter than the modulator, the modulator will be trimmed to thee length of the carrier.""")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tempdir:
        global TEMP_DIR
        TEMP_DIR = Path(tempdir)
        process(**vars(args))


if __name__ == '__main__':
    main()
