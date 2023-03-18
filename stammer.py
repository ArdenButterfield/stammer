from argparse import ArgumentParser
import numpy as np
from scipy.io import wavfile
from pathlib import Path
import shutil
import subprocess
import sys

TEMP_DIR = Path('temp')

MAX_BASIS_WIDTH = 6
DEFAULT_FRAME_LENGTH = 1/25 # Seconds

BAND_WIDTH = 1.2
INTERNAL_SAMPLERATE = 44100 # Hz

def test_command(cmd):
    try:
        subprocess.run(cmd, capture_output=True)
    except FileNotFoundError as error:
        print(f"ERROR: '{cmd[0]}' not found. Please install it.")
        raise error

def make_normalized_bands(frames_input,band_width):
    transforms = np.fft.fft(frames_input)
    spectra = abs(transforms[:,1:len(transforms[0])//2])
    split_points = [0]
    i = 2
    while i < len(spectra[0]):
        if int(i) > split_points[-1]:
            split_points.append(int(i))
        i *= band_width
    section_lengths = []
    for i in range(len(split_points) - 1):
        section_lengths.append(split_points[i+1]-split_points[i])
    section_lengths.append(len(spectra[0]) - split_points[-1])
    bands = np.divide(np.add.reduceat(spectra, split_points, axis=1), section_lengths) # average value in each vand
    vector_magnitudes = np.sqrt((bands * bands).sum(axis=1))
    vector_magnitudes[vector_magnitudes==0]=1
    normalized_bands = bands / vector_magnitudes[:,None]
    
    return normalized_bands

def make_frames(input_audio, frame_length):
    if input_audio.dtype != float:
        intmax = np.iinfo(input_audio.dtype).max
        input_audio = input_audio.astype(float) / intmax

    # todo: this will get slightly off over time.
    num_frames = (len(input_audio) // frame_length) - 2
    
    window = np.hanning(frame_length * 2)
    frames = np.zeros((num_frames, frame_length * 2), dtype=input_audio.dtype)

    for i in range(num_frames):
        a = input_audio[i*frame_length:i*frame_length+frame_length*2]
        frames[i] = (window * input_audio[i*frame_length:i*frame_length+frame_length*2])
        
    return frames

def find_best_match(carrier_bands, modulator_band):
    proj_indices = []
    coeffs = []
    pre, post, delta = None, None, None
    basis_epsilon = 5e-16
    while (delta is None or delta < 0) and ((not coeffs) or basis_epsilon < np.abs(coeffs[-1])) and ((not proj_indices) or len(proj_indices) == 1 or len(proj_indices) != MAX_BASIS_WIDTH): 
        dot_products = np.sum(carrier_bands * modulator_band, axis=1)
        max = np.argmax(dot_products)
        proj_indices.append(max)
        orth_band = carrier_bands[proj_indices[-1]]
        coeffs.append(np.sum(orth_band * modulator_band))
        decrement = coeffs[-1] * orth_band
        if not post is None:
            pre = post
        else:
            pre = np.sum(np.ones(len(modulator_band))* np.abs(modulator_band))
        modulator_band -= decrement
        post = np.sum(np.ones(len(modulator_band))* np.abs(modulator_band))
        delta = post - pre
    if np.abs(proj_indices[-1]) < basis_epsilon or np.abs(coeffs[-1]) < basis_epsilon:
        proj_indices.pop()
        coeffs.pop()
    padding = [0] * (MAX_BASIS_WIDTH - len(proj_indices))
    proj_indices = proj_indices + padding
    basis_array = np.asarray(proj_indices, dtype=np.int32)
    return (basis_array, coeffs + padding)


def file_type(path):
    # is the file at path an audio file, video file, or neither?
    return subprocess.run(
        [
            'ffprobe',
            '-loglevel', 'error',
            '-show_entries', 'stream=codec_type',
            '-of', 'csv=p=0',
            path
        ],
        capture_output=True,
        check=True,
        text=True
    ).stdout

def get_duration(path):
    return subprocess.run(
            [
                'ffprobe',
                '-i', path,
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
                path
            ],
            capture_output=True,
            check=True,
            text=True
        ).stdout

def build_output_video(frames_dir, outframes_dir, best_matches, basis_coefficients, framerate, output_path):
    print("building output video")
        
    for i, match_num in enumerate(best_matches):
        shutil.copy(frames_dir / f'frame{match_num+1:06d}.png', outframes_dir / f'frame{i:06d}.png')
    subprocess.run(
        [
            'ffmpeg',
            '-hide_banner',
            '-loglevel', 'error',
            '-y',
            '-framerate', str(framerate),
            '-i', outframes_dir / 'frame%06d.png',
            '-i', TEMP_DIR / 'out.wav',
            '-c:a', 'aac',
            '-shortest',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            output_path
        ],
        check=True
    )

def create_output_audio(best_matches, coefficients, modulator_audio, carrier_frames, modulator_frames, samples_per_frame):
    output_audio = np.zeros(modulator_audio.shape, dtype=float)
    def get_carrier(k,c):
        composite_carrier = None
        for index, element in enumerate(c):
            if element == 0:
                break
            if index == 0:
                composite_carrier = carrier_frames[k[index]]*element
            else:
                composite_carrier += carrier_frames[k[index]]*element
        return composite_carrier

    for i in range(len(modulator_frames)):
        composed_frame = get_carrier(best_matches[i],coefficients[i])
        output_audio[i*samples_per_frame : i*samples_per_frame + samples_per_frame*2] += composed_frame

    wavfile.write(TEMP_DIR / 'out.wav', INTERNAL_SAMPLERATE, output_audio)

def process(carrier_path, modulator_path, output_path):
    carrier_type = file_type(carrier_path)
    modulator_type = file_type(modulator_path)

    if 'video' in carrier_type:
        carrier_is_video = True
        print("Separating video frames")
        frames_dir = TEMP_DIR / 'frames'
        frames_dir.mkdir()
        subprocess.run(
            [
                'ffmpeg',
                '-loglevel', 'error',
                '-i', carrier_path,
                frames_dir / 'frame%06d.png'
            ],
            check=True
        )
        carrier_duration = float(get_duration(carrier_path))
        carrier_framecount = float(get_framecount(carrier_path))

        frame_length = carrier_duration / carrier_framecount        

    elif 'audio' in carrier_type:
        carrier_is_video = False
        frame_length = DEFAULT_FRAME_LENGTH
    else:
        print(f"Unrecognized file type: {carrier_path}. Should be audio or video")
        return

    if not (('video' in modulator_type) or ('audio' in modulator_type)):
        print(f"Unrecognized file type: {modulator_path}. Should be audio or video")
        return

    print("copying audio")
    subprocess.run(
        [
            'ffmpeg',
            '-loglevel', 'error',
            '-i', carrier_path,
            '-ac', '1',
            '-ar', str(INTERNAL_SAMPLERATE),
            TEMP_DIR / 'carrier.wav'
        ],
        check=True
    )
    subprocess.run(
        [
            'ffmpeg',
            '-loglevel', 'error',
            '-i', modulator_path,
            '-ac', '1',
            '-ar', str(INTERNAL_SAMPLERATE),
            TEMP_DIR / 'modulator.wav'
        ],
        check=True
    )

    print("reading audio")
    _, carrier_audio = wavfile.read(TEMP_DIR / 'carrier.wav')
    _, modulator_audio = wavfile.read(TEMP_DIR / 'modulator.wav')

    print("analyzing audio")
    samples_per_frame = int(frame_length * INTERNAL_SAMPLERATE)
    carrier_frames = make_frames(carrier_audio, samples_per_frame)
    modulator_frames = make_frames(modulator_audio, samples_per_frame)

    carrier_bands = make_normalized_bands(carrier_frames, BAND_WIDTH)
    modulator_bands = make_normalized_bands(modulator_frames, BAND_WIDTH)

    print("finding best matches")
    basis_coefficients = {}
    best_matches_MATRIX = np.zeros((len(modulator_bands), MAX_BASIS_WIDTH), np.int32) - np.ones((len(modulator_bands), MAX_BASIS_WIDTH), np.int32)
    for i in range(len(modulator_bands)):
        (basis, scalars) =find_best_match(carrier_bands,modulator_bands[i])
        best_matches_MATRIX[i] = basis
        basis_coefficients[i]= scalars


    print("creating output audio")
    create_output_audio(best_matches_MATRIX, basis_coefficients, modulator_audio, carrier_frames, modulator_frames, samples_per_frame)
    

    if carrier_is_video:
        outframes_dir = TEMP_DIR / 'outframes'
        outframes_dir.mkdir()
        build_output_video(frames_dir, outframes_dir, best_matches_MATRIX, basis_coefficients, 1/frame_length, output_path)
    else:
        subprocess.run(
            [
                'ffmpeg',
                '-loglevel', 'error',
                '-y', '-i', TEMP_DIR / 'out.wav',
                output_path
            ],
            check=True
        )

def main():
    # check required command line tools
    test_command(['ffmpeg', '-version'])
    test_command(['ffprobe', '-version'])
    
    parser = ArgumentParser()
    parser.add_argument('carrier_path', type=Path, metavar='carrier_track', help='path to an audio or video file that frames will be taken from')
    parser.add_argument('modulator_path', type=Path, metavar='modulator_track', help='path to an audio or video file that will be reconstructed using the carrier track')
    parser.add_argument('output_path', type=Path, metavar='output_file', help='path to file that will be written to; should have an audio or video file extension (such as .wav, .mp3, .mp4, etc.)')
    args = parser.parse_args()
    
    if not len(sys.argv) in (4,):
        print("Usage: python stammer.py <carrier track> <modulator track> <ouptut file>")
        return
    try:
        TEMP_DIR.mkdir()
    except FileExistsError:
        shutil.rmtree(TEMP_DIR)
        TEMP_DIR.mkdir()
    try:
        process(**vars(args))
        shutil.rmtree(TEMP_DIR)
    except Exception:
        shutil.rmtree(TEMP_DIR, ignore_errors=True)  # no guarantee that temp/ was created
        raise

if __name__ == '__main__':
    main()
