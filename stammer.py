import numpy as np
from scipy.io import wavfile
from os import system, popen
import sys

DEFAULT_FRAME_LENGTH = 1/25 # Seconds

BAND_WIDTH = 1.2

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

def make_frames(input_audio, fs, frame_length):
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

def find_best_match(source_bands, dest_band):
    dot_products = np.sum(source_bands * dest_band, axis=1)
    return np.argmax(dot_products)

def main():
    if not len(sys.argv) in (4,):
        print("Usage: python stammer.py <carrier track> <modulator track> <ouptut file>")
        return
    source_filename = sys.argv[1]
    destination_filename = sys.argv[2]
    output_filename = sys.argv[3]
    system("mkdir temp")

    source_type = popen(f"ffprobe -loglevel error -show_entries stream=codec_type -of csv=p=0 {source_filename}").read()
    dest_type = popen(f"ffprobe -loglevel error -show_entries stream=codec_type -of csv=p=0 {destination_filename}").read()

    source_is_video = False
    frame_length = DEFAULT_FRAME_LENGTH

    if 'video' in source_type:
        source_is_video = True
        print("Separating video frames")
        system("mkdir temp/frames")
        system(f"ffmpeg -loglevel error -i {source_filename} temp/frames/frame%06d.png")
        source_duration = popen(f"ffprobe -i {source_filename} -show_entries format=duration -v quiet -of csv='p=0'").read()
        source_framecount = popen(f"ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -print_format csv='p=0' {source_filename}").read()

        source_duration = float(source_duration)
        source_framecount = float(source_framecount)

        frame_length = source_duration / source_framecount        

    elif not 'audio' in source_type:
        print(f"Unrecognized file type: {source_filename}. Should be audio or video")
        return
    if not (('video' in dest_type) or ('audio' in dest_type)):
        print(f"Unrecognized file type: {destination_filename}. Should be audio or video")
        return

    print("copying audio")
    system(f"ffmpeg -loglevel error -i {source_filename} -ac 1 -ar 44100 temp/src.wav")
    system(f"ffmpeg -loglevel error -i {destination_filename} -ac 1 -ar 44100 temp/dest.wav")

    print("reading audio")
    fs, source_audio = wavfile.read('temp/src.wav')
    fs, dest_audio = wavfile.read('temp/dest.wav')

    print("analyzing audio")
    samples_per_frame = int(frame_length * fs)
    source_frames = make_frames(source_audio, fs, samples_per_frame)
    dest_frames = make_frames(dest_audio, fs, samples_per_frame)

    source_bands = make_normalized_bands(source_frames, BAND_WIDTH)
    dest_bands = make_normalized_bands(dest_frames, BAND_WIDTH)

    print("finding best matches")
    best_matches = []
    for i in range(len(dest_bands)):
        best_matches.append(find_best_match(source_bands,dest_bands[i]))


    print("creating output audio")
    output_audio = np.zeros(dest_audio.shape, dtype=float)

    for i in range(len(dest_frames)):
        source_frame = source_frames[best_matches[i]]
        dest_frame = dest_frames[i]
        dest_frame_amp = np.sqrt(np.sum(dest_frame*dest_frame))
        source_frame_amp = np.sqrt(np.sum(source_frame*source_frame))
        if (source_frame_amp == 0):
            continue
        rescaled_frame = source_frame * (dest_frame_amp / source_frame_amp)

        if (max(abs(rescaled_frame))) > 1:
            rescaled_frame /= max(abs(rescaled_frame))
        output_audio[i*samples_per_frame : i*samples_per_frame + samples_per_frame*2] += rescaled_frame


    wavfile.write('temp/out.wav',fs, output_audio)

    if source_is_video:
        print("building output video")
        system(f"mkdir temp/outframes")
        for i, match_num in enumerate(best_matches):
            system(f"cp temp/frames/frame{match_num+1:06d}.png temp/outframes/frame{i:06d}.png")
        system(f"ffmpeg -hide_banner -loglevel error -y -framerate {1/frame_length} -pattern_type glob -i 'temp/outframes/*.png' -i temp/out.wav -c:a aac -shortest -c:v libx264 -pix_fmt yuv420p {output_filename}")
    else:
        system(f"ffmpeg -i temp/out.wav {output_filename}")


if __name__ == '__main__':
    main()
    system("rm -rf temp")
