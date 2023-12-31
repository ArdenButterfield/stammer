from audiosplitter import AudioSplitter
from stammer import stammer
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np

def plotwindow():
    a = AudioSplitter(44100, 1 / 30)

    plt.plot(a.long_window)
    plt.show()

def test_identify_shortbands():
    a = AudioSplitter(44100, 1 / 30)
    fs, audio = wavfile.read("./testinput/drums10sec.wav")
    audio = np.sum(audio, axis=1)
    bands = a.make_normalized_bands(audio, True)
    shorts = a.identify_short_blocks(bands)
    shortsmask = np.repeat(shorts, a.samples_per_frame)
    plt.plot(audio[:len(shortsmask)] * 1 - shortsmask)
    plt.plot(audio[:len(shortsmask)] * shortsmask)
    plt.show()

def test_base_frames():
    a = AudioSplitter(44100, 1 / 3000)
    fs, audio = wavfile.read("./testinput/kyoto2sec.wav")
    audio = np.sum(audio, axis=1)[:1000]
    frames = a.make_base_frames(audio)
    plt.imshow(frames)
    plt.show()


def test_normalized_bands():
    a = AudioSplitter(44100, 1 / 30)
    fs, audio = wavfile.read("./testinput/kyoto2sec.wav")
    audio = np.sum(audio, axis=1)
    bands = a.make_normalized_bands(audio, True)
    plt.imshow(bands)
    plt.show()

def test_frames():
    a = AudioSplitter(44100, 1 / 3000)
    fs, audio = wavfile.read("./testinput/kyoto2sec.wav")
    audio = np.sum(audio, axis=1)[:1000]
    frames = a.make_frames(audio)
    plt.imshow(frames)
    plt.show()

def test_short_frames():
    a = AudioSplitter(44100, 1 / 3000)
    fs, audio = wavfile.read("./testinput/kyoto2sec.wav")
    audio = np.sum(audio, axis=1)[:1000]
    frames = a.make_short_frames(audio)
    plt.imshow(frames[0])
    plt.show()

def test_stammer():
    stammer("./testinput/kyoto10sec.wav", "./testinput/drums10sec.wav", "testout.wav")

test_stammer()