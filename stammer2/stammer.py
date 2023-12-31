import numpy as np
from scipy.io import wavfile

from lookup import Lookup
from audiosplitter import AudioSplitter
from modulatormanager import ModulatorManager

def reconstruct(carrier: np.ndarray, modulator: np.ndarray, audiosplitter: AudioSplitter):
    lookup = Lookup(audiosplitter)
    lookup.load_audio(carrier)

    modulator_manager = ModulatorManager(audiosplitter, modulator)

    output = np.zeros(modulator.shape)
    pos = 0
    for frame in range(modulator_manager.get_num_frames()):
        if modulator_manager.is_short_block(frame):
            for sub in range(audiosplitter.shortframe_resolution):
                match = lookup.find_short_match(modulator_manager.get_short_bands(frame, sub))
                output[pos:pos+len(match)] += modulator_manager.scale_amp_to_frame(match, frame, True, sub)
                pos += audiosplitter.samples_per_frame // audiosplitter.shortframe_resolution
        else:
            match = lookup.find_long_match(modulator_manager.get_long_bands(frame))
            output[pos:pos+len(match)] += modulator_manager.scale_amp_to_frame(match, frame)
            pos += audiosplitter.samples_per_frame
    return output

def stammer(carrier_filename, modulator_filename, output_filename):
    fsc, carrier = wavfile.read(carrier_filename)
    fsm, modulator = wavfile.read(modulator_filename)

    carrier = carrier[:,0]
    modulator = modulator[:,0]

    audiosplitter = AudioSplitter(fsm, 1 / 30)
    output = reconstruct(carrier, modulator, audiosplitter)
    wavfile.write(output_filename, fsm, output)