import matplotlib.pyplot as plt
import numpy as np


def _normalize_bands(bands):
    vector_magnitudes = np.sqrt((bands * bands).sum(axis=1))
    vector_magnitudes[vector_magnitudes == 0] = 1
    normalized_bands = bands / vector_magnitudes[:, None]
    return normalized_bands


def _identify_short_blocks(frames):
    transforms = np.fft.fft(frames)
    short_blocks = np.zeros(len(transforms), dtype=bool)
    for i in range(1, len(transforms)):
        prev = abs(transforms[i - 1])
        curr = abs(transforms[i])
        currmax = np.max(curr)
        diff = prev - curr
        diff /= currmax
        # if np.sum(diff * diff) > 500:
        #      short_blocks[i] = 1
        if max(abs(frames[i])) > max(abs(frames[i - 1])) * 2:
            short_blocks[i] = 1
    return short_blocks

class AudioSplitter:
    def __init__(self, samplerate, frame_length):
        self.frame_length = frame_length  # seconds

        # Let's try splitting on critical bands
        self.long_window_cutoffs = (0, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320,
                                    2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500)  # Hz

        self.shortframe_resolution = 3
        self.short_window_cutoffs = [self.long_window_cutoffs[i] for i in range(0,
                                                                                len(self.long_window_cutoffs),
                                                                                self.shortframe_resolution)]

        self.samplerate = samplerate
        self.samples_per_frame = int(self.frame_length * self.samplerate)

        self.samples_per_frame -= self.samples_per_frame % self.shortframe_resolution

        transition_length = self.samples_per_frame // self.shortframe_resolution
        self.short_window = np.hanning(transition_length * 2)
        transition = self.short_window[:transition_length]
        self.long_window = np.zeros(self.samples_per_frame * 2)
        self.long_window[:transition_length] = transition
        self.long_window[transition_length:self.samples_per_frame] = 1
        self.long_window[self.samples_per_frame:self.samples_per_frame + transition_length] = 1 - transition
        self.short_window_starts = [i * transition_length for i in range(self.shortframe_resolution)]

    def make_base_frames(self, input_audio):
        if input_audio.dtype != float:
            intmax = np.iinfo(input_audio.dtype).max
            input_audio = input_audio.astype(float) / intmax
        num_frames = (len(input_audio) // self.samples_per_frame) - 2
        left = np.resize(input_audio, (num_frames, self.samples_per_frame))
        right = np.resize(input_audio[self.samples_per_frame:], (num_frames, self.samples_per_frame))
        return np.hstack((left, right))

    def make_frames(self, input_audio):
        base_frames = self.make_base_frames(input_audio)
        windowed_frames = base_frames * self.long_window
        return windowed_frames

    def make_short_frames(self, input_audio):
        base_frames = self.make_base_frames(input_audio)

        shortframe_length = self.samples_per_frame // self.shortframe_resolution

        short_frames = []

        for i in range(self.shortframe_resolution):
            start = i * shortframe_length
            column = base_frames[:, start:start + len(self.short_window)]
            short_frames.append(column * self.short_window)

        return short_frames

    def make_bands(self, frames_input, cutoffs):
        transforms = np.fft.fft(frames_input)
        spectra = abs(transforms[:, 1:len(transforms[0]) // 2])
        split_points = [int(freq * len(transforms[0]) / self.samplerate) for freq in cutoffs]
        section_lengths = [split_points[i + 1] - split_points[i] for i in range(len(split_points) - 1)] + [
            len(spectra[0]) - split_points[-1]]

        return np.divide(np.add.reduceat(spectra, split_points, axis=1), section_lengths)

    def identify_short_blocks(self, normalized_short_bands):
        shortframe_length = len(normalized_short_bands[0]) // self.shortframe_resolution

        first_subblock = normalized_short_bands[:, :shortframe_length]
        differences = np.zeros(len(normalized_short_bands))
        for i in range(1, self.shortframe_resolution):
            subblock = normalized_short_bands[:, i * shortframe_length : i * shortframe_length + shortframe_length]
            diff = subblock - first_subblock
            avg_diff = np.sum(diff * diff, axis=1) / len(diff)
            differences += avg_diff
        return differences > 0.001

    def make_normalized_bands(self, input_audio, make_short_frames=False):
        if make_short_frames:
            frames_inputs = self.make_short_frames(input_audio)
            bands = np.hstack([self.make_bands(f, self.short_window_cutoffs) for f in frames_inputs])
        else:
            frames_input = self.make_frames(input_audio)
            bands = self.make_bands(frames_input,
                                    self.short_window_cutoffs if make_short_frames else self.long_window_cutoffs)
        return _normalize_bands(bands)

    