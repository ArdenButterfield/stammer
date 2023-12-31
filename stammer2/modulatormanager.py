from audiosplitter import AudioSplitter

class ModulatorManager:
    def __init__(self, audiosplitter, audio):
        self.audiosplitter = audiosplitter
        self.audio = audio
        self.long_frames = audiosplitter.make_frames(audio)
        self.short_frames = audiosplitter.make_short_frames(audio)
        self.long_bands = audiosplitter.make_normalized_bands(audio, False)
        self.short_bands = audiosplitter.make_normalized_bands(audio, True)
        self.short_blocks = audiosplitter.identify_short_blocks(self.short_bands)

    def get_long_frame(self, frame):
        return self.long_frames[frame]

    def get_short_frame(self, frame, subframe):
        return self.short_frames[subframe][frame]

    def get_short_bands(self, frame, subframe):
        blocks = self.short_bands[frame]
        step = len(blocks) // self.audiosplitter.shortframe_resolution
        start = step * subframe
        end = start + step
        return blocks[start:end]
    def get_long_bands(self, frame):
        return self.long_bands[frame]

    def get_amplitude(self, frame):
        return max(abs(self.get_long_frame(frame)))

    def get_shortframe_amplitude(self, frame, subframe):
        return max(abs(self.get_short_frame(frame, subframe)))

    def is_short_block(self, frame):
        return self.short_blocks[frame]

    def get_num_frames(self):
        return len(self.long_frames)

    def scale_amp_to_frame(self, carrierframe, frame, short=False, subframe=None):
        if short:
            frame_amp = self.get_shortframe_amplitude(frame, subframe)
        else:
            frame_amp = self.get_amplitude(frame)
        carrier_amp = max(abs(carrierframe))
        return carrierframe * frame_amp / carrier_amp
