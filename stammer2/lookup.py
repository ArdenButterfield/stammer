from annoy import AnnoyIndex
from audiosplitter import AudioSplitter
class Lookup:
    def __init__(self, audiosplitter):
        self.long_band_db = None
        self.short_band_db = None
        self.short_bands = None
        self.short_frames = None
        self.long_frames = None
        self.long_bands = None
        self.loaded = False
        self.audiosplitter = audiosplitter

    def load_audio(self, audio):
        self.long_frames = self.audiosplitter.make_frames(audio)
        self.short_frames = self.audiosplitter.make_short_frames(audio)

        self.long_bands = self.audiosplitter.make_normalized_bands(audio, False)
        shorts = self.audiosplitter.make_normalized_bands(audio, True)
        bands_per_row = self.audiosplitter.shortframe_resolution
        self.short_bands = shorts.reshape((len(shorts) * bands_per_row, len(shorts[0]) // bands_per_row))

        self.short_band_db = AnnoyIndex(len(self.short_bands[0]), 'angular')
        self.long_band_db = AnnoyIndex(len(self.long_bands[0]), 'angular')

        for i, band in enumerate(self.long_bands):
            self.long_band_db.add_item(i, band)
        for i, band in enumerate(self.short_bands):
            self.short_band_db.add_item(i, band)

        self.long_band_db.build(10)
        self.short_band_db.build(10)
        self.loaded = True

    def find_long_match(self, band):
        if self.loaded:
            match = self.long_band_db.get_nns_by_vector(band, 1)[0]
            return self.long_frames[match]
        raise ValueError("attempt to find match when no audio is loaded")

    def find_short_match(self, band):
        if self.loaded:
            match = self.short_band_db.get_nns_by_vector(band, 1)[0]
            col = match % self.audiosplitter.shortframe_resolution
            row = match // self.audiosplitter.shortframe_resolution
            return self.short_frames[col][row]
        raise ValueError("attempt to find match when no audio is loaded")
