from pathlib import Path
from decay_cache import DecayCache
from audio_matching import AudioMatcher
import subprocess
import io

def apply_color_mode(ffmpeg_call,color_mode):
    color_strs = []
    if color_mode == '8fast':
        # color_strs = ['-pix_fmt', 'pal8', '-sws_dither', 'ed']
        color_strs = ['-pix_fmt', 'pal8']
    elif color_mode == '8full':
        color_strs = ['-vf', 'split[s0][s1];[s0]palettegen=256:0:stats_mode=single[p];[s1][p]paletteuse=new=1:dither=bayer']

    idx = ffmpeg_call.index('include_color_mode')
    ffmpeg_call.pop(idx)
    if color_mode != "full":
        for i, s in enumerate(color_strs):
            ffmpeg_call.insert(idx+i,s)
    return ffmpeg_call

class VideoHandler:
    def __init__(self, carrier_path: Path, output_path: Path, temp_dir: Path, matcher: AudioMatcher, framecount: int, frame_length: float, color_mode):
        self.matcher = matcher
        self.best_match_count = int(len(matcher.get_best_matches()) * matcher.frame_length / frame_length)

        self.carrier_path = carrier_path
        self.output_path = output_path
        self.temp_dir = temp_dir
        self.frames_dir = self.temp_dir / 'frames' 
        self.outframes_dir = self.temp_dir / 'outframes'

        self.framecount = int(framecount)
        self.frame_length = frame_length

        self.color_mode = color_mode

        self.frames_written = 0
    
    def get_frame(self,idx):
        try:
            assert(idx < self.framecount)
        except AssertionError:
            print("ERROR:")
            print(f"STAMMER just tried to use carrier frame {idx}")
            print(f"but carrier only has {self.framecount} frames.")
            print()
            print("This is a critical known issue with how carrier frames are handled.")
            print("Please report STAMMER's output at this link:\nhttps://github.com/ArdenButterfield/stammer/issues/62")
            print("\nQuitting.")
            quit()
    
    def write_frame(self):
        self.frames_written += 1
        self.print_progress()

    def complete(self):
        print(end="\n")

    def get_progress_strings(self) -> list[str]:
        strings: list[str] = []
        strings.append(str(self.frames_written) + "/" + str(self.best_match_count))
        
        return strings
    
    def progress_strings_separated(self):
        ps = self.get_progress_strings()
        if len(ps) == 1: return ps[0]
        return " . ".join(self.get_progress_strings())
    
    def print_progress(self):
        print(self.progress_strings_separated(),end='      \r')

def get_output_cmd(handler: VideoHandler,input):
    cmd = [
        'ffmpeg',
        '-v', 'quiet',
        '-y',
        '-framerate', str(1.0/handler.frame_length),
        '!inputs!',
        '-c:a', 'aac',
        '-c:v', 'libx264',
        '-crf', '20',
        '-pix_fmt', 'yuv420p',
        '-shortest',
        str(handler.output_path)
    ]

    def replace(value, list):
        idx = cmd.index(value)
        cmd.pop(idx)
        for i, x in enumerate(list): cmd.insert(idx+i,x)

    replace('!inputs!',input)
    
    return cmd

class VideoHandlerDisk(VideoHandler):
    def get_frame(self,idx):
        super().get_frame(idx)
        f = open(self.frames_dir / f"frame{idx:06d}.png", 'rb')
        return f

    def write_frame(self,idx,frame: io.BytesIO):
        super().write_frame()
        frame.seek(0)
        f = open(self.outframes_dir / f"frame{idx:06d}.png", 'wb')
        f.write(frame.read())
        f.close()
    
    def complete(self):
        super().complete()
        call = get_output_cmd(
            self,
            input=[
                '-stats',
                '-i', str(self.outframes_dir / 'frame%06d.png'),
                '-i', str(self.temp_dir / 'out.wav')
            ]
        )
        subprocess.run(call,check=True)


PNG_MAGIC = int("89504e47",16).to_bytes(4,byteorder='big')
JPG_MAGIC = int("ffd8ffe0",16).to_bytes(4,byteorder='big')

class VideoHandlerMem(VideoHandler):
    def __init__(self, *args):
        super().__init__(*args)
        self.cache = DecayCache(self.framecount)
        self.cache_hits = 0

        self.frame_length_max = self.frame_length / max(self.frame_length,self.matcher.frame_length)
        self.frames_backtrack = 0
        self.frames_lookahead = int(max(1.0/self.frame_length_max,2))
        
        self.out_proc = self.__create_output_proc()

    def set_min_cached_frames(self,mcf):
        # if a decayed frame is about to be used, we fetch the frame + this amount of frames around it
        # it's likely that the modulator will generally fetch similar frames
        self.frames_backtrack = 0
        self.frames_lookahead = int(max(1.0/self.frame_length_max,mcf))
        
        # This enforces that cached frame count cannot exceed decay time
        # i.e. if decay time is 500 frames, max cached frames will be 500
        self.cache.decay /= self.frames_lookahead

    def __get_video_frames_mem(self,start_frame: int,end_frame: int):
        start_time = start_frame * self.frame_length
        call = apply_color_mode([
                'ffmpeg',
                '-loglevel', 'error',
                '-ss', str(start_time),
                '-i', self.carrier_path, '-c:v', 'png',
                'include_color_mode',
                '-frames:v', str(end_frame-start_frame),
                '-f', 'image2pipe',
                '-'
            ],self.color_mode)
        
        return subprocess.check_output(call)
    
    def __get_frame_ofs_index(frames: bytes, index):
        cur = 0
        total_idx = 0
        idx = 0
        while True:
            check = frames.find(PNG_MAGIC, cur)
            if check == -1: break
            idx = check
            if idx > cur: total_idx += 1
            if total_idx == index: break
            cur = max(cur,idx+1)

        return idx

    def __get_frame_slice(frames: bytes, index: int):
        start = VideoHandlerMem.__get_frame_ofs_index(frames,index)
        end = VideoHandlerMem.__get_frame_ofs_index(frames,index+1)
        if start == end: end = len(frames)

        return frames[start:end]

    def __cache_decayed_frames(self,match_id):
        def grow_to_nondecayed(min,max):
            for idx in range(min,max):
                if self.cache.item_usable(idx): return idx
            return max

        min_f = max(match_id-self.frames_backtrack,0)
        max_f = min(match_id+self.frames_lookahead,self.framecount)

        min_f = grow_to_nondecayed(min_f,match_id)
        max_f = grow_to_nondecayed(match_id,max_f)

        decoded_frames = self.__get_video_frames_mem(min_f,max_f)
        
        new_frame_ids = range(min_f, max_f)
        self.cache.clear(new_frame_ids)
        for idx in new_frame_ids:
            frame_slice = VideoHandlerMem.__get_frame_slice(decoded_frames,idx-min_f)
            self.cache.set_item(idx,frame_slice)
    
    def get_frame(self,idx) -> io.BytesIO:
        super().get_frame(idx)
        self.cache.process()
        
        if self.cache.item_usable(idx):
            self.cache_hits += 1
            self.cache.clear([idx])
        else:
            self.__cache_decayed_frames(idx)
        
        frame = self.cache.items[idx].item
        return io.BytesIO(frame)

    def write_frame(self,idx,frame: io.BytesIO):
        super().write_frame()
        frame.seek(0)
        f = frame.read()
        self.out_proc.stdin.write(f)

    def get_progress_strings(self):
        strs = super().get_progress_strings()
        strs.append(f"{self.cache_hits} cache hits")
        strs.append(f"{self.framecount-self.cache.decayed_items}/{self.framecount} cached frames")
        return strs
    
    def __create_output_proc(self):
        call = get_output_cmd(
            self,
            [
                '-f', 'image2pipe', '-i', 'pipe:',
                '-i', str(self.temp_dir / 'out.wav')
            ]
        )

        return subprocess.Popen(
            call,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL
        )
    
    def complete(self):
        super().complete()

        # properly close ffmpeg
        self.out_proc.communicate()