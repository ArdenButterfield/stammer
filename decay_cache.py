import gc

class Frame:
    frame: bytes = None
    timer: int = 0

    def __init__(self,
                 frame: bytes = None,
                 timer: int = 0):
        self.frame = frame
        self.timer = timer

class DecayCache:
    array: dict[Frame] = {}
    decay: int = 100
    bad_frames = 0

    def __init__(self,size: int):
        for i in range(size):
            self.array[i] = Frame()
    
    def reinit(self):
        self.__init__(len(self.array))
    
    def item_decayed(self, i: int):
        return self.array[i].timer == 0

    def item_usable(self, i: int):
        return self.array[i].frame != None
    
    def __set_timer(self,i: int,time: int):
        self.array[i].timer = time
    
    def reset_timer(self,i: int):
        self.__set_timer(i,self.decay)
    
    def process(self):
        self.decayed_frames = 0
        for i in range(len(self.array)):
            self.array[i].timer = max(0,self.array[i].timer-1)
            
            if self.item_decayed(i):
                self.decayed_frames += 1
                self.array[i] = Frame()
        
        # free decayed items from memory
        gc.collect()
    

    def clear(self,frame_ids: list[int]):
        for id in frame_ids:
            if self.item_usable(id):
                self.reset_timer(id)
    
    def set_frame(self,i: int,frame: bytes,):
        self.array[i] = Frame(frame,self.decay)
    
