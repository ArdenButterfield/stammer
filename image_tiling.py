from enum import Enum
import numpy as np
from PIL import Image
from scratch_fractions.from_scratch import as_array
from pathlib import Path

class Side(Enum):
    LEFT = 1
    BOTTOM = 2
    RIGHT = 3
    TOP = 4

    @classmethod
    def next_ccw(cls, current):
        if current == cls.LEFT:
            return cls(2)
        if current == cls.BOTTOM:
            return cls(3)
        if current == cls.RIGHT:
            return cls(4)
        if current == cls.TOP:
            return cls(1)


class Tiling:
    def __init__(self, height: int, width: int) -> None:
        self.top = 0
        self.bottom = height
        self.left = 0
        self.right = width
        if width > height:
            self.side = Side.LEFT
        else:
            self.side = Side.BOTTOM
        
    def get_image_placement(self):
        pos_height, pos_width, pos_x0, pos_y0 = None, None, None, None
        if self.side in [Side.LEFT, Side.RIGHT]:
            pos_height = self.bottom - self.top
            pos_width = (self.right - self.left) // 2
            pos_y0 = self.top
            if self.side == Side.LEFT:
                pos_x0 = self.left
                self.left += pos_width
            else:
                self.right -= pos_width
                pos_x0 = self.right
        else:
            pos_width = self.right - self.left
            pos_height = (self.bottom - self.top) // 2
            pos_x0 = self.left
            if self.side == Side.TOP:
                pos_y0 = self.top
                self.top += pos_height
            else:
                self.bottom -= pos_height
                pos_y0 = self.bottom
        self.side = Side.next_ccw(self.side)
        return (pos_x0, pos_y0, pos_width, pos_height)
    
def main():
    a,_ = as_array(2.0/7.0)
    b,_ = as_array(3.0/7.0)
    c,_ = as_array(4.0/7.0)
    sum_list = [x + y for x,y in zip(a,b)]
    sum_list = [x + y for x,y in zip(sum_list,c)]
    
    #get ready to do some image processing
    output = Image.new('RGB', (320, 240))
    
    directory = Path(__file__).parent.absolute()
    kb_path = directory / 'frame000026.png'
    ch_path = directory / 'frame000013.png'
    sk_path = directory / 'frame000001.png'

    output_path = directory / 'composite_image.png'

    krusty_burger = Image.open(kb_path)
    chalmers = Image.open(ch_path)
    skinner = Image.open(sk_path)

    print(f'Zeros: {sum_list.count(0)}')
    print(f'Ones: {sum_list.count(1)}')
    print(f'Total = {sum_list.count(0)+sum_list.count(1)}')
    print(f'Number of elements: {len(sum_list)}')

    tiling = Tiling(240,320)

    for i in np.arange(1,8):
        debug_path = directory / f'debug_{i:>02}.png'
        if a[i]:
            x0, y0, w, h = tiling.get_image_placement()
            tb = krusty_burger.copy()
            tb.thumbnail((w,h))
            output.paste(tb, (x0,y0))
            output.save(debug_path)
            continue
        elif b[i]:
            x0, y0, w, h = tiling.get_image_placement()
            tb = chalmers.copy()
            tb.thumbnail((w,h))
            output.paste(tb, (x0,y0))
            output.save(debug_path)
            continue
        if c[i]:
            x0, y0, w, h = tiling.get_image_placement()
            tb = skinner.copy()
            tb.thumbnail((w,h))
            output.paste(tb, (x0,y0))
            output.save(debug_path)

    output.save(output_path)    




if __name__ == '__main__':
    main()

            
            

