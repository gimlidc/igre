class Image:
    def __init__(self, height, width, c_x=-1, c_y=-1):
        self.height = height
        self.width = width
        self.c_x = c_x*2./height-1 if c_x != -1 else 0.
        self.c_y = c_y*2./width-1 if c_y != -1 else 0.


def init(h, w, x=-1, y=-1):
    global image
    image = Image(h, w, x, y)
