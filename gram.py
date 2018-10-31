# diagram creator

# native format is SVG + CSS
# use cairosvg to convert to PDF/PNG

class Scene:
    def __init__(self, root=None):
        if root is not None:
            self.set_root(root)

    def set_root(root):
        self.root = root

    def render(format='svg'):
        return self.root.render()

# the element contains its ID and positioning information relative to the parent
# positioning: absolute (negative is relative to top/right) and percentage (of parent size)
class Element:
    def __init__(self, i=None, x=None, y=None, h=None, w=None):
        self.i = i
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def render(self):
        pass

class Area(Element):
    def __init__(self, **kwargs):
        super().__init__(self, **kwargs)
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def render(self):
        pass

class Rows(Element):
    def __init__(self, **kwargs):
        super().__init__(self, **kwargs)
        self.rows = []

    def add_row(self, row):
        self.rows.append(row)

    def render(self):
        pass

class Cols(Element):
    def __init__(self, **kwargs):
        super().__init__(self, **kwargs)
        self.cols = []

    def add_row(self, col):
        self.cols.append(col)

    def render(self):
        pass

class Grid(Element):
    pass

class Text(Element):
    def __init__(self, text, **kwargs):
        super().__init__(self, **kwargs)
        self.text = text

    def render(self):
        pass

class Image(Element):
    pass

class Plot(Area):
    pass