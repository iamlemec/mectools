# diagram creator

# native format is SVG + CSS
# use cairosvg to convert to PDF/PNG

# generate a random id
def gen_id(tag):
    uid = hex(random.getrandbits(32))[2:].zfill(8)
    return f'{tag}_{uid}'

def gen_css(sel, rules):
    css = f'{sel} {{\n'
    for k, v in rules.items():
        css += f'{k}: {v};'
    css += f'}}'
    return css

def measure(x):
    tx = type(x)
    if tx is float:
        value = f'{100*x}%'
    elif tx is int:
        value = f'{x}px'
    else:
        value = x
    return value

class Canvas:
    def __init__(self, root=None):
        if root is not None:
            self.set_root(root)

    def set_root(self, root):
        self.root = root

    def render(self, fmt='svg'):
        core = self.root.render()
        svg = f'<svg>\n{core}\n</svg>'
        if fmt == 'svg':
            return svg
        else:
            raise

# the element contains its ID and positioning information relative to the parent
# positioning: absolute (negative is relative to top/right) and percentage (of parent size)
class Element:
    def __init__(self, eid=None, tag='element', x=None, y=None, h=None, w=None):
        self.eid = eid
        self.tag = tag
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def position(self):
        eid = self.eid if self.eid is not None else gen_id(self.tag)
        style = {}

        # handle position
        if self.x is not None:
            key = 'left' if self.x >= 0 else 'right'
            style[key] = measure(self.x)
        if self.y is not None:
            key = 'top' if self.y >= 0 else 'bottom'
            style[key] = measure(self.y)

        # handle sizing
        if self.h is not None:
            style['height'] = measure(self.h)
        if self.w is not None:
            style['width'] = measure(self.w)

        return gen_css(eid, style)

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
        return f'<text>{self.text}</text>'

class Line(Element):
    pass

class Image(Element):
    pass

class Plot(Area):
    pass